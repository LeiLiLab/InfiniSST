#!/usr/bin/env python3
"""
Autoresearch experiment runner for dual-tower retrieval model.

Submits train.sh via sbatch, polls until completion, extracts metrics
from the Slurm output log, and prints a standardized summary.

Usage:
    python experiment.py                          # submit and wait
    python experiment.py --dry_run                # print what would run
    python experiment.py --dependency 43116       # wait for job 43116 first
    python experiment.py > run.log 2>&1           # redirect for agent use
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

# ======Configuration=====
TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.sh")
LOG_DIR = "/mnt/gemini/data1/jiaxuanluo/logs/autoresearch"
POLL_INTERVAL_SECONDS = 60
MAX_WAIT_SECONDS = 14400
# ======Configuration=====


def run_cmd(cmd: str) -> str:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip()


def submit_job(dependency: str = "") -> int:
    """Submit train.sh via sbatch, return job ID."""
    cmd = "sbatch"
    if dependency:
        cmd += f" --dependency=afterok:{dependency}"
    cmd += f" --parsable {TRAIN_SCRIPT}"

    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, (
        f"sbatch failed (rc={result.returncode}): {result.stderr}"
    )
    job_id_str = result.stdout.strip().split(";")[0]
    job_id = int(job_id_str)
    return job_id


def wait_for_job(job_id: int) -> bool:
    """Poll squeue until job disappears. Returns True if completed, False if timeout."""
    t0 = time.time()
    while (time.time() - t0) < MAX_WAIT_SECONDS:
        output = run_cmd(f"squeue -j {job_id} -h -o '%T' 2>/dev/null")
        if not output:
            return True
        state = output.strip().split("\n")[0].strip()
        elapsed_min = (time.time() - t0) / 60
        print(
            f"[POLL] job={job_id} state={state} elapsed={elapsed_min:.1f}min",
            flush=True,
        )
        time.sleep(POLL_INTERVAL_SECONDS)
    return False


def find_log_file(job_id: int) -> str:
    """Find the Slurm output log file for a given job ID."""
    pattern = os.path.join(LOG_DIR, f"{job_id}_ar_wr.out")
    if os.path.isfile(pattern):
        return pattern
    # Fallback: search for any file starting with the job ID
    for fname in os.listdir(LOG_DIR):
        if fname.startswith(str(job_id)) and fname.endswith(".out"):
            return os.path.join(LOG_DIR, fname)
    return ""


def parse_summary(log_path: str) -> dict[str, str]:
    """Parse the --- summary block from training output."""
    metrics: dict[str, str] = {}
    if not log_path or not os.path.isfile(log_path):
        return metrics

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    in_summary = False
    for line in lines:
        line = line.strip()
        if line == "---":
            in_summary = True
            continue
        if in_summary and ":" in line:
            key, _, val = line.partition(":")
            metrics[key.strip()] = val.strip()
        elif in_summary and not line:
            break
    return metrics


def parse_best_metrics_from_log(log_path: str) -> dict[str, float]:
    """Scan the full log for [BEST] and [BEST_SECONDARY] lines to get peak metrics."""
    best: dict[str, float] = {
        "best_acl6060_r10_gs1000": 0.0,
        "best_acl6060_r10_gs10000": 0.0,
    }
    if not log_path or not os.path.isfile(log_path):
        return best

    best_re = re.compile(
        r"\[BEST\]\s+([\w/@ _]+)=([\d.]+)"
    )
    best_sec_re = re.compile(
        r"\[BEST_SECONDARY\]\s+([\w/@ _]+)=([\d.]+)"
    )
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = best_re.search(line)
            if m:
                best["best_acl6060_r10_gs1000"] = max(
                    best["best_acl6060_r10_gs1000"], float(m.group(2))
                )
            m2 = best_sec_re.search(line)
            if m2:
                best["best_acl6060_r10_gs10000"] = max(
                    best["best_acl6060_r10_gs10000"], float(m2.group(2))
                )
    return best


def check_exit_code(job_id: int) -> int:
    """Get the Slurm job exit code."""
    output = run_cmd(
        f"sacct -j {job_id} --format=ExitCode --noheader -P 2>/dev/null"
    )
    if not output:
        return -1
    first_line = output.strip().split("\n")[0]
    code_str = first_line.split(":")[0]
    try:
        return int(code_str)
    except ValueError:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autoresearch experiment runner"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be submitted without actually running",
    )
    parser.add_argument(
        "--dependency", type=str, default="",
        help="Slurm job ID to wait for before starting (afterok dependency)",
    )
    args = parser.parse_args()

    assert os.path.isfile(TRAIN_SCRIPT), f"train.sh not found: {TRAIN_SCRIPT}"
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.dry_run:
        print(f"[DRY RUN] Would submit: sbatch {TRAIN_SCRIPT}")
        if args.dependency:
            print(f"[DRY RUN] With dependency: afterok:{args.dependency}")
        print(f"[DRY RUN] Log dir: {LOG_DIR}")
        return

    # Submit
    print(f"[SUBMIT] Submitting {TRAIN_SCRIPT}...", flush=True)
    job_id = submit_job(dependency=args.dependency)
    print(f"[SUBMIT] Job ID: {job_id}", flush=True)

    # Wait
    print(f"[WAIT] Polling every {POLL_INTERVAL_SECONDS}s...", flush=True)
    completed = wait_for_job(job_id)
    if not completed:
        print(f"[TIMEOUT] Job {job_id} did not complete within {MAX_WAIT_SECONDS}s")
        print("---")
        print("status: timeout")
        sys.exit(1)

    # Check exit code
    exit_code = check_exit_code(job_id)
    log_path = find_log_file(job_id)
    print(f"[DONE] Job {job_id} exit_code={exit_code} log={log_path}", flush=True)

    if exit_code != 0:
        print("---")
        print("status: crash")
        print(f"exit_code: {exit_code}")
        if log_path:
            print(f"log_file: {log_path}")
            # Print last 50 lines of error log for debugging
            err_path = log_path.replace(".out", ".err")
            if os.path.isfile(err_path):
                with open(err_path, "r", errors="replace") as f:
                    lines = f.readlines()
                print("--- last 50 lines of stderr ---")
                for line in lines[-50:]:
                    print(line, end="")
        sys.exit(1)

    # Parse metrics from the summary block
    summary = parse_summary(log_path)
    best_from_log = parse_best_metrics_from_log(log_path)

    acl_gs10k = summary.get(
        "best_acl6060_recall10_gs10000",
        str(best_from_log.get("best_acl6060_r10_gs10000", "0.0")),
    )
    dev_gs10k = summary.get(
        "best_dev_recall10_gs10000",
        "0.0",
    )
    peak_vram = summary.get("peak_vram_mb", "0.0")
    total_steps = summary.get("total_steps", "0")
    wiki_rank = summary.get("wiki_rank", "0")
    clean_ratio = summary.get("clean_ratio", "-1.0")
    training_sec = summary.get("training_seconds", "0.0")

    memory_gb = float(peak_vram) / 1024 if peak_vram != "0.0" else 0.0

    # Print standardized summary
    print("---")
    print(f"acl6060_r10_gs10k: {acl_gs10k}")
    print(f"dev_r10_gs10k:     {dev_gs10k}")
    print(f"memory_gb:         {memory_gb:.1f}")
    print(f"wiki_rank:         {wiki_rank}")
    print(f"clean_ratio:       {clean_ratio}")
    print(f"total_steps:       {total_steps}")
    print(f"training_seconds:  {training_sec}")
    print(f"status:            ok")
    print(f"log_file:          {log_path}")


if __name__ == "__main__":
    main()
