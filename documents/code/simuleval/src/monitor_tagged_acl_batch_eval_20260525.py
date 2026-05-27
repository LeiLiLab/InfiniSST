#!/usr/bin/env python3
"""Monitor a detached same-LM batched SimulEval run.

The monitor is intentionally read-only: it polls launcher logs and output
artifacts, writes a compact TSV status stream, and optionally sends a
notification after the top-level launcher exits.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import signal
import subprocess
import time
from pathlib import Path


METRIC_ALIASES = {
    "BLEU": ("BLEU", "bleu"),
    "StreamLAAL": ("StreamLAAL", "stream_laal", "LAAL", "laal"),
    "StreamLAAL_CA": ("StreamLAAL_CA", "stream_laal_ca", "LAAL_CA", "laal_ca"),
    "TERM_ACC": ("TERM_ACC", "Term_Acc", "term_acc", "TERM_ACCURACY"),
    "TERM_CORRECT": ("TERM_CORRECT", "term_correct", "CORRECT"),
    "TERM_TOTAL": ("TERM_TOTAL", "term_total", "TOTAL"),
}

ERROR_RE = re.compile(
    r"(Traceback|CUDA out of memory|RuntimeError|ValueError|ZMQError|No space left|Killed)",
    re.IGNORECASE,
)
SHARD_RE = re.compile(r"Loading safetensors checkpoint shards:.*?(\d+)%.*?\|\s*(\d+)/(\d+)")
STEP_RE = re.compile(r"\[STEP\]\s+(\d+)\s+ready=(\d+)\s+active=(\d+)")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pgrep(pattern: str) -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def read_tail(path: Path, max_bytes: int = 65536) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="replace")


def line_count(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        return str(sum(1 for _ in handle))


def find_lm_files(output_base: Path, lm: int, name: str) -> list[Path]:
    if not output_base.exists():
        return []
    matches = []
    lm_dir_re = re.compile(rf"(^|_)lm{lm}(_|$)")
    for path in output_base.rglob(name):
        if lm_dir_re.search(path.parent.name):
            matches.append(path)
    return sorted(matches)


def parse_eval_results(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        row = next(reader, None)
    if not row:
        return {}
    out: dict[str, str] = {}
    for canonical, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in row and row[alias] not in (None, ""):
                out[canonical] = row[alias]
                break
        else:
            out[canonical] = ""
    return out


def latest_shard_status(text: str) -> str:
    matches = SHARD_RE.findall(text.replace("\r", "\n"))
    if not matches:
        return ""
    pct, cur, total = matches[-1]
    return f"loading_shards_{cur}_of_{total}_{pct}pct"


def latest_step_status(text: str) -> str:
    matches = STEP_RE.findall(text)
    if not matches:
        return ""
    step, ready, active = matches[-1]
    return f"generation_step_{step}_ready_{ready}_active_{active}"


def summarize_lm(output_base: Path, log_root: Path, lm: int) -> dict[str, str]:
    log_dir = log_root / f"lm{lm}"
    err_tail = read_tail(log_dir / "launcher.err")
    out_tail = read_tail(log_dir / "launcher.out")
    combined = f"{out_tail}\n{err_tail}"

    eval_files = find_lm_files(output_base, lm, "eval_results.tsv")
    instances_files = find_lm_files(output_base, lm, "instances.log")
    strip_files = find_lm_files(output_base, lm, "instances.strip_term.log")
    eval_path = eval_files[-1] if eval_files else None
    instances_path = instances_files[-1] if instances_files else None
    strip_path = strip_files[-1] if strip_files else None

    error_match = ERROR_RE.search(combined)
    stage = ""
    note = ""
    if eval_path:
        stage = "completed_eval_results"
    elif error_match:
        stage = "error_seen_in_log"
        note = error_match.group(1)
    elif "[ALL DONE]" in combined:
        stage = "all_done_marker_no_eval_results"
    elif strip_path:
        stage = "strip_log_written"
    elif instances_path:
        stage = "instances_log_written"
    else:
        stage = latest_step_status(combined) or latest_shard_status(combined) or "running_no_result_yet"

    metrics = parse_eval_results(eval_path) if eval_path else {}
    pids = pgrep(rf"batched_vllm_rag_eval.py .*(--lms {lm})( |$)")
    engine_pids = pgrep(f"VLLM::EngineCore")

    return {
        "timestamp_utc": utc_now(),
        "lm": str(lm),
        "status": "done" if eval_path else ("error" if error_match else "running"),
        "alive_pids": ",".join(pids),
        "engine_pids_all": ",".join(engine_pids),
        "stage": stage,
        "eval_results": str(eval_path) if eval_path else "",
        "instances_rows": line_count(instances_path) if instances_path else "",
        "strip_rows": line_count(strip_path) if strip_path else "",
        "BLEU": metrics.get("BLEU", ""),
        "StreamLAAL": metrics.get("StreamLAAL", ""),
        "StreamLAAL_CA": metrics.get("StreamLAAL_CA", ""),
        "TERM_ACC": metrics.get("TERM_ACC", ""),
        "TERM_CORRECT": metrics.get("TERM_CORRECT", ""),
        "TERM_TOTAL": metrics.get("TERM_TOTAL", ""),
        "note": note,
    }


def append_rows(report_tsv: Path, rows: list[dict[str, str]]) -> None:
    report_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "lm",
        "status",
        "alive_pids",
        "engine_pids_all",
        "stage",
        "eval_results",
        "instances_rows",
        "strip_rows",
        "BLEU",
        "StreamLAAL",
        "StreamLAAL_CA",
        "TERM_ACC",
        "TERM_CORRECT",
        "TERM_TOTAL",
        "note",
    ]
    exists = report_tsv.exists()
    with report_tsv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def notify(cmd: str, workspace: str, message: str) -> None:
    if not cmd:
        return
    try:
        subprocess.run(
            [cmd, "--workspace", workspace, message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-pid", type=int, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--report-tsv", type=Path, required=True)
    parser.add_argument("--lms", default="1,2,3,4")
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--workspace", default=os.getcwd())
    parser.add_argument("--notify-cmd", default="")
    parser.add_argument("--exit-on-error", action="store_true")
    args = parser.parse_args()

    lms = [int(x) for x in args.lms.split(",") if x.strip()]
    final_status = "unknown"
    while True:
        rows = [summarize_lm(args.output_base, args.log_root, lm) for lm in lms]
        append_rows(args.report_tsv, rows)
        done = sum(row["status"] == "done" for row in rows)
        errors = [row for row in rows if row["status"] == "error"]
        print(
            f"[{utc_now()}] monitor tick done={done}/{len(rows)} "
            f"errors={len(errors)} top_alive={pid_alive(args.top_pid)}",
            flush=True,
        )
        if done == len(rows):
            final_status = "all_lms_done"
            break
        if errors and args.exit_on_error:
            final_status = "error_seen"
            break
        if not pid_alive(args.top_pid):
            final_status = f"top_pid_exited_done_{done}_of_{len(rows)}"
            break
        time.sleep(args.interval_sec)

    notify(
        args.notify_cmd,
        args.workspace,
        f"Tagged ACL new_v10 sample50 de lm1-4 monitor: {final_status}; {args.report_tsv}",
    )
    print(f"[{utc_now()}] monitor exit status={final_status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
