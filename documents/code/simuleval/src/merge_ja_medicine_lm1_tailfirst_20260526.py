#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_jsonl_prefix(path: Path, n: int) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(line)
            if len(rows) == n:
                break
    if len(rows) != n:
        raise RuntimeError(f"expected {n} rows in {path}, got {len(rows)}")
    return rows


def sample_marker(line: str) -> str:
    row = json.loads(line)
    source = row.get("source")
    if isinstance(source, list) and source:
        return str(source[0])
    return str(source)


def validate_markers(rows: list[str], expected: list[str], label: str) -> None:
    actual = [sample_marker(row) for row in rows]
    missing = [
        (want, got) for want, got in zip(expected, actual, strict=True) if want not in got
    ]
    if missing:
        raise RuntimeError(f"{label} sample order mismatch: {missing}")


def terminate_process_group(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-instances", type=Path, required=True)
    ap.add_argument("--tailfirst-instances", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--source-file", type=Path, required=True)
    ap.add_argument("--ref-file", type=Path, required=True)
    ap.add_argument("--audio-yaml", type=Path, required=True)
    ap.add_argument("--glossary", type=Path, required=True)
    ap.add_argument("--offline-eval-script", type=Path, required=True)
    ap.add_argument("--tailfirst-pid-file", type=Path)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--timeout-seconds", type=int, default=10800)
    ap.add_argument("--kill-tailfirst-after-merge", action="store_true")
    args = ap.parse_args()

    deadline = time.time() + args.timeout_seconds
    while time.time() < deadline:
        orig_n = line_count(args.orig_instances)
        tail_n = line_count(args.tailfirst_instances)
        print(
            f"[WAIT] orig_rows={orig_n}/3 tailfirst_rows={tail_n}/2",
            flush=True,
        )
        if orig_n >= 3 and tail_n >= 2:
            break
        time.sleep(args.poll_seconds)
    else:
        raise SystemExit("timeout waiting for mergeable instances.log rows")

    orig_rows = read_jsonl_prefix(args.orig_instances, 3)
    tail_rows = read_jsonl_prefix(args.tailfirst_instances, 2)
    validate_markers(orig_rows, ["sample_404", "sample_545006", "sample_596001"], "orig")
    validate_markers(tail_rows, ["sample_605000", "sample_606"], "tailfirst")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_instances = args.output_dir / "instances.log"
    merged_instances.write_text("".join(orig_rows + tail_rows), encoding="utf-8")
    meta = {
        "merged_instances": str(merged_instances),
        "orig_instances": str(args.orig_instances),
        "tailfirst_instances": str(args.tailfirst_instances),
        "orig_rows": 3,
        "tailfirst_rows": 2,
        "order": ["404", "545006", "596001", "605000", "606"],
    }
    (args.output_dir / "merge_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    eval_tsv = args.output_dir / "eval_results.tsv"
    eval_log = args.output_dir / "eval_results.log"
    cmd = [
        "python3",
        str(args.offline_eval_script),
        "--mode",
        "acl6060",
        "--instances-log",
        str(merged_instances),
        "--lang-code",
        "ja",
        "--source-file",
        str(args.source_file),
        "--ref-file",
        str(args.ref_file),
        "--audio-yaml",
        str(args.audio_yaml),
        "--glossary-acl6060",
        str(args.glossary),
        "--strip-output-tags",
        "term_t",
        "--term-fcr-policy",
        "term_map_source_ref_negative_sentence",
        "--output-tsv",
        str(eval_tsv),
        "--output-log",
        str(eval_log),
        "--work-dir",
        str(args.output_dir / "offline_work"),
    ]
    print("[RUN] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    rows = list(csv.DictReader(eval_tsv.open("r", encoding="utf-8"), delimiter="\t"))
    if len(rows) != 1:
        raise RuntimeError(f"expected one eval row in {eval_tsv}, got {len(rows)}")
    row = rows[0]
    print(
        "RESULT "
        f"BLEU={float(row['BLEU']):.4f} "
        f"StreamLAAL={float(row['StreamLAAL']):.4f} "
        f"StreamLAAL_CA={float(row['StreamLAAL_CA']):.4f} "
        f"TERM_ACC={float(row['TERM_ACC']):.4f} "
        f"TERM={row.get('TERM_CORRECT', '')}/{row.get('TERM_TOTAL', '')} "
        f"eval={eval_tsv}",
        flush=True,
    )

    if args.kill_tailfirst_after_merge and args.tailfirst_pid_file:
        terminate_process_group(args.tailfirst_pid_file)


if __name__ == "__main__":
    main()
