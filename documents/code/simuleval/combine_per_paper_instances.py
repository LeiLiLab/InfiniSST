#!/usr/bin/env python3
"""
Combine per-paper instances.log files into one aligned with dev.source.

Reads dev.source to determine the paper order, then concatenates
per-paper instances.log files in that order so the combined log
aligns line-by-line with the full dev.yaml / reference.

All user-facing strings are in English.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


def _paper_id_from_wav(wav_path: str) -> str:
    base = os.path.basename(wav_path.strip())
    if base.lower().endswith(".wav"):
        base = base[: -len(".wav")]
    assert base, f"Could not extract paper_id from wav path: {wav_path}"
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-source", required=True, help="Path to dev.source")
    ap.add_argument(
        "--paper-instances-dir", required=True,
        help="Directory containing per-paper instances.log files (named instances__{paper_id}.log or under subdirs)"
    )
    ap.add_argument(
        "--per-paper-output-pattern", default="",
        help="Pattern to locate instances.log for each paper. "
             "Use {paper_id} as placeholder. E.g. '/path/to/pp{paper_id}/instances.log'"
    )
    ap.add_argument("--output", required=True, help="Output combined instances.log")
    args = ap.parse_args()

    dev_source = Path(args.dev_source)
    assert dev_source.is_file(), f"dev.source not found: {dev_source}"

    src_lines = dev_source.read_text(encoding="utf-8").strip().splitlines()
    paper_order = [_paper_id_from_wav(line) for line in src_lines]
    print(f"[INFO] dev.source has {len(paper_order)} lines, paper order: {paper_order}")

    paper_instances: Dict[str, List[str]] = {}
    for paper_id in set(paper_order):
        if args.per_paper_output_pattern:
            inst_path = Path(args.per_paper_output_pattern.replace("{paper_id}", paper_id))
        else:
            inst_path = Path(args.paper_instances_dir) / f"instances__{paper_id}.log"
        if not inst_path.is_file():
            print(f"[ERROR] Missing instances.log for paper {paper_id}: {inst_path}", file=sys.stderr)
            return 1
        lines = inst_path.read_text(encoding="utf-8").strip().splitlines()
        paper_instances[paper_id] = lines
        print(f"[INFO] Paper {paper_id}: {len(lines)} instances")

    paper_counters: Dict[str, int] = {pid: 0 for pid in set(paper_order)}
    combined_lines: List[str] = []
    for paper_id in paper_order:
        idx = paper_counters[paper_id]
        assert idx < len(paper_instances[paper_id]), (
            f"Ran out of instances for paper {paper_id} "
            f"(needed index {idx}, have {len(paper_instances[paper_id])})"
        )
        combined_lines.append(paper_instances[paper_id][idx])
        paper_counters[paper_id] = idx + 1

    for paper_id, count in paper_counters.items():
        total = len(paper_instances[paper_id])
        assert count == total, (
            f"Not all instances used for paper {paper_id}: used {count}/{total}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(combined_lines) + "\n", encoding="utf-8")
    print(f"[INFO] Combined {len(combined_lines)} instances -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
