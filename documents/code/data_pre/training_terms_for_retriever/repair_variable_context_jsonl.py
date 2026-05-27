#!/usr/bin/env python3
"""Strictly filter invalid rows from a variable-context retriever JSONL."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def duration_tag(sec: float) -> str:
    return f"{sec:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_duration_secs(value: str) -> List[float]:
    out = []
    for item in value.replace(",", " ").split():
        dur = round(float(item), 4)
        if dur not in out:
            out.append(dur)
    if not out:
        raise ValueError("--expected-duration-secs must not be empty")
    return out


def parse_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def nearest_expected_duration(value: float | None, expected: Iterable[float], eps: float) -> float | None:
    if value is None:
        return None
    best = min(expected, key=lambda d: abs(d - value))
    if abs(best - value) <= eps:
        return best
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--source-stats-json", default="")
    parser.add_argument("--expected-duration-secs", default="2.88 3.84 4.80 5.76")
    parser.add_argument("--duration-eps", type=float, default=0.015)
    args = parser.parse_args()

    expected = parse_duration_secs(args.expected_duration_secs)
    expected_tags = [duration_tag(d) for d in expected]

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats_json)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_output.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    kept_total = 0
    read_total = 0
    drop_reasons = Counter()
    duration_counts = Counter()
    context_build_counts = Counter()

    with open(input_path, "r", encoding="utf-8") as fin, open(tmp_output, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            read_total += 1
            try:
                row: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                drop_reasons["json_error"] += 1
                continue

            dur = parse_float(row.get("context_duration_sec"))
            if dur is None:
                dur = parse_float(row.get("chunk_duration_sec"))
            matched = nearest_expected_duration(dur, expected, args.duration_eps)
            if matched is None:
                drop_reasons["unknown_duration"] += 1
                continue

            term = str(row.get("term_key") or row.get("term") or "").strip()
            if term:
                start = parse_float(row.get("mfa_term_start_in_chunk"))
                end = parse_float(row.get("mfa_term_end_in_chunk"))
                if (
                    start is None
                    or end is None
                    or start < -1e-4
                    or end > matched + 1e-4
                    or end <= start
                ):
                    drop_reasons["invalid_mfa_span"] += 1
                    continue

            tag = duration_tag(matched)
            row["chunk_duration_sec"] = matched
            row["context_duration_sec"] = matched
            row["context_duration_tag"] = tag
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept_total += 1
            duration_counts[tag] += 1
            context_build_counts[str(row.get("context_build") or "missing")] += 1

    os.replace(tmp_output, output_path)

    source_stats = {}
    if args.source_stats_json and Path(args.source_stats_json).is_file():
        with open(args.source_stats_json, "r", encoding="utf-8") as fin:
            source_stats = json.load(fin)

    stats = dict(source_stats)
    stats.update(
        {
            "output": str(output_path),
            "repair_source_input": str(input_path),
            "repair_source_stats_json": args.source_stats_json,
            "repair_read_total_rows": read_total,
            "repair_kept_total_rows": kept_total,
            "repair_dropped_total_rows": read_total - kept_total,
            "repair_drop_reasons": dict(drop_reasons),
            "written_total_rows": kept_total,
            "duration_secs": expected,
            "duration_tags": expected_tags,
            "context_build_counts_after_repair": dict(context_build_counts),
        }
    )
    for tag in expected_tags:
        stats[f"duration_row_count_{tag}"] = duration_counts[tag]

    with open(stats_path, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[REPAIR] input={input_path}")
    print(f"[REPAIR] output={output_path}")
    print(f"[REPAIR] stats={stats_path}")
    print(f"[REPAIR] read_total_rows={read_total}")
    print(f"[REPAIR] kept_total_rows={kept_total}")
    print(f"[REPAIR] dropped_total_rows={read_total - kept_total} reasons={dict(drop_reasons)}")
    for tag in expected_tags:
        print(f"[REPAIR] duration_row_count_{tag}={duration_counts[tag]}")


if __name__ == "__main__":
    main()
