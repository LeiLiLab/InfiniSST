#!/usr/bin/env python3
"""Filter a mixed GigaSpeech/wiki_synth retriever JSONL to GigaSpeech rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


WIKI_PREFIX = "wiki_synth_"


def infer_domain(row: dict[str, Any]) -> str:
    utter_id = str(row.get("utter_id") or "")
    context_build = str(row.get("context_build") or "")
    chunk_audio_path = str(row.get("chunk_audio_path") or "")
    if utter_id.startswith(WIKI_PREFIX) or context_build.startswith("wiki_synth") or "/wiki_synth/" in chunk_audio_path:
        return "wiki_synth"
    if context_build.startswith("gigaspeech") or utter_id:
        return "gigaspeech"
    return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input mixed-domain JSONL")
    p.add_argument("--output", required=True, help="Output GigaSpeech-only JSONL")
    p.add_argument("--stats-json", required=True, help="Output stats JSON")
    p.add_argument("--fail-on-unknown", action="store_true", help="Fail if a row domain cannot be inferred")
    p.add_argument("--progress-every", type=int, default=250_000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats_json)
    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = dropped = json_errors = unknown = 0
    domain_counts: Counter[str] = Counter()
    kept_context_build: Counter[str] = Counter()
    kept_duration_tags: Counter[str] = Counter()
    drop_reasons: Counter[str] = Counter()

    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with input_path.open("r", encoding="utf-8") as src, tmp_output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                drop_reasons["json_error"] += 1
                continue
            domain = infer_domain(row)
            domain_counts[domain] += 1
            if domain == "unknown":
                unknown += 1
                drop_reasons["unknown_domain"] += 1
                if args.fail_on_unknown:
                    raise ValueError(f"unknown domain at line {line_no}: {line[:200]}")
                continue
            if domain != "gigaspeech":
                dropped += 1
                drop_reasons[domain] += 1
                continue
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
            kept_context_build[str(row.get("context_build") or "missing")] += 1
            kept_duration_tags[str(row.get("context_duration_tag") or "missing")] += 1
            if args.progress_every and total % args.progress_every == 0:
                print(f"[filter] total={total:,} kept={kept:,} dropped={dropped:,}", flush=True)

    tmp_output.replace(output_path)
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "total_rows": total,
        "kept_rows": kept,
        "dropped_rows": dropped,
        "drop_rate": dropped / total if total else 0.0,
        "json_errors": json_errors,
        "unknown_domain_rows": unknown,
        "domain_counts": dict(domain_counts),
        "drop_reasons": dict(drop_reasons),
        "kept_context_build_counts": dict(kept_context_build),
        "kept_duration_tag_counts": dict(kept_duration_tags),
    }
    tmp_stats = stats_path.with_suffix(stats_path.suffix + ".tmp")
    tmp_stats.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_stats.replace(stats_path)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
