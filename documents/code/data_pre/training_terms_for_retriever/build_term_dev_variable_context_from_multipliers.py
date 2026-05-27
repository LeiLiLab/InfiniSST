#!/usr/bin/env python3
"""Build a balanced variable-duration term-dev JSONL from multiplier outputs.

The legacy dev extractor writes one JSONL per latency multiplier m, where
duration = 0.96 * m seconds.  This script annotates those rows with the common
variable-context metadata used by the train/ACL builders and balances row counts
across the requested durations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


UNIT_SEC = 0.96


def duration_tag(sec: float) -> str:
    return f"{sec:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def stable_u64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def row_key(row: Dict[str, Any]) -> str:
    return "\t".join(
        [
            str(row.get("utter_id") or ""),
            str(row.get("chunk_idx") or ""),
            str(row.get("term_key") or row.get("term") or ""),
            str(row.get("chunk_src_text") or ""),
            str(row.get("chunk_audio_path") or ""),
        ]
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def parse_multiplier_specs(values: Iterable[str]) -> List[Tuple[int, Path]]:
    specs: List[Tuple[int, Path]] = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"bad --multiplier-jsonl item {value!r}; expected M=PATH")
        m_raw, path_raw = value.split("=", 1)
        m = int(m_raw)
        if m <= 0:
            raise ValueError(f"multiplier must be positive: {value!r}")
        if m in seen:
            raise ValueError(f"duplicate multiplier: m={m}")
        seen.add(m)
        specs.append((m, Path(path_raw)))
    if not specs:
        raise ValueError("at least one --multiplier-jsonl item is required")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multiplier-jsonl",
        nargs="+",
        required=True,
        help="Items like 3=/path/m3.jsonl 4=/path/m4.jsonl.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument(
        "--balance",
        choices=["min", "all"],
        default="min",
        help="'min' keeps the same row count for every duration.",
    )
    parser.add_argument(
        "--context-build",
        default="term_dev_multiplier_varctx_2p88_3p84_4p80_5p76",
    )
    args = parser.parse_args()

    specs = parse_multiplier_specs(args.multiplier_jsonl)
    rows_by_tag: Dict[str, List[Dict[str, Any]]] = {}
    source_counts = {}
    for m, path in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        dur = round(m * UNIT_SEC, 4)
        tag = duration_tag(dur)
        rows = load_jsonl(path)
        source_counts[tag] = len(rows)
        annotated: List[Dict[str, Any]] = []
        for row in rows:
            out = dict(row)
            out["chunk_duration_sec"] = dur
            out["context_duration_sec"] = dur
            out["context_duration_tag"] = tag
            out["duration_multiplier"] = m
            out["context_build"] = args.context_build
            annotated.append(out)
        rows_by_tag[tag] = annotated

    target_per_tag = None
    if args.balance == "min":
        target_per_tag = min(len(rows) for rows in rows_by_tag.values())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_json)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    written_counts = Counter()
    unique_terms = defaultdict(set)
    with open(output, "w", encoding="utf-8") as fout:
        for m, _path in sorted(specs):
            tag = duration_tag(round(m * UNIT_SEC, 4))
            rows = rows_by_tag[tag]
            rows.sort(key=lambda r: stable_u64(f"{tag}\t{row_key(r)}"))
            keep_n = len(rows) if target_per_tag is None else target_per_tag
            for row in rows[:keep_n]:
                term = str(row.get("term_key") or row.get("term") or "").casefold().strip()
                if term:
                    unique_terms[tag].add(term)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written_counts[tag] += 1

    stats: Dict[str, Any] = {
        "output": str(output),
        "balance": args.balance,
        "context_build": args.context_build,
        "unit_sec": UNIT_SEC,
        "multiplier_jsonl": {str(m): str(path) for m, path in specs},
        "source_row_counts": source_counts,
        "written_total_rows": sum(written_counts.values()),
        "duration_secs": [round(m * UNIT_SEC, 4) for m, _ in sorted(specs)],
        "duration_tags": [duration_tag(round(m * UNIT_SEC, 4)) for m, _ in sorted(specs)],
    }
    for tag, count in sorted(written_counts.items()):
        stats[f"duration_row_count_{tag}"] = count
        stats[f"unique_terms_{tag}"] = len(unique_terms[tag])

    with open(stats_path, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[DEV-VARCTX-MERGE] output={output}")
    print(f"[DEV-VARCTX-MERGE] stats={stats_path}")
    print(f"[DEV-VARCTX-MERGE] written_total_rows={stats['written_total_rows']}")
    for tag in stats["duration_tags"]:
        print(f"[DEV-VARCTX-MERGE] duration_row_count_{tag}={stats.get('duration_row_count_' + tag, 0)}")


if __name__ == "__main__":
    main()
