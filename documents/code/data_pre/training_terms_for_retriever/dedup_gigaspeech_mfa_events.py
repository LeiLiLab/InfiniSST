#!/usr/bin/env python3
"""Deduplicate overlapping GigaSpeech retriever-training term events.

GigaSpeech retriever rows are cut as 1.92s chunks with 0.96s stride. A term
whose absolute MFA span lies in the overlap can therefore appear in two adjacent
chunks. For MFA-supervised retriever training, those duplicate rows describe the
same acoustic event, so this script keeps one random row per event and leaves
wiki-synth rows untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, Hashable, Optional, Tuple


WIKI_SYNTH_PREFIX = "wiki_synth_"
DEFAULT_STRIDE_SEC = 0.96
PROGRESS_EVERY = 500_000


def _term_key(row: Dict[str, Any]) -> str:
    return str(row.get("term_key") or row.get("term") or "").strip().casefold()


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def gigaspeech_event_key(
    row: Dict[str, Any],
    *,
    chunk_stride_sec: float,
    round_digits: int,
) -> Optional[Tuple[Hashable, ...]]:
    """Return a stable absolute-MFA event key for a GigaSpeech row.

    Returns None when the row lacks enough metadata; such rows are preserved as
    unique rows rather than being accidentally merged.
    """

    utter_id = str(row.get("utter_id") or "").strip()
    term = _term_key(row)
    chunk_idx = _parse_int(row.get("chunk_idx"))
    mfa_start = _parse_float(row.get("mfa_term_start_in_chunk"))
    mfa_end = _parse_float(row.get("mfa_term_end_in_chunk"))
    if not utter_id or not term or chunk_idx is None or mfa_start is None or mfa_end is None:
        return None

    abs_start = chunk_idx * chunk_stride_sec + mfa_start
    abs_end = chunk_idx * chunk_stride_sec + mfa_end
    return (
        utter_id,
        term,
        round(abs_start, round_digits),
        round(abs_end, round_digits),
    )


def choose_rows(args: argparse.Namespace) -> Tuple[Dict[Tuple[Hashable, ...], int], Counter]:
    """First pass: reservoir-sample one row number per GigaSpeech event key."""

    rng = random.Random(args.seed)
    selected_row_by_key: Dict[Tuple[Hashable, ...], int] = {}
    seen_by_key: Counter = Counter()
    stats: Counter = Counter()

    with open(args.input, "r", encoding="utf-8") as fin:
        for row_no, line in enumerate(fin, start=1):
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["json_parse_errors"] += 1
                continue

            stats["input_rows"] += 1
            utter_id = str(row.get("utter_id") or "")
            if utter_id.startswith(WIKI_SYNTH_PREFIX):
                stats["wiki_rows"] += 1
                continue

            stats["gigaspeech_rows"] += 1
            key = gigaspeech_event_key(
                row,
                chunk_stride_sec=args.chunk_stride_sec,
                round_digits=args.round_digits,
            )
            if key is None:
                stats["gigaspeech_missing_event_key"] += 1
                key = ("__missing_event_key__", row_no)

            seen_by_key[key] += 1
            if rng.randrange(seen_by_key[key]) == 0:
                selected_row_by_key[key] = row_no

            if stats["input_rows"] % PROGRESS_EVERY == 0:
                print(
                    "[PASS1] "
                    f"rows={stats['input_rows']:,} "
                    f"gs={stats['gigaspeech_rows']:,} "
                    f"unique_gs_events={len(selected_row_by_key):,}",
                    flush=True,
                )

    duplicate_rows = sum(count - 1 for count in seen_by_key.values())
    duplicate_groups = sum(1 for count in seen_by_key.values() if count > 1)
    stats["gigaspeech_unique_events"] = len(selected_row_by_key)
    stats["gigaspeech_duplicate_rows"] = duplicate_rows
    stats["gigaspeech_duplicate_groups"] = duplicate_groups
    return selected_row_by_key, stats


def write_selected(
    args: argparse.Namespace,
    selected_row_by_key: Dict[Tuple[Hashable, ...], int],
    stats: Counter,
) -> Counter:
    """Second pass: write all wiki rows and the selected GigaSpeech rows."""

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for row_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            utter_id = str(row.get("utter_id") or "")
            if utter_id.startswith(WIKI_SYNTH_PREFIX):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_wiki_rows"] += 1
                continue

            key = gigaspeech_event_key(
                row,
                chunk_stride_sec=args.chunk_stride_sec,
                round_digits=args.round_digits,
            )
            if key is None:
                key = ("__missing_event_key__", row_no)
            if selected_row_by_key.get(key) == row_no:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_gigaspeech_rows"] += 1

            if (stats["written_wiki_rows"] + stats["written_gigaspeech_rows"]) % PROGRESS_EVERY == 0:
                print(
                    "[PASS2] "
                    f"written={stats['written_wiki_rows'] + stats['written_gigaspeech_rows']:,} "
                    f"gs={stats['written_gigaspeech_rows']:,} "
                    f"wiki={stats['written_wiki_rows']:,}",
                    flush=True,
                )

    stats["written_total_rows"] = stats["written_wiki_rows"] + stats["written_gigaspeech_rows"]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly keep one row per absolute GigaSpeech MFA term event."
    )
    parser.add_argument("--input", required=True, help="Input retriever train JSONL.")
    parser.add_argument("--output", required=True, help="Output deduplicated JSONL.")
    parser.add_argument("--stats-json", default="", help="Optional stats JSON path.")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--chunk-stride-sec", type=float, default=DEFAULT_STRIDE_SEC)
    parser.add_argument("--round-digits", type=int, default=4)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    print(f"[DEDUP] input={args.input}", flush=True)
    print(f"[DEDUP] output={args.output}", flush=True)
    print(
        f"[DEDUP] seed={args.seed} stride={args.chunk_stride_sec} "
        f"round_digits={args.round_digits}",
        flush=True,
    )

    selected_row_by_key, stats = choose_rows(args)
    stats = write_selected(args, selected_row_by_key, stats)

    stats_payload = dict(sorted(stats.items()))
    stats_payload["input"] = args.input
    stats_payload["output"] = args.output
    stats_payload["seed"] = args.seed
    stats_payload["chunk_stride_sec"] = args.chunk_stride_sec
    stats_payload["round_digits"] = args.round_digits
    stats_path = args.stats_json or args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fout:
        json.dump(stats_payload, fout, indent=2, ensure_ascii=False)

    print("[DONE]", flush=True)
    for key, value in stats_payload.items():
        print(f"  {key}: {value}", flush=True)
    print(f"  stats_json: {stats_path}", flush=True)


if __name__ == "__main__":
    main()
