#!/usr/bin/env python3
"""Supplement partial GSV2 wiki training data with old clean wiki rows.

The partial GSV2 scout replaces the wiki_synth portion with clean-only GSV2
speaker-pool TTS, but the partial shard set has fewer active wiki rows than the
baseline 3variant train JSONL. This builder keeps the partial train JSONL as-is
and copies enough old clean wiki rows from the baseline to match the baseline's
active clean wiki count under the training filters:

  - wiki_rank < 1,000,000
  - noisy rows are not used

Rows whose term is absent from the partial active wiki set are copied first to
recover term coverage before adding overlapping-term rows.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Iterable, TextIO


DEFAULT_BASELINE_TRAIN = "/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
DEFAULT_PARTIAL_TRAIN = (
    "/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_partial0_20_clean_mfa.jsonl"
)
DEFAULT_OUTPUT_TRAIN = (
    "/mnt/gemini/home/jiaxuanluo/"
    "term_train_3variant_gsv2_partial0_20_oldwiki_clean_mfa.jsonl"
)


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def term_key(row: dict) -> str:
    return str(row.get("term_key", row.get("term", ""))).strip().lower()


def is_wiki(row: dict) -> bool:
    return str(row.get("utter_id", "")).startswith("wiki_synth_")


def p31_rank(row: dict) -> int:
    try:
        return int(row.get("p31_rank", -1) or -1)
    except (TypeError, ValueError):
        return -1


def is_active_clean_wiki(row: dict, wiki_rank_cutoff: int) -> bool:
    return (
        is_wiki(row)
        and row.get("audio_type") == "clean"
        # Training keeps p31_rank=-1 rows; it only skips non-negative ranks
        # greater than or equal to the cutoff.
        and p31_rank(row) < wiki_rank_cutoff
    )


def write_row(f: TextIO, row: dict) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_active_wiki_terms(path: str, wiki_rank_cutoff: int) -> tuple[set[str], int]:
    terms: set[str] = set()
    count = 0
    for row in iter_jsonl(path):
        if is_active_clean_wiki(row, wiki_rank_cutoff):
            terms.add(term_key(row))
            count += 1
    return terms, count


def count_baseline_active_clean_wiki(path: str, wiki_rank_cutoff: int) -> int:
    return sum(1 for row in iter_jsonl(path) if is_active_clean_wiki(row, wiki_rank_cutoff))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-train", default=DEFAULT_BASELINE_TRAIN)
    parser.add_argument("--partial-train", default=DEFAULT_PARTIAL_TRAIN)
    parser.add_argument("--output-train", default=DEFAULT_OUTPUT_TRAIN)
    parser.add_argument("--wiki-rank-cutoff", type=int, default=1_000_000)
    args = parser.parse_args()

    for path in [args.baseline_train, args.partial_train]:
        assert os.path.isfile(path), f"Missing input JSONL: {path}"

    partial_terms, partial_active_wiki = collect_active_wiki_terms(
        args.partial_train, args.wiki_rank_cutoff
    )
    target_active_wiki = count_baseline_active_clean_wiki(
        args.baseline_train, args.wiki_rank_cutoff
    )
    rows_needed = target_active_wiki - partial_active_wiki
    assert rows_needed > 0, (
        f"Partial active wiki already reaches target: "
        f"partial={partial_active_wiki} target={target_active_wiki}"
    )

    os.makedirs(os.path.dirname(args.output_train) or ".", exist_ok=True)
    tmp_path = args.output_train + ".tmp"

    stats = Counter()
    supplement_missing_term: list[dict] = []
    supplement_overlap_term: list[dict] = []

    for row in iter_jsonl(args.baseline_train):
        if not is_active_clean_wiki(row, args.wiki_rank_cutoff):
            continue
        if term_key(row) in partial_terms:
            supplement_overlap_term.append(row)
        else:
            supplement_missing_term.append(row)

    selected: list[dict] = []
    selected.extend(supplement_missing_term[:rows_needed])
    if len(selected) < rows_needed:
        selected.extend(supplement_overlap_term[: rows_needed - len(selected)])

    assert len(selected) == rows_needed, (
        f"Could not find enough old clean wiki rows: need={rows_needed}, "
        f"selected={len(selected)}"
    )

    with open(tmp_path, "w", encoding="utf-8") as fout:
        for row in iter_jsonl(args.partial_train):
            write_row(fout, row)
            stats["partial_rows"] += 1
            if is_wiki(row):
                stats["partial_wiki_rows"] += 1
            else:
                stats["partial_non_wiki_rows"] += 1

        selected_terms: set[str] = set()
        for row in selected:
            row = dict(row)
            row["source_mix"] = "oldwiki_clean_supplement"
            write_row(fout, row)
            selected_terms.add(term_key(row))
            stats["oldwiki_supplement_rows"] += 1
            if term_key(row) in partial_terms:
                stats["oldwiki_supplement_overlap_term_rows"] += 1
            else:
                stats["oldwiki_supplement_missing_term_rows"] += 1

    os.replace(tmp_path, args.output_train)

    stats.update(
        {
            "target_active_wiki": target_active_wiki,
            "partial_active_wiki": partial_active_wiki,
            "output_expected_active_wiki": partial_active_wiki + len(selected),
            "partial_active_wiki_unique_terms": len(partial_terms),
            "oldwiki_supplement_unique_terms": len(selected_terms),
            "oldwiki_missing_term_rows_available": len(supplement_missing_term),
            "oldwiki_overlap_term_rows_available": len(supplement_overlap_term),
            "wiki_rank_cutoff": args.wiki_rank_cutoff,
        }
    )

    stats_path = args.output_train.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, indent=2, sort_keys=True)

    print("[DONE] wrote", args.output_train, flush=True)
    print(json.dumps(dict(stats), indent=2, sort_keys=True), flush=True)
    print("[STATS]", stats_path, flush=True)


if __name__ == "__main__":
    main()
