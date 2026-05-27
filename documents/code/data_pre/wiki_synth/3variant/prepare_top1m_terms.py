#!/usr/bin/env python3
"""
Extract top N terms from the P31-ranked list, enriched with short_description
from the full pool file, and output as a JSON list for generate_term_utterances.py.

Usage:
    python prepare_top1m_terms.py
    python prepare_top1m_terms.py --top_n 500000
"""

from __future__ import annotations

import argparse
import json
import os
import time

# ======Configuration=====
RANKED_JSONL = "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/p31_full/wiki_synth_terms_p31_ranked.jsonl"
FULL_POOL_JSON = "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/p31_full/final_train_terms.json"
OUTPUT_DIR = "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant"
DEFAULT_TOP_N = 1_000_000
# ======Configuration=====


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract top N P31-ranked terms with descriptions")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--ranked_jsonl", type=str, default=RANKED_JSONL)
    parser.add_argument("--full_pool_json", type=str, default=FULL_POOL_JSON)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    assert os.path.isfile(args.ranked_jsonl), f"Ranked JSONL not found: {args.ranked_jsonl}"
    assert os.path.isfile(args.full_pool_json), f"Full pool JSON not found: {args.full_pool_json}"

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"top_{args.top_n // 1000}k_terms.json")

    # Load full pool for short_description lookup
    print(f"[1/3] Loading full pool: {args.full_pool_json}")
    t0 = time.time()
    with open(args.full_pool_json, "r", encoding="utf-8") as f:
        full_pool = json.load(f)
    desc_lookup: dict[str, str] = {}
    for entry in full_pool:
        desc_lookup[entry["term_key"]] = entry.get("short_description", entry["term"])
    del full_pool
    print(f"    Loaded {len(desc_lookup):,} descriptions in {time.time() - t0:.1f}s")

    # Read top N from ranked JSONL
    print(f"[2/3] Reading top {args.top_n:,} from: {args.ranked_jsonl}")
    t0 = time.time()
    terms: list[dict] = []
    missing_desc = 0
    with open(args.ranked_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.top_n:
                break
            entry = json.loads(line)
            term_key = entry["term_key"]
            desc = desc_lookup.get(term_key, "")
            if not desc:
                desc = entry["term"]
                missing_desc += 1
            terms.append({
                "term": entry["term"],
                "term_key": term_key,
                "short_description": desc,
                "p31_rank": entry["rank"],
            })
    print(f"    Got {len(terms):,} terms in {time.time() - t0:.1f}s "
          f"({missing_desc} with fallback description)")
    assert len(terms) == args.top_n, (
        f"Expected {args.top_n} terms but got {len(terms)}. "
        f"Ranked file may have fewer entries."
    )

    # Write output
    print(f"[3/3] Writing: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(terms, f, ensure_ascii=False, indent=0)
    file_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"    Done. {len(terms):,} terms, {file_mb:.1f} MB")
    print(f"    First: {terms[0]['term']} (rank {terms[0]['p31_rank']})")
    print(f"    Last:  {terms[-1]['term']} (rank {terms[-1]['p31_rank']})")


if __name__ == "__main__":
    main()
