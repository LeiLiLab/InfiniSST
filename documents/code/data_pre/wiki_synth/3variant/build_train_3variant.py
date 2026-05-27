#!/usr/bin/env python3
"""
Build 3-variant training dataset from MFA output, with P31 ranking + inference leakage filter.

Analogous to build_train_v1_0.py but uses 3-variant MFA output directory.

Steps:
  1. Load P31 ranking → term_key→rank mapping
  2. Load inference glossary terms → blocked set (cs + medicine 10k)
  3. Stream 3-variant MFA shard outputs, attach p31_rank, filter blocked terms
  4. Stream Gigaspeech (p31_rank=-1), filter blocked terms
  5. Output unified JSONL

Usage:
    python build_train_3variant.py
    python build_train_3variant.py --smoke-test 3
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Set

# ======Configuration=====
P31_RANKING = (
    "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/p31_full/"
    "wiki_synth_terms_p31_ranked.jsonl"
)

WIKI_3VAR_MFA_DIR = "/mnt/aries/data4/jiaxuanluo/MFA/3variant/output"
WIKI_3VAR_SHARD_PATTERN = "wiki_synth_train_shard_{:02d}.jsonl"
WIKI_3VAR_NUM_SHARDS = 20

GIGASPEECH_TRAIN = "/mnt/gemini/data1/jiaxuanluo/term_train_dataset_final.jsonl"

INFERENCE_GLOSSARIES = [
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_medicine.json",
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json",
]

OUTPUT_TRAIN = "/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m.jsonl"
OUTPUT_STATS = "/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_stats.json"
# ======Configuration=====


def load_p31_ranking(path: str) -> Dict[str, int]:
    assert os.path.isfile(path), f"P31 ranking not found: {path}"
    mapping: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            mapping[obj["term_key"]] = obj["rank"]
    return mapping


def load_inference_terms(glossary_paths: List[str]) -> Set[str]:
    terms: Set[str] = set()
    for path in glossary_paths:
        assert os.path.isfile(path), f"Glossary not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                terms.add(item["term"].strip().lower())
    print(f"  Loaded {len(terms):,} inference glossary terms to block", flush=True)
    return terms


def load_3variant_mfa_shards(
    mfa_dir: str,
    num_shards: int,
    shard_pattern: str,
    p31_ranking: Dict[str, int],
    blocked: Set[str],
    max_shards: int = 0,
) -> List[dict]:
    all_entries: List[dict] = []
    actual_shards = max_shards if max_shards > 0 else num_shards
    missing_rank = 0
    filtered = 0

    for shard_id in range(actual_shards):
        shard_filename = shard_pattern.format(shard_id)
        shard_path = os.path.join(mfa_dir, shard_filename)
        assert os.path.isfile(shard_path), f"Missing MFA shard: {shard_path}"

        count = 0
        shard_filtered = 0
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                tk = entry.get("term_key", entry["term"].strip().lower())
                if tk in blocked:
                    shard_filtered += 1
                    continue
                rank = p31_ranking.get(tk, -1)
                if rank == -1:
                    missing_rank += 1
                entry["p31_rank"] = rank
                all_entries.append(entry)
                count += 1
        filtered += shard_filtered
        print(f"    Shard {shard_id:2d}: {count:>8,} kept, {shard_filtered:>5,} filtered", flush=True)

    if missing_rank > 0:
        print(f"  [WARN] {missing_rank:,} entries had no P31 rank (set to -1)", flush=True)
    if filtered > 0:
        print(f"  [INFO] {filtered:,} entries blocked by inference glossary", flush=True)
    return all_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3-variant training dataset")
    parser.add_argument("--mfa-dir", type=str, default=WIKI_3VAR_MFA_DIR)
    parser.add_argument("--gigaspeech-train", type=str, default=GIGASPEECH_TRAIN)
    parser.add_argument(
        "--gigaspeech-skip-wiki-synth",
        action="store_true",
        help="When --gigaspeech-train is a combined train JSONL, skip old wiki_synth rows.",
    )
    parser.add_argument("--output-train", type=str, default=OUTPUT_TRAIN)
    parser.add_argument("--num-shards", type=int, default=WIKI_3VAR_NUM_SHARDS)
    parser.add_argument("--shard-pattern", type=str, default=WIKI_3VAR_SHARD_PATTERN)
    parser.add_argument("--smoke-test", type=int, default=0)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("[Step 1] Loading P31 ranking...", flush=True)
    p31_ranking = load_p31_ranking(P31_RANKING)
    print(f"  Ranked terms: {len(p31_ranking):,}", flush=True)

    print(f"\n[Step 2] Loading inference glossary terms (leakage filter)...", flush=True)
    blocked = load_inference_terms(INFERENCE_GLOSSARIES)

    max_shards = args.smoke_test if args.smoke_test > 0 else 0
    print(f"\n[Step 3] Loading 3-variant MFA shards from {args.mfa_dir}...", flush=True)
    wiki_entries = load_3variant_mfa_shards(
        args.mfa_dir, args.num_shards, args.shard_pattern, p31_ranking, blocked, max_shards,
    )
    print(f"  Total 3-variant wiki entries: {len(wiki_entries):,}", flush=True)

    unique_terms = set()
    for e in wiki_entries:
        unique_terms.add(e.get("term_key", e["term"].strip().lower()))
    print(f"  Unique wiki terms: {len(unique_terms):,}", flush=True)

    print(f"\n[Step 4] Writing output: {args.output_train}", flush=True)
    os.makedirs(os.path.dirname(args.output_train) or ".", exist_ok=True)

    gs_count = 0
    gs_filtered = 0

    with open(args.output_train, "w", encoding="utf-8") as fout:
        assert os.path.isfile(args.gigaspeech_train), \
            f"Gigaspeech train not found: {args.gigaspeech_train}"
        with open(args.gigaspeech_train, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                if args.smoke_test > 0 and i >= args.smoke_test * 1000:
                    break
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if (
                    args.gigaspeech_skip_wiki_synth
                    and str(entry.get("utter_id", "")).startswith("wiki_synth_")
                ):
                    continue
                tk = entry.get("term_key", entry["term"].strip().lower())
                if tk in blocked:
                    gs_filtered += 1
                    continue
                entry["p31_rank"] = -1
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                gs_count += 1

        for entry in wiki_entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = gs_count + len(wiki_entries)

    rank_bins = Counter()
    for e in wiki_entries:
        r = e["p31_rank"]
        if r < 0:
            rank_bins["no_rank"] += 1
        elif r < 100_000:
            rank_bins["0-100K"] += 1
        elif r < 500_000:
            rank_bins["100K-500K"] += 1
        elif r < 1_000_000:
            rank_bins["500K-1M"] += 1
        elif r < 2_000_000:
            rank_bins["1M-2M"] += 1
        else:
            rank_bins["2M+"] += 1

    stats = {
        "gigaspeech_kept": gs_count,
        "gigaspeech_filtered": gs_filtered,
        "wiki_3var_kept": len(wiki_entries),
        "wiki_3var_unique_terms": len(unique_terms),
        "total": total,
        "wiki_rank_distribution": dict(rank_bins),
    }
    stats_path = args.output_train.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 70}", flush=True)
    print("[SUMMARY]", flush=True)
    print(f"  Gigaspeech:     {gs_count:>9,} kept, {gs_filtered:>6,} filtered", flush=True)
    print(f"  Wiki 3-variant: {len(wiki_entries):>9,} ({len(unique_terms):,} unique terms)", flush=True)
    print(f"  Total:          {total:>9,}", flush=True)
    print(f"  Wiki rank distribution:", flush=True)
    for bucket in ["0-100K", "100K-500K", "500K-1M", "1M-2M", "2M+", "no_rank"]:
        if bucket in rank_bins:
            print(f"    {bucket:>12s}: {rank_bins[bucket]:>9,}", flush=True)
    print(f"  Output: {args.output_train}", flush=True)
    print(f"  Stats:  {stats_path}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
