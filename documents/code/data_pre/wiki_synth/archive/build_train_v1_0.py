#!/usr/bin/env python3
"""
Build the v1.0 full training dataset with P31 ranking support.

Each wiki_synth entry gets a `p31_rank` field (0 = rarest type, higher = more common).
Gigaspeech entries get p31_rank = -1 (always included regardless of wiki_rank cutoff).

At training time, --wiki_rank N includes only wiki entries with p31_rank < N.
Default wiki_rank=1000000 (1M terms), max=~4500000 (all terms).

Steps:
  1. Load P31 ranking → term_key→rank mapping
  2. Load inference glossary terms → blocked set
  3. Stream all training data sources, attach p31_rank, filter blocked terms
  4. Output unified JSONL

Usage:
    python build_train_v1_0.py
    python build_train_v1_0.py --smoke-test 5
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

WIKI_SYNTH_MFA_DIRS = [
    "/mnt/data/jiaxuanluo/wiki_synth_mfa_p31_output",
    "/mnt/data/jiaxuanluo/wiki_synth_mfa_leftover_output",
    "/mnt/aries/data6/jiaxuanluo/MFA/aries/output",
    "/mnt/aries/data6/jiaxuanluo/MFA/1third_aries/output",
    "/mnt/data/jiaxuanluo/wiki_synth_mfa_1third_output",
]
WIKI_SYNTH_SHARD_PATTERN = "wiki_synth_train_shard_{:02d}.jsonl"
WIKI_SYNTH_NUM_SHARDS = 20

GIGASPEECH_TRAIN = "/mnt/gemini/data1/jiaxuanluo/term_train_dataset_final.jsonl"

INFERENCE_GLOSSARIES = [
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_medicine.json",
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json",
]

TAURUS_LOCAL_PREFIX = "/mnt/data/"
TAURUS_GLOBAL_PREFIX = "/mnt/taurus/data/"

OUTPUT_TRAIN = "/mnt/gemini/data1/jiaxuanluo/term_train_v1_0.jsonl"
OUTPUT_STATS = "/mnt/gemini/data1/jiaxuanluo/term_train_v1_0_stats.json"
# ======Configuration=====


def normalize_audio_path(path: str) -> str:
    if path.startswith(TAURUS_LOCAL_PREFIX) and not path.startswith("/mnt/data6/"):
        return TAURUS_GLOBAL_PREFIX + path[len(TAURUS_LOCAL_PREFIX):]
    return path


def load_p31_ranking(path: str) -> Dict[str, int]:
    """Load term_key → p31_rank mapping from ranked JSONL."""
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
    return terms


def load_wiki_synth_shards(
    shard_dirs: List[str],
    num_shards: int,
    p31_ranking: Dict[str, int],
    blocked: Set[str],
    max_shards: int = 0,
) -> List[dict]:
    """Load wiki_synth MFA shard outputs from ALL directories, normalize paths, attach p31_rank.

    Each directory may contain shards 0..num_shards-1 with the same filenames.
    We load from every directory (not first-match), since different batches
    (e.g. p31 1M vs leftover 2/3) use the same shard numbering.
    """
    all_entries: List[dict] = []
    actual_shards = max_shards if max_shards > 0 else num_shards
    missing_rank = 0
    filtered = 0

    for d in shard_dirs:
        dir_count = 0
        dir_filtered = 0
        for shard_id in range(actual_shards):
            shard_filename = WIKI_SYNTH_SHARD_PATTERN.format(shard_id)
            shard_path = os.path.join(d, shard_filename)
            if not os.path.isfile(shard_path):
                continue

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
                    entry["chunk_audio_path"] = normalize_audio_path(entry["chunk_audio_path"])
                    rank = p31_ranking.get(tk, -1)
                    if rank == -1:
                        missing_rank += 1
                    entry["p31_rank"] = rank
                    all_entries.append(entry)
                    count += 1
            dir_count += count
            dir_filtered += shard_filtered
            filtered += shard_filtered

        print(
            f"  {d}:\n"
            f"    {dir_count:>9,} kept, {dir_filtered:>6,} filtered",
            flush=True,
        )

    if missing_rank > 0:
        print(f"  [WARN] {missing_rank:,} wiki entries had no P31 rank (set to -1)", flush=True)
    if filtered > 0:
        print(f"  [INFO] {filtered:,} wiki entries blocked by inference glossary", flush=True)
    return all_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v1.0 training dataset with P31 ranking")
    parser.add_argument("--gigaspeech-train", type=str, default=GIGASPEECH_TRAIN)
    parser.add_argument("--output-train", type=str, default=OUTPUT_TRAIN)
    parser.add_argument(
        "--smoke-test", type=int, default=0,
        help="If > 0, limit shards and gigaspeech lines for quick testing",
    )
    args = parser.parse_args()

    # Step 1: Load P31 ranking
    print("=" * 70, flush=True)
    print("[Step 1] Loading P31 ranking...", flush=True)
    p31_ranking = load_p31_ranking(P31_RANKING)
    print(f"  Ranked terms: {len(p31_ranking):,}", flush=True)

    # Step 2: Load inference blocked terms
    print(f"\n[Step 2] Loading inference glossary terms...", flush=True)
    blocked = load_inference_terms(INFERENCE_GLOSSARIES)
    print(f"  Blocked terms: {len(blocked):,}", flush=True)

    # Step 3: Load wiki_synth MFA shards
    max_shards = args.smoke_test if args.smoke_test > 0 else 0
    print(f"\n[Step 3] Loading wiki_synth MFA shards...", flush=True)
    wiki_entries = load_wiki_synth_shards(
        WIKI_SYNTH_MFA_DIRS, WIKI_SYNTH_NUM_SHARDS, p31_ranking, blocked, max_shards
    )
    print(f"  Total wiki_synth entries: {len(wiki_entries):,}", flush=True)

    # Step 4: Write output = gigaspeech (p31_rank=-1) + wiki_synth (with p31_rank)
    print(f"\n[Step 4] Writing output: {args.output_train}", flush=True)
    os.makedirs(os.path.dirname(args.output_train) or ".", exist_ok=True)

    gs_count = 0
    gs_filtered = 0

    with open(args.output_train, "w", encoding="utf-8") as fout:
        # 4a: Gigaspeech
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
                tk = entry.get("term_key", entry["term"].strip().lower())
                if tk in blocked:
                    gs_filtered += 1
                    continue
                entry["p31_rank"] = -1
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                gs_count += 1

        # 4b: Wiki_synth (already has p31_rank)
        for entry in wiki_entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total = gs_count + len(wiki_entries)

    # Compute rank distribution for wiki entries
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
        "wiki_synth_kept": len(wiki_entries),
        "total": total,
        "wiki_rank_distribution": dict(rank_bins),
    }
    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}", flush=True)
    print("[SUMMARY]", flush=True)
    print(f"  Gigaspeech:   {gs_count:>9,} kept, {gs_filtered:>6,} filtered", flush=True)
    print(f"  Wiki_synth:   {len(wiki_entries):>9,}", flush=True)
    print(f"  Total:        {total:>9,}", flush=True)
    print(f"  Wiki rank distribution:", flush=True)
    for bucket in ["0-100K", "100K-500K", "500K-1M", "1M-2M", "2M+", "no_rank"]:
        if bucket in rank_bins:
            print(f"    {bucket:>12s}: {rank_bins[bucket]:>9,}", flush=True)
    print(f"  Output: {args.output_train}", flush=True)
    print(f"  Stats:  {OUTPUT_STATS}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
