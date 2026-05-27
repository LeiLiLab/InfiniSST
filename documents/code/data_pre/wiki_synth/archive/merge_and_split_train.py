#!/usr/bin/env python3
"""
Merge MFA-processed wiki_synth shard outputs, split off a test set,
and combine with existing gigaspeech training data.

Steps:
  1. Merge all shard JSONL files from MFA processing
  2. Verify audio chunk files exist
  3. Split off 1k entries as test set (stratified: diverse terms)
  4. Merge remaining wiki_synth train entries with existing gigaspeech train data
  5. Output: merged train JSONL + test JSONL

Usage:
    python merge_and_split_train.py
    python merge_and_split_train.py --test-size 1000
    python merge_and_split_train.py --skip-audio-check
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

# ======Configuration=====
SHARD_JSONL_DIRS = [
    "/mnt/data/jiaxuanluo/wiki_synth_mfa_p31_output",
]
SHARD_PATTERN = "wiki_synth_train_shard_{:02d}.jsonl"
NUM_SHARDS = 20

ORIGINAL_TRAIN_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_train_dataset_final.jsonl"

OUTPUT_DIR = "/mnt/gemini/data1/jiaxuanluo"

TEST_SIZE = 1000
RANDOM_SEED = 42
# ======Configuration=====


def load_shard_jsonls(shard_dirs: List[str], num_shards: int) -> List[dict]:
    """Load and merge all shard JSONL files, searching across multiple directories."""
    all_entries = []
    for shard_id in range(num_shards):
        shard_filename = SHARD_PATTERN.format(shard_id)
        shard_path = None
        for d in shard_dirs:
            candidate = os.path.join(d, shard_filename)
            if os.path.isfile(candidate):
                shard_path = candidate
                break
        assert shard_path is not None, (
            f"Missing shard {shard_id}: searched {[os.path.join(d, shard_filename) for d in shard_dirs]}"
        )
        count = 0
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_entries.append(json.loads(line))
                count += 1
        print(f"  Shard {shard_id:2d}: {count:6d} entries  <- {shard_path}", flush=True)
    return all_entries


def verify_audio_exists(entries: List[dict]) -> int:
    """Check that all chunk_audio_path files exist. Returns count of missing."""
    missing = 0
    for entry in entries:
        path = entry.get("chunk_audio_path", "")
        if not path or not os.path.isfile(path):
            missing += 1
    return missing


def split_test_set(
    entries: List[dict],
    test_size: int,
    rng: random.Random,
) -> tuple:
    """Split entries into train and test, ensuring diverse term coverage.

    Selects test_size unique terms, then includes ALL rows for those terms
    (both clean and noisy versions).  Returns (train_entries, test_entries).
    """
    from collections import defaultdict

    term_to_entries: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        tk = entry.get("term_key", entry["term"].lower())
        term_to_entries[tk].append(entry)

    all_term_keys = list(term_to_entries.keys())
    rng.shuffle(all_term_keys)

    test_term_keys = set(all_term_keys[:test_size])
    test_entries = []
    remaining = []

    for entry in entries:
        tk = entry.get("term_key", entry["term"].lower())
        if tk in test_term_keys:
            test_entries.append(entry)
        else:
            remaining.append(entry)

    assert len(test_term_keys) == test_size, (
        f"Expected {test_size} test terms, got {len(test_term_keys)}"
    )
    return remaining, test_entries


def write_jsonl(entries: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Written: {path} ({len(entries)} entries)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge wiki_synth shards, split test set, merge with gigaspeech"
    )
    parser.add_argument(
        "--shard-dirs", type=str, nargs="+", default=SHARD_JSONL_DIRS,
        help="Directories to search for shard JSONL files (checked in order)",
    )
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    parser.add_argument("--original-train", type=str, default=ORIGINAL_TRAIN_JSONL)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-audio-check", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Step 1: Merge all shard JSONLs
    print("=" * 60, flush=True)
    print("[Step 1] Loading shard JSONL files...", flush=True)
    all_wiki = load_shard_jsonls(args.shard_dirs, args.num_shards)
    print(f"  Total wiki_synth entries: {len(all_wiki)}", flush=True)

    # Step 2: Verify audio
    if not args.skip_audio_check:
        print("\n[Step 2] Verifying audio files exist...", flush=True)
        missing = verify_audio_exists(all_wiki)
        assert missing == 0, f"{missing} entries have missing audio files!"
        print(f"  All {len(all_wiki)} audio files verified.", flush=True)
    else:
        print("\n[Step 2] Skipping audio verification.", flush=True)

    # Save the full wiki_synth JSONL (before split)
    wiki_all_path = os.path.join(args.output_dir, "wiki_synth_train_all.jsonl")
    write_jsonl(all_wiki, wiki_all_path)

    # Step 3: Split test set
    print(f"\n[Step 3] Splitting {args.test_size} test entries...", flush=True)
    wiki_train, wiki_test = split_test_set(all_wiki, args.test_size, rng)
    print(f"  Train: {len(wiki_train)}, Test: {len(wiki_test)}", flush=True)

    unique_test_terms = len({e.get("term_key", e["term"].lower()) for e in wiki_test})
    test_clean = sum(1 for e in wiki_test if e.get("audio_type") == "clean")
    test_noisy = sum(1 for e in wiki_test if e.get("audio_type") == "noisy")
    print(f"  Test unique terms: {unique_test_terms}", flush=True)
    print(f"  Test rows: {len(wiki_test)} (clean={test_clean}, noisy={test_noisy})", flush=True)

    wiki_test_path = os.path.join(args.output_dir, "wiki_synth_test_1k.jsonl")
    write_jsonl(wiki_test, wiki_test_path)

    # Step 4: Merge with original training data
    print(f"\n[Step 4] Merging with original training data...", flush=True)
    merged_path = os.path.join(args.output_dir, "term_train_dataset_final_with_wiki_synth.jsonl")

    if os.path.isfile(args.original_train):
        original_count = 0
        with open(merged_path, "w", encoding="utf-8") as out_f:
            with open(args.original_train, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    original_count += 1
            for entry in wiki_train:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        total = original_count + len(wiki_train)
        print(f"  Original: {original_count}, Wiki_synth train: {len(wiki_train)}", flush=True)
        print(f"  Merged total: {total}", flush=True)
        print(f"  Written: {merged_path}", flush=True)
    else:
        print(f"  [WARN] Original train not found: {args.original_train}", flush=True)
        print(f"  Writing wiki_synth-only as merged output.", flush=True)
        write_jsonl(wiki_train, merged_path)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("[SUMMARY]", flush=True)
    print(f"  Wiki_synth total:     {len(all_wiki)}", flush=True)
    print(f"  Wiki_synth train:     {len(wiki_train)}", flush=True)
    print(f"  Wiki_synth test:      {len(wiki_test)}", flush=True)
    if os.path.isfile(args.original_train):
        print(f"  Original train:       {original_count}", flush=True)
        print(f"  Merged train total:   {total}", flush=True)
    print(f"\n  All wiki_synth:       {wiki_all_path}", flush=True)
    print(f"  Test set:             {wiki_test_path}", flush=True)
    print(f"  Merged train:         {merged_path}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
