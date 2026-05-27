#!/usr/bin/env python3
"""
Build leftover training terms by filtering the full P31-balanced pool.

Pipeline:
  full_pool (4.5M JSONL)
    - remove medicine 10k glossary (for inference eval)
    - remove CS 10k glossary (for inference eval)
    → final_train_terms
    - remove existing 1M terms (already have TTS/MFA/chunks)
    → leftover_train_terms (ready for Gemini utterance generation)

Output: two JSON files (for downstream generate_term_utterances.py):
  1. final_train_terms.json     — all training-eligible terms
  2. leftover_train_terms.json  — only the new ones needing TTS pipeline

Usage:
    python build_leftover_train_terms.py
    python build_leftover_train_terms.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import time

# ======Configuration=====
FULL_POOL_JSONL = (
    "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/"
    "wiki_synth_terms_p31_balanced_full.jsonl"
)
MEDICINE_GLOSSARY = (
    "/home/jiaxuanluo/InfiniSST/"
    "documents/code/data_pre/glossary_scale/wiki_glossary_medicine.json"
)
CS_GLOSSARY = (
    "/home/jiaxuanluo/InfiniSST/"
    "documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
)
EXISTING_1M_JSON = (
    "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/"
    "wiki_synth_terms_p31_balanced_1000k.json"
)
OUTPUT_DIR = "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/p31_full"
# ======Configuration=====


def load_glossary_keys(path: str) -> set[str]:
    """Load a JSON glossary file and return a set of lowercased term keys."""
    assert os.path.isfile(path), f"Glossary not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = set()
    for entry in data:
        term = entry.get("term", entry.get("term_key", ""))
        keys.add(term.strip().lower())
    return keys


def load_jsonl_keys(path: str) -> set[str]:
    """Load a JSONL file and return a set of term_key values."""
    assert os.path.isfile(path), f"JSONL not found: {path}"
    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            keys.add(obj.get("term_key", obj.get("term", "").lower()))
    return keys


def load_json_keys(path: str) -> set[str]:
    """Load a JSON array file and return a set of term_key values."""
    assert os.path.isfile(path), f"JSON not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = set()
    for entry in data:
        keys.add(entry.get("term_key", entry.get("term", "").lower()))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leftover training terms (full pool minus eval glossaries minus existing 1M)"
    )
    parser.add_argument("--full_pool", type=str, default=FULL_POOL_JSONL)
    parser.add_argument("--medicine_glossary", type=str, default=MEDICINE_GLOSSARY)
    parser.add_argument("--cs_glossary", type=str, default=CS_GLOSSARY)
    parser.add_argument("--existing_1m", type=str, default=EXISTING_1M_JSON)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load exclusion sets
    print("=" * 60)
    print("[Step 1] Loading exclusion sets...")

    medicine_keys = load_glossary_keys(args.medicine_glossary)
    print(f"  Medicine glossary: {len(medicine_keys):,} terms")

    cs_keys = load_glossary_keys(args.cs_glossary)
    print(f"  CS glossary:       {len(cs_keys):,} terms")

    existing_keys = load_json_keys(args.existing_1m)
    print(f"  Existing 1M:       {len(existing_keys):,} terms")

    eval_keys = medicine_keys | cs_keys
    print(f"  Eval exclusion (medicine ∪ CS): {len(eval_keys):,} unique terms")
    overlap = medicine_keys & cs_keys
    if overlap:
        print(f"  [INFO] Medicine ∩ CS overlap: {len(overlap)} terms")

    # Step 2: Load full pool and filter
    print(f"\n{'=' * 60}")
    print("[Step 2] Loading full pool and filtering...")
    t0 = time.time()

    full_pool = []
    with open(args.full_pool, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            full_pool.append(json.loads(line))

    print(f"  Full pool: {len(full_pool):,} terms ({time.time()-t0:.1f}s)")

    removed_medicine = 0
    removed_cs = 0
    final_train = []
    for entry in full_pool:
        key = entry.get("term_key", entry.get("term", "").lower())
        if key in medicine_keys:
            removed_medicine += 1
            continue
        if key in cs_keys:
            removed_cs += 1
            continue
        final_train.append(entry)

    print(f"  Removed (medicine): {removed_medicine:,}")
    print(f"  Removed (CS):       {removed_cs:,}")
    print(f"  final_train_terms:  {len(final_train):,}")

    # Step 3: Remove existing 1M to get leftover
    print(f"\n{'=' * 60}")
    print("[Step 3] Removing existing 1M terms...")

    leftover = []
    removed_existing = 0
    for entry in final_train:
        key = entry.get("term_key", entry.get("term", "").lower())
        if key in existing_keys:
            removed_existing += 1
            continue
        leftover.append(entry)

    print(f"  Removed (existing 1M): {removed_existing:,}")
    print(f"  leftover_train_terms:  {len(leftover):,}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without writing files.")
        return

    # Step 4: Write outputs
    print(f"\n{'=' * 60}")
    print("[Step 4] Writing output files...")

    final_path = os.path.join(args.output_dir, "final_train_terms.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_train, f, ensure_ascii=False, indent=2)
    print(f"  final_train_terms:    {final_path} ({len(final_train):,} terms)")

    leftover_path = os.path.join(args.output_dir, "leftover_train_terms.json")
    with open(leftover_path, "w", encoding="utf-8") as f:
        json.dump(leftover, f, ensure_ascii=False, indent=2)
    print(f"  leftover_train_terms: {leftover_path} ({len(leftover):,} terms)")

    # Summary
    print(f"\n{'=' * 60}")
    print("[SUMMARY]")
    print(f"  Full pool:             {len(full_pool):,}")
    print(f"  - Medicine eval:       {removed_medicine:,}")
    print(f"  - CS eval:             {removed_cs:,}")
    print(f"  = final_train_terms:   {len(final_train):,}")
    print(f"  - Existing 1M:         {removed_existing:,}")
    print(f"  = leftover_train_terms:{len(leftover):,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
