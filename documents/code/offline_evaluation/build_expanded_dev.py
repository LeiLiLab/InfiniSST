#!/usr/bin/env python3
"""
Build an expanded dev set by adding wiki synth entries from rank 1M-2M.
These terms were NOT used in training (wiki_rank cutoff = 1M).

Adds:
  1. has_term entries: wiki synth clean audio with their terms
  2. no_term entries: wiki synth clean audio with term="" (the chunk does contain
     a term, but since that term is from rank 1M-2M and won't appear in the
     glossary during evaluation, the retriever should NOT find a match.
     This simulates domain-diverse "background speech" without glossary terms.)

Output: expanded dev JSONL with more balanced has_term / no_term ratio
and broader domain coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

# ======Configuration=====
TRAIN_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m.jsonl"
ORIG_DEV_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
OUTPUT_DIR = "/mnt/gemini/data1/jiaxuanluo"

WIKI_RANK_LO = 1000000
WIKI_RANK_HI = 2000000

N_HAS_TERM_SAMPLE = 2000
N_NO_TERM_SAMPLE = 2000

SEED = 42
# ======Configuration=====


def main():
    random.seed(SEED)

    print("[INFO] Loading original dev set...")
    orig_lines = []
    orig_has = orig_no = 0
    for line in open(ORIG_DEV_JSONL, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        orig_lines.append(line)
        obj = json.loads(line)
        if obj.get("term", ""):
            orig_has += 1
        else:
            orig_no += 1
    print(f"  Original dev: has_term={orig_has}, no_term={orig_no}, total={len(orig_lines)}")

    print(f"[INFO] Scanning training JSONL for wiki entries with rank [{WIKI_RANK_LO}, {WIKI_RANK_HI})...")
    wiki_entries_by_term = defaultdict(list)
    scanned = 0
    for line in open(TRAIN_JSONL, "r", encoding="utf-8"):
        scanned += 1
        obj = json.loads(line.strip())
        rank = obj.get("p31_rank", -1)
        if WIKI_RANK_LO <= rank < WIKI_RANK_HI:
            if obj.get("audio_type") == "clean":
                wiki_entries_by_term[obj["term_key"]].append(obj)
        if scanned % 500000 == 0:
            print(f"  scanned {scanned:,} lines, found {len(wiki_entries_by_term):,} unique terms so far")

    print(f"  Total unique wiki terms in rank [{WIKI_RANK_LO}, {WIKI_RANK_HI}): {len(wiki_entries_by_term)}")

    all_term_keys = list(wiki_entries_by_term.keys())
    random.shuffle(all_term_keys)

    assert len(all_term_keys) >= N_HAS_TERM_SAMPLE + N_NO_TERM_SAMPLE, (
        f"Not enough wiki terms: {len(all_term_keys)} < {N_HAS_TERM_SAMPLE + N_NO_TERM_SAMPLE}"
    )

    has_term_keys = all_term_keys[:N_HAS_TERM_SAMPLE]
    no_term_keys = all_term_keys[N_HAS_TERM_SAMPLE:N_HAS_TERM_SAMPLE + N_NO_TERM_SAMPLE]

    new_has_term_lines = []
    for tk in has_term_keys:
        entries = wiki_entries_by_term[tk]
        entry = random.choice(entries)
        new_obj = {
            "term": entry["term"],
            "term_key": entry["term_key"],
            "chunk_src_text": entry.get("chunk_src_text", ""),
            "utter_id": entry["utter_id"],
            "chunk_idx": entry.get("chunk_idx", 0),
            "chunk_audio_path": entry["chunk_audio_path"],
        }
        new_has_term_lines.append(json.dumps(new_obj, ensure_ascii=False))

    new_no_term_lines = []
    for tk in no_term_keys:
        entries = wiki_entries_by_term[tk]
        entry = random.choice(entries)
        new_obj = {
            "term": "",
            "term_key": "",
            "chunk_src_text": entry.get("chunk_src_text", ""),
            "utter_id": entry["utter_id"],
            "chunk_idx": entry.get("chunk_idx", 0),
            "chunk_audio_path": entry["chunk_audio_path"],
        }
        new_no_term_lines.append(json.dumps(new_obj, ensure_ascii=False))

    all_lines = orig_lines + new_has_term_lines + new_no_term_lines
    random.shuffle(all_lines)

    total_has = orig_has + len(new_has_term_lines)
    total_no = orig_no + len(new_no_term_lines)

    output_path = os.path.join(OUTPUT_DIR, "term_dev_expanded_wiki1m2m.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    print(f"\n[INFO] Expanded dev set written to: {output_path}")
    print(f"  Original: has_term={orig_has}, no_term={orig_no}")
    print(f"  Added:    has_term={len(new_has_term_lines)}, no_term={len(new_no_term_lines)}")
    print(f"  Total:    has_term={total_has}, no_term={total_no}, all={len(all_lines)}")
    print(f"  has_term ratio: {total_has / len(all_lines):.3f}")


if __name__ == "__main__":
    main()
