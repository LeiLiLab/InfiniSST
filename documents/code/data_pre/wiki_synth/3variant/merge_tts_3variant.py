#!/usr/bin/env python3
"""
Merge TTS shard outputs from 3-variant pipeline into a single JSONL.

The TTS script saves per-shard JSONLs:
  <data_dir>/wiki_synth_3variant_with_tts_shard{id}.jsonl
Each entry has {term, utterance, variant_idx, clean_audio_path, [noisy_audio_path]}.
(`noisy_audio_path` is optional: present iff TTS was run with --noise-dir set.)

This merges all shards, verifies audio files exist, and produces a
unified JSONL ready for MFA alignment.

Usage:
    python merge_tts_3variant.py
    python merge_tts_3variant.py --smoke_test 100
"""

from __future__ import annotations

import argparse
import json
import os
import random

# ======Configuration=====
SHARD_JSONL_DIR = "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant"
SHARD_PREFIX = "wiki_synth_3variant_with_tts"
TOTAL_SHARDS = 8
OUTPUT_JSONL = (
    "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant/"
    "wiki_synth_3variant_dual.jsonl"
)
VERIFY_SAMPLE_SIZE = 200
RANDOM_SEED = 42
# ======Configuration=====


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge 3-variant TTS shard outputs")
    parser.add_argument("--shard_dir", type=str, default=SHARD_JSONL_DIR)
    parser.add_argument("--shard_prefix", type=str, default=SHARD_PREFIX)
    parser.add_argument("--total_shards", type=int, default=TOTAL_SHARDS)
    parser.add_argument("--output", type=str, default=OUTPUT_JSONL)
    parser.add_argument("--smoke_test", type=int, default=0)
    args = parser.parse_args()

    all_entries: list[dict] = []

    for shard_id in range(args.total_shards):
        shard_path = os.path.join(
            args.shard_dir,
            f"{args.shard_prefix}_shard{shard_id}.jsonl",
        )
        assert os.path.isfile(shard_path), f"Missing shard {shard_id}: {shard_path}"

        count = 0
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                assert "clean_audio_path" in entry, (
                    f"Shard {shard_id}: missing clean_audio_path"
                )
                all_entries.append(entry)
                count += 1
                if args.smoke_test > 0 and count >= args.smoke_test:
                    break
        print(f"  Shard {shard_id}: {count:>8,} entries", flush=True)

    print(f"\nTotal entries: {len(all_entries):,}")

    total_with_noisy = sum(1 for e in all_entries if "noisy_audio_path" in e)
    has_noisy = total_with_noisy == len(all_entries)
    if not has_noisy and total_with_noisy > 0:
        # Partially noisy: refuse — indicates a corrupt / mixed-mode shard set.
        raise RuntimeError(
            f"Inconsistent TTS output: {total_with_noisy}/{len(all_entries)} "
            f"entries have noisy_audio_path. Either all or none must be noisy."
        )
    mode = "dual (clean+noisy)" if has_noisy else "clean-only"
    print(f"TTS output mode: {mode}")

    # Verify random sample of audio files
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(all_entries, min(VERIFY_SAMPLE_SIZE, len(all_entries)))
    clean_ok = 0
    noisy_ok = 0
    for entry in sample:
        assert os.path.isfile(entry["clean_audio_path"]), (
            f"Clean audio missing: {entry['clean_audio_path']}"
        )
        clean_ok += 1
        if has_noisy:
            assert os.path.isfile(entry["noisy_audio_path"]), (
                f"Noisy audio missing: {entry['noisy_audio_path']}"
            )
            noisy_ok += 1
    if has_noisy:
        print(f"Verified {clean_ok} clean + {noisy_ok} noisy audio files (sample)")
    else:
        print(f"Verified {clean_ok} clean audio files (sample)")

    # Count unique terms / variants
    term_variants: dict[str, set[int]] = {}
    for entry in all_entries:
        t = entry["term"]
        v = entry.get("variant_idx", 0)
        term_variants.setdefault(t, set()).add(v)
    total_terms = len(term_variants)
    avg_variants = sum(len(vs) for vs in term_variants.values()) / max(total_terms, 1)
    print(f"Unique terms: {total_terms:,}, avg variants/term: {avg_variants:.1f}")

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in all_entries:
            out = {
                "term": entry["term"],
                "utterance": entry["utterance"],
                "variant_idx": entry.get("variant_idx", 0),
                "clean_audio_path": entry["clean_audio_path"],
            }
            if has_noisy:
                out["noisy_audio_path"] = entry["noisy_audio_path"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nOutput: {args.output} ({len(all_entries):,} entries)")


if __name__ == "__main__":
    main()
