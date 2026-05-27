"""Rewrite clean_audio_path in teammate's back-half shard JSONLs.

Teammate ran shards 22-31 on a different machine with a local
${TEAMMATE_OUTPUT_DIR} (e.g. /scratch/teammate/wav_out). Their shard*.jsonl
files therefore contain clean_audio_path values like:
    /scratch/teammate/wav_out/clean/0093/934567.wav

After we rsync those WAVs into our canonical taurus/gemini location
    /mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_full
we need the JSONLs to point there instead, so that merge_tts_3variant.py
and the downstream MFA step can actually open the files.

This script just does a prefix swap on the clean_audio_path field and
(optionally) spot-checks that the target file now exists.

Usage:
    python rewrite_teammate_wav_paths.py \\
        --jsonl-dir /mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant \\
        --shard-prefix wiki_synth_3variant_gs_v2_clean_with_tts \\
        --shards 22-31 \\
        --from-prefix /scratch/teammate/wav_out \\
        --to-prefix /mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_full \\
        --verify-sample 50

By default the script writes back in-place (.jsonl -> .jsonl) and keeps a
.bak copy. Use --no-backup to skip the .bak.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys


def parse_shards(spec: str) -> list[int]:
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def rewrite_one(
    path: str,
    from_prefix: str,
    to_prefix: str,
    backup: bool,
) -> tuple[int, int]:
    """Return (n_rewritten, n_total)."""
    if backup:
        bak = f"{path}.bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)

    tmp = f"{path}.tmp"
    n_rewritten = 0
    n_total = 0
    with open(path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            entry = json.loads(line)
            n_total += 1
            cap = entry.get("clean_audio_path")
            if isinstance(cap, str) and cap.startswith(from_prefix):
                entry["clean_audio_path"] = to_prefix + cap[len(from_prefix):]
                n_rewritten += 1
            # noisy_audio_path is unused for CLEAN-only runs, but handle it
            # defensively in case a future variant turns it back on.
            nap = entry.get("noisy_audio_path")
            if isinstance(nap, str) and nap.startswith(from_prefix):
                entry["noisy_audio_path"] = to_prefix + nap[len(from_prefix):]
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return n_rewritten, n_total


def verify_sample(path: str, sample_size: int) -> tuple[int, int]:
    """Spot-check that rewritten WAV paths actually exist after rsync.

    Returns (n_present, n_checked).
    """
    if sample_size <= 0:
        return 0, 0
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rng = random.Random(42)
    sample = rng.sample(lines, k=min(sample_size, len(lines)))
    n_present = 0
    for line in sample:
        entry = json.loads(line)
        cap = entry.get("clean_audio_path")
        if isinstance(cap, str) and os.path.isfile(cap):
            n_present += 1
    return n_present, len(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-dir", required=True,
                        help="Directory holding the shard JSONLs")
    parser.add_argument("--shard-prefix", default="wiki_synth_3variant_gs_v2_clean_with_tts",
                        help="Shard filename prefix (before `_shard{N}.jsonl`)")
    parser.add_argument("--shards", default="22-31",
                        help="Shard ids to rewrite, e.g. '22-31' or '22,24,26-31'")
    parser.add_argument("--from-prefix", required=True,
                        help="Teammate's OUTPUT_DIR (absolute path, no trailing /)")
    parser.add_argument("--to-prefix", required=True,
                        help="Your canonical OUTPUT_DIR (absolute path, no trailing /)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't keep .bak copies (default: keep)")
    parser.add_argument("--verify-sample", type=int, default=0,
                        help="After rewrite, randomly sample N lines per shard "
                             "and check the rewritten WAV exists on disk. "
                             "Set to 0 to skip.")
    args = parser.parse_args()

    from_prefix = args.from_prefix.rstrip("/")
    to_prefix = args.to_prefix.rstrip("/")

    shard_ids = parse_shards(args.shards)
    print(f"Rewriting {len(shard_ids)} shard(s) in {args.jsonl_dir}")
    print(f"  {from_prefix}  ->  {to_prefix}")
    print()

    total_rewritten = 0
    total_lines = 0
    total_present = 0
    total_checked = 0
    missing_files: list[str] = []

    for sid in shard_ids:
        path = os.path.join(args.jsonl_dir, f"{args.shard_prefix}_shard{sid}.jsonl")
        if not os.path.isfile(path):
            print(f"[WARN] shard {sid}: file not found, skipping: {path}")
            missing_files.append(path)
            continue

        n_rw, n_tot = rewrite_one(
            path, from_prefix, to_prefix, backup=not args.no_backup,
        )
        total_rewritten += n_rw
        total_lines += n_tot

        msg = f"  shard {sid:>2}: rewrote {n_rw:>7,}/{n_tot:>7,} lines"

        if args.verify_sample > 0:
            n_pres, n_chk = verify_sample(path, args.verify_sample)
            total_present += n_pres
            total_checked += n_chk
            msg += f"  (verified {n_pres}/{n_chk} wavs present)"
            if n_pres < n_chk:
                msg += "  !! MISSING WAVS"

        print(msg)

    print()
    print(f"TOTAL: rewrote {total_rewritten:,} / {total_lines:,} lines across "
          f"{len(shard_ids) - len(missing_files)} shard(s).")
    if args.verify_sample > 0:
        print(f"VERIFY: {total_present} / {total_checked} sampled WAVs exist on disk.")
        if total_present < total_checked:
            print("!! Some sampled WAVs are missing — rsync probably didn't land them "
                  "at the expected --to-prefix. Re-check paths before running MFA.")
            sys.exit(1)
    if missing_files:
        print("!! Some shard jsonls were missing:")
        for p in missing_files:
            print(f"   {p}")
        sys.exit(1)


if __name__ == "__main__":
    main()
