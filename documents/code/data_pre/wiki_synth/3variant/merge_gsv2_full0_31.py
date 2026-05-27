#!/usr/bin/env python3
"""Merge local GSV2 shards 0-21 with teammate GSV2 shards 22-31.

The teammate handoff is a single merged JSONL whose clean_audio_path values
still point to the teammate machine. This script rewrites that prefix while
streaming all rows into a canonical full 0-31 JSONL for MFA.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Iterable


DEFAULT_LOCAL_DIR = "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant"
DEFAULT_LOCAL_PREFIX = "wiki_synth_3variant_gs_v2_clean_with_tts"
DEFAULT_TEAMMATE_JSONL = (
    "/mnt/data2/siqiouyang/datasets/gigaspeech/"
    "wiki_synth_3variant_gs_v2_clean_with_tts_merged.jsonl"
)
DEFAULT_TEAMMATE_FROM_PREFIX = (
    "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/"
    "wiki_synth_utterances_tts_gigaspk_22-31"
)
DEFAULT_TEAMMATE_TO_PREFIX = (
    "/mnt/data2/siqiouyang/datasets/gigaspeech/"
    "wiki_synth_utterances_tts_gigaspk_22-31"
)
DEFAULT_OUTPUT = (
    "/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_full0_31/"
    "wiki_synth_3variant_gs_v2_clean_full0_31.jsonl"
)


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


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def maybe_sample(
    rng: random.Random,
    sample: list[str],
    seen: int,
    candidate: str,
    sample_size: int,
) -> None:
    if sample_size <= 0:
        return
    if len(sample) < sample_size:
        sample.append(candidate)
        return
    j = rng.randrange(seen)
    if j < sample_size:
        sample[j] = candidate


def rewrite_teammate_path(path: str, from_prefix: str, to_prefix: str) -> tuple[str, bool]:
    if path.startswith(from_prefix):
        return to_prefix + path[len(from_prefix):], True
    return path, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--local-prefix", default=DEFAULT_LOCAL_PREFIX)
    parser.add_argument("--local-shards", default="0-21")
    parser.add_argument("--teammate-jsonl", default=DEFAULT_TEAMMATE_JSONL)
    parser.add_argument("--teammate-from-prefix", default=DEFAULT_TEAMMATE_FROM_PREFIX)
    parser.add_argument("--teammate-to-prefix", default=DEFAULT_TEAMMATE_TO_PREFIX)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-sample", type=int, default=1000)
    parser.add_argument("--expected-total", type=int, default=2_998_703)
    args = parser.parse_args()

    local_shards = parse_shards(args.local_shards)
    for sid in local_shards:
        path = os.path.join(args.local_dir, f"{args.local_prefix}_shard{sid}.jsonl")
        assert os.path.isfile(path), f"Missing local shard {sid}: {path}"
    assert os.path.isfile(args.teammate_jsonl), f"Missing teammate JSONL: {args.teammate_jsonl}"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp_path = args.output + ".tmp"
    rng = random.Random(42)
    wav_sample: list[str] = []
    stats = Counter()
    unique_terms: set[str] = set()

    with open(tmp_path, "w", encoding="utf-8") as fout:
        for sid in local_shards:
            path = os.path.join(args.local_dir, f"{args.local_prefix}_shard{sid}.jsonl")
            shard_count = 0
            for entry in iter_jsonl(path):
                clean_path = entry.get("clean_audio_path")
                assert isinstance(clean_path, str) and clean_path, (
                    f"{path}: missing clean_audio_path"
                )
                if "noisy_audio_path" in entry:
                    raise RuntimeError(f"{path}: unexpected noisy_audio_path in clean-only run")
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stats["local_rows"] += 1
                stats[f"local_shard_{sid:02d}_rows"] += 1
                shard_count += 1
                unique_terms.add(str(entry.get("term", "")).strip().lower())
                maybe_sample(rng, wav_sample, stats["total_seen"] + 1, clean_path, args.verify_sample)
                stats["total_seen"] += 1
            print(f"[LOCAL] shard {sid:02d}: {shard_count:,} rows", flush=True)

        teammate_rewritten = 0
        teammate_unrewritten = 0
        for entry in iter_jsonl(args.teammate_jsonl):
            clean_path = entry.get("clean_audio_path")
            assert isinstance(clean_path, str) and clean_path, (
                f"{args.teammate_jsonl}: missing clean_audio_path"
            )
            clean_path, did_rewrite = rewrite_teammate_path(
                clean_path,
                args.teammate_from_prefix.rstrip("/"),
                args.teammate_to_prefix.rstrip("/"),
            )
            entry["clean_audio_path"] = clean_path
            if did_rewrite:
                teammate_rewritten += 1
            else:
                teammate_unrewritten += 1
            if "noisy_audio_path" in entry:
                raise RuntimeError(
                    f"{args.teammate_jsonl}: unexpected noisy_audio_path in clean-only run"
                )
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stats["teammate_rows"] += 1
            unique_terms.add(str(entry.get("term", "")).strip().lower())
            maybe_sample(rng, wav_sample, stats["total_seen"] + 1, clean_path, args.verify_sample)
            stats["total_seen"] += 1

    os.replace(tmp_path, args.output)

    missing_wavs: list[str] = []
    for wav_path in wav_sample:
        if not os.path.isfile(wav_path):
            missing_wavs.append(wav_path)

    stats.update(
        {
            "total_rows": stats["local_rows"] + stats["teammate_rows"],
            "unique_terms": len(unique_terms),
            "teammate_rewritten_rows": teammate_rewritten,
            "teammate_unrewritten_rows": teammate_unrewritten,
            "verify_sample": len(wav_sample),
            "verify_missing_wavs": len(missing_wavs),
        }
    )

    stats_path = args.output.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, indent=2, sort_keys=True)

    print("[TEAMMATE] rows:", f"{stats['teammate_rows']:,}", flush=True)
    print("[TEAMMATE] rewritten:", f"{teammate_rewritten:,}", flush=True)
    print("[DONE] output:", args.output, flush=True)
    print("[DONE] stats:", stats_path, flush=True)
    print(json.dumps(dict(stats), indent=2, sort_keys=True), flush=True)

    if args.expected_total > 0 and stats["total_rows"] != args.expected_total:
        raise RuntimeError(
            f"Expected {args.expected_total:,} rows, got {stats['total_rows']:,}"
        )
    if teammate_unrewritten:
        raise RuntimeError(f"{teammate_unrewritten:,} teammate rows were not rewritten")
    if missing_wavs:
        print("[ERROR] sampled missing WAV paths:", flush=True)
        for path in missing_wavs[:20]:
            print(path, flush=True)
        raise RuntimeError(f"{len(missing_wavs)} sampled WAVs are missing")


if __name__ == "__main__":
    main()
