#!/usr/bin/env python3
"""
Script to extract REAL NO_TERM chunks from the TSV.
A chunk is considered a valid negative sample ONLY IF:
1. It is not already in the positive dataset.
2. It contains NO known terms from the global term library (Regex/N-gram check).
"""

import os
import sys
import json
import argparse
import logging
import ast
from pathlib import Path
from tqdm import tqdm  # type: ignore
import soundfile as sf  # type: ignore

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
UNIT_DURATION_SEC = 0.96
SAMPLES_PER_UNIT = int(SAMPLE_RATE * UNIT_DURATION_SEC)  # 15360

def get_term_key(text: str) -> str:
    if not text: return ""
    return " ".join(text.lower().split())

def parse_audio_spec(audio_spec: str):
    if not audio_spec:
        return "", None, None
    parts = audio_spec.split(":")
    path = parts[0]
    def _parse_int(idx: int):
        if len(parts) <= idx or not parts[idx]:
            return None
        try:
            return int(parts[idx])
        except ValueError:
            return None
    return path, _parse_int(1), _parse_int(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", default="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv")
    parser.add_argument("--input-jsonl", default="/mnt/gemini/data1/jiaxuanluo/term_train_dataset_v2.jsonl", help="The sampled positive term dataset")
    parser.add_argument("--full-term-jsonl", default="/mnt/gemini/data1/jiaxuanluo/term_train_dataset_v2.jsonl", help="The full original term library for strict filtering")
    parser.add_argument("--output-dir", default="/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks")
    parser.add_argument("--output-jsonl", default="/mnt/gemini/data1/jiaxuanluo/term_train_dataset_all_neg.jsonl")
    parser.add_argument("--multiplier-merge", type=int, default=2, help="How many 0.96s units to merge for one chunk (must match positive dataset).")
    parser.add_argument("--write-audio", action="store_true", help="If set, materialize chunk wavs into --output-dir for no-term samples.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard id for TSV line-level sharding.")
    parser.add_argument("--total-shards", type=int, default=1, help="Total shards for TSV line-level sharding.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 1. Load existing positive chunks (to skip them)
    logger.info(f"Loading existing positive chunks from {args.input_jsonl}...")
    positive_chunk_ids = set()
    if os.path.exists(args.input_jsonl):
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    positive_chunk_ids.add((data["utter_id"], data["chunk_idx"]))
                except: continue
    logger.info(f"Loaded {len(positive_chunk_ids)} sampled positive chunks.")

    # 2. Build Global Term Library (The Guard Set)
    # We use the full original extraction to ensure NO 'leaked' terms enter negatives
    logger.info(f"Building global term library from {args.full_term_jsonl}...")
    known_terms_set = set()
    if os.path.exists(args.full_term_jsonl):
        with open(args.full_term_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    t = data.get("term")
                    if t:
                        known_terms_set.add(get_term_key(t))
                except: continue
    logger.info(f"Global term library size: {len(known_terms_set)} unique term keys.")

    # 3. Iterate through TSV and find pure negative chunks
    logger.info(f"Streaming TSV and verifying negative chunks...")
    
    neg_count = 0
    with open(args.input_tsv, "r", encoding="utf-8") as f_in, \
         open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        
        header = f_in.readline().rstrip("\n").split("\t")
        col_map = {name: i for i, name in enumerate(header)}
        
        for line_idx, line in enumerate(tqdm(f_in, desc="Scanning TSV")):
            if args.total_shards > 1 and (line_idx % args.total_shards) != args.shard_id:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header): continue
            
            uid = parts[col_map["id"]]
            try:
                src_traj = ast.literal_eval(parts[col_map["src_trajectory"]])
            except: continue
            
            N = len(src_traj)
            audio_spec = parts[col_map["audio"]] if "audio" in col_map else ""
            audio_path, start_frame, total_frames_spec = parse_audio_spec(audio_spec)
            if args.write_audio:
                if not audio_path or not os.path.exists(audio_path):
                    continue
                os.makedirs(args.output_dir, exist_ok=True)
            if N < 1:
                continue
            
            m = int(args.multiplier_merge)
            if m <= 0:
                continue
            if N < m:
                continue
            for j in range(N - m + 1):
                # Rule 1: Skip if it's already a positive sample
                if (uid, j) in positive_chunk_ids:
                    continue
                
                chunk_src_text = " ".join(src_traj[j : j + m]).strip()
                if not chunk_src_text: continue
                
                # Rule 2: Strict N-gram check against Global Term Library
                # This catches 'technology' even if the extractor missed it this time
                words = chunk_src_text.lower().split()
                has_leaked_term = False
                # Check 1-gram to 4-gram
                for n in range(1, 5):
                    for i in range(len(words) - n + 1):
                        gram = " ".join(words[i : i+n])
                        if gram in known_terms_set:
                            has_leaked_term = True
                            break
                    if has_leaked_term: break
                
                if has_leaked_term:
                    continue # This chunk contains a term we know about! Skip.

                # Passed both rules -> True NO_TERM sample
                chunk_wav_path = os.path.join(args.output_dir, f"{uid}_chunk_{j}.wav")
                if args.write_audio:
                    # Ensure wav exists (write if missing). If we fail to materialize, skip emitting JSONL.
                    if not os.path.exists(chunk_wav_path):
                        try:
                            with sf.SoundFile(audio_path) as src:
                                seg_start = j * SAMPLES_PER_UNIT
                                seg_end = (j + m) * SAMPLES_PER_UNIT
                                absolute_start = (start_frame or 0) + seg_start
                                frames_to_read = seg_end - seg_start

                                if total_frames_spec is not None and seg_start >= total_frames_spec:
                                    continue
                                if absolute_start < 0:
                                    continue

                                src.seek(absolute_start)
                                data = src.read(frames_to_read, dtype="float32", always_2d=False)
                                if getattr(data, "size", 0) == 0:
                                    continue
                                if getattr(data, "ndim", 1) > 1:
                                    data = data.mean(axis=1)
                                sf.write(chunk_wav_path, data, SAMPLE_RATE, subtype="PCM_16")
                        except Exception:
                            continue
                neg_item = {
                    "term": "", "term_key": "",
                    "chunk_src_text": chunk_src_text,
                    "utter_id": uid, "chunk_idx": j,
                    "chunk_audio_path": chunk_wav_path
                }
                f_out.write(json.dumps(neg_item, ensure_ascii=False) + "\n")
                neg_count += 1

    logger.info(f"Done. Extracted {neg_count} PURE negative chunks to {args.output_jsonl}.")

if __name__ == "__main__":
    main()
