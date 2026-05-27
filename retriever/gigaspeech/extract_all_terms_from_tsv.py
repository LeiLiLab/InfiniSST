#!/usr/bin/env python3
"""
Script to extract (term, chunk_src_text, chunk_audio_path) using pre-extracted NER candidates.
Optimized for CPU multiprocessing.
"""

import os
import json
import re
import argparse
import logging
import ast
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp

logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
DEFAULT_INPUT_TSV = "/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
DEFAULT_OUTPUT_DIR = "/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks"
DEFAULT_OUTPUT_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_train_dataset_v2.jsonl"

SAMPLE_RATE = 16000
UNIT_DURATION_SEC = 0.96
SAMPLES_PER_UNIT = int(SAMPLE_RATE * UNIT_DURATION_SEC) # 15360

def get_term_key(text: str) -> str:
    if not text: return ""
    return " ".join(text.lower().split())

def parse_audio_spec(audio_spec: str) -> Tuple[str, Optional[int], Optional[int]]:
    if not audio_spec:
        return "", None, None
    parts = audio_spec.split(":")
    path = parts[0]
    def _parse_int(idx: int) -> Optional[int]:
        if len(parts) <= idx or not parts[idx]: return None
        try: return int(parts[idx])
        except ValueError: return None
    return path, _parse_int(1), _parse_int(2)

def process_row(args_tuple):
    row, ner_candidates, multiplier_merge, output_dir, output_jsonl_path = args_tuple
    uid = row["id"]
    src_traj_raw = row["src_trajectory"]
    audio_spec = row.get("audio", "")

    if not ner_candidates:
        return []

    try: 
        src_traj = ast.literal_eval(src_traj_raw)
        if not isinstance(src_traj, list): src_traj = []
    except:
        return []
    
    if not src_traj:
        return []

    audio_path, start_frame, total_frames_spec = parse_audio_spec(audio_spec)
    if not audio_path or not os.path.exists(audio_path):
        return []

    term_regexes = []
    for t in ner_candidates:
        try: 
            pattern = re.compile(r'(?<![A-Za-z0-9])' + re.escape(t) + r'(?![A-Za-z0-9])', re.IGNORECASE)
            term_regexes.append((t, pattern))
        except: continue

    if not term_regexes:
        return []

    results = []
    try:
        with sf.SoundFile(audio_path) as src:
            for j in range(len(src_traj) - multiplier_merge + 1):
                chunk_text = " ".join(src_traj[j : j + multiplier_merge]).strip()
                if not chunk_text: continue
                
                chunk_terms = [t for t, p in term_regexes if p.search(chunk_text)]
                if not chunk_terms: continue
                
                seg_start = j * SAMPLES_PER_UNIT
                seg_end = (j + multiplier_merge) * SAMPLES_PER_UNIT
                
                absolute_start = (start_frame or 0) + seg_start
                frames_to_read = seg_end - seg_start
                
                if seg_start >= (total_frames_spec or len(src)): continue
                
                src.seek(absolute_start)
                data = src.read(frames_to_read, dtype="float32", always_2d=False)
                
                if data.size == 0: continue
                if data.ndim > 1: data = data.mean(axis=1)
                
                chunk_wav_name = f"{uid}_chunk_{j}.wav"
                chunk_wav_path = os.path.join(output_dir, chunk_wav_name)
                sf.write(chunk_wav_path, data, SAMPLE_RATE, subtype="PCM_16")

                for term in chunk_terms:
                    results.append({
                        "term": term, 
                        "term_key": get_term_key(term),
                        "chunk_src_text": chunk_text, 
                        "utter_id": uid, 
                        "chunk_idx": j,
                        "chunk_audio_path": chunk_wav_path
                    })
    except Exception as e:
        # logger.debug(f"Error processing {uid}: {e}")
        pass
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", default=DEFAULT_INPUT_TSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--multiplier-merge", type=int, default=2)
    parser.add_argument("--ner-candidates-jsonl", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    if not args.output_dir.endswith(f"_m{args.multiplier_merge}"):
        args.output_dir = args.output_dir.rstrip("/") + f"_m{args.multiplier_merge}"
    
    os.makedirs(args.output_dir, exist_ok=True)

    ner_map = {}
    logger.info(f"Loading NER candidates from {args.ner_candidates_jsonl}...")
    with open(args.ner_candidates_jsonl, "r") as f:
        for line in tqdm(f, desc="Loading NER"):
            try:
                obj = json.loads(line)
                ner_map[obj["utter_id"]] = obj["ner_candidates"]
            except: continue
    logger.info(f"Loaded {len(ner_map)} utterances' candidates.")

    output_jsonl = args.output_jsonl
    if args.total_shards > 1:
        output_jsonl = output_jsonl.replace(".jsonl", f"_shard{args.shard_id}.jsonl")

    rows_to_process = []
    with open(args.input_tsv, "r", encoding="utf-8") as f_in:
        header = f_in.readline().rstrip("\n").split("\t")
        col_map = {name: i for i, name in enumerate(header)}
        for line_idx, line in enumerate(f_in):
            if args.total_shards > 1 and line_idx % args.total_shards != args.shard_id: continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header): continue
            
            row = {name: parts[i] for name, i in col_map.items()}
            uid = row["id"]
            if uid in ner_map:
                rows_to_process.append((row, ner_map[uid], args.multiplier_merge, args.output_dir, output_jsonl))
            
            if args.max_rows and len(rows_to_process) >= args.max_rows:
                break

    logger.info(f"Starting Multiprocessing with {args.num_workers} workers for {len(rows_to_process)} rows...")
    
    with mp.Pool(args.num_workers) as pool:
        results_iterator = pool.imap_unordered(process_row, rows_to_process)
        
        with open(output_jsonl, "w", encoding="utf-8") as out_f:
            for results in tqdm(results_iterator, total=len(rows_to_process), desc="Processing"):
                for res in results:
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")

    logger.info("Done.")

if __name__ == "__main__":
    main()
