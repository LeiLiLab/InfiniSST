#!/usr/bin/env python3
"""
Script to extract (term, chunk_src_text, chunk_audio_path) from TSV.
Refactored to remove LLM alignment and improve term extraction/filtering.
"""

import os
import sys
import json
import re
import argparse
import logging
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import soundfile as sf
import spacy
from tqdm import tqdm

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

# ----------------------------
# Helpers
# ----------------------------
def _load_spacy(use_gpu=False):
    import time
    
    max_retries = 3
    for attempt in range(max_retries):
        if use_gpu:
            try:
                spacy.require_gpu()
                logger.info(f"spaCy SUCCESS: GPU activated (attempt {attempt+1}).")
            except Exception as e:
                logger.warning(f"spaCy WARNING: Could not activate GPU ({e}). attempt {attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                logger.error("Falling back to CPU.")
        
        try:
            logger.info("Attempting to load en_core_web_trf...")
            return spacy.load("en_core_web_trf")
        except Exception as e:
            logger.warning(f"Failed to load en_core_web_trf: {e}")
            try:
                logger.info("Trying en_core_web_lg...")
                return spacy.load("en_core_web_lg")
            except Exception:
                logger.info("Trying en_core_web_sm...")
                return spacy.load("en_core_web_sm")

def get_term_key(text: str) -> str:
    """
    Lower + whitespace normalize.
    """
    if not text:
        return ""
    return " ".join(text.lower().split())

def is_acronym(text: str) -> bool:
    """
    Check if a string is an acronym (all caps, len > 1).
    """
    return text.isupper() and len(text) > 1 and any(c.isalpha() for c in text)

def is_valid_term(span_or_token) -> bool:
    """
    Check if a spaCy span or token is a valid term based on POS and content rules.
    """
    text = span_or_token.text.strip()
    if not text or len(text) < 2:
        return False
    
    # 1. Must contain at least one alphanumeric character
    if not any(c.isalnum() for c in text):
        return False
    
    # 2. Filter by tokens
    # Handle both spacy.tokens.Span and spacy.tokens.Token
    if hasattr(span_or_token, "__iter__"):
        tokens = [t for t in span_or_token if not t.is_punct and not t.is_space]
    else:
        tokens = [span_or_token] if not span_or_token.is_punct and not span_or_token.is_space else []
    
    if not tokens:
        return False
        
    # 3. Discard pure pronouns/determiners/疑问词 (e.g., "you", "she", "what", "which")
    if all(t.pos_ in {"PRON", "DET"} for t in tokens):
        return False
        
    # 4. Discard pure stopwords
    if all(t.is_stop for t in tokens):
        return False

    # 5. For single-token terms, be even stricter
    if len(tokens) == 1:
        t = tokens[0]
        # Discard common function words
        if t.is_stop or t.pos_ in {"PRON", "DET", "PART", "ADP", "CONJ", "CCONJ", "SCONJ", "AUX"}:
            return False
            
    # 6. Discard if it starts with an apostrophe (fragments like "'re")
    if text.startswith("'"):
        return False

    return True

def get_filtered_candidates(doc) -> List[str]:
    candidates = []
    
    # 1. Noun chunks (Primary)
    for chunk in doc.noun_chunks:
        if is_valid_term(chunk):
            text = chunk.text.strip()
            # Strip leading determiners manually if spaCy didn't
            toks = text.split()
            if toks and toks[0].lower() in {"the", "a", "an", "this", "that", "these", "those"}:
                text = " ".join(toks[1:])
            if text and len(text) >= 2:
                candidates.append(text)
            
    # 2. NER Entities (Supplementary)
    for ent in doc.ents:
        if is_valid_term(ent):
            candidates.append(ent.text.strip())
    
    # 3. Fallback: Single tokens (Strict: PROPN or Acronym only)
    for token in doc:
        is_propn = token.pos_ == "PROPN"
        is_abbr = is_acronym(token.text)
        if (is_propn or is_abbr) and is_valid_term(token):
            candidates.append(token.text.strip())

    # 4. Hyphenated compounds
    try:
        src_text = doc.text
        for m in re.finditer(r"\b[A-Za-z]{2,}(?:[-–—][A-Za-z]{2,})+\b", src_text):
            candidates.append(m.group(0))
    except Exception:
        pass
    
    return list(set(candidates))

def deduplicate_terms(terms_list: List[str]) -> List[str]:
    """
    Prioritize longer terms based on CONTINUOUS TOKEN SEQUENCE substring match.
    If a short term is a continuous word-subsequence of a longer term, discard it.
    """
    if not terms_list:
        return []
    
    # Use term_key for normalization during comparison
    unique_terms = list(set(t.strip() for t in terms_list if t.strip()))
    if not unique_terms:
        return []
        
    # Sort by number of words (descending), then by length
    sorted_terms = sorted(unique_terms, key=lambda x: (len(x.split()), len(x)), reverse=True)
    
    keep = []
    keep_keys = [] # Stores (term_key, list_of_words)
    
    for t in sorted_terms:
        key = get_term_key(t)
        words = key.split()
        
        is_substring = False
        for _, longer_words in keep_keys:
            # Check if current words are a continuous subsequence of longer_words
            n = len(words)
            m = len(longer_words)
            found = False
            for i in range(m - n + 1):
                if longer_words[i : i + n] == words:
                    found = True
                    break
            
            if found:
                is_substring = True
                break
        
        if not is_substring:
            keep.append(t)
            keep_keys.append((key, words))
            
    return keep

# ----------------------------
# Main Logic
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", default=DEFAULT_INPUT_TSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    nlp = _load_spacy(use_gpu=True)
    
    # Read TSV rows
    rows = []
    if not os.path.exists(args.input_tsv):
        logger.error(f"Input TSV not found: {args.input_tsv}")
        return

    with open(args.input_tsv, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        col_map = {name: i for i, name in enumerate(header)}
        for line_idx, line in enumerate(f):
            if args.total_shards > 1 and line_idx % args.total_shards != args.shard_id:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header): continue
            row_dict = {name: parts[i] for name, i in col_map.items()}
            rows.append(row_dict)

    if args.max_rows:
        rows = rows[:args.max_rows]

    logger.info(f"Loaded {len(rows)} rows for shard {args.shard_id}.")

    # Output file handling
    output_jsonl = args.output_jsonl
    if args.total_shards > 1:
        output_jsonl = output_jsonl.replace(".jsonl", f"_shard{args.shard_id}.jsonl")
    
    processed_ids = set()
    open_mode = "w"
    
    if args.resume and not args.force:
        if os.path.exists(output_jsonl):
            with open(output_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_ids.add(data["utter_id"])
                    except:
                        continue
            open_mode = "a"
            logger.info(f"Resuming: found {len(processed_ids)} already processed IDs.")

    # In this new version, we process row by row and filter
    rows = [r for r in rows if r["id"] not in processed_ids]
    if not rows:
        logger.info("No rows left to process.")
        return

    logger.info(f"Starting processing {len(rows)} rows...")

    with open(output_jsonl, open_mode, encoding="utf-8") as out_f:
        for w_start in range(0, len(rows), args.window_size):
            w_end = min(w_start + args.window_size, len(rows))
            window_rows = rows[w_start:w_end]
            
            window_texts = [row["src_text"] for row in window_rows]
            
            # Process with spaCy pipe
            for idx, doc in enumerate(nlp.pipe(window_texts, batch_size=args.batch_size)):
                row = window_rows[idx]
                uid = row["id"]
                
                # 1. Extract all potential terms for this utterance
                all_utterance_terms = get_filtered_candidates(doc)
                if not all_utterance_terms:
                    continue
                
                # 2. Setup audio and trajectories
                audio_info = row["audio"]
                try:
                    audio_path, start_frame, total_frames = audio_info.split(":")
                    start_frame = int(start_frame)
                    total_frames = int(total_frames)
                except:
                    continue

                src_traj = ast.literal_eval(row["src_trajectory"])
                N = len(src_traj)
                if N < 2: continue

                try:
                    full_audio_data, _ = sf.read(audio_path, start=start_frame, frames=total_frames)
                except:
                    continue

                # 3. Iterate through chunks
                for j in range(N - 1):
                    chunk_src_text = (src_traj[j] + " " + src_traj[j+1]).strip()
                    if not chunk_src_text:
                        continue
                        
                    # Find which terms appear in this chunk
                    chunk_terms = []
                    for term in all_utterance_terms:
                        # Case insensitive match
                        if re.search(r'\b' + re.escape(term) + r'\b', chunk_src_text, re.IGNORECASE):
                            chunk_terms.append(term)
                    
                    if not chunk_terms:
                        continue
                        
                    # 4. Deduplicate terms within this chunk
                    final_chunk_terms = deduplicate_terms(chunk_terms)
                    if not final_chunk_terms:
                        continue

                    # 5. Prepare audio chunk
                    chunk_start_rel = j * SAMPLES_PER_UNIT
                    chunk_end_rel = (j + 2) * SAMPLES_PER_UNIT
                    if chunk_start_rel >= len(full_audio_data): continue
                    
                    chunk_data = full_audio_data[chunk_start_rel : min(chunk_end_rel, len(full_audio_data))]
                    if len(chunk_data) == 0: continue
                    
                    chunk_filename = f"{uid}_chunk_{j}.wav"
                    chunk_audio_path = os.path.join(args.output_dir, chunk_filename)
                    if not os.path.exists(chunk_audio_path):
                        sf.write(chunk_audio_path, chunk_data, SAMPLE_RATE)

                    # 6. Write to JSONL
                    for term in final_chunk_terms:
                        output_obj = {
                            "term": term,
                            "term_key": get_term_key(term),
                            "chunk_src_text": chunk_src_text,
                            "chunk_audio_path": chunk_audio_path,
                            "utter_id": uid,
                            "chunk_idx": j
                        }
                        out_f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
                
                out_f.flush()

    logger.info("Done.")

if __name__ == "__main__":
    main()
