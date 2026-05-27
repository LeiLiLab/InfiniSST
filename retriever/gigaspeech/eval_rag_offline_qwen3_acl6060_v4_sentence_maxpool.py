#!/usr/bin/env python3
"""
Sentence-level offline retriever diagnostic (easy-to-interpret upper bound).

For each gold sentence wav:
  1) Run sliding windows over the full sentence (chunk_size / hop_size).
  2) Retrieve top-K2 candidates per window from FAISS.
  3) Max-pool scores over windows for each term.
  4) Keep terms with score >= threshold, then take top-K1 by pooled score.
  5) Count GT hits (unique terms by default; also report occurrence-level hits).

This intentionally overestimates online performance, but is simple and stable for sweeps.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4
from retriever.gigaspeech.acl_eval_utils import (
    _canonicalize_plural_english,
    build_keyword_processor,
    extract_gt_term_occurrences_from_text,
    l2_distance_to_score,
)


def _load_text_lines(txt_path: str) -> List[str]:
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def _sorted_wavs(wav_dir: str) -> List[str]:
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    # ACL dev segmented naming: sent_123.wav
    def _key(p: str) -> int:
        m = re.search(r"sent_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(wav_files, key=_key)


def _make_windows(audio: np.ndarray, sr: int, chunk_size: float, hop_size: float) -> List[np.ndarray]:
    chunk_samples = int(round(chunk_size * sr))
    hop_samples = int(round(hop_size * sr))
    if chunk_samples <= 0 or hop_samples <= 0:
        return []
    if audio.size == 0:
        return []

    windows: List[np.ndarray] = []
    # Start at 0, slide by hop; pad the last window.
    for start in range(0, max(1, len(audio)), hop_samples):
        end = start + chunk_samples
        w = audio[start:end]
        if w.shape[0] < chunk_samples:
            w = np.pad(w, (0, chunk_samples - w.shape[0]), mode="constant")
        windows.append(w)
        if end >= len(audio):
            break
    return windows


def _maxpool_retrieve(
    retriever_model: Any,
    faiss_index: Any,
    glossary_terms: List[str],
    windows: List[np.ndarray],
    feature_extractor: WhisperFeatureExtractor,
    device: torch.device,
    top_k2: int,
    score_threshold: float,
    merge_plural_terms: bool,
    term_canonical_map: Dict[str, str] | None,
    batch_size: int = 32,
) -> Dict[str, float]:
    if not windows:
        return {}

    pooled: Dict[str, float] = {}
    sr = 16000

    for j in range(0, len(windows), batch_size):
        batch_win = windows[j : j + batch_size]
        inputs = feature_extractor(batch_win, sampling_rate=sr, return_tensors="pt", padding=False)
        input_features = inputs.input_features.to(device).to(torch.bfloat16)
        f_lens = torch.full((input_features.shape[0],), input_features.shape[-1], device=device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = retriever_model(input_features, f_lens)
            audio_embs_np = audio_embs.cpu().float().numpy()
            import faiss  # local import
            faiss.normalize_L2(audio_embs_np)
            D, I = faiss_index.search(audio_embs_np, top_k2)

        for b in range(D.shape[0]):
            for rank in range(I.shape[1]):
                idx = int(I[b, rank])
                if idx < 0 or idx >= len(glossary_terms):
                    continue
                term = glossary_terms[idx]
                if merge_plural_terms:
                    if term_canonical_map is None:
                        term = _canonicalize_plural_english(term)
                    else:
                        term = term_canonical_map.get(term, term)
                score = l2_distance_to_score(float(D[b, rank]))
                if score < score_threshold:
                    continue
                prev = pooled.get(term)
                if prev is None or score > prev:
                    pooled[term] = score

    return pooled


def main() -> None:
    ap = argparse.ArgumentParser(description="Sentence-level offline max-pool retrieval diagnostic (Qwen3 RAG V4).")
    ap.add_argument("--model_path", type=str, required=True, help="Path to .pt checkpoint")
    ap.add_argument("--index_path", type=str, required=True, help="Path to .pkl FAISS index")
    ap.add_argument("--glossary_path", type=str, required=True, help="Path to glossary.json")
    ap.add_argument("--wav_dir", type=str, required=True, help="Directory with gold wav files")
    ap.add_argument("--txt_path", type=str, required=True, help="Path to gold transcript txt")
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--top_k", type=int, default=5, help="K1: keep top-K after max-pooling")
    ap.add_argument("--rag_voting_k", type=int, default=20, help="K2: FAISS top-K per window (kept for compatibility)")
    ap.add_argument("--rag_chunk_size", type=float, default=1.92)
    ap.add_argument("--rag_hop_size", type=float, default=0.96)
    ap.add_argument("--score_threshold", type=float, default=0.5)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--merge_plural_terms", action="store_true", default=False)
    ap.add_argument("--acl_audio_batch_size", type=int, default=32)
    ap.add_argument("--debug_print_limit", type=int, default=0)
    ap.add_argument("--debug_miss_limit", type=int, default=0)
    ap.add_argument("--rag_lora_r", type=int, default=32)
    ap.add_argument("--rag_text_lora_r", type=int, default=16)
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load glossary terms for GT extraction and eval.
    with open(args.glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
    glossary_terms_for_gt = [k.lower() for k in glossary.keys()]
    kp = build_keyword_processor(glossary_terms_for_gt)

    # Init retriever to reuse model + FAISS index (and ensure glossary order matches index).
    rag = StreamingQwen3RAGRetrieverV4(
        index_path=args.index_path,
        model_path=args.model_path,
        device=str(device),
        lora_r=int(args.rag_lora_r),
        text_lora_r=int(args.rag_text_lora_r),
        top_k=int(args.top_k),
        voting_k=int(args.rag_voting_k),
        voting_min_votes=2,
        score_threshold=float(args.score_threshold),
        chunk_size=float(args.rag_chunk_size),
        hop_size=float(args.rag_hop_size),
        aggregation_strategy="max_pool",
        debug_audio_dir=None,
        verbose=False,
    )

    # Use the retriever's internal term_list order (must match FAISS).
    index_glossary_keys = [item["key"] for item in rag.term_list]
    term_canonical_map = None
    if args.merge_plural_terms:
        term_canonical_map = {k: _canonicalize_plural_english(k) for k in index_glossary_keys}

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    wav_files = _sorted_wavs(args.wav_dir)
    text_lines = _load_text_lines(args.txt_path)
    if args.max_samples and args.max_samples > 0:
        wav_files = wav_files[: args.max_samples]
        text_lines = text_lines[: args.max_samples]

    sr = 16000
    total_gt_unique = 0
    total_hit_unique = 0
    total_gt_occ = 0
    total_hit_occ = 0
    total_pred = 0
    used = 0
    skipped_no_gt = 0

    debug_print_limit = int(args.debug_print_limit or 0)
    debug_miss_limit = int(args.debug_miss_limit or 0)
    dbg_printed = 0
    dbg_missed = 0

    for wav_path, txt in tqdm(list(zip(wav_files, text_lines)), desc="Sentence MaxPool Eval", leave=False):
        gt_occ = extract_gt_term_occurrences_from_text(
            txt,
            glossary_terms_for_gt,
            kp=kp,
            term_canonical_map=None,
            merge_plural_terms=False,
        )
        if not gt_occ:
            skipped_no_gt += 1
            continue

        # Optionally canonicalize GT in the same way as prediction canonicalization.
        if args.merge_plural_terms:
            gt_occ = [(_canonicalize_plural_english(t), s, e) for (t, s, e) in gt_occ]

        gt_seq = [t for (t, _s, _e) in gt_occ]
        gt_unique = set(gt_seq)

        audio, got_sr = sf.read(wav_path)
        if got_sr != sr:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)

        windows = _make_windows(audio, sr=sr, chunk_size=float(args.rag_chunk_size), hop_size=float(args.rag_hop_size))
        pooled_scores = _maxpool_retrieve(
            retriever_model=rag.model,
            faiss_index=rag.index,
            glossary_terms=index_glossary_keys,
            windows=windows,
            feature_extractor=feature_extractor,
            device=device,
            top_k2=int(args.rag_voting_k),
            score_threshold=float(args.score_threshold),
            merge_plural_terms=bool(args.merge_plural_terms),
            term_canonical_map=term_canonical_map,
            batch_size=int(args.acl_audio_batch_size),
        )

        # Final prediction set after max-pool: threshold then top-K1.
        pred_sorted = sorted(pooled_scores.items(), key=lambda x: x[1], reverse=True)
        pred_terms = [t for (t, _s) in pred_sorted[: int(args.top_k)]]
        pred_set = set(pred_terms)

        hit_unique = len(gt_unique & pred_set)
        total_gt_unique += len(gt_unique)
        total_hit_unique += hit_unique

        # Occurrence-level: count an occurrence as hit if its term appears in final pred_set.
        gt_cnt = Counter(gt_seq)
        hit_occ = sum(c for t, c in gt_cnt.items() if t in pred_set)
        total_gt_occ += len(gt_seq)
        total_hit_occ += hit_occ

        total_pred += len(pred_terms)
        used += 1

        if dbg_printed < debug_print_limit:
            print("\n" + "-" * 60)
            print("[DBG] Sentence preview")
            print(f"[DBG] wav: {wav_path}")
            print(f"[DBG] text: {txt.strip()}")
            print(f"[DBG] gt_unique({len(gt_unique)}): {sorted(list(gt_unique))[:50]}")
            print(f"[DBG] pred_topk({len(pred_terms)}): {pred_terms}")
            print(f"[DBG] hit_unique={hit_unique}/{len(gt_unique)} hit_occ={hit_occ}/{len(gt_seq)}")
            print("-" * 60)
            dbg_printed += 1

        if dbg_missed < debug_miss_limit and hit_unique < len(gt_unique):
            missed = sorted(list(gt_unique - pred_set))
            print("\n" + "!" * 60)
            print("[MISS] Sentence has missing GT terms")
            print(f"[MISS] wav: {wav_path}")
            print(f"[MISS] missing_unique({len(missed)}): {missed[:80]}")
            print(f"[MISS] pred_topk({len(pred_terms)}): {pred_terms}")
            print("!" * 60)
            dbg_missed += 1

    recall_unique = (total_hit_unique / total_gt_unique) if total_gt_unique > 0 else 0.0
    recall_occ = (total_hit_occ / total_gt_occ) if total_gt_occ > 0 else 0.0
    precision = (total_hit_unique / total_pred) if total_pred > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"{'FINAL SENTENCE MAX-POOL SUMMARY':^60}")
    print("=" * 60)
    print(f"Used Samples:   {used}")
    print(f"Skipped No-GT:  {skipped_no_gt}")
    print("-" * 60)
    print(f"GT Unique:      {total_gt_unique}")
    print(f"Hit Unique:     {total_hit_unique}")
    print(f"Recall (uniq):  {recall_unique:.2%}")
    print("-" * 60)
    print(f"GT Occurrences: {total_gt_occ}")
    print(f"Hit Occurrences:{total_hit_occ}")
    print(f"Recall (occ):   {recall_occ:.2%}")
    print("-" * 60)
    print(f"Pred Terms:     {total_pred}")
    print(f"Precision (uniq hits / pred): {precision:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()









