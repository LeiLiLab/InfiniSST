#!/usr/bin/env python3
"""
Analyze how GT term ranking changes as glossary size scales up.

For each glossary size, retrieves the FULL ranked list from FAISS
(not just top-10), then reports:
  - Mean / Median / P25 / P75 rank of GT terms
  - MRR (Mean Reciprocal Rank)
  - Recall@K for K = 1, 3, 5, 10, 20, 50, 100

Usage:
  CUDA_VISIBLE_DEVICES=0 python analyze_gt_rank_vs_glossary_size.py

Env vars:
  GLOSSARY_SIZES  comma-separated list (default: "95,1000,10000")
"""

from __future__ import annotations

import gc
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

# ======Configuration=====
DEV_JSONL = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
)

TEXT_MODEL_NAME = "text_ttsw0.0_epoch5"
TEXT_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_tts_lora-r32-tr16_bs4k_ttsw0.0_ttm=query key value_temperature=0.03_v2_epoch_5.pt"
)
TEXT_AUDIO_BASE_MODEL_NAME = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_AUDIO_LORA_R = 32
TEXT_AUDIO_LORA_ALPHA = 64
TEXT_LORA_R = 16
TARGET_LANG_CODE = "zh"

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SECONDS = 1.92

EVAL_BATCH_SIZE = 32
VOTING_MIN_VOTES = 1
SCORE_THRESHOLD = 0.0

INDEX_DIR = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_offline_eval_extracted_paper_glossary_xeus_tts_qwen3_text"
)
INDEX_BASENAME_TEMPLATE = "index_v4_tr{lora_r}_{model_name}{suffix}.pkl"

DEVICE = "cuda:0"

DEFAULT_GLOSSARY_SIZES = [95, 1000, 10000]
RECALL_AT_K_VALUES = [1, 3, 5, 10, 20, 50, 100]
# ======Configuration=====


def _detect_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "retriever" / "gigaspeech" / "build_index_v4.py").exists():
            return parent
    raise RuntimeError(f"Cannot locate repository root from: {current}")


_REPO_ROOT = _detect_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from documents.code.offline_evaluation.tts.xeus_tts_text_intersection_eval import (
    _log,
    _warn,
    _load_full_dev_dataset,
    _load_audio_mono_16k,
    _is_cuda_device,
    _safe_name,
    ChunkData,
)


def _resolve_index_path(glossary_size: int, gt_size: int) -> Path:
    """Build the index file path for a given glossary size."""
    suffix = f"_gs{glossary_size}" if glossary_size != gt_size else ""
    name = INDEX_BASENAME_TEMPLATE.format(
        lora_r=TEXT_LORA_R,
        model_name=_safe_name(TEXT_MODEL_NAME),
        suffix=suffix,
    )
    return Path(INDEX_DIR) / name


def _load_index(index_path: Path) -> Tuple[object, Dict[str, int], Dict[int, str]]:
    """Load FAISS index + term mappings from pickle."""
    import faiss

    assert index_path.exists(), f"Index not found: {index_path}"
    with index_path.open("rb") as f:
        data = pickle.load(f)

    index = faiss.deserialize_index(data["faiss_index"])
    term_list = data["term_list"]

    term_to_idx: Dict[str, int] = {}
    idx_to_term: Dict[int, str] = {}
    for i, item in enumerate(term_list):
        key = str(item.get("key", "")).strip().lower()
        assert key, f"Empty key at index {i}"
        term_to_idx[key] = i
        idx_to_term[i] = key

    return index, term_to_idx, idx_to_term


def encode_all_chunks(
    chunks: Sequence[ChunkData],
    effective_device: str,
) -> np.ndarray:
    """Encode all chunks once, return (N, D) normalized embedding matrix."""
    import faiss
    import torch
    from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4

    _log("Loading Qwen3-Omni model for audio encoding ...")

    dummy_index_path = _resolve_index_path(DEFAULT_GLOSSARY_SIZES[0], DEFAULT_GLOSSARY_SIZES[0])
    retriever = StreamingQwen3RAGRetrieverV4(
        index_path=str(dummy_index_path),
        model_path=str(TEXT_MODEL_PATH),
        base_model_name=TEXT_AUDIO_BASE_MODEL_NAME,
        device=effective_device,
        lora_r=TEXT_AUDIO_LORA_R,
        lora_alpha=TEXT_AUDIO_LORA_ALPHA,
        text_lora_r=TEXT_LORA_R,
        top_k=10,
        voting_k=10,
        voting_min_votes=VOTING_MIN_VOTES,
        target_lang=TARGET_LANG_CODE,
        score_threshold=SCORE_THRESHOLD,
        chunk_size=EXPECTED_CHUNK_SECONDS,
        hop_size=EXPECTED_CHUNK_SECONDS,
        aggregation_strategy="max_pool",
        sample_rate=EXPECTED_SAMPLE_RATE,
        debug_audio_dir=None,
        verbose=False,
    )
    retriever.model = retriever.model.float()

    use_cuda_amp = _is_cuda_device(effective_device)
    all_embs: List[np.ndarray] = []

    _log(f"Encoding {len(chunks)} chunks (batch_size={EVAL_BATCH_SIZE}) ...")
    for start in range(0, len(chunks), EVAL_BATCH_SIZE):
        batch = chunks[start: start + EVAL_BATCH_SIZE]
        audios = [_load_audio_mono_16k(c.audio_path) for c in batch]
        inputs = retriever.feature_extractor(
            audios, sampling_rate=EXPECTED_SAMPLE_RATE,
            return_tensors="pt", padding=False,
        )
        features = inputs.input_features
        batch_size, channels, mel_len = features.shape
        input_features = (
            features.transpose(0, 1).reshape(channels, -1)
            .to(retriever.device).float()
        )
        feature_lens = torch.full(
            (batch_size,), mel_len, dtype=torch.long, device=retriever.device,
        )
        with torch.no_grad():
            if use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    embs = retriever.model(input_features, feature_lens)
            else:
                embs = retriever.model(input_features, feature_lens)
            embs = embs.detach().cpu().float().numpy()

        faiss.normalize_L2(embs)
        all_embs.append(embs)

    del retriever
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    all_embs_np = np.concatenate(all_embs, axis=0)
    _log(f"Audio encoding done: shape={all_embs_np.shape}")
    return all_embs_np


def analyze_ranks_for_size(
    glossary_size: int,
    gt_size: int,
    all_embs: np.ndarray,
    chunks: Sequence[ChunkData],
) -> Dict:
    """Search FAISS with K=glossary_size, compute GT rank statistics."""
    import faiss

    index_path = _resolve_index_path(glossary_size, gt_size)
    _log(f"\n{'='*70}")
    _log(f"Glossary size = {glossary_size}")
    _log(f"Index: {index_path}")
    index, term_to_idx, idx_to_term = _load_index(index_path)

    search_k = min(glossary_size, index.ntotal)
    _log(f"Searching with K={search_k} (index.ntotal={index.ntotal})")

    dists, indices = index.search(all_embs, search_k)

    gt_ranks: List[int] = []
    gt_reciprocal_ranks: List[float] = []
    missed_count = 0
    total_gt = 0

    for ci, chunk in enumerate(chunks):
        if not chunk.gt_terms:
            continue
        retrieved_indices = indices[ci]
        for term in chunk.gt_terms:
            tidx = term_to_idx.get(term)
            if tidx is None:
                continue
            total_gt += 1
            positions = np.where(retrieved_indices == tidx)[0]
            if len(positions) > 0:
                rank = int(positions[0]) + 1
                gt_ranks.append(rank)
                gt_reciprocal_ranks.append(1.0 / rank)
            else:
                missed_count += 1
                gt_ranks.append(search_k + 1)
                gt_reciprocal_ranks.append(0.0)

    ranks_arr = np.array(gt_ranks)
    recall_at_k = {}
    for k in RECALL_AT_K_VALUES:
        if k <= search_k:
            recall_at_k[k] = float(np.mean(ranks_arr <= k))

    stats = {
        "glossary_size": glossary_size,
        "total_gt": total_gt,
        "found": total_gt - missed_count,
        "missed": missed_count,
        "mean_rank": float(np.mean(ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
        "p25_rank": float(np.percentile(ranks_arr, 25)),
        "p75_rank": float(np.percentile(ranks_arr, 75)),
        "mrr": float(np.mean(gt_reciprocal_ranks)),
        "recall_at_k": recall_at_k,
    }
    return stats


def print_comparison_table(all_stats: List[Dict]) -> str:
    """Pretty-print a comparison table across glossary sizes."""
    lines: List[str] = []

    def _add(s: str = "") -> None:
        lines.append(s)
        print(s, flush=True)

    _add("=" * 90)
    _add("GT TERM RANK ANALYSIS vs GLOSSARY SIZE")
    _add("=" * 90)

    header_parts = [f"{'Metric':<25s}"]
    for s in all_stats:
        header_parts.append(f"{'gs=' + str(s['glossary_size']):>14s}")
    _add("  ".join(header_parts))
    _add("-" * 90)

    def _row(label: str, key: str, fmt: str = ".2f") -> None:
        parts = [f"{label:<25s}"]
        for s in all_stats:
            val = s[key]
            parts.append(f"{val:>14{fmt}}")
        _add("  ".join(parts))

    _row("Total GT instances", "total_gt", "d")
    _row("Found in index", "found", "d")
    _row("Missed (not in top-K)", "missed", "d")
    _add("")
    _row("Mean Rank", "mean_rank", ".2f")
    _row("Median Rank", "median_rank", ".1f")
    _row("P25 Rank", "p25_rank", ".1f")
    _row("P75 Rank", "p75_rank", ".1f")
    _row("MRR", "mrr", ".4f")
    _add("")

    all_k_values = sorted({
        k for s in all_stats for k in s["recall_at_k"]
    })
    for k in all_k_values:
        parts = [f"{'Recall@' + str(k):<25s}"]
        for s in all_stats:
            val = s["recall_at_k"].get(k)
            if val is not None:
                parts.append(f"{val:>14.4f}")
            else:
                parts.append(f"{'n/a':>14s}")
        _add("  ".join(parts))

    _add("=" * 90)
    return "\n".join(lines)


def main() -> int:
    global DEVICE

    env_device = os.environ.get("OFFLINE_EVAL_DEVICE", "").strip()
    if env_device:
        DEVICE = env_device

    env_sizes = os.environ.get("GLOSSARY_SIZES", "").strip()
    if env_sizes:
        glossary_sizes = [int(x.strip()) for x in env_sizes.split(",")]
    else:
        glossary_sizes = DEFAULT_GLOSSARY_SIZES

    _log(f"Glossary sizes to analyze: {glossary_sizes}")

    import torch
    effective_device = DEVICE
    if not torch.cuda.is_available():
        _warn("CUDA not available, falling back to CPU.")
        effective_device = "cpu"
    _log(f"DEVICE: {effective_device}")

    _log("=== Loading dataset ===")
    all_chunks = _load_full_dev_dataset(Path(DEV_JSONL))
    gt_terms = sorted({term for chunk in all_chunks for term in chunk.gt_terms})
    gt_size = len(gt_terms)
    _log(f"Dataset: {len(all_chunks)} chunks, {gt_size} unique GT terms")

    for gs in glossary_sizes:
        idx_path = _resolve_index_path(gs, gt_size)
        assert idx_path.exists(), (
            f"Index for gs={gs} not found: {idx_path}\n"
            f"Run acl6060_extracted_paper_glossary_eval.py with "
            f"GLOSSARY_SIZE={gs} first to build it."
        )

    _log("=== Encoding chunk audio (one-time) ===")
    all_embs = encode_all_chunks(all_chunks, effective_device)

    _log("=== Analyzing GT ranks per glossary size ===")
    all_stats: List[Dict] = []
    for gs in glossary_sizes:
        stats = analyze_ranks_for_size(gs, gt_size, all_embs, all_chunks)
        all_stats.append(stats)

    _log("\n")
    table_text = print_comparison_table(all_stats)

    out_path = Path(INDEX_DIR) / "gt_rank_analysis.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(table_text + "\n")
    _log(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit("Missing dependency: torch.")
    raise SystemExit(main())
