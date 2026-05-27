#!/usr/bin/env python3
"""
Dual-encoder offline evaluation for ACL6060 dev:
  Qwen3-Omni (text path) + XEUS (TTS path).

Supports glossary scaling via GLOSSARY_SIZE env var:
  - GLOSSARY_SIZE=0 (default): GT terms only
  - GLOSSARY_SIZE=N: pad with wiki terms to N total

TTS data for expanded glossaries comes from ADDITIONAL_TTS_MAPPING.

Reuses the evaluation logic from xeus_tts_text_intersection_eval.py
with ACL6060-specific data paths and TTS bank from /mnt/gemini/data/siqiouyang/acl_terms/.
"""

from __future__ import annotations

# ======Configuration=====
DEV_JSONL = "/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval/acl6060_dev_dataset.jsonl"
DEV_JSONL_WITH_TTS = "/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval/acl6060_dev_dataset_with_tts.jsonl"
TTS_ROOT_DIR = "/mnt/gemini/data/siqiouyang/acl_terms"

TEXT_MODEL_NAME = "scale_lora-r32-tr128_best"
TEXT_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_scale_lora-r32-tr128_bs4k_t=0.03_v1_best.pt"
)
TEXT_AUDIO_BASE_MODEL_NAME = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_AUDIO_LORA_R = 32
TEXT_AUDIO_LORA_ALPHA = 64
TEXT_LORA_R = 128
TARGET_LANG_CODE = "zh"
INDEX_BUILD_BATCH_SIZE = 1024

TTS_MODEL_NAME = "xeus_ttsw1.0_epoch2"
TTS_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "xeus_rag_xeus_tts_lora-r32-tr16_bs512_ttsw1.0_ttm=query key value_temperature=0.03_v1_epoch_2.pt"
)
XEUS_CHECKPOINT_PATH = "/mnt/gemini/data/jiaxuanluo/XEUS/model/xeus_checkpoint_new.pth"
XEUS_HIDDEN_DIM = 1024
XEUS_LORA_RANK = 32
XEUS_LORA_ALPHA = 64
XEUS_LORA_TARGET_MODULES = [
    "linear_q", "linear_k", "linear_v", "linear_out",
    "w_1", "w_2", "merge_proj",
]
XEUS_LORA_DROPOUT = 0.05

TARGET_DIM = 1024
DEVICE = "cuda:0"
TOP_K = 10

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SECONDS = 1.92
EXPECTED_CHUNK_SAMPLES = 30720

EVAL_BATCH_SIZE = 32
TTS_EMB_BATCH_SIZE = 64
MAX_TTS_PROTOTYPES_PER_TERM = 0
MAX_CHUNKS = 0

OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/acl6060_offline_eval_xeus_tts_qwen3_text"
GLOSSARY_JSON_NAME = "acl6060_dev_terms_glossary.json"

GLOSSARY_SIZE = 0
WIKI_GLOSSARY_PATH = (
    "/home/jiaxuanluo/InfiniSST/documents/code/data_pre/"
    "glossary_scale/wiki_glossary_nlp_ai_cs.json"
)
ADDITIONAL_TTS_MAPPING = ""
SKIP_TTS = False

VOTING_MIN_VOTES = 1
SCORE_THRESHOLD = 0.0
NUM_QUALITATIVE_SAMPLES_PER_CATEGORY = 8

CSV_DELIMITER = "\t"
CSV_LINE_TERMINATOR = "\n"
FLOAT_DECIMALS = 6
# ======Configuration=====

import os
import sys
from pathlib import Path


def _detect_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "retriever" / "gigaspeech" / "build_index_v4.py").exists():
            return parent
    raise RuntimeError(f"Cannot locate repository root from: {current}")


_REPO_ROOT = _detect_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gc
import json
import re
from typing import Dict, List, Sequence, Set, Tuple

from documents.code.offline_evaluation.tts.xeus_tts_text_intersection_eval import (  # noqa: E402
    _log,
    _warn,
    _ensure_dir,
    _safe_name,
    _build_glossary_json,
    _maybe_build_index,
    _load_index_data,
    _load_full_dev_dataset,
    _load_tts_paths,
    _load_audio_mono_16k,
    _run_tts_model_retrieval,
    _compute_metrics,
    _print_qualitative_samples,
    _print_metrics_table,
    _write_tsv,
    ChunkData,
    TopKResult,
    _is_cuda_device,
)


def _model_tag(model_path: str) -> str:
    stem = Path(model_path).stem
    return _safe_name(stem)


def _load_wiki_glossary(wiki_path: str) -> List[str]:
    assert os.path.isfile(wiki_path), f"Wiki glossary not found: {wiki_path}"
    with open(wiki_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    assert isinstance(entries, list), f"Expected list, got {type(entries)}"
    terms: List[str] = []
    seen: Set[str] = set()
    for e in entries:
        term = e["term"].strip().lower()
        assert term, f"Empty term: {e!r}"
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def _expand_glossary_with_wiki(
    gt_terms: List[str],
    target_size: int,
    wiki_glossary_path: str,
) -> List[str]:
    gt_set: Set[str] = set(gt_terms)
    if target_size <= len(gt_terms):
        _log(f"GLOSSARY_SIZE={target_size} <= GT terms ({len(gt_terms)}), using GT only.")
        return sorted(gt_terms)
    wiki_terms = _load_wiki_glossary(wiki_glossary_path)
    _log(f"Wiki glossary loaded: {len(wiki_terms)} terms")
    expanded = list(gt_terms)
    seen: Set[str] = set(gt_set)
    for wt in wiki_terms:
        if wt not in seen:
            expanded.append(wt)
            seen.add(wt)
        if len(expanded) >= target_size:
            break
    actual_wiki_added = len(expanded) - len(gt_terms)
    if len(expanded) < target_size:
        _warn(f"Wiki glossary exhausted: {len(expanded)} terms (wanted {target_size})")
    else:
        _log(f"Expanded glossary: {len(gt_terms)} GT + {actual_wiki_added} wiki = {len(expanded)}")
    return expanded


def _load_additional_tts_mapping(jsonl_path: str) -> Dict[str, List[str]]:
    """Load additional TTS mapping (from generated wiki glossary TTS)."""
    mapping: Dict[str, List[str]] = {}
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return mapping
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("term", "").strip().lower()
            p = obj.get("tts_audio_path", "").strip()
            if t and p:
                paths = mapping.setdefault(t, [])
                if p not in paths:
                    paths.append(p)
    _log(f"Additional TTS mapping loaded: {len(mapping)} terms from {jsonl_path}")
    return mapping


def _run_text_model_retrieval_f32(
    chunks: Sequence[ChunkData],
    term_to_idx: Dict[str, int],
    index_path,
    model_path: str,
    effective_device: str,
) -> Dict[str, TopKResult]:
    """
    Patched text retrieval that keeps the model in float32 and relies on
    autocast for mixed-precision.  Avoids the bfloat16 conv2d dtype mismatch
    inside Qwen3-Omni's AudioEncoder.
    """
    _log(f"=== Phase 2: Text Model Retrieval ({TEXT_MODEL_NAME}) — Qwen3-Omni (f32 fix) ===")

    from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4
    import faiss
    import torch
    import numpy as np

    use_cuda_amp = _is_cuda_device(effective_device)

    retriever = StreamingQwen3RAGRetrieverV4(
        index_path=str(index_path),
        model_path=str(model_path),
        base_model_name=TEXT_AUDIO_BASE_MODEL_NAME,
        device=effective_device,
        lora_r=TEXT_AUDIO_LORA_R,
        lora_alpha=TEXT_AUDIO_LORA_ALPHA,
        text_lora_r=TEXT_LORA_R,
        top_k=TOP_K,
        voting_k=TOP_K,
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

    results: Dict[str, TopKResult] = {}
    _log(f"Text model: encoding {len(chunks)} chunks -> Top-{TOP_K} ...")

    for start in range(0, len(chunks), EVAL_BATCH_SIZE):
        batch = chunks[start: start + EVAL_BATCH_SIZE]
        audios = [_load_audio_mono_16k(c.audio_path) for c in batch]
        inputs = retriever.feature_extractor(
            audios, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False
        )
        features = inputs.input_features
        batch_size, channels, mel_len = features.shape
        input_features = features.transpose(0, 1).reshape(channels, -1).to(retriever.device).float()
        feature_lens = torch.full((batch_size,), mel_len, dtype=torch.long, device=retriever.device)

        with torch.no_grad():
            if use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    audio_embs = retriever.model(input_features, feature_lens)
            else:
                audio_embs = retriever.model(input_features, feature_lens)
            audio_embs = audio_embs.detach().cpu().float().numpy()

        faiss.normalize_L2(audio_embs)
        dists, indices = retriever.index.search(audio_embs, TOP_K)

        for i, chunk in enumerate(batch):
            cid = chunk.key.as_id()
            term_indices = [int(idx) for idx in indices[i] if int(idx) >= 0]
            scores = [float(dists[i][j]) for j in range(len(term_indices))]
            results[cid] = TopKResult(term_indices=term_indices, scores=scores)

    _log(f"Text model retrieval done: {len(results)} chunks.")

    del retriever
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def main() -> int:
    global DEVICE, OUTPUT_DIR, TTS_ROOT_DIR, TOP_K
    global GLOSSARY_SIZE, WIKI_GLOSSARY_PATH, ADDITIONAL_TTS_MAPPING, SKIP_TTS
    global TEXT_MODEL_PATH, TEXT_MODEL_NAME

    env_device = os.environ.get("OFFLINE_EVAL_DEVICE", "").strip()
    if env_device:
        DEVICE = env_device
    env_out = os.environ.get("OFFLINE_EVAL_OUTPUT_DIR", "").strip()
    if env_out:
        OUTPUT_DIR = env_out
    env_tts_root = os.environ.get("OFFLINE_EVAL_TTS_ROOT_DIR", "").strip()
    if env_tts_root:
        TTS_ROOT_DIR = env_tts_root
    env_topk = os.environ.get("OFFLINE_EVAL_TOP_K", "").strip()
    if env_topk:
        TOP_K = int(env_topk)
        assert TOP_K > 0, f"TOP_K must be > 0, got {TOP_K}"
    env_glossary_size = os.environ.get("GLOSSARY_SIZE", "").strip()
    if env_glossary_size:
        GLOSSARY_SIZE = int(env_glossary_size)
        assert GLOSSARY_SIZE >= 0, f"GLOSSARY_SIZE must be >= 0, got {GLOSSARY_SIZE}"
    env_wiki_path = os.environ.get("WIKI_GLOSSARY_PATH", "").strip()
    if env_wiki_path:
        WIKI_GLOSSARY_PATH = env_wiki_path
    env_additional_tts = os.environ.get("ADDITIONAL_TTS_MAPPING", "").strip()
    if env_additional_tts:
        ADDITIONAL_TTS_MAPPING = env_additional_tts
    env_skip_tts = os.environ.get("SKIP_TTS", "").strip().lower()
    if env_skip_tts in ("1", "true", "yes"):
        SKIP_TTS = True
    env_text_model = os.environ.get("TEXT_MODEL_PATH", "").strip()
    if env_text_model:
        TEXT_MODEL_PATH = env_text_model
        TEXT_MODEL_NAME = _model_tag(TEXT_MODEL_PATH)

    model_tag = _model_tag(TEXT_MODEL_PATH)
    gs_tag = f"_gs{GLOSSARY_SIZE}" if GLOSSARY_SIZE > 0 else ""
    result_tsv_name = f"acl6060_xeus_tts_qwen3_text_metrics_top{TOP_K}{gs_tag}_{model_tag}.tsv"
    samples_txt_name = f"acl6060_qualitative_samples_top{TOP_K}{gs_tag}_{model_tag}.txt"
    _log(f"[ACL6060] TOP_K={TOP_K}, GLOSSARY_SIZE={GLOSSARY_SIZE}, SKIP_TTS={SKIP_TTS}")
    _log(f"TEXT_MODEL_PATH={TEXT_MODEL_PATH}")
    _log(f"TEXT_MODEL_NAME={TEXT_MODEL_NAME} (tag={model_tag})")

    import torch
    effective_device = DEVICE
    if not torch.cuda.is_available():
        _warn("CUDA not available, falling back to CPU.")
        effective_device = "cpu"
    _log(f"DEVICE: {effective_device}")

    out_dir = Path(OUTPUT_DIR)
    _ensure_dir(out_dir)

    import documents.code.offline_evaluation.tts.xeus_tts_text_intersection_eval as _eval_mod
    _eval_mod.TEXT_MODEL_NAME = TEXT_MODEL_NAME
    _eval_mod.TEXT_MODEL_PATH = TEXT_MODEL_PATH
    _eval_mod.TEXT_AUDIO_BASE_MODEL_NAME = TEXT_AUDIO_BASE_MODEL_NAME
    _eval_mod.TEXT_AUDIO_LORA_R = TEXT_AUDIO_LORA_R
    _eval_mod.TEXT_AUDIO_LORA_ALPHA = TEXT_AUDIO_LORA_ALPHA
    _eval_mod.TEXT_LORA_R = TEXT_LORA_R
    _eval_mod.TARGET_LANG_CODE = TARGET_LANG_CODE
    _eval_mod.INDEX_BUILD_BATCH_SIZE = INDEX_BUILD_BATCH_SIZE
    _eval_mod.TTS_MODEL_NAME = TTS_MODEL_NAME
    _eval_mod.TTS_MODEL_PATH = TTS_MODEL_PATH
    _eval_mod.XEUS_CHECKPOINT_PATH = XEUS_CHECKPOINT_PATH
    _eval_mod.XEUS_HIDDEN_DIM = XEUS_HIDDEN_DIM
    _eval_mod.XEUS_LORA_RANK = XEUS_LORA_RANK
    _eval_mod.XEUS_LORA_ALPHA = XEUS_LORA_ALPHA
    _eval_mod.XEUS_LORA_TARGET_MODULES = XEUS_LORA_TARGET_MODULES
    _eval_mod.XEUS_LORA_DROPOUT = XEUS_LORA_DROPOUT
    _eval_mod.TARGET_DIM = TARGET_DIM
    _eval_mod.TOP_K = TOP_K
    _eval_mod.EXPECTED_SAMPLE_RATE = EXPECTED_SAMPLE_RATE
    _eval_mod.EXPECTED_CHUNK_SECONDS = EXPECTED_CHUNK_SECONDS
    _eval_mod.EXPECTED_CHUNK_SAMPLES = EXPECTED_CHUNK_SAMPLES
    _eval_mod.EVAL_BATCH_SIZE = EVAL_BATCH_SIZE
    _eval_mod.TTS_EMB_BATCH_SIZE = TTS_EMB_BATCH_SIZE
    _eval_mod.MAX_TTS_PROTOTYPES_PER_TERM = MAX_TTS_PROTOTYPES_PER_TERM
    _eval_mod.VOTING_MIN_VOTES = VOTING_MIN_VOTES
    _eval_mod.SCORE_THRESHOLD = SCORE_THRESHOLD
    _eval_mod.NUM_QUALITATIVE_SAMPLES_PER_CATEGORY = NUM_QUALITATIVE_SAMPLES_PER_CATEGORY
    _eval_mod.TTS_ROOT_DIR = TTS_ROOT_DIR

    # ---- Phase 1: Load data ----
    _log("=== Phase 1: Loading ACL6060 data ===")
    all_chunks = _load_full_dev_dataset(Path(DEV_JSONL))
    if MAX_CHUNKS > 0:
        all_chunks = all_chunks[:MAX_CHUNKS]
        _log(f"Applied MAX_CHUNKS={MAX_CHUNKS}, chunks={len(all_chunks)}")

    term_to_tts_paths = _load_tts_paths(Path(DEV_JSONL_WITH_TTS))

    if ADDITIONAL_TTS_MAPPING:
        additional_tts = _load_additional_tts_mapping(ADDITIONAL_TTS_MAPPING)
        for term_key, paths in additional_tts.items():
            existing = term_to_tts_paths.setdefault(term_key, [])
            for p in paths:
                if p not in existing:
                    existing.append(p)
        _log(f"Merged additional TTS: now {len(term_to_tts_paths)} terms with TTS")

    gt_terms = sorted({term for chunk in all_chunks for term in chunk.gt_terms})
    num_gt_terms = len(gt_terms)
    _log(f"GT unique terms: {num_gt_terms}")

    if GLOSSARY_SIZE > 0:
        expanded_terms = _expand_glossary_with_wiki(
            gt_terms, GLOSSARY_SIZE, WIKI_GLOSSARY_PATH,
        )
    else:
        expanded_terms = gt_terms

    glossary_size = len(expanded_terms)
    _log(f"Final glossary size: {glossary_size}")

    glossary_json_name = (
        f"glossary_gs{glossary_size}.json" if GLOSSARY_SIZE > 0
        else GLOSSARY_JSON_NAME
    )
    glossary_path = out_dir / glossary_json_name
    if not glossary_path.exists():
        _build_glossary_json(expanded_terms, glossary_path)

    index_suffix = f"_gs{glossary_size}" if GLOSSARY_SIZE > 0 else ""
    text_index_path = out_dir / f"index_v4_tr{TEXT_LORA_R}_{model_tag}{index_suffix}.pkl"
    assert Path(TEXT_MODEL_PATH).exists(), f"Text model not found: {TEXT_MODEL_PATH}"

    _maybe_build_index(glossary_path, text_index_path, TEXT_MODEL_PATH, effective_device)
    term_to_idx, idx_to_term = _load_index_data(text_index_path)

    # ---- Phase 2: Text retrieval — Qwen3-Omni ----
    text_results = _run_text_model_retrieval_f32(
        chunks=all_chunks,
        term_to_idx=term_to_idx,
        index_path=text_index_path,
        model_path=TEXT_MODEL_PATH,
        effective_device=effective_device,
    )

    # ---- Phase 3: TTS retrieval — XEUS ----
    if SKIP_TTS:
        _log("=== Phase 3: TTS skipped (SKIP_TTS=1) ===")
        tts_results: Dict[str, TopKResult] = {
            chunk.key.as_id(): TopKResult(term_indices=[], scores=[])
            for chunk in all_chunks
        }
        tts_bank_terms = 0
        tts_bank_prototypes = 0
    else:
        assert Path(TTS_MODEL_PATH).exists(), f"TTS model not found: {TTS_MODEL_PATH}"
        tts_results, tts_bank_terms, tts_bank_prototypes = _run_tts_model_retrieval(
            chunks=all_chunks,
            term_to_idx=term_to_idx,
            term_to_tts_paths_raw=term_to_tts_paths,
            effective_device=effective_device,
        )

    # ---- Phase 4: Analysis ----
    _log("=== Phase 4: Analysis ===")
    no_term_m, with_term_m, overall_m = _compute_metrics(
        all_chunks, text_results, tts_results, term_to_idx,
    )

    samples_text = _print_qualitative_samples(
        all_chunks, text_results, tts_results, idx_to_term, term_to_idx,
        out_dir / samples_txt_name,
    )
    print(samples_text, flush=True)

    _print_metrics_table(
        no_term_m, with_term_m, overall_m,
        tts_bank_terms, tts_bank_prototypes, glossary_size,
    )
    _write_tsv(out_dir / result_tsv_name, no_term_m, with_term_m, overall_m)

    _log(f"Done. Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit("Missing dependency: torch.")
    raise SystemExit(main())
