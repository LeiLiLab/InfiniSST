#!/usr/bin/env python3
"""
Dual-encoder offline evaluation for ACL6060 dev using the
*extracted paper glossary*.

Supports scaling the FAISS index with wiki-sourced terms via GLOSSARY_SIZE:
  - GLOSSARY_SIZE=0 (default): use GT terms only (baseline)
  - GLOSSARY_SIZE=N (N > #GT): pad index with wiki NLP/AI/CS terms to N

Same evaluation logic as acl6060_xeus_tts_text_eval.py but pointing to
the dataset produced by prepare_acl6060_extracted_paper_glossary.py.
"""

from __future__ import annotations

# ======Configuration=====
DEV_JSONL = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
)
DEV_JSONL_WITH_TTS = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset_with_tts.jsonl"
)
TTS_ROOT_DIR = "/mnt/gemini/data/siqiouyang/acl_terms"

TEXT_MODEL_NAME = "scale_lora-r32-tr64_best"
TEXT_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_scale_lora-r32-tr64_bs4k_t=0.03_v1_best.pt"
)
TEXT_AUDIO_BASE_MODEL_NAME = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_AUDIO_LORA_R = 32
TEXT_AUDIO_LORA_ALPHA = 64
TEXT_LORA_R = 64
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
INTERSECTION_TOP_K = 20

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SECONDS = 1.92
EXPECTED_CHUNK_SAMPLES = 30720

EVAL_BATCH_SIZE = 32
TTS_EMB_BATCH_SIZE = 64
MAX_TTS_PROTOTYPES_PER_TERM = 0
MAX_CHUNKS = 0

OUTPUT_DIR = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_offline_eval_extracted_paper_glossary_xeus_tts_qwen3_text"
)
GLOSSARY_JSON_NAME = "acl6060_extracted_paper_glossary_terms.json"

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

import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set


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


def _load_wiki_glossary(wiki_path: str) -> List[str]:
    """Load pre-cleaned wiki glossary JSON and return deduplicated lowercase terms."""
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
    """
    Expand glossary to target_size by padding with wiki terms.
    GT terms are always included first; wiki terms fill the rest.
    """
    gt_set: Set[str] = set(gt_terms)
    if target_size <= len(gt_terms):
        _log(
            f"GLOSSARY_SIZE={target_size} <= GT terms ({len(gt_terms)}), "
            f"using GT terms only."
        )
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
        _warn(
            f"Wiki glossary exhausted: only {len(expanded)} terms "
            f"(wanted {target_size}). "
            f"GT={len(gt_terms)}, wiki_added={actual_wiki_added}"
        )
    else:
        _log(
            f"Expanded glossary: {len(gt_terms)} GT + "
            f"{actual_wiki_added} wiki = {len(expanded)} total"
        )

    return expanded


def _run_text_model_retrieval_f32(
    chunks: Sequence[ChunkData],
    term_to_idx: Dict[str, int],
    index_path,
    model_path: str,
    effective_device: str,
) -> Dict[str, TopKResult]:
    """
    Text retrieval in float32 to avoid bfloat16 conv2d dtype mismatch
    inside Qwen3-Omni's AudioEncoder.
    """
    retrieval_k = max(TOP_K, INTERSECTION_TOP_K)
    _log(f"=== Phase 2: Text Model Retrieval ({TEXT_MODEL_NAME}) — Qwen3-Omni (f32 fix) ===")
    _log(f"  retrieval_k={retrieval_k} (TOP_K={TOP_K}, INTERSECTION_TOP_K={INTERSECTION_TOP_K})")

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
        top_k=retrieval_k,
        voting_k=retrieval_k,
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
    _log(f"Text model: encoding {len(chunks)} chunks -> Top-{retrieval_k} ...")

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
        dists, indices = retriever.index.search(audio_embs, retrieval_k)

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


def _model_tag(model_path: str) -> str:
    """Derive a short, filesystem-safe tag from model checkpoint filename."""
    stem = Path(model_path).stem
    return _safe_name(stem)


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


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _compute_multi_k_metrics(
    chunks: Sequence[ChunkData],
    text_results: Dict[str, TopKResult],
    tts_results: Dict[str, TopKResult],
    term_to_idx: Dict[str, int],
    text_k: int,
    inter_k: int,
) -> dict:
    """
    Compute text@text_k and intersection@inter_k metrics.
    Text results and TTS results should each contain at least inter_k items.
    """
    wt_chunks = 0
    wt_gt_total = 0
    wt_text_tp = 0
    wt_text_pred = 0
    wt_inter_tp = 0
    wt_inter_pred = 0

    nt_chunks = 0
    nt_inter_pred = 0

    for chunk in chunks:
        cid = chunk.key.as_id()
        text_r = text_results[cid]
        tts_r = tts_results[cid]

        text_set_at_k = set(text_r.term_indices[:text_k])
        text_set_at_inter_k = set(text_r.term_indices[:inter_k])
        tts_set_at_inter_k = set(tts_r.term_indices[:inter_k])
        inter_set = text_set_at_inter_k & tts_set_at_inter_k

        gt_indices: Set[int] = set()
        for term in chunk.gt_terms:
            idx = term_to_idx.get(term)
            if idx is not None:
                gt_indices.add(idx)

        if chunk.has_term:
            wt_chunks += 1
            wt_gt_total += len(gt_indices)
            wt_text_tp += len(text_set_at_k & gt_indices)
            wt_text_pred += len(text_set_at_k)
            wt_inter_tp += len(inter_set & gt_indices)
            wt_inter_pred += len(inter_set)
        else:
            nt_chunks += 1
            nt_inter_pred += len(inter_set)

    text_recall = wt_text_tp / wt_gt_total if wt_gt_total > 0 else 0.0
    text_prec = wt_text_tp / wt_text_pred if wt_text_pred > 0 else 0.0
    text_f1v = _f1(text_prec, text_recall)

    inter_recall = wt_inter_tp / wt_gt_total if wt_gt_total > 0 else 0.0
    inter_prec = wt_inter_tp / wt_inter_pred if wt_inter_pred > 0 else 0.0
    inter_f1v = _f1(inter_prec, inter_recall)

    avg_inter_size = wt_inter_pred / wt_chunks if wt_chunks > 0 else 0.0
    nt_avg_inter = nt_inter_pred / nt_chunks if nt_chunks > 0 else 0.0

    return {
        "wt_chunks": wt_chunks,
        "wt_gt": wt_gt_total,
        "text_k": text_k,
        "inter_k": inter_k,
        "text_recall": text_recall,
        "text_precision": text_prec,
        "text_f1": text_f1v,
        "text_tp": wt_text_tp,
        "text_pred": wt_text_pred,
        "inter_recall": inter_recall,
        "inter_precision": inter_prec,
        "inter_f1": inter_f1v,
        "inter_tp": wt_inter_tp,
        "inter_pred": wt_inter_pred,
        "avg_inter_size": avg_inter_size,
        "nt_chunks": nt_chunks,
        "nt_avg_inter": nt_avg_inter,
        "nt_total_inter": nt_inter_pred,
    }


def _print_multi_k_table(m: dict) -> str:
    """Print the multi-K evaluation table (Text@K vs Intersection@K)."""
    lines = []

    def _add(s: str = "") -> None:
        lines.append(s)

    _add("\n" + "=" * 90)
    _add(f"WITH-TERM CHUNKS: Text@{m['text_k']} vs Intersection@{m['inter_k']}")
    _add("=" * 90)
    _add(f"  Chunks: {m['wt_chunks']}, GT positives: {m['wt_gt']}")
    _add("")
    _add(f"  {'Method':<30s} {'Recall':>8s} {'Prec':>8s} {'F1':>8s} {'TP':>6s} {'Pred':>6s} {'AvgInter':>9s}")
    _add(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*9}")
    _add(f"  {'Text@' + str(m['text_k']):<30s} {m['text_recall']:>8.4f} {m['text_precision']:>8.4f} "
         f"{m['text_f1']:>8.4f} {m['text_tp']:>6d} {m['text_pred']:>6d} {m['text_k']:>9.2f}")
    _add(f"  {'Intersection@' + str(m['inter_k']):<30s} {m['inter_recall']:>8.4f} {m['inter_precision']:>8.4f} "
         f"{m['inter_f1']:>8.4f} {m['inter_tp']:>6d} {m['inter_pred']:>6d} {m['avg_inter_size']:>9.2f}")

    _add("\n" + "=" * 90)
    _add(f"NO-TERM CHUNKS (Noise, Intersection@{m['inter_k']})")
    _add("=" * 90)
    _add(f"  No-term chunks: {m['nt_chunks']}")
    _add(f"  Total noise preds: {m['nt_total_inter']}")
    _add(f"  Avg inter terms per no-term chunk: {m['nt_avg_inter']:.4f}")

    output = "\n".join(lines)
    print(output, flush=True)
    return output


def main() -> int:
    global DEVICE, OUTPUT_DIR, TTS_ROOT_DIR, TOP_K, INTERSECTION_TOP_K
    global GLOSSARY_SIZE, WIKI_GLOSSARY_PATH
    global ADDITIONAL_TTS_MAPPING, SKIP_TTS
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
    env_inter_topk = os.environ.get("INTERSECTION_TOP_K", "").strip()
    if env_inter_topk:
        INTERSECTION_TOP_K = int(env_inter_topk)
        assert INTERSECTION_TOP_K > 0, f"INTERSECTION_TOP_K must be > 0, got {INTERSECTION_TOP_K}"
    assert INTERSECTION_TOP_K >= TOP_K, (
        f"INTERSECTION_TOP_K ({INTERSECTION_TOP_K}) must be >= TOP_K ({TOP_K})"
    )
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
    env_text_model_path = os.environ.get("TEXT_MODEL_PATH", "").strip()
    if env_text_model_path:
        TEXT_MODEL_PATH = env_text_model_path
        TEXT_MODEL_NAME = _model_tag(TEXT_MODEL_PATH)

    gs_tag = f"_gs{GLOSSARY_SIZE}" if GLOSSARY_SIZE > 0 else ""
    model_tag = _model_tag(TEXT_MODEL_PATH)
    result_tsv_name = f"acl6060_extracted_paper_glossary_metrics_top{TOP_K}{gs_tag}_{model_tag}.tsv"
    samples_txt_name = f"acl6060_extracted_paper_glossary_samples_top{TOP_K}{gs_tag}_{model_tag}.txt"
    _log(f"[ACL6060-ExtractedPaperGlossary] TOP_K={TOP_K}, INTERSECTION_TOP_K={INTERSECTION_TOP_K}, GLOSSARY_SIZE={GLOSSARY_SIZE}")
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
    _eval_mod.TOP_K = max(TOP_K, INTERSECTION_TOP_K)
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
    _log("=== Phase 1: Loading ACL6060 extracted-paper-glossary data ===")
    all_chunks = _load_full_dev_dataset(Path(DEV_JSONL))
    if MAX_CHUNKS > 0:
        all_chunks = all_chunks[:MAX_CHUNKS]

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
    assert Path(TTS_MODEL_PATH).exists(), f"TTS model not found: {TTS_MODEL_PATH}"

    _maybe_build_index(glossary_path, text_index_path, TEXT_MODEL_PATH, effective_device)
    term_to_idx, idx_to_term = _load_index_data(text_index_path)

    # ---- Phase 2: Text retrieval ----
    text_results = _run_text_model_retrieval_f32(
        chunks=all_chunks,
        term_to_idx=term_to_idx,
        index_path=text_index_path,
        model_path=TEXT_MODEL_PATH,
        effective_device=effective_device,
    )

    # ---- Phase 3: TTS retrieval (optional) ----
    if SKIP_TTS:
        _log("=== Phase 3: TTS skipped (SKIP_TTS=1) ===")
        tts_results: Dict[str, TopKResult] = {
            chunk.key.as_id(): TopKResult(term_indices=[], scores=[])
            for chunk in all_chunks
        }
        tts_bank_terms = 0
        tts_bank_prototypes = 0
    else:
        tts_results, tts_bank_terms, tts_bank_prototypes = _run_tts_model_retrieval(
            chunks=all_chunks,
            term_to_idx=term_to_idx,
            term_to_tts_paths_raw=term_to_tts_paths,
            effective_device=effective_device,
        )

    # ---- Phase 4: Analysis (multi-K: Text@TOP_K vs Intersection@INTERSECTION_TOP_K) ----
    _log(f"=== Phase 4: Analysis (Text@{TOP_K} vs Intersection@{INTERSECTION_TOP_K}) ===")

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

    multi_k_m = _compute_multi_k_metrics(
        all_chunks, text_results, tts_results, term_to_idx,
        text_k=TOP_K,
        inter_k=INTERSECTION_TOP_K,
    )
    _print_multi_k_table(multi_k_m)

    _write_tsv(out_dir / result_tsv_name, no_term_m, with_term_m, overall_m)

    _log(f"Done. Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit("Missing dependency: torch.")
    raise SystemExit(main())
