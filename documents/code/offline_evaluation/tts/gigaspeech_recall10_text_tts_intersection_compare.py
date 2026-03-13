#!/usr/bin/env python3

"""
Offline evaluation: fixed Top-10 comparison for text retrieval, TTS retrieval, and their intersection.

Protocol:
  - Filter rows with non-empty term from DEV_JSONL and group by (utter_id, chunk_idx).
  - Build glossary once from unique terms.
  - For each model:
      1) Build/load text FAISS index from the model's text encoder.
      2) Retrieve text Top-10 per speech chunk.
      3) Build TTS term embedding bank (if TTS audio exists) and retrieve TTS Top-10 by cosine similarity.
      4) Compute:
         - text recall@10 + precision@10 (no TTS)
         - tts recall@10 + precision@10
         - intersection (text Top-10 ∩ tts Top-10): remaining ratio, recall, precision
  - Write a model-level comparison TSV.

All log messages are in English.
"""

from __future__ import annotations

# ======Configuration=====
# Use the TTS-augmented dev dataset for speech->TTS evaluation.
DEV_JSONL = "/mnt/gemini/data/siqiouyang/term_dev_dataset_final_with_tts.jsonl"
# Reference only (not used in this script):
# TRAIN_JSONL = "/mnt/gemini/data/siqiouyang/term_train_dataset_final_with_tts.jsonl"
TTS_ROOT_DIR = "/mnt/gemini/data/siqiouyang/term_dev_tts"

# Compare models in one run.
# old_model_v1: /mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt
# new_model_tts_v2: /mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_v2_epoch_0.pt
MODEL_SPECS = [
    {
        "model_name": "old_model_v1",
        "model_path": "/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt",
        "enable_tts_eval": False,
    },
    {
        "model_name": "new_model_tts_v2",
        "model_path": "/mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_v2_epoch_3.pt",
        "enable_tts_eval": True,
    },
]

# Index build settings (match build_index_v4.py expectations).
TEXT_LORA_R = 16
TARGET_LANG_CODE = "zh"
INDEX_BUILD_BATCH_SIZE = 1024

# Audio encoder settings (match StreamingQwen3RAGRetrieverV4 defaults).
DEVICE = "cuda:0"
AUDIO_BASE_MODEL_NAME = "Atotti/Qwen3-Omni-AudioTransformer"
AUDIO_LORA_R = 32
AUDIO_LORA_ALPHA = 64

# Fixed no-threshold Top-K evaluation.
TOP_K = 10

# Audio chunk assumptions for this dataset.
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SECONDS = 1.92
EXPECTED_CHUNK_SAMPLES = 30720  # 1.92s * 16kHz

# Runtime
EVAL_BATCH_SIZE = 32
TTS_EMB_BATCH_SIZE = 256
MAX_TTS_PROTOTYPES_PER_TERM = 0  # <=0 means use all available prototypes per term
MAX_CHUNKS = 0  # 0 means no limit

# Output
OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/offline_eval_recall10_text_tts_intersection_compare"
GLOSSARY_JSON_NAME = "gigaspeech_dev_terms_glossary.json"
RESULT_TSV_NAME = "model_compare_recall10_text_tts_intersection.tsv"
PLOT_PNG_NAME = "model_compare_recall10_text_tts_intersection.png"

# Retriever behavior
VOTING_MIN_VOTES = 1
SCORE_THRESHOLD = 0.0

# CSV
CSV_DELIMITER = "\t"
CSV_LINE_TERMINATOR = "\n"
FLOAT_DECIMALS = 6

# Plot
PLOT_DPI = 180
PLOT_FIGSIZE_W = 11.5
PLOT_FIGSIZE_H = 4.8
# ======Configuration=====

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


def _detect_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "retriever" / "gigaspeech" / "build_index_v4.py"
        if candidate.exists():
            return parent
    raise RuntimeError(f"Cannot locate repository root from script path: {current}")


_REPO_ROOT = _detect_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def _err(msg: str) -> None:
    raise RuntimeError(msg)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _format_float(x: float) -> str:
    return f"{x:.{FLOAT_DECIMALS}f}"


def _resolve_effective_device(requested_device: str) -> str:
    device = requested_device.strip().lower()
    if not device.startswith("cuda:"):
        return requested_device

    try:
        requested_idx = int(device.split(":", 1)[1])
    except Exception:
        _warn(f"Invalid DEVICE format={requested_device!r}. Fallback to cuda:0.")
        return "cuda:0"

    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_raw:
        visible_list = [item.strip() for item in visible_raw.split(",") if item.strip()]
        if visible_list and requested_idx >= len(visible_list):
            mapped = "cuda:0"
            _warn(
                f"Requested DEVICE={requested_device} with CUDA_VISIBLE_DEVICES={visible_raw}. "
                f"Inside process, visible GPUs are remapped to 0..{len(visible_list) - 1}. "
                f"Auto-adjust device to {mapped}."
            )
            return mapped

    return requested_device


@dataclass(frozen=True)
class ChunkKey:
    utter_id: str
    chunk_idx: str

    def as_id(self) -> str:
        return f"{self.utter_id}::{self.chunk_idx}"


@dataclass
class ChunkExample:
    key: ChunkKey
    audio_path: str
    gt_terms: Set[str]


@dataclass
class ChunkEvalItem:
    chunk_id: str
    utter_id: str
    chunk_idx: str
    audio_path: str
    pos_indices_all: Set[int]


def _load_and_group_dev_jsonl(dev_jsonl: Path) -> Tuple[List[ChunkExample], Dict[str, List[str]]]:
    _log(f"Loading DEV_JSONL: {dev_jsonl}")

    groups: Dict[str, ChunkExample] = {}
    term_to_tts_paths: Dict[str, List[str]] = {}
    term_tts_conflicts = 0
    total_rows = 0
    kept_rows = 0

    for obj in _read_jsonl(dev_jsonl):
        total_rows += 1
        term = str(obj.get("term", "")).strip().lower()
        if not term:
            continue
        kept_rows += 1

        utter_id = str(obj.get("utter_id", "")).strip()
        chunk_idx = str(obj.get("chunk_idx", "")).strip()
        audio_path = str(obj.get("chunk_audio_path", "")).strip()

        if not utter_id or not chunk_idx or not audio_path:
            _warn(
                "Skip row with missing fields: "
                f"utter_id={utter_id!r} chunk_idx={chunk_idx!r} audio_path={audio_path!r}"
            )
            continue

        ck = ChunkKey(utter_id=utter_id, chunk_idx=chunk_idx)
        cid = ck.as_id()
        if cid not in groups:
            groups[cid] = ChunkExample(key=ck, audio_path=audio_path, gt_terms=set())
        groups[cid].gt_terms.add(term)

        # Collect all explicit term-level TTS paths from dataset.
        row_tts_path = str(obj.get("tts_audio_path", "") or obj.get("term_tts_path", "")).strip()
        if row_tts_path:
            paths = term_to_tts_paths.setdefault(term, [])
            if row_tts_path not in paths:
                if paths:
                    term_tts_conflicts += 1
                paths.append(row_tts_path)

    examples = list(groups.values())
    examples.sort(key=lambda x: (x.key.utter_id, int(x.key.chunk_idx) if x.key.chunk_idx.isdigit() else x.key.chunk_idx))
    _log(f"Loaded rows={total_rows}, kept_rows_with_term={kept_rows}, unique_chunks={len(examples)}")
    _log(
        f"Collected term-level TTS paths from dataset: mapped_terms={len(term_to_tts_paths)} "
        f"conflicts={term_tts_conflicts}"
    )

    if not examples:
        _err("No valid chunks after filtering. Check DEV_JSONL and term field.")
    return examples, term_to_tts_paths


def _build_glossary_json(unique_terms: Sequence[str], glossary_path: Path) -> None:
    glossary: Dict[str, str] = {term: "" for term in unique_terms}
    with glossary_path.open("w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)


def _maybe_build_index(glossary_path: Path, index_path: Path, model_path: str, device: str) -> None:
    if index_path.exists():
        _log(f"Using existing index: {index_path}")
        return

    script_path = _REPO_ROOT / "retriever" / "gigaspeech" / "build_index_v4.py"
    if not script_path.exists():
        _err(f"build_index_v4.py not found: {script_path}")

    _log(f"Building index for model={model_path} -> {index_path}")
    cmd = [
        sys.executable,
        str(script_path),
        "--glossary_path",
        str(glossary_path),
        "--model_path",
        model_path,
        "--output_path",
        str(index_path),
        "--text_lora_r",
        str(TEXT_LORA_R),
        "--device",
        str(device),
        "--batch_size",
        str(INDEX_BUILD_BATCH_SIZE),
        "--target_lang_code",
        str(TARGET_LANG_CODE),
    ]

    import subprocess

    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        _err(f"Index build failed (rc={proc.returncode}). stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    _log("Index build finished.")


def _load_index_term_map(index_path: Path) -> Dict[str, int]:
    import pickle
    import faiss  # type: ignore

    with index_path.open("rb") as f:
        data = pickle.load(f)
    term_list = data["term_list"]
    _ = faiss.deserialize_index(data["faiss_index"])

    term_to_idx: Dict[str, int] = {}
    for i, item in enumerate(term_list):
        key = str(item.get("key", "")).strip().lower()
        if key:
            term_to_idx[key] = i

    if not term_to_idx:
        _err(f"Empty term_list in index: {index_path}")
    return term_to_idx


def _load_audio_mono_16k(path: str) -> np.ndarray:
    import soundfile as sf  # type: ignore

    audio, sr = sf.read(path)
    if sr != EXPECTED_SAMPLE_RATE:
        _warn(f"Unexpected sample rate: path={path} sr={sr} expected={EXPECTED_SAMPLE_RATE}")

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).flatten()

    max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_val > 0.0:
        audio = audio / max_val

    if audio.shape[0] < EXPECTED_CHUNK_SAMPLES:
        audio = np.pad(audio, (0, EXPECTED_CHUNK_SAMPLES - audio.shape[0]), mode="constant")
    elif audio.shape[0] > EXPECTED_CHUNK_SAMPLES:
        audio = audio[:EXPECTED_CHUNK_SAMPLES]
    return audio


def _resolve_tts_path(raw_path: str, tts_root_dir: str) -> str:
    p = Path(raw_path)
    if p.exists():
        return str(p)
    if not p.is_absolute():
        candidate = Path(tts_root_dir) / p
        if candidate.exists():
            return str(candidate)
    return ""


def _map_term_to_tts_paths_from_dataset(
    term_to_idx: Dict[str, int],
    term_to_tts_path_from_data: Dict[str, List[str]],
    tts_root_dir: str,
) -> Dict[int, List[str]]:
    term_to_tts_path: Dict[int, List[str]] = {}
    for term, raw_paths in term_to_tts_path_from_data.items():
        idx = term_to_idx.get(term)
        if idx is None:
            continue
        resolved_paths: List[str] = []
        for raw_path in raw_paths:
            resolved = _resolve_tts_path(raw_path, tts_root_dir)
            if not resolved:
                continue
            if resolved in resolved_paths:
                continue
            resolved_paths.append(resolved)
            if MAX_TTS_PROTOTYPES_PER_TERM > 0 and len(resolved_paths) >= MAX_TTS_PROTOTYPES_PER_TERM:
                break
        if resolved_paths:
            term_to_tts_path[idx] = resolved_paths
    return term_to_tts_path


def _prepare_eval_items(examples: Sequence[ChunkExample], term_to_idx: Dict[str, int]) -> Tuple[List[ChunkEvalItem], int, int]:
    items: List[ChunkEvalItem] = []
    total_valid_pos = 0
    missing_terms = 0

    for ex in examples:
        pos_indices: Set[int] = set()
        for term in ex.gt_terms:
            idx = term_to_idx.get(term)
            if idx is None:
                missing_terms += 1
                continue
            pos_indices.add(idx)
        if not pos_indices:
            continue

        total_valid_pos += len(pos_indices)
        items.append(
            ChunkEvalItem(
                chunk_id=ex.key.as_id(),
                utter_id=ex.key.utter_id,
                chunk_idx=ex.key.chunk_idx,
                audio_path=ex.audio_path,
                pos_indices_all=pos_indices,
            )
        )

    return items, total_valid_pos, missing_terms


def _evaluate_single_model(
    model_name: str,
    model_path: str,
    enable_tts_eval: bool,
    effective_device: str,
    out_dir: Path,
    glossary_path: Path,
    examples: Sequence[ChunkExample],
    term_to_tts_path_from_data: Dict[str, List[str]],
    unique_terms_count: int,
) -> Dict[str, Any]:
    _log(f"Evaluating model_name={model_name} model_path={model_path}")
    if not Path(model_path).exists():
        _err(f"MODEL_PATH not found: {model_path}")

    model_dir = out_dir / _safe_name(model_name)
    _ensure_dir(model_dir)
    index_name = f"gigaspeech_dev_terms_index_v4_tr{TEXT_LORA_R}_{_safe_name(model_name)}.pkl"
    index_path = model_dir / index_name

    _maybe_build_index(glossary_path, index_path, model_path, effective_device)
    term_to_idx = _load_index_term_map(index_path)
    items, total_valid_pos_all, missing_terms = _prepare_eval_items(examples, term_to_idx)

    if missing_terms > 0:
        _warn(f"Model={model_name}: missing terms not found in index: {missing_terms}")
    if not items or total_valid_pos_all <= 0:
        _err(f"Model={model_name}: no positives to evaluate after term mapping.")

    _log(
        f"Model={model_name}: chunks_with_pos={len(items)} total_pos_all={total_valid_pos_all} "
        f"top_k={TOP_K}"
    )

    from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4
    import faiss  # type: ignore
    import torch

    retriever = StreamingQwen3RAGRetrieverV4(
        index_path=str(index_path),
        model_path=str(model_path),
        base_model_name=AUDIO_BASE_MODEL_NAME,
        device=effective_device,
        lora_r=AUDIO_LORA_R,
        lora_alpha=AUDIO_LORA_ALPHA,
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

    text_pred_by_chunk: Dict[str, Set[int]] = {}
    speech_emb_by_chunk: Dict[str, np.ndarray] = {}

    _log(f"Model={model_name}: encoding speech and retrieving text Top-{TOP_K} ...")
    for start in range(0, len(items), EVAL_BATCH_SIZE):
        batch = items[start : start + EVAL_BATCH_SIZE]
        audios = [_load_audio_mono_16k(item.audio_path) for item in batch]
        inputs = retriever.feature_extractor(audios, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False)
        features = inputs.input_features
        batch_size, channels, mel_len = features.shape
        input_features = features.transpose(0, 1).reshape(channels, -1).to(retriever.device).to(torch.bfloat16)
        feature_lens = torch.full((batch_size,), mel_len, dtype=torch.long, device=retriever.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = retriever.model(input_features, feature_lens)
            audio_embs = audio_embs.detach().cpu().float().numpy()

        faiss.normalize_L2(audio_embs)
        dists, indices = retriever.index.search(audio_embs, TOP_K)
        _ = dists

        for i, item in enumerate(batch):
            pred_set: Set[int] = set(int(idx) for idx in indices[i] if int(idx) >= 0)
            text_pred_by_chunk[item.chunk_id] = pred_set
            speech_emb_by_chunk[item.chunk_id] = audio_embs[i].astype(np.float32, copy=False)

    tts_pred_by_chunk: Dict[str, Set[int]] = {item.chunk_id: set() for item in items}
    tts_valid_term_indices: Set[int] = set()
    if enable_tts_eval:
        _log(f"Model={model_name}: preparing TTS term bank ...")
        term_to_tts_path = _map_term_to_tts_paths_from_dataset(
            term_to_idx=term_to_idx,
            term_to_tts_path_from_data=term_to_tts_path_from_data,
            tts_root_dir=TTS_ROOT_DIR,
        )
        if not term_to_tts_path:
            _warn(
                f"Model={model_name}: no valid term-level TTS paths found "
                f"(dataset map size={len(term_to_tts_path_from_data)}, TTS_ROOT_DIR={TTS_ROOT_DIR})"
            )

        tts_valid_term_indices = set(term_to_tts_path.keys())
        if tts_valid_term_indices:
            sorted_term_indices = sorted(tts_valid_term_indices)
            prototype_term_indices: List[int] = []
            prototype_audio_paths: List[str] = []
            for term_idx in sorted_term_indices:
                for tts_path in term_to_tts_path.get(term_idx, []):
                    prototype_term_indices.append(term_idx)
                    prototype_audio_paths.append(tts_path)

            if not prototype_term_indices:
                _warn(f"Model={model_name}: no valid TTS prototypes after path resolution.")
            else:
                _log(
                    f"Model={model_name}: building multi-prototype TTS bank "
                    f"(terms={len(sorted_term_indices)}, prototypes={len(prototype_term_indices)}, "
                    f"max_per_term={MAX_TTS_PROTOTYPES_PER_TERM})"
                )

            prototype_embeddings: List[np.ndarray] = []
            for start in range(0, len(prototype_audio_paths), TTS_EMB_BATCH_SIZE):
                path_batch = prototype_audio_paths[start : start + TTS_EMB_BATCH_SIZE]
                audios = [_load_audio_mono_16k(path) for path in path_batch]
                inputs = retriever.feature_extractor(audios, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False)
                features = inputs.input_features
                bsz, channels, mel_len = features.shape
                input_features = features.transpose(0, 1).reshape(channels, -1).to(retriever.device).to(torch.bfloat16)
                feature_lens = torch.full((bsz,), mel_len, dtype=torch.long, device=retriever.device)

                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        emb_batch = retriever.model(input_features, feature_lens)
                    emb_batch = emb_batch.detach().cpu().float().numpy()
                faiss.normalize_L2(emb_batch)
                prototype_embeddings.append(emb_batch.astype(np.float32, copy=False))

            if prototype_embeddings:
                prototype_embs_np = np.concatenate(prototype_embeddings, axis=0)
                prototype_term_indices_np = np.asarray(prototype_term_indices, dtype=np.int64)
                unique_term_indices_np = np.asarray(sorted_term_indices, dtype=np.int64)
                term_position = {int(term_idx): i for i, term_idx in enumerate(unique_term_indices_np)}

                _log(f"Model={model_name}: retrieving TTS Top-{TOP_K} (term score=max over prototypes) ...")
                for item in items:
                    speech_vec = speech_emb_by_chunk[item.chunk_id]
                    proto_scores = prototype_embs_np @ speech_vec
                    if proto_scores.size == 0:
                        tts_pred_by_chunk[item.chunk_id] = set()
                        continue

                    # Aggregate prototype scores into term scores by max pooling.
                    term_scores = np.full(unique_term_indices_np.shape[0], -np.inf, dtype=np.float32)
                    for proto_idx, score in enumerate(proto_scores):
                        term_idx = int(prototype_term_indices_np[proto_idx])
                        pos = term_position.get(term_idx)
                        if pos is not None and score > term_scores[pos]:
                            term_scores[pos] = float(score)

                    valid_mask = np.isfinite(term_scores)
                    if not np.any(valid_mask):
                        tts_pred_by_chunk[item.chunk_id] = set()
                        continue
                    valid_term_indices = unique_term_indices_np[valid_mask]
                    valid_term_scores = term_scores[valid_mask]

                    k = min(TOP_K, valid_term_scores.shape[0])
                    top_idx = np.argpartition(-valid_term_scores, k - 1)[:k]
                    top_idx = top_idx[np.argsort(-valid_term_scores[top_idx])]
                    pred_set = set(int(valid_term_indices[i]) for i in top_idx)
                    tts_pred_by_chunk[item.chunk_id] = pred_set
    else:
        _log(f"Model={model_name}: skipping TTS and intersection evaluation (enable_tts_eval=False).")

    tp_text = 0
    pred_total_text = 0
    tp_tts = 0
    pred_total_tts = 0
    tp_inter = 0
    pred_total_inter = 0
    total_valid_pos_tts = 0
    chunks_with_tts_pos = 0

    for item in items:
        pos_all = item.pos_indices_all
        text_pred = text_pred_by_chunk.get(item.chunk_id, set())
        tts_pred = tts_pred_by_chunk.get(item.chunk_id, set())
        pos_tts = pos_all & tts_valid_term_indices

        tp_text += len(text_pred & pos_all)
        pred_total_text += len(text_pred)

        if pos_tts:
            total_valid_pos_tts += len(pos_tts)
            chunks_with_tts_pos += 1
            tp_tts += len(tts_pred & pos_tts)
            tp_inter += len((text_pred & tts_pred) & pos_tts)
        pred_total_tts += len(tts_pred)
        pred_total_inter += len(text_pred & tts_pred)

    recall_text = (tp_text / total_valid_pos_all) if total_valid_pos_all > 0 else 0.0
    precision_text = (tp_text / pred_total_text) if pred_total_text > 0 else 0.0

    recall_tts = (tp_tts / total_valid_pos_tts) if total_valid_pos_tts > 0 else 0.0
    precision_tts = (tp_tts / pred_total_tts) if pred_total_tts > 0 else 0.0

    recall_inter = (tp_inter / total_valid_pos_tts) if total_valid_pos_tts > 0 else 0.0
    precision_inter = (tp_inter / pred_total_inter) if pred_total_inter > 0 else 0.0
    keep_ratio_vs_text = (pred_total_inter / pred_total_text) if pred_total_text > 0 else 0.0
    keep_ratio_vs_tts = (pred_total_inter / pred_total_tts) if pred_total_tts > 0 else 0.0

    if enable_tts_eval:
        tts_recall_value = _format_float(recall_tts)
        tts_precision_value = _format_float(precision_tts)
        intersection_recall_value = _format_float(recall_inter)
        intersection_precision_value = _format_float(precision_inter)
        keep_ratio_vs_text_value = _format_float(keep_ratio_vs_text)
        keep_ratio_vs_tts_value = _format_float(keep_ratio_vs_tts)
    else:
        tts_recall_value = "NA"
        tts_precision_value = "NA"
        intersection_recall_value = "NA"
        intersection_precision_value = "NA"
        keep_ratio_vs_text_value = "NA"
        keep_ratio_vs_tts_value = "NA"

    return {
        "model_name": model_name,
        "model_path": model_path,
        "enable_tts_eval": int(enable_tts_eval),
        "top_k": TOP_K,
        "num_chunks_all": len(items),
        "chunks_with_tts_pos": chunks_with_tts_pos,
        "unique_terms": unique_terms_count,
        "tts_valid_terms": len(tts_valid_term_indices),
        "total_pos_all": total_valid_pos_all,
        "total_pos_tts": total_valid_pos_tts,
        "text_recall_at_k": _format_float(recall_text),
        "text_precision_at_k": _format_float(precision_text),
        "text_tp": tp_text,
        "text_pred_total": pred_total_text,
        "tts_recall_at_k": tts_recall_value,
        "tts_precision_at_k": tts_precision_value,
        "tts_tp": tp_tts,
        "tts_pred_total": pred_total_tts,
        "intersection_recall_at_k": intersection_recall_value,
        "intersection_precision_at_k": intersection_precision_value,
        "intersection_tp": tp_inter,
        "intersection_pred_total": pred_total_inter,
        "intersection_keep_ratio_vs_text": keep_ratio_vs_text_value,
        "intersection_keep_ratio_vs_tts": keep_ratio_vs_tts_value,
    }


def _write_tsv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        _err("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=CSV_DELIMITER,
            lineterminator=CSV_LINE_TERMINATOR,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _log_delta(rows: Sequence[Dict[str, Any]]) -> None:
    if len(rows) != 2:
        return

    left = rows[0]
    right = rows[1]
    metric_keys = [
        "text_recall_at_k",
        "text_precision_at_k",
        "tts_recall_at_k",
        "tts_precision_at_k",
        "intersection_recall_at_k",
        "intersection_precision_at_k",
        "intersection_keep_ratio_vs_text",
    ]

    _log(f"Delta report: {right['model_name']} - {left['model_name']}")
    for key in metric_keys:
        try:
            delta = float(right[key]) - float(left[key])
            _log(f"delta_{key}={_format_float(delta)}")
        except Exception:
            _warn(f"Skip delta for key={key}")


def _parse_float_or_none(v: str) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _write_plot(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    by_name = {str(r.get("model_name", "")): r for r in rows}
    old_row = by_name.get("old_model_v1")
    new_row = by_name.get("new_model_tts_v2")
    if old_row is None or new_row is None:
        _warn("Plot skipped: expected rows for old_model_v1 and new_model_tts_v2.")
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, axes = plt.subplots(1, 2, figsize=(PLOT_FIGSIZE_W, PLOT_FIGSIZE_H))

        # Panel 1: speech->text comparison (old vs new).
        labels = ["Recall@10", "Precision@10"]
        old_vals = [
            _parse_float_or_none(str(old_row.get("text_recall_at_k", ""))) or 0.0,
            _parse_float_or_none(str(old_row.get("text_precision_at_k", ""))) or 0.0,
        ]
        new_vals = [
            _parse_float_or_none(str(new_row.get("text_recall_at_k", ""))) or 0.0,
            _parse_float_or_none(str(new_row.get("text_precision_at_k", ""))) or 0.0,
        ]
        xs = [0, 1]
        bar_w = 0.36
        axes[0].bar([x - bar_w / 2 for x in xs], old_vals, width=bar_w, label="old_model_v1", color="tab:blue")
        axes[0].bar([x + bar_w / 2 for x in xs], new_vals, width=bar_w, label="new_model_tts_v2", color="tab:orange")
        axes[0].set_xticks(xs)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("Speech -> Text Top-10")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].legend()

        # Panel 2: new model tts/intersection metrics.
        metric_keys = [
            "tts_recall_at_k",
            "tts_precision_at_k",
            "intersection_recall_at_k",
            "intersection_precision_at_k",
        ]
        metric_labels = [
            "TTS Recall@10",
            "TTS Precision@10",
            "Inter Recall@10",
            "Inter Precision@10",
        ]
        metric_vals = [_parse_float_or_none(str(new_row.get(key, ""))) for key in metric_keys]
        bar_vals = [v if v is not None else 0.0 for v in metric_vals]
        axes[1].bar(range(4), bar_vals, color=["tab:green", "tab:green", "tab:red", "tab:red"])
        axes[1].set_xticks(range(4))
        axes[1].set_xticklabels(metric_labels, rotation=18, ha="right")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("New Model TTS / Intersection")
        axes[1].grid(axis="y", alpha=0.3)

        for i, value in enumerate(metric_vals):
            if value is None:
                axes[1].text(i, 0.02, "NA", ha="center", va="bottom", fontsize=9)

        keep_ratio = _parse_float_or_none(str(new_row.get("intersection_keep_ratio_vs_text", "")))
        if keep_ratio is not None:
            axes[1].text(
                0.98,
                0.95,
                f"Inter keep ratio vs text: {keep_ratio:.3f}",
                transform=axes[1].transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="0.8"),
            )

        fig.suptitle("GigaSpeech Top-10 Comparison (No threshold)")
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)
        _log(f"Wrote plot: {path}")
    except Exception as e:
        _warn(f"Plot skipped (matplotlib not available or failed): {e}")


def main() -> int:
    global DEVICE, OUTPUT_DIR, TTS_ROOT_DIR

    env_device = os.environ.get("OFFLINE_EVAL_DEVICE", "").strip()
    if env_device:
        DEVICE = env_device
    env_out = os.environ.get("OFFLINE_EVAL_OUTPUT_DIR", "").strip()
    if env_out:
        OUTPUT_DIR = env_out
    env_tts_root = os.environ.get("OFFLINE_EVAL_TTS_ROOT_DIR", "").strip()
    if env_tts_root:
        TTS_ROOT_DIR = env_tts_root

    effective_device = _resolve_effective_device(DEVICE)
    if effective_device != DEVICE:
        _log(f"DEVICE adjusted: requested={DEVICE} effective={effective_device}")
    else:
        _log(f"DEVICE: {effective_device}")

    out_dir = Path(OUTPUT_DIR)
    _ensure_dir(out_dir)

    dev_jsonl = Path(DEV_JSONL)
    if not dev_jsonl.exists():
        _err(f"DEV_JSONL not found: {dev_jsonl}")

    examples, term_to_tts_path_from_data = _load_and_group_dev_jsonl(dev_jsonl)
    if MAX_CHUNKS > 0:
        examples = examples[:MAX_CHUNKS]
        _log(f"Applied MAX_CHUNKS={MAX_CHUNKS}, evaluating chunks={len(examples)}")

    unique_terms = sorted({term for ex in examples for term in ex.gt_terms})
    unique_terms_count = len(unique_terms)
    _log(f"Unique terms from dataset: {unique_terms_count}")

    glossary_path = out_dir / GLOSSARY_JSON_NAME
    if not glossary_path.exists():
        _log(f"Writing glossary JSON: {glossary_path}")
        _build_glossary_json(unique_terms, glossary_path)
    else:
        _log(f"Using existing glossary JSON: {glossary_path}")

    rows: List[Dict[str, Any]] = []
    for spec in MODEL_SPECS:
        model_name = str(spec["model_name"])
        model_path = str(spec["model_path"])
        row = _evaluate_single_model(
            model_name=model_name,
            model_path=model_path,
            enable_tts_eval=bool(spec.get("enable_tts_eval", True)),
            effective_device=effective_device,
            out_dir=out_dir,
            glossary_path=glossary_path,
            examples=examples,
            term_to_tts_path_from_data=term_to_tts_path_from_data,
            unique_terms_count=unique_terms_count,
        )
        rows.append(row)

    tsv_path = out_dir / RESULT_TSV_NAME
    _log(f"Writing comparison TSV: {tsv_path}")
    _write_tsv(tsv_path, rows)
    png_path = out_dir / PLOT_PNG_NAME
    _write_plot(png_path, rows)
    _log_delta(rows)

    _log("Done.")
    _log(f"TSV: {tsv_path}")
    _log(f"Glossary: {glossary_path}")
    return 0


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit(
            "Missing dependency: torch. Run this script in an environment that has PyTorch installed."
        )
    raise SystemExit(main())
