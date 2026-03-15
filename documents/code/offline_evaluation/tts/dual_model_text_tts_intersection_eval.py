#!/usr/bin/env python3

"""
Dual-model offline evaluation: separate Text and TTS models for intersection analysis.

Two separate models:
  - Text model (tts_weight=0.0): speech -> text FAISS index retrieval (semantic similarity)
  - TTS model (tts_weight=1.0): speech -> TTS audio bank retrieval (acoustic similarity)

Analysis:
  1. TTS bank size statistics
  2. Qualitative samples: top-k comparison to illustrate acoustic vs semantic similarity
  3. Quantitative metrics by chunk category:
     a. no-term chunks: noise reduction analysis (intersection removes false positives)
     b. existing-term chunks: recall/precision/F1 before and after intersection
"""

from __future__ import annotations

# ======Configuration=====
DEV_JSONL = "/mnt/gemini/data/siqiouyang/term_dev_dataset_final.jsonl"
DEV_JSONL_WITH_TTS = "/mnt/gemini/data/siqiouyang/term_dev_dataset_final_with_tts.jsonl"
TTS_ROOT_DIR = "/mnt/gemini/data/siqiouyang/term_dev_tts"

TEXT_MODEL_NAME = "text_ttsw0.0_epoch5"
TEXT_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_tts_lora-r32-tr16_bs4k_ttsw0.0_ttm=query key value_temperature=0.03_v2_epoch_5.pt"
)

TTS_MODEL_NAME = "tts_ttsw1.0_step2000_retrained"
TTS_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_tts_lora-r32-tr16_bs4k_ttsw1.0_ttm=query key value_temperature=0.03_v2_step_2000.pt"
)

TEXT_LORA_R = 16
TARGET_LANG_CODE = "zh"
INDEX_BUILD_BATCH_SIZE = 1024

DEVICE = "cuda:0"
AUDIO_BASE_MODEL_NAME = "Atotti/Qwen3-Omni-AudioTransformer"
AUDIO_LORA_R = 32
AUDIO_LORA_ALPHA = 64
ENABLE_CPU_FALLBACK_WHEN_NO_CUDA = True
CPU_FALLBACK_DEVICE = "cpu"

TOP_K = 10

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SECONDS = 1.92
EXPECTED_CHUNK_SAMPLES = 30720  # 1.92s * 16kHz

EVAL_BATCH_SIZE = 32
TTS_EMB_BATCH_SIZE = 256
MAX_TTS_PROTOTYPES_PER_TERM = 0  # <=0 means use all
MAX_CHUNKS = 0  # 0 means no limit; >0 for smoke test

OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/offline_eval_dual_model_intersection_v2_retrained"
GLOSSARY_JSON_NAME = "gigaspeech_dev_terms_glossary.json"
RESULT_TSV_NAME = "dual_model_intersection_metrics.tsv"
SAMPLES_TXT_NAME = "qualitative_samples.txt"

VOTING_MIN_VOTES = 1
SCORE_THRESHOLD = 0.0

NUM_QUALITATIVE_SAMPLES_PER_CATEGORY = 8

CSV_DELIMITER = "\t"
CSV_LINE_TERMINATOR = "\n"
FLOAT_DECIMALS = 6
# ======Configuration=====

import csv
import gc
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


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
                f"Auto-adjust device to {mapped}."
            )
            return mapped
    return requested_device


def _is_cuda_device(device: str) -> bool:
    return str(device).strip().lower().startswith("cuda:")


def _ensure_runtime_device(requested_device: str) -> str:
    import torch

    if not _is_cuda_device(requested_device):
        return requested_device
    if torch.cuda.is_available():
        return requested_device
    if ENABLE_CPU_FALLBACK_WHEN_NO_CUDA:
        _warn(f"CUDA not available. Fallback to {CPU_FALLBACK_DEVICE!r}.")
        return CPU_FALLBACK_DEVICE
    _err("CUDA device is requested but no CUDA GPU is available.")
    return requested_device


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkKey:
    utter_id: str
    chunk_idx: str

    def as_id(self) -> str:
        return f"{self.utter_id}::{self.chunk_idx}"


@dataclass
class ChunkData:
    key: ChunkKey
    audio_path: str
    chunk_src_text: str
    gt_terms: Set[str]

    @property
    def has_term(self) -> bool:
        return len(self.gt_terms) > 0


@dataclass
class TopKResult:
    """Top-K retrieval result for a single chunk from one model."""
    term_indices: List[int]
    scores: List[float]

    def index_set(self) -> Set[int]:
        return set(self.term_indices)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_full_dev_dataset(
    dev_jsonl: Path,
) -> List[ChunkData]:
    _log(f"Loading full DEV_JSONL (including no-term rows): {dev_jsonl}")
    groups: Dict[str, ChunkData] = {}
    total_rows = 0
    no_term_rows = 0
    with_term_rows = 0

    for obj in _read_jsonl(dev_jsonl):
        total_rows += 1
        term = str(obj.get("term", "")).strip().lower()
        utter_id = str(obj.get("utter_id", "")).strip()
        chunk_idx = str(obj.get("chunk_idx", "")).strip()
        audio_path = str(obj.get("chunk_audio_path", "")).strip()
        src_text = str(obj.get("chunk_src_text", "")).strip()

        assert utter_id and chunk_idx and audio_path, (
            f"Missing fields: utter_id={utter_id!r} chunk_idx={chunk_idx!r} audio_path={audio_path!r}"
        )

        ck = ChunkKey(utter_id=utter_id, chunk_idx=chunk_idx)
        cid = ck.as_id()
        if cid not in groups:
            groups[cid] = ChunkData(key=ck, audio_path=audio_path, chunk_src_text=src_text, gt_terms=set())

        if term:
            groups[cid].gt_terms.add(term)
            with_term_rows += 1
        else:
            no_term_rows += 1

    chunks = list(groups.values())
    chunks.sort(key=lambda x: (x.key.utter_id, int(x.key.chunk_idx) if x.key.chunk_idx.isdigit() else x.key.chunk_idx))
    _log(
        f"Loaded rows={total_rows}, no_term_rows={no_term_rows}, with_term_rows={with_term_rows}, "
        f"unique_chunks={len(chunks)} "
        f"(no_term_chunks={sum(1 for c in chunks if not c.has_term)}, "
        f"with_term_chunks={sum(1 for c in chunks if c.has_term)})"
    )
    assert chunks, "No valid chunks loaded."
    return chunks


def _load_tts_paths(dev_jsonl_with_tts: Path) -> Dict[str, List[str]]:
    _log(f"Loading TTS paths from: {dev_jsonl_with_tts}")
    term_to_tts_paths: Dict[str, List[str]] = {}
    for obj in _read_jsonl(dev_jsonl_with_tts):
        term = str(obj.get("term", "")).strip().lower()
        tts_path = str(obj.get("tts_audio_path", "")).strip()
        if not term or not tts_path:
            continue
        paths = term_to_tts_paths.setdefault(term, [])
        if tts_path not in paths:
            paths.append(tts_path)
    _log(f"TTS paths loaded: unique_terms={len(term_to_tts_paths)}, total_paths={sum(len(v) for v in term_to_tts_paths.values())}")
    return term_to_tts_paths


# ---------------------------------------------------------------------------
# Index building / loading
# ---------------------------------------------------------------------------

def _build_glossary_json(unique_terms: Sequence[str], glossary_path: Path) -> None:
    glossary: Dict[str, str] = {term: "" for term in unique_terms}
    with glossary_path.open("w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)


def _maybe_build_index(glossary_path: Path, index_path: Path, model_path: str, device: str) -> None:
    if index_path.exists():
        _log(f"Using existing index: {index_path}")
        return

    script_path = _REPO_ROOT / "retriever" / "gigaspeech" / "build_index_v4.py"
    assert script_path.exists(), f"build_index_v4.py not found: {script_path}"

    _log(f"Building index for model={model_path} -> {index_path}")
    import subprocess
    cmd = [
        sys.executable, str(script_path),
        "--glossary_path", str(glossary_path),
        "--model_path", model_path,
        "--output_path", str(index_path),
        "--text_lora_r", str(TEXT_LORA_R),
        "--device", str(device),
        "--batch_size", str(INDEX_BUILD_BATCH_SIZE),
        "--target_lang_code", str(TARGET_LANG_CODE),
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    assert proc.returncode == 0, (
        f"Index build failed (rc={proc.returncode}). stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    )
    _log("Index build finished.")


def _load_index_data(index_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    import pickle

    with index_path.open("rb") as f:
        data = pickle.load(f)
    term_list = data["term_list"]

    term_to_idx: Dict[str, int] = {}
    idx_to_term: Dict[int, str] = {}
    for i, item in enumerate(term_list):
        key = str(item.get("key", "")).strip().lower()
        assert key, f"Empty key at index {i} in {index_path}"
        term_to_idx[key] = i
        idx_to_term[i] = key

    assert term_to_idx, f"Empty term_list in index: {index_path}"
    return term_to_idx, idx_to_term


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _load_audio_mono_16k(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path)
    assert sr == EXPECTED_SAMPLE_RATE, f"Unexpected sample rate: path={path} sr={sr} expected={EXPECTED_SAMPLE_RATE}"

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


# ---------------------------------------------------------------------------
# Phase 2: Text model evaluation (speech -> text FAISS index)
# ---------------------------------------------------------------------------

def _run_text_model_retrieval(
    chunks: Sequence[ChunkData],
    term_to_idx: Dict[str, int],
    index_path: Path,
    model_path: str,
    effective_device: str,
) -> Dict[str, TopKResult]:
    _log(f"=== Phase: Text Model Retrieval ({TEXT_MODEL_NAME}) ===")

    from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4
    import faiss
    import torch

    use_cuda_amp = _is_cuda_device(effective_device)
    feature_dtype = torch.bfloat16 if use_cuda_amp else torch.float32

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

    results: Dict[str, TopKResult] = {}
    _log(f"Text model: encoding {len(chunks)} chunks and retrieving Top-{TOP_K} ...")

    for start in range(0, len(chunks), EVAL_BATCH_SIZE):
        batch = chunks[start : start + EVAL_BATCH_SIZE]
        audios = [_load_audio_mono_16k(c.audio_path) for c in batch]
        inputs = retriever.feature_extractor(
            audios, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False
        )
        features = inputs.input_features
        batch_size, channels, mel_len = features.shape
        input_features = features.transpose(0, 1).reshape(channels, -1).to(retriever.device).to(feature_dtype)
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

    _log(f"Text model retrieval done: {len(results)} chunks processed.")

    del retriever
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Phase 3: TTS model evaluation (speech -> TTS audio bank)
#
# Directly loads the Qwen3OmniRetriever audio encoder only, matching the
# training eval protocol exactly. Does NOT go through
# StreamingQwen3RAGRetrieverV4 -- no text encoder, no FAISS index needed.
# ---------------------------------------------------------------------------

def _encode_audio_batch(
    model: Any,
    feature_extractor: Any,
    audio_arrays: Sequence[np.ndarray],
    device: Any,
) -> np.ndarray:
    """Encode a batch of raw audio arrays via Qwen3OmniRetriever and return
    L2-normalised float32 embeddings of shape ``(B, D)``."""
    import torch
    import faiss

    inputs = feature_extractor(
        list(audio_arrays),
        sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    features = inputs.input_features
    bsz, channels, mel_len = features.shape
    input_features = (
        features.transpose(0, 1)
        .reshape(channels, -1)
        .to(device)
        .to(torch.bfloat16)
    )
    feature_lens = torch.full(
        (bsz,), mel_len, dtype=torch.long, device=device
    )
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = model(input_features, feature_lens)
        embs = embs.detach().cpu().float().numpy()
    faiss.normalize_L2(embs)
    return embs.astype(np.float32, copy=False)


def _run_tts_model_retrieval(
    chunks: Sequence[ChunkData],
    term_to_idx: Dict[str, int],
    term_to_tts_paths_raw: Dict[str, List[str]],
    model_path: str,
    effective_device: str,
) -> Tuple[Dict[str, TopKResult], int, int]:
    """TTS recall using audio encoder only (same as training eval)."""
    _log(f"=== Phase: TTS Model Retrieval ({TTS_MODEL_NAME}) -- direct encoder ===")

    from agents.streaming_qwen3_rag_retriever_v4 import Qwen3OmniRetriever
    from transformers import WhisperFeatureExtractor
    import torch

    device = torch.device(effective_device)

    _log("Loading Qwen3OmniRetriever audio encoder directly (strict=True) ...")
    model = Qwen3OmniRetriever(
        model_id=AUDIO_BASE_MODEL_NAME,
        target_dim=1024,
        use_lora=True,
        lora_rank=AUDIO_LORA_R,
        lora_alpha=AUDIO_LORA_ALPHA,
    ).to(device).to(torch.bfloat16)

    ckpt = torch.load(model_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    _log("Audio encoder loaded (strict=True, all keys matched).")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    # ---- Build TTS prototype embedding bank ----
    proto_term_idx_list: List[int] = []
    proto_audio_paths: List[str] = []
    tts_valid_term_set: Set[int] = set()

    for term, raw_paths in term_to_tts_paths_raw.items():
        idx = term_to_idx.get(term)
        if idx is None:
            continue
        resolved: List[str] = []
        for rp in raw_paths:
            rr = _resolve_tts_path(rp, TTS_ROOT_DIR)
            if not rr or rr in resolved:
                continue
            resolved.append(rr)
            if MAX_TTS_PROTOTYPES_PER_TERM > 0 and len(resolved) >= MAX_TTS_PROTOTYPES_PER_TERM:
                break
        for p in resolved:
            proto_term_idx_list.append(idx)
            proto_audio_paths.append(p)
        if resolved:
            tts_valid_term_set.add(idx)

    assert proto_audio_paths, "No valid TTS prototypes found."
    tts_bank_terms = len(tts_valid_term_set)
    tts_bank_prototypes = len(proto_audio_paths)
    _log(
        f"TTS Bank: unique_terms={tts_bank_terms}, "
        f"total_prototypes={tts_bank_prototypes}, "
        f"avg_prototypes_per_term={tts_bank_prototypes / tts_bank_terms:.2f}"
    )

    proto_embs_parts: List[np.ndarray] = []
    for start in range(0, len(proto_audio_paths), TTS_EMB_BATCH_SIZE):
        batch_paths = proto_audio_paths[start : start + TTS_EMB_BATCH_SIZE]
        audios = [_load_audio_mono_16k(p) for p in batch_paths]
        proto_embs_parts.append(
            _encode_audio_batch(model, feature_extractor, audios, device)
        )
    proto_embs = np.concatenate(proto_embs_parts, axis=0)
    proto_term_idx_np = np.array(proto_term_idx_list, dtype=np.int64)

    sorted_term_list = sorted(tts_valid_term_set)
    term_pos_map = {ti: pos for pos, ti in enumerate(sorted_term_list)}

    # ---- Encode speech chunks & search TTS bank (max-pool per term) ----
    _log(f"TTS model: encoding {len(chunks)} chunks and searching TTS bank Top-{TOP_K} ...")
    results: Dict[str, TopKResult] = {}

    for start in range(0, len(chunks), TTS_EMB_BATCH_SIZE):
        batch = chunks[start : start + TTS_EMB_BATCH_SIZE]
        audios = [_load_audio_mono_16k(c.audio_path) for c in batch]
        speech_embs = _encode_audio_batch(model, feature_extractor, audios, device)

        for i, chunk in enumerate(batch):
            cid = chunk.key.as_id()
            scores = proto_embs @ speech_embs[i]

            term_scores = np.full(tts_bank_terms, -np.inf, dtype=np.float32)
            for pi in range(scores.shape[0]):
                ti = int(proto_term_idx_np[pi])
                pos = term_pos_map.get(ti)
                if pos is not None and scores[pi] > term_scores[pos]:
                    term_scores[pos] = float(scores[pi])

            valid_mask = np.isfinite(term_scores)
            assert np.any(valid_mask), f"No valid term scores for chunk {cid}"
            valid_positions = np.where(valid_mask)[0]
            valid_scores = term_scores[valid_positions]

            k = min(TOP_K, valid_scores.shape[0])
            top_pos = np.argpartition(-valid_scores, k - 1)[:k]
            top_pos = top_pos[np.argsort(-valid_scores[top_pos])]

            term_indices = [sorted_term_list[valid_positions[j]] for j in top_pos]
            score_vals = [float(valid_scores[j]) for j in top_pos]
            results[cid] = TopKResult(term_indices=term_indices, scores=score_vals)

    _log(f"TTS model retrieval done: {len(results)} chunks processed.")

    del model, ckpt, sd
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results, tts_bank_terms, tts_bank_prototypes


# ---------------------------------------------------------------------------
# Phase 4: Analysis
# ---------------------------------------------------------------------------

@dataclass
class CategoryMetrics:
    category: str
    num_chunks: int = 0

    # For with-term chunks: micro-averaged metrics
    total_gt: int = 0
    text_tp: int = 0
    text_pred_total: int = 0
    tts_tp: int = 0
    tts_pred_total: int = 0
    inter_tp: int = 0
    inter_pred_total: int = 0

    # For no-term chunks: noise counts
    total_text_preds: int = 0
    total_tts_preds: int = 0
    total_inter_preds: int = 0

    def text_recall(self) -> float:
        return self.text_tp / self.total_gt if self.total_gt > 0 else 0.0

    def text_precision(self) -> float:
        return self.text_tp / self.text_pred_total if self.text_pred_total > 0 else 0.0

    def text_f1(self) -> float:
        return _f1(self.text_precision(), self.text_recall())

    def tts_recall(self) -> float:
        return self.tts_tp / self.total_gt if self.total_gt > 0 else 0.0

    def tts_precision(self) -> float:
        return self.tts_tp / self.tts_pred_total if self.tts_pred_total > 0 else 0.0

    def tts_f1(self) -> float:
        return _f1(self.tts_precision(), self.tts_recall())

    def inter_recall(self) -> float:
        return self.inter_tp / self.total_gt if self.total_gt > 0 else 0.0

    def inter_precision(self) -> float:
        return self.inter_tp / self.inter_pred_total if self.inter_pred_total > 0 else 0.0

    def inter_f1(self) -> float:
        return _f1(self.inter_precision(), self.inter_recall())

    def avg_text_preds(self) -> float:
        return self.total_text_preds / self.num_chunks if self.num_chunks > 0 else 0.0

    def avg_tts_preds(self) -> float:
        return self.total_tts_preds / self.num_chunks if self.num_chunks > 0 else 0.0

    def avg_inter_preds(self) -> float:
        return self.total_inter_preds / self.num_chunks if self.num_chunks > 0 else 0.0

    def noise_reduction_vs_text(self) -> float:
        return 1.0 - (self.total_inter_preds / self.total_text_preds) if self.total_text_preds > 0 else 0.0

    def noise_reduction_vs_tts(self) -> float:
        return 1.0 - (self.total_inter_preds / self.total_tts_preds) if self.total_tts_preds > 0 else 0.0


def _compute_metrics(
    chunks: Sequence[ChunkData],
    text_results: Dict[str, TopKResult],
    tts_results: Dict[str, TopKResult],
    term_to_idx: Dict[str, int],
) -> Tuple[CategoryMetrics, CategoryMetrics, CategoryMetrics]:
    """Compute metrics split by no-term / with-term / overall."""
    no_term_m = CategoryMetrics(category="no_term")
    with_term_m = CategoryMetrics(category="with_term")
    overall_m = CategoryMetrics(category="overall")

    for chunk in chunks:
        cid = chunk.key.as_id()
        text_r = text_results.get(cid)
        tts_r = tts_results.get(cid)
        assert text_r is not None, f"Missing text result for chunk {cid}"
        assert tts_r is not None, f"Missing TTS result for chunk {cid}"

        text_set = text_r.index_set()
        tts_set = tts_r.index_set()
        inter_set = text_set & tts_set

        gt_indices: Set[int] = set()
        for term in chunk.gt_terms:
            idx = term_to_idx.get(term)
            if idx is not None:
                gt_indices.add(idx)

        if chunk.has_term:
            m = with_term_m
        else:
            m = no_term_m

        m.num_chunks += 1
        overall_m.num_chunks += 1

        m.total_text_preds += len(text_set)
        m.total_tts_preds += len(tts_set)
        m.total_inter_preds += len(inter_set)
        overall_m.total_text_preds += len(text_set)
        overall_m.total_tts_preds += len(tts_set)
        overall_m.total_inter_preds += len(inter_set)

        if gt_indices:
            m.total_gt += len(gt_indices)
            m.text_tp += len(text_set & gt_indices)
            m.text_pred_total += len(text_set)
            m.tts_tp += len(tts_set & gt_indices)
            m.tts_pred_total += len(tts_set)
            m.inter_tp += len(inter_set & gt_indices)
            m.inter_pred_total += len(inter_set)

            overall_m.total_gt += len(gt_indices)
            overall_m.text_tp += len(text_set & gt_indices)
            overall_m.text_pred_total += len(text_set)
            overall_m.tts_tp += len(tts_set & gt_indices)
            overall_m.tts_pred_total += len(tts_set)
            overall_m.inter_tp += len(inter_set & gt_indices)
            overall_m.inter_pred_total += len(inter_set)

    return no_term_m, with_term_m, overall_m


# ---------------------------------------------------------------------------
# Qualitative samples
# ---------------------------------------------------------------------------

def _print_qualitative_samples(
    chunks: Sequence[ChunkData],
    text_results: Dict[str, TopKResult],
    tts_results: Dict[str, TopKResult],
    idx_to_term: Dict[int, str],
    term_to_idx: Dict[str, int],
    out_file: Optional[Path] = None,
) -> str:
    lines: List[str] = []

    def _add(s: str = "") -> None:
        lines.append(s)

    with_term_chunks = [c for c in chunks if c.has_term]
    no_term_chunks = [c for c in chunks if not c.has_term]

    np.random.seed(42)

    _add("=" * 90)
    _add("QUALITATIVE SAMPLES: WITH-TERM CHUNKS (existing terms)")
    _add("=" * 90)
    sample_wt = np.random.choice(
        len(with_term_chunks),
        size=min(NUM_QUALITATIVE_SAMPLES_PER_CATEGORY, len(with_term_chunks)),
        replace=False,
    )
    for sample_idx, ci in enumerate(sorted(sample_wt)):
        chunk = with_term_chunks[ci]
        cid = chunk.key.as_id()
        text_r = text_results[cid]
        tts_r = tts_results[cid]

        gt_idx_set = {term_to_idx[t] for t in chunk.gt_terms if t in term_to_idx}
        inter_set = set(text_r.term_indices) & set(tts_r.term_indices)
        inter_gt = inter_set & gt_idx_set

        _add(f"\n--- Sample {sample_idx + 1} (with-term) ---")
        _add(f"  chunk_id:       {cid}")
        _add(f"  chunk_src_text: \"{chunk.chunk_src_text}\"")
        _add(f"  gt_terms:       {chunk.gt_terms}")

        _add(f"  Text Model Top-{TOP_K} (semantic):")
        for rank, (ti, sc) in enumerate(zip(text_r.term_indices, text_r.scores)):
            term_name = idx_to_term.get(ti, f"?idx={ti}")
            hit = " << GT HIT" if ti in gt_idx_set else ""
            _add(f"    {rank+1:2d}. {term_name:<30s} score={sc:.4f}{hit}")

        _add(f"  TTS Model Top-{TOP_K} (acoustic):")
        for rank, (ti, sc) in enumerate(zip(tts_r.term_indices, tts_r.scores)):
            term_name = idx_to_term.get(ti, f"?idx={ti}")
            hit = " << GT HIT" if ti in gt_idx_set else ""
            _add(f"    {rank+1:2d}. {term_name:<30s} score={sc:.4f}{hit}")

        _add(f"  Intersection ({len(inter_set)} terms):")
        if inter_set:
            for ti in sorted(inter_set):
                term_name = idx_to_term.get(ti, f"?idx={ti}")
                hit = " << GT HIT" if ti in gt_idx_set else ""
                _add(f"    - {term_name}{hit}")
        else:
            _add("    (empty)")
        _add(f"  => GT in intersection: {len(inter_gt)}/{len(gt_idx_set)}")

    _add("\n" + "=" * 90)
    _add("QUALITATIVE SAMPLES: NO-TERM CHUNKS (noise removal)")
    _add("=" * 90)
    sample_nt = np.random.choice(
        len(no_term_chunks),
        size=min(NUM_QUALITATIVE_SAMPLES_PER_CATEGORY, len(no_term_chunks)),
        replace=False,
    )
    for sample_idx, ci in enumerate(sorted(sample_nt)):
        chunk = no_term_chunks[ci]
        cid = chunk.key.as_id()
        text_r = text_results[cid]
        tts_r = tts_results[cid]
        inter_set = set(text_r.term_indices) & set(tts_r.term_indices)

        _add(f"\n--- Sample {sample_idx + 1} (no-term) ---")
        _add(f"  chunk_id:       {cid}")
        _add(f"  chunk_src_text: \"{chunk.chunk_src_text}\"")
        _add(f"  gt_terms:       (none)")

        _add(f"  Text Model Top-{TOP_K} (semantic) = {len(text_r.term_indices)} preds:")
        for rank, (ti, sc) in enumerate(zip(text_r.term_indices, text_r.scores)):
            term_name = idx_to_term.get(ti, f"?idx={ti}")
            _add(f"    {rank+1:2d}. {term_name:<30s} score={sc:.4f}")

        _add(f"  TTS Model Top-{TOP_K} (acoustic) = {len(tts_r.term_indices)} preds:")
        for rank, (ti, sc) in enumerate(zip(tts_r.term_indices, tts_r.scores)):
            term_name = idx_to_term.get(ti, f"?idx={ti}")
            _add(f"    {rank+1:2d}. {term_name:<30s} score={sc:.4f}")

        _add(f"  Intersection ({len(inter_set)} terms) = ALL NOISE:")
        if inter_set:
            for ti in sorted(inter_set):
                _add(f"    - {idx_to_term.get(ti, f'?idx={ti}')}")
        else:
            _add("    (empty) << intersection successfully removed all noise")

    output = "\n".join(lines)
    if out_file is not None:
        with out_file.open("w", encoding="utf-8") as f:
            f.write(output)
        _log(f"Wrote qualitative samples: {out_file}")
    return output


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_metrics_table(
    no_term_m: CategoryMetrics,
    with_term_m: CategoryMetrics,
    overall_m: CategoryMetrics,
    tts_bank_terms: int,
    tts_bank_prototypes: int,
    glossary_size: int,
) -> str:
    lines: List[str] = []

    def _add(s: str = "") -> None:
        lines.append(s)

    _add("\n" + "=" * 90)
    _add("TTS BANK STATISTICS")
    _add("=" * 90)
    _add(f"  Glossary size (unique terms):       {glossary_size}")
    _add(f"  TTS bank terms (with TTS audio):    {tts_bank_terms}")
    _add(f"  TTS bank prototypes (total audios):  {tts_bank_prototypes}")
    _add(f"  Avg prototypes per term:             {tts_bank_prototypes/tts_bank_terms:.2f}" if tts_bank_terms > 0 else "  Avg prototypes per term:             N/A")
    _add(f"  TTS coverage:                        {tts_bank_terms}/{glossary_size} = {tts_bank_terms/glossary_size*100:.1f}%" if glossary_size > 0 else "")

    _add("\n" + "=" * 90)
    _add("NO-TERM CHUNKS: Noise Reduction Analysis")
    _add("=" * 90)
    _add(f"  Number of chunks: {no_term_m.num_chunks}")
    _add(f"  Total text predictions:         {no_term_m.total_text_preds}  (avg per chunk: {no_term_m.avg_text_preds():.2f})")
    _add(f"  Total TTS predictions:          {no_term_m.total_tts_preds}  (avg per chunk: {no_term_m.avg_tts_preds():.2f})")
    _add(f"  Total intersection predictions: {no_term_m.total_inter_preds}  (avg per chunk: {no_term_m.avg_inter_preds():.2f})")
    _add(f"  Noise reduction vs text:        {no_term_m.noise_reduction_vs_text()*100:.1f}%")
    _add(f"  Noise reduction vs TTS:         {no_term_m.noise_reduction_vs_tts()*100:.1f}%")

    _add("\n" + "=" * 90)
    _add("WITH-TERM CHUNKS: Recall / Precision / F1")
    _add("=" * 90)
    _add(f"  Number of chunks: {with_term_m.num_chunks}")
    _add(f"  Total GT positives: {with_term_m.total_gt}")
    _add("")
    _add(f"  {'Method':<25s} {'Recall':>10s} {'Precision':>10s} {'F1':>10s} {'TP':>6s} {'Pred':>6s}")
    _add(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*6}")
    _add(
        f"  {'Text (semantic)':<25s} "
        f"{with_term_m.text_recall():>10.4f} {with_term_m.text_precision():>10.4f} {with_term_m.text_f1():>10.4f} "
        f"{with_term_m.text_tp:>6d} {with_term_m.text_pred_total:>6d}"
    )
    _add(
        f"  {'TTS (acoustic)':<25s} "
        f"{with_term_m.tts_recall():>10.4f} {with_term_m.tts_precision():>10.4f} {with_term_m.tts_f1():>10.4f} "
        f"{with_term_m.tts_tp:>6d} {with_term_m.tts_pred_total:>6d}"
    )
    _add(
        f"  {'Intersection':<25s} "
        f"{with_term_m.inter_recall():>10.4f} {with_term_m.inter_precision():>10.4f} {with_term_m.inter_f1():>10.4f} "
        f"{with_term_m.inter_tp:>6d} {with_term_m.inter_pred_total:>6d}"
    )
    _add("")
    _add(f"  Noise reduction (pred count): text={with_term_m.text_pred_total} -> inter={with_term_m.inter_pred_total} "
         f"(-{with_term_m.noise_reduction_vs_text()*100:.1f}%)")

    _add("\n" + "=" * 90)
    _add("OVERALL (all chunks)")
    _add("=" * 90)
    _add(f"  Number of chunks: {overall_m.num_chunks}")
    _add(f"  Total GT positives: {overall_m.total_gt}")
    _add(f"  Total text predictions:         {overall_m.total_text_preds}")
    _add(f"  Total TTS predictions:          {overall_m.total_tts_preds}")
    _add(f"  Total intersection predictions: {overall_m.total_inter_preds}")
    _add(f"  Noise reduction vs text:        {overall_m.noise_reduction_vs_text()*100:.1f}%")
    if overall_m.total_gt > 0:
        _add("")
        _add(f"  {'Method':<25s} {'Recall':>10s} {'Precision':>10s} {'F1':>10s}")
        _add(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        _add(
            f"  {'Text (semantic)':<25s} "
            f"{overall_m.text_recall():>10.4f} {overall_m.text_precision():>10.4f} {overall_m.text_f1():>10.4f}"
        )
        _add(
            f"  {'TTS (acoustic)':<25s} "
            f"{overall_m.tts_recall():>10.4f} {overall_m.tts_precision():>10.4f} {overall_m.tts_f1():>10.4f}"
        )
        _add(
            f"  {'Intersection':<25s} "
            f"{overall_m.inter_recall():>10.4f} {overall_m.inter_precision():>10.4f} {overall_m.inter_f1():>10.4f}"
        )

    output = "\n".join(lines)
    print(output, flush=True)
    return output


def _write_tsv(path: Path, no_term_m: CategoryMetrics, with_term_m: CategoryMetrics, overall_m: CategoryMetrics) -> None:
    fieldnames = [
        "category", "num_chunks", "total_gt",
        "text_recall", "text_precision", "text_f1",
        "text_tp", "text_pred_total",
        "tts_recall", "tts_precision", "tts_f1",
        "tts_tp", "tts_pred_total",
        "inter_recall", "inter_precision", "inter_f1",
        "inter_tp", "inter_pred_total",
        "avg_text_preds", "avg_tts_preds", "avg_inter_preds",
        "noise_reduction_vs_text", "noise_reduction_vs_tts",
    ]

    def _row(m: CategoryMetrics) -> Dict[str, Any]:
        return {
            "category": m.category,
            "num_chunks": m.num_chunks,
            "total_gt": m.total_gt,
            "text_recall": _format_float(m.text_recall()),
            "text_precision": _format_float(m.text_precision()),
            "text_f1": _format_float(m.text_f1()),
            "text_tp": m.text_tp,
            "text_pred_total": m.text_pred_total,
            "tts_recall": _format_float(m.tts_recall()),
            "tts_precision": _format_float(m.tts_precision()),
            "tts_f1": _format_float(m.tts_f1()),
            "tts_tp": m.tts_tp,
            "tts_pred_total": m.tts_pred_total,
            "inter_recall": _format_float(m.inter_recall()),
            "inter_precision": _format_float(m.inter_precision()),
            "inter_f1": _format_float(m.inter_f1()),
            "inter_tp": m.inter_tp,
            "inter_pred_total": m.inter_pred_total,
            "avg_text_preds": _format_float(m.avg_text_preds()),
            "avg_tts_preds": _format_float(m.avg_tts_preds()),
            "avg_inter_preds": _format_float(m.avg_inter_preds()),
            "noise_reduction_vs_text": _format_float(m.noise_reduction_vs_text()),
            "noise_reduction_vs_tts": _format_float(m.noise_reduction_vs_tts()),
        }

    rows = [_row(no_term_m), _row(with_term_m), _row(overall_m)]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=CSV_DELIMITER, lineterminator=CSV_LINE_TERMINATOR)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _log(f"Wrote TSV: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    effective_device = _ensure_runtime_device(effective_device)
    _log(f"DEVICE: {effective_device}")

    out_dir = Path(OUTPUT_DIR)
    _ensure_dir(out_dir)

    # ---- Phase 1: Load data ----
    _log("=== Phase 1: Loading data ===")
    dev_jsonl = Path(DEV_JSONL)
    assert dev_jsonl.exists(), f"DEV_JSONL not found: {dev_jsonl}"

    all_chunks = _load_full_dev_dataset(dev_jsonl)
    if MAX_CHUNKS > 0:
        all_chunks = all_chunks[:MAX_CHUNKS]
        _log(f"Applied MAX_CHUNKS={MAX_CHUNKS}, evaluating chunks={len(all_chunks)}")

    dev_jsonl_with_tts = Path(DEV_JSONL_WITH_TTS)
    assert dev_jsonl_with_tts.exists(), f"DEV_JSONL_WITH_TTS not found: {dev_jsonl_with_tts}"
    term_to_tts_paths = _load_tts_paths(dev_jsonl_with_tts)

    unique_terms = sorted({term for chunk in all_chunks for term in chunk.gt_terms})
    glossary_size = len(unique_terms)
    _log(f"Unique terms from dataset: {glossary_size}")

    glossary_path = out_dir / GLOSSARY_JSON_NAME
    if not glossary_path.exists():
        _log(f"Writing glossary JSON: {glossary_path}")
        _build_glossary_json(unique_terms, glossary_path)
    else:
        _log(f"Using existing glossary JSON: {glossary_path}")

    # ---- Build FAISS text index (only needed for text model) ----
    text_model_dir = out_dir / _safe_name(TEXT_MODEL_NAME)
    _ensure_dir(text_model_dir)
    text_index_path = text_model_dir / f"index_v4_tr{TEXT_LORA_R}_{_safe_name(TEXT_MODEL_NAME)}.pkl"

    assert Path(TEXT_MODEL_PATH).exists(), f"Text model not found: {TEXT_MODEL_PATH}"
    assert Path(TTS_MODEL_PATH).exists(), f"TTS model not found: {TTS_MODEL_PATH}"

    _maybe_build_index(glossary_path, text_index_path, TEXT_MODEL_PATH, effective_device)

    term_to_idx, idx_to_term = _load_index_data(text_index_path)

    # ---- Phase 2: Text model retrieval ----
    text_results = _run_text_model_retrieval(
        chunks=all_chunks,
        term_to_idx=term_to_idx,
        index_path=text_index_path,
        model_path=TEXT_MODEL_PATH,
        effective_device=effective_device,
    )

    # ---- Phase 3: TTS model retrieval (direct audio encoder, no FAISS) ----
    tts_results, tts_bank_terms, tts_bank_prototypes = _run_tts_model_retrieval(
        chunks=all_chunks,
        term_to_idx=term_to_idx,
        term_to_tts_paths_raw=term_to_tts_paths,
        model_path=TTS_MODEL_PATH,
        effective_device=effective_device,
    )

    # ---- Phase 4: Analysis ----
    _log("=== Phase 4: Analysis ===")

    no_term_m, with_term_m, overall_m = _compute_metrics(
        chunks=all_chunks,
        text_results=text_results,
        tts_results=tts_results,
        term_to_idx=term_to_idx,
    )

    samples_text = _print_qualitative_samples(
        chunks=all_chunks,
        text_results=text_results,
        tts_results=tts_results,
        idx_to_term=idx_to_term,
        term_to_idx=term_to_idx,
        out_file=out_dir / SAMPLES_TXT_NAME,
    )
    print(samples_text, flush=True)

    _print_metrics_table(
        no_term_m=no_term_m,
        with_term_m=with_term_m,
        overall_m=overall_m,
        tts_bank_terms=tts_bank_terms,
        tts_bank_prototypes=tts_bank_prototypes,
        glossary_size=glossary_size,
    )

    _write_tsv(out_dir / RESULT_TSV_NAME, no_term_m, with_term_m, overall_m)

    _log("Done.")
    _log(f"Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit("Missing dependency: torch.")
    raise SystemExit(main())
