#!/usr/bin/env python3

"""
Compare two retrieval architectures for term retrieval:

  A) Dual-model: separate Text encoder (ttsw=0.0) and TTS encoder (ttsw=1.0),
     intersection of their top-k results.
  B) Single-model: ONE shared encoder (ttsw=0.5) used for both text-FAISS
     retrieval AND TTS prototype bank search, then intersection.

The same dev dataset, glossary, and TTS prototype bank are used for both so the
comparison is apples-to-apples.
"""

from __future__ import annotations

# ======Configuration=====
DEV_JSONL = "/mnt/gemini/data/siqiouyang/term_dev_dataset_final.jsonl"
DEV_JSONL_WITH_TTS = "/mnt/gemini/data/siqiouyang/term_dev_dataset_final_with_tts.jsonl"
TTS_ROOT_DIR = "/mnt/gemini/data/siqiouyang/term_dev_tts"

DUAL_TEXT_MODEL_NAME = "dual_text_ttsw0.0_epoch5"
DUAL_TEXT_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_tts_lora-r32-tr16_bs4k_ttsw0.0_ttm=query key value_temperature=0.03_v2_epoch_5.pt"
)

DUAL_TTS_MODEL_NAME = "dual_tts_ttsw1.0_step2000"
DUAL_TTS_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_tts_lora-r32-tr16_bs4k_ttsw1.0_ttm=query key value_temperature=0.03_v2_step_2000.pt"
)

SINGLE_MODEL_NAME = "single_ttsw0.5_epoch3"
SINGLE_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_v2_epoch_3.pt"
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
EXPECTED_CHUNK_SAMPLES = 30720

EVAL_BATCH_SIZE = 32
TTS_EMB_BATCH_SIZE = 256
MAX_TTS_PROTOTYPES_PER_TERM = 0
MAX_CHUNKS = 0

OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/offline_eval_dual_vs_single_model"
GLOSSARY_JSON_NAME = "gigaspeech_dev_terms_glossary.json"
RESULT_TSV_NAME = "dual_vs_single_metrics.tsv"
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
    term_indices: List[int]
    scores: List[float]

    def index_set(self) -> Set[int]:
        return set(self.term_indices)


# ---------------------------------------------------------------------------
# Data loading (reused for both configurations)
# ---------------------------------------------------------------------------

def _load_full_dev_dataset(dev_jsonl: Path) -> List[ChunkData]:
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
# Shared encoder helpers
# ---------------------------------------------------------------------------

def _encode_audio_batch(
    model: Any,
    feature_extractor: Any,
    audio_arrays: Sequence[np.ndarray],
    device: Any,
) -> np.ndarray:
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


def _build_tts_proto_bank(
    model: Any,
    feature_extractor: Any,
    term_to_tts_paths_raw: Dict[str, List[str]],
    term_to_idx: Dict[str, int],
    device: Any,
) -> Tuple[np.ndarray, np.ndarray, List[int], Set[int], Dict[int, int], int, int]:
    """Build TTS prototype embedding bank using the given audio encoder.

    Returns (proto_embs, proto_term_idx_np, sorted_term_list, tts_valid_term_set,
             term_pos_map, tts_bank_terms, tts_bank_prototypes).
    """
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
        f"  TTS Bank: unique_terms={tts_bank_terms}, "
        f"total_prototypes={tts_bank_prototypes}, "
        f"avg_prototypes_per_term={tts_bank_prototypes / tts_bank_terms:.2f}"
    )

    proto_embs_parts: List[np.ndarray] = []
    for start in range(0, len(proto_audio_paths), TTS_EMB_BATCH_SIZE):
        batch_paths = proto_audio_paths[start: start + TTS_EMB_BATCH_SIZE]
        audios = [_load_audio_mono_16k(p) for p in batch_paths]
        proto_embs_parts.append(
            _encode_audio_batch(model, feature_extractor, audios, device)
        )
    proto_embs = np.concatenate(proto_embs_parts, axis=0)
    proto_term_idx_np = np.array(proto_term_idx_list, dtype=np.int64)

    sorted_term_list = sorted(tts_valid_term_set)
    term_pos_map = {ti: pos for pos, ti in enumerate(sorted_term_list)}

    return (proto_embs, proto_term_idx_np, sorted_term_list,
            tts_valid_term_set, term_pos_map, tts_bank_terms, tts_bank_prototypes)


def _search_tts_bank(
    speech_embs_batch: np.ndarray,
    proto_embs: np.ndarray,
    proto_term_idx_np: np.ndarray,
    sorted_term_list: List[int],
    term_pos_map: Dict[int, int],
    tts_bank_terms: int,
) -> List[TopKResult]:
    """Search TTS prototype bank for a batch of speech embeddings."""
    results: List[TopKResult] = []
    for i in range(speech_embs_batch.shape[0]):
        scores = proto_embs @ speech_embs_batch[i]
        term_scores = np.full(tts_bank_terms, -np.inf, dtype=np.float32)
        for pi in range(scores.shape[0]):
            ti = int(proto_term_idx_np[pi])
            pos = term_pos_map.get(ti)
            if pos is not None and scores[pi] > term_scores[pos]:
                term_scores[pos] = float(scores[pi])

        valid_mask = np.isfinite(term_scores)
        assert np.any(valid_mask), "No valid term scores"
        valid_positions = np.where(valid_mask)[0]
        valid_scores = term_scores[valid_positions]

        k = min(TOP_K, valid_scores.shape[0])
        top_pos = np.argpartition(-valid_scores, k - 1)[:k]
        top_pos = top_pos[np.argsort(-valid_scores[top_pos])]

        term_indices = [sorted_term_list[valid_positions[j]] for j in top_pos]
        score_vals = [float(valid_scores[j]) for j in top_pos]
        results.append(TopKResult(term_indices=term_indices, scores=score_vals))
    return results


# ---------------------------------------------------------------------------
# Config A: Dual-model retrieval (separate text & TTS encoders)
# ---------------------------------------------------------------------------

def _run_dual_model_eval(
    chunks: Sequence[ChunkData],
    term_to_idx: Dict[str, int],
    term_to_tts_paths_raw: Dict[str, List[str]],
    glossary_path: Path,
    out_dir: Path,
    effective_device: str,
) -> Tuple[Dict[str, TopKResult], Dict[str, TopKResult], int, int]:
    _log("=" * 70)
    _log("CONFIG A: Dual-Model (text=ttsw0.0 + tts=ttsw1.0)")
    _log("=" * 70)

    import torch
    import faiss
    from agents.streaming_qwen3_rag_retriever_v4 import (
        StreamingQwen3RAGRetrieverV4,
        Qwen3OmniRetriever,
    )
    from transformers import WhisperFeatureExtractor

    use_cuda_amp = _is_cuda_device(effective_device)
    feature_dtype = torch.bfloat16 if use_cuda_amp else torch.float32
    device = torch.device(effective_device)

    # --- Text pathway ---
    _log("[Dual] Phase: Text model FAISS retrieval")
    text_model_dir = out_dir / _safe_name(DUAL_TEXT_MODEL_NAME)
    _ensure_dir(text_model_dir)
    text_index_path = text_model_dir / f"index_v4_tr{TEXT_LORA_R}_{_safe_name(DUAL_TEXT_MODEL_NAME)}.pkl"
    _maybe_build_index(glossary_path, text_index_path, DUAL_TEXT_MODEL_PATH, effective_device)

    retriever = StreamingQwen3RAGRetrieverV4(
        index_path=str(text_index_path),
        model_path=str(DUAL_TEXT_MODEL_PATH),
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

    text_results: Dict[str, TopKResult] = {}
    for start in range(0, len(chunks), EVAL_BATCH_SIZE):
        batch = chunks[start: start + EVAL_BATCH_SIZE]
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
            text_results[cid] = TopKResult(term_indices=term_indices, scores=scores)

    _log(f"[Dual] Text retrieval done: {len(text_results)} chunks")

    del retriever
    gc.collect()
    torch.cuda.empty_cache()

    # --- TTS pathway ---
    _log("[Dual] Phase: TTS model prototype bank retrieval")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    tts_model = Qwen3OmniRetriever(
        model_id=AUDIO_BASE_MODEL_NAME,
        target_dim=1024,
        use_lora=True,
        lora_rank=AUDIO_LORA_R,
        lora_alpha=AUDIO_LORA_ALPHA,
    ).to(device).to(torch.bfloat16)

    ckpt = torch.load(DUAL_TTS_MODEL_PATH, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    tts_model.load_state_dict(sd, strict=True)
    tts_model.eval()
    _log("[Dual] TTS audio encoder loaded.")

    (proto_embs, proto_term_idx_np, sorted_term_list,
     tts_valid_term_set, term_pos_map,
     tts_bank_terms, tts_bank_prototypes) = _build_tts_proto_bank(
        tts_model, feature_extractor, term_to_tts_paths_raw, term_to_idx, device
    )

    tts_results: Dict[str, TopKResult] = {}
    for start in range(0, len(chunks), TTS_EMB_BATCH_SIZE):
        batch = chunks[start: start + TTS_EMB_BATCH_SIZE]
        audios = [_load_audio_mono_16k(c.audio_path) for c in batch]
        speech_embs = _encode_audio_batch(tts_model, feature_extractor, audios, device)
        batch_results = _search_tts_bank(
            speech_embs, proto_embs, proto_term_idx_np,
            sorted_term_list, term_pos_map, tts_bank_terms,
        )
        for chunk, res in zip(batch, batch_results):
            tts_results[chunk.key.as_id()] = res

    _log(f"[Dual] TTS retrieval done: {len(tts_results)} chunks")

    del tts_model, ckpt, sd, proto_embs
    gc.collect()
    torch.cuda.empty_cache()

    return text_results, tts_results, tts_bank_terms, tts_bank_prototypes


# ---------------------------------------------------------------------------
# Config B: Single-model retrieval (shared encoder for both pathways)
# ---------------------------------------------------------------------------

def _run_single_model_eval(
    chunks: Sequence[ChunkData],
    term_to_idx: Dict[str, int],
    term_to_tts_paths_raw: Dict[str, List[str]],
    glossary_path: Path,
    out_dir: Path,
    effective_device: str,
) -> Tuple[Dict[str, TopKResult], Dict[str, TopKResult], int, int]:
    _log("=" * 70)
    _log("CONFIG B: Single-Model (ttsw=0.5 for BOTH text-FAISS and TTS-bank)")
    _log("=" * 70)

    import torch
    import faiss
    from agents.streaming_qwen3_rag_retriever_v4 import Qwen3OmniRetriever
    from transformers import WhisperFeatureExtractor

    device = torch.device(effective_device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    # --- Load the single shared audio encoder ---
    _log("[Single] Loading shared audio encoder ...")
    model = Qwen3OmniRetriever(
        model_id=AUDIO_BASE_MODEL_NAME,
        target_dim=1024,
        use_lora=True,
        lora_rank=AUDIO_LORA_R,
        lora_alpha=AUDIO_LORA_ALPHA,
    ).to(device).to(torch.bfloat16)

    ckpt = torch.load(SINGLE_MODEL_PATH, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    _log("[Single] Audio encoder loaded.")

    # --- Build FAISS index using single model's text encoder ---
    single_model_dir = out_dir / _safe_name(SINGLE_MODEL_NAME)
    _ensure_dir(single_model_dir)
    single_index_path = single_model_dir / f"index_v4_tr{TEXT_LORA_R}_{_safe_name(SINGLE_MODEL_NAME)}.pkl"
    _maybe_build_index(glossary_path, single_index_path, SINGLE_MODEL_PATH, effective_device)

    import pickle
    with single_index_path.open("rb") as f:
        index_data = pickle.load(f)
    faiss_index = faiss.deserialize_index(index_data["faiss_index"])

    # --- Build TTS prototype bank using the SAME audio encoder ---
    _log("[Single] Building TTS prototype bank with shared encoder ...")
    (proto_embs, proto_term_idx_np, sorted_term_list,
     tts_valid_term_set, term_pos_map,
     tts_bank_terms, tts_bank_prototypes) = _build_tts_proto_bank(
        model, feature_extractor, term_to_tts_paths_raw, term_to_idx, device
    )

    # --- Encode speech ONCE, search both pathways ---
    _log(f"[Single] Encoding {len(chunks)} chunks and searching both pathways ...")
    text_results: Dict[str, TopKResult] = {}
    tts_results: Dict[str, TopKResult] = {}

    for start in range(0, len(chunks), EVAL_BATCH_SIZE):
        batch = chunks[start: start + EVAL_BATCH_SIZE]
        audios = [_load_audio_mono_16k(c.audio_path) for c in batch]
        speech_embs = _encode_audio_batch(model, feature_extractor, audios, device)

        # Text pathway: FAISS search
        dists, indices = faiss_index.search(speech_embs, TOP_K)
        for i, chunk in enumerate(batch):
            cid = chunk.key.as_id()
            term_indices = [int(idx) for idx in indices[i] if int(idx) >= 0]
            scores = [float(dists[i][j]) for j in range(len(term_indices))]
            text_results[cid] = TopKResult(term_indices=term_indices, scores=scores)

        # TTS pathway: prototype bank search
        batch_tts_results = _search_tts_bank(
            speech_embs, proto_embs, proto_term_idx_np,
            sorted_term_list, term_pos_map, tts_bank_terms,
        )
        for chunk, res in zip(batch, batch_tts_results):
            tts_results[chunk.key.as_id()] = res

    _log(f"[Single] Done: {len(text_results)} text + {len(tts_results)} TTS results")

    del model, ckpt, sd, proto_embs, faiss_index
    gc.collect()
    torch.cuda.empty_cache()

    return text_results, tts_results, tts_bank_terms, tts_bank_prototypes


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

@dataclass
class CategoryMetrics:
    category: str
    num_chunks: int = 0
    total_gt: int = 0
    text_tp: int = 0
    text_pred_total: int = 0
    tts_tp: int = 0
    tts_pred_total: int = 0
    inter_tp: int = 0
    inter_pred_total: int = 0
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

    def noise_reduction_vs_text(self) -> float:
        return 1.0 - (self.total_inter_preds / self.total_text_preds) if self.total_text_preds > 0 else 0.0


def _compute_metrics(
    chunks: Sequence[ChunkData],
    text_results: Dict[str, TopKResult],
    tts_results: Dict[str, TopKResult],
    term_to_idx: Dict[str, int],
) -> Tuple[CategoryMetrics, CategoryMetrics, CategoryMetrics]:
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

        m = with_term_m if chunk.has_term else no_term_m

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
# Output
# ---------------------------------------------------------------------------

def _print_comparison_table(
    dual_wt: CategoryMetrics,
    dual_nt: CategoryMetrics,
    single_wt: CategoryMetrics,
    single_nt: CategoryMetrics,
    dual_tts_bank_terms: int,
    dual_tts_bank_protos: int,
    single_tts_bank_terms: int,
    single_tts_bank_protos: int,
    glossary_size: int,
) -> str:
    lines: List[str] = []

    def _add(s: str = "") -> None:
        lines.append(s)

    _add("\n" + "=" * 100)
    _add("COMPARISON: Dual-Model vs Single-Model")
    _add("=" * 100)

    _add("\n--- TTS Bank Statistics ---")
    _add(f"  Glossary size (unique terms):   {glossary_size}")
    _add(f"  Dual   TTS bank: {dual_tts_bank_terms} terms, {dual_tts_bank_protos} prototypes")
    _add(f"  Single TTS bank: {single_tts_bank_terms} terms, {single_tts_bank_protos} prototypes")

    _add("\n--- WITH-TERM CHUNKS: Recall / Precision / F1 ---")
    _add(f"  Dual   chunks={dual_wt.num_chunks}  GT={dual_wt.total_gt}")
    _add(f"  Single chunks={single_wt.num_chunks} GT={single_wt.total_gt}")
    _add("")

    header = f"  {'Method':<35s} {'Recall':>8s} {'Prec':>8s} {'F1':>8s} {'TP':>6s} {'Pred':>6s}"
    sep = f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}"

    _add(header)
    _add(sep)

    def _row(label: str, m: CategoryMetrics, kind: str) -> str:
        if kind == "text":
            return (f"  {label:<35s} {m.text_recall():>8.4f} {m.text_precision():>8.4f} "
                    f"{m.text_f1():>8.4f} {m.text_tp:>6d} {m.text_pred_total:>6d}")
        elif kind == "tts":
            return (f"  {label:<35s} {m.tts_recall():>8.4f} {m.tts_precision():>8.4f} "
                    f"{m.tts_f1():>8.4f} {m.tts_tp:>6d} {m.tts_pred_total:>6d}")
        else:
            return (f"  {label:<35s} {m.inter_recall():>8.4f} {m.inter_precision():>8.4f} "
                    f"{m.inter_f1():>8.4f} {m.inter_tp:>6d} {m.inter_pred_total:>6d}")

    _add(_row("Dual  | Text (ttsw=0.0)", dual_wt, "text"))
    _add(_row("Dual  | TTS  (ttsw=1.0)", dual_wt, "tts"))
    _add(_row("Dual  | Intersection", dual_wt, "inter"))
    _add(sep)
    _add(_row("Single| Text (ttsw=0.5)", single_wt, "text"))
    _add(_row("Single| TTS  (ttsw=0.5)", single_wt, "tts"))
    _add(_row("Single| Intersection", single_wt, "inter"))

    _add("\n--- NO-TERM CHUNKS: Noise Reduction ---")
    _add(f"  {'Metric':<35s} {'Dual':>12s} {'Single':>12s}")
    _add(f"  {'-'*35} {'-'*12} {'-'*12}")
    _add(f"  {'Chunks':<35s} {dual_nt.num_chunks:>12d} {single_nt.num_chunks:>12d}")
    _add(f"  {'Text predictions':<35s} {dual_nt.total_text_preds:>12d} {single_nt.total_text_preds:>12d}")
    _add(f"  {'TTS predictions':<35s} {dual_nt.total_tts_preds:>12d} {single_nt.total_tts_preds:>12d}")
    _add(f"  {'Intersection predictions':<35s} {dual_nt.total_inter_preds:>12d} {single_nt.total_inter_preds:>12d}")
    _add(f"  {'Noise reduction vs text':<35s} {dual_nt.noise_reduction_vs_text()*100:>11.1f}% {single_nt.noise_reduction_vs_text()*100:>11.1f}%")

    _add("\n--- DELTA (Single - Dual) on with-term chunks ---")
    delta_text_recall = single_wt.text_recall() - dual_wt.text_recall()
    delta_tts_recall = single_wt.tts_recall() - dual_wt.tts_recall()
    delta_inter_recall = single_wt.inter_recall() - dual_wt.inter_recall()
    delta_inter_prec = single_wt.inter_precision() - dual_wt.inter_precision()
    delta_inter_f1 = single_wt.inter_f1() - dual_wt.inter_f1()
    _add(f"  Text Recall:          {delta_text_recall:+.4f}")
    _add(f"  TTS  Recall:          {delta_tts_recall:+.4f}")
    _add(f"  Intersection Recall:  {delta_inter_recall:+.4f}")
    _add(f"  Intersection Prec:    {delta_inter_prec:+.4f}")
    _add(f"  Intersection F1:      {delta_inter_f1:+.4f}")

    output = "\n".join(lines)
    print(output, flush=True)
    return output


def _write_comparison_tsv(
    path: Path,
    dual_nt: CategoryMetrics, dual_wt: CategoryMetrics, dual_ov: CategoryMetrics,
    single_nt: CategoryMetrics, single_wt: CategoryMetrics, single_ov: CategoryMetrics,
) -> None:
    fieldnames = [
        "config", "category", "num_chunks", "total_gt",
        "text_recall", "text_precision", "text_f1", "text_tp", "text_pred_total",
        "tts_recall", "tts_precision", "tts_f1", "tts_tp", "tts_pred_total",
        "inter_recall", "inter_precision", "inter_f1", "inter_tp", "inter_pred_total",
        "noise_reduction_vs_text",
    ]

    def _row(config: str, m: CategoryMetrics) -> Dict[str, Any]:
        return {
            "config": config, "category": m.category,
            "num_chunks": m.num_chunks, "total_gt": m.total_gt,
            "text_recall": _format_float(m.text_recall()),
            "text_precision": _format_float(m.text_precision()),
            "text_f1": _format_float(m.text_f1()),
            "text_tp": m.text_tp, "text_pred_total": m.text_pred_total,
            "tts_recall": _format_float(m.tts_recall()),
            "tts_precision": _format_float(m.tts_precision()),
            "tts_f1": _format_float(m.tts_f1()),
            "tts_tp": m.tts_tp, "tts_pred_total": m.tts_pred_total,
            "inter_recall": _format_float(m.inter_recall()),
            "inter_precision": _format_float(m.inter_precision()),
            "inter_f1": _format_float(m.inter_f1()),
            "inter_tp": m.inter_tp, "inter_pred_total": m.inter_pred_total,
            "noise_reduction_vs_text": _format_float(m.noise_reduction_vs_text()),
        }

    rows = [
        _row("dual", dual_nt), _row("dual", dual_wt), _row("dual", dual_ov),
        _row("single", single_nt), _row("single", single_wt), _row("single", single_ov),
    ]
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

    # ---- Load data (shared) ----
    _log("=== Loading data (shared between both configs) ===")
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

    assert Path(DUAL_TEXT_MODEL_PATH).exists(), f"Dual text model not found: {DUAL_TEXT_MODEL_PATH}"
    assert Path(DUAL_TTS_MODEL_PATH).exists(), f"Dual TTS model not found: {DUAL_TTS_MODEL_PATH}"
    assert Path(SINGLE_MODEL_PATH).exists(), f"Single model not found: {SINGLE_MODEL_PATH}"

    # Both configs share the same term_to_idx from the glossary.
    # Each config builds its own FAISS index (with its own text encoder),
    # but the term vocabulary is identical.
    # We load index data from the dual-text index just to get term_to_idx
    # (the mapping is glossary-derived and order-identical across configs).
    dual_text_model_dir = out_dir / _safe_name(DUAL_TEXT_MODEL_NAME)
    _ensure_dir(dual_text_model_dir)
    dual_text_index_path = (
        dual_text_model_dir / f"index_v4_tr{TEXT_LORA_R}_{_safe_name(DUAL_TEXT_MODEL_NAME)}.pkl"
    )
    _maybe_build_index(glossary_path, dual_text_index_path, DUAL_TEXT_MODEL_PATH, effective_device)
    term_to_idx, idx_to_term = _load_index_data(dual_text_index_path)

    # ---- Config A: Dual-model ----
    dual_text_res, dual_tts_res, dual_bank_terms, dual_bank_protos = _run_dual_model_eval(
        chunks=all_chunks,
        term_to_idx=term_to_idx,
        term_to_tts_paths_raw=term_to_tts_paths,
        glossary_path=glossary_path,
        out_dir=out_dir,
        effective_device=effective_device,
    )

    # ---- Config B: Single-model ----
    single_text_res, single_tts_res, single_bank_terms, single_bank_protos = _run_single_model_eval(
        chunks=all_chunks,
        term_to_idx=term_to_idx,
        term_to_tts_paths_raw=term_to_tts_paths,
        glossary_path=glossary_path,
        out_dir=out_dir,
        effective_device=effective_device,
    )

    # ---- Metrics ----
    _log("=== Computing metrics ===")
    dual_nt, dual_wt, dual_ov = _compute_metrics(all_chunks, dual_text_res, dual_tts_res, term_to_idx)
    single_nt, single_wt, single_ov = _compute_metrics(all_chunks, single_text_res, single_tts_res, term_to_idx)

    _print_comparison_table(
        dual_wt=dual_wt, dual_nt=dual_nt,
        single_wt=single_wt, single_nt=single_nt,
        dual_tts_bank_terms=dual_bank_terms, dual_tts_bank_protos=dual_bank_protos,
        single_tts_bank_terms=single_bank_terms, single_tts_bank_protos=single_bank_protos,
        glossary_size=glossary_size,
    )

    _write_comparison_tsv(
        out_dir / RESULT_TSV_NAME,
        dual_nt, dual_wt, dual_ov,
        single_nt, single_wt, single_ov,
    )

    _log("Done.")
    _log(f"Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit("Missing dependency: torch.")
    raise SystemExit(main())
