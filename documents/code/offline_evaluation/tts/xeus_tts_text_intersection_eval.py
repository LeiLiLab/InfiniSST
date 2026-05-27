#!/usr/bin/env python3
"""
Dual-encoder offline evaluation: Qwen3-Omni (text path) + XEUS (TTS path).

Two different audio encoders:
  - Text path:  Qwen3-Omni AuT (ttsw=0.0 epoch5) -> speech emb -> FAISS text index (BGE-M3)
  - TTS path:   XEUS E-Branchformer (ttsw=1.0 epoch2) -> speech emb -> TTS proto bank (XEUS)

Analysis:
  1. TTS bank size statistics
  2. Qualitative samples: top-k text vs TTS, intersection behaviour
  3. Quantitative: recall / precision / F1 for text, TTS, intersection
"""

from __future__ import annotations

# ======Configuration=====
DEV_JSONL = "/mnt/gemini/data/siqiouyang/term_dev_dataset_final.jsonl"
DEV_JSONL_WITH_TTS = "/mnt/gemini/data/siqiouyang/term_dev_dataset_final_with_tts.jsonl"
TTS_ROOT_DIR = "/mnt/gemini/data/siqiouyang/term_dev_tts"

# --- Text path: Qwen3-Omni (same as dual_model_text_tts_intersection_eval.py) ---
TEXT_MODEL_NAME = "text_ttsw0.0_epoch5"
TEXT_MODEL_PATH = (
    "/mnt/gemini/data/jiaxuanluo/"
    "q3rag_unfrozen_lora-r32-tr16_bs4k_w1.0-0.0-ttm=query key value-temperature=0.03_enriched_v2_full_best.pt"
    #"q3rag_tts_lora-r32-tr16_bs4k_ttsw0.0_ttm=query key value_temperature=0.03_v2_epoch_5.pt"
)

TEXT_AUDIO_BASE_MODEL_NAME = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_AUDIO_LORA_R = 32
TEXT_AUDIO_LORA_ALPHA = 64
TEXT_LORA_R = 16
TARGET_LANG_CODE = "zh"
INDEX_BUILD_BATCH_SIZE = 1024

# --- TTS path: XEUS ---
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

OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/offline_eval_xeus_tts_qwen3_text_intersection"
GLOSSARY_JSON_NAME = "gigaspeech_dev_terms_glossary.json"
# RESULT_TSV_NAME and SAMPLES_TXT_NAME are derived from TOP_K at runtime (see main())

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


def _detect_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "retriever" / "gigaspeech" / "build_index_v4.py").exists():
            return parent
    raise RuntimeError(f"Cannot locate repository root from: {current}")


_REPO_ROOT = _detect_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _format_float(x: float) -> str:
    return f"{x:.{FLOAT_DECIMALS}f}"


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _is_cuda_device(device: str) -> bool:
    return str(device).strip().lower().startswith("cuda:")


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
# Data loading
# ---------------------------------------------------------------------------

def _load_full_dev_dataset(dev_jsonl: Path) -> List[ChunkData]:
    _log(f"Loading DEV_JSONL: {dev_jsonl}")
    groups: Dict[str, ChunkData] = {}
    total_rows = 0

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

    chunks = list(groups.values())
    chunks.sort(key=lambda x: (x.key.utter_id, int(x.key.chunk_idx) if x.key.chunk_idx.isdigit() else x.key.chunk_idx))
    with_term = sum(1 for c in chunks if c.has_term)
    _log(f"Loaded rows={total_rows}, unique_chunks={len(chunks)} (with_term={with_term}, no_term={len(chunks)-with_term})")
    assert chunks, "No valid chunks loaded."
    return chunks


def _load_tts_paths(dev_jsonl_with_tts: Path) -> Dict[str, List[str]]:
    _log(f"Loading TTS paths: {dev_jsonl_with_tts}")
    result: Dict[str, List[str]] = {}
    for obj in _read_jsonl(dev_jsonl_with_tts):
        term = str(obj.get("term", "")).strip().lower()
        tts_path = str(obj.get("tts_audio_path", "")).strip()
        if not term or not tts_path:
            continue
        paths = result.setdefault(term, [])
        if tts_path not in paths:
            paths.append(tts_path)
    _log(f"TTS terms={len(result)}, total_paths={sum(len(v) for v in result.values())}")
    return result


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _load_audio_mono_16k(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path)
    assert sr == EXPECTED_SAMPLE_RATE, f"Bad SR: path={path} sr={sr}"

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
# Text index (FAISS) building / loading
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

    _log(f"Building FAISS text index -> {index_path}")
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
        assert key, f"Empty key at index {i}"
        term_to_idx[key] = i
        idx_to_term[i] = key

    assert term_to_idx, f"Empty term_list in index: {index_path}"
    return term_to_idx, idx_to_term


# ---------------------------------------------------------------------------
# Phase 2: Text model retrieval — Qwen3-Omni (unchanged from dual_model)
# ---------------------------------------------------------------------------

def _run_text_model_retrieval(
    chunks: Sequence[ChunkData],
    term_to_idx: Dict[str, int],
    index_path: Path,
    model_path: str,
    effective_device: str,
) -> Dict[str, TopKResult]:
    _log(f"=== Phase 2: Text Model Retrieval ({TEXT_MODEL_NAME}) — Qwen3-Omni ===")

    from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4
    import faiss
    import torch

    use_cuda_amp = _is_cuda_device(effective_device)
    feature_dtype = torch.bfloat16 if use_cuda_amp else torch.float32

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

    _log(f"Text model retrieval done: {len(results)} chunks.")

    del retriever
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Phase 3: TTS model retrieval — XEUS E-Branchformer
# ---------------------------------------------------------------------------

def _load_xeus_audio_encoder(device_str: str):
    """Load XeusRetriever from trained XEUS checkpoint."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from espnet2.tasks.ssl import SSLTask

    _log(f"Loading XEUS base model from: {XEUS_CHECKPOINT_PATH}")
    xeus_model, _ = SSLTask.build_model_from_file(None, XEUS_CHECKPOINT_PATH, "cpu")
    xeus_model = xeus_model.to(dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=XEUS_LORA_RANK,
        lora_alpha=XEUS_LORA_ALPHA,
        target_modules=XEUS_LORA_TARGET_MODULES,
        lora_dropout=XEUS_LORA_DROPOUT,
        bias="none",
        task_type=None,
    )
    xeus_model = get_peft_model(xeus_model, lora_config)

    class AttentivePooling(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.Tanh(),
                nn.Linear(input_dim // 2, 1),
            )

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            scores = self.attention(x)
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            weights = F.softmax(scores, dim=1)
            return torch.sum(x * weights, dim=1)

    class XeusRetrieverInference(nn.Module):
        def __init__(self, xeus, hidden_dim: int, target_dim: int):
            super().__init__()
            self.xeus = xeus
            self.pooler = AttentivePooling(hidden_dim)
            self.projector = nn.Linear(hidden_dim, target_dim)
            self.register_buffer("logit_scale", torch.tensor(0.0))

        def forward(self, wavs: torch.Tensor, wav_lengths: torch.Tensor) -> torch.Tensor:
            feats = self.xeus.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0][-1]
            B, T_out, _ = feats.shape
            T_in = wavs.shape[1]
            assert T_in > 0
            ratio = T_out / T_in
            output_lens = (wav_lengths.float() * ratio).long().clamp(min=1, max=T_out)
            mask = torch.arange(T_out, device=feats.device).unsqueeze(0).expand(B, -1) < output_lens.unsqueeze(1)
            pooled = self.pooler(feats, mask)
            projected = self.projector(pooled)
            return F.normalize(projected, p=2, dim=-1)

    model = XeusRetrieverInference(xeus_model, XEUS_HIDDEN_DIM, TARGET_DIM)

    _log(f"Loading trained XEUS weights from: {TTS_MODEL_PATH}")
    device = torch.device(device_str)
    ckpt = torch.load(TTS_MODEL_PATH, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    _log("XeusRetriever loaded (strict=True).")

    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model, device


def _encode_audio_batch_xeus(
    model, audio_arrays: Sequence[np.ndarray], device,
) -> np.ndarray:
    """Encode raw waveform batch via XeusRetriever -> L2-normalised float32 (B, D)."""
    import torch
    import faiss

    wavs_np = np.stack(audio_arrays, axis=0)
    wavs = torch.from_numpy(wavs_np).float().to(device)
    wav_lens = torch.tensor(
        [a.shape[0] for a in audio_arrays], dtype=torch.long, device=device,
    )

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = model(wavs, wav_lens)
        embs = embs.detach().cpu().float().numpy()
    faiss.normalize_L2(embs)
    return embs.astype(np.float32, copy=False)


def _run_tts_model_retrieval(
    chunks: Sequence[ChunkData],
    term_to_idx: Dict[str, int],
    term_to_tts_paths_raw: Dict[str, List[str]],
    effective_device: str,
) -> Tuple[Dict[str, TopKResult], int, int]:
    _log(f"=== Phase 3: TTS Model Retrieval ({TTS_MODEL_NAME}) — XEUS ===")
    import torch

    xeus_model, device = _load_xeus_audio_encoder(effective_device)

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
    _log(f"TTS Bank: terms={tts_bank_terms}, prototypes={tts_bank_prototypes}, avg={tts_bank_prototypes/tts_bank_terms:.2f}")

    proto_embs_parts: List[np.ndarray] = []
    for start in range(0, len(proto_audio_paths), TTS_EMB_BATCH_SIZE):
        batch_paths = proto_audio_paths[start: start + TTS_EMB_BATCH_SIZE]
        audios = [_load_audio_mono_16k(p) for p in batch_paths]
        proto_embs_parts.append(_encode_audio_batch_xeus(xeus_model, audios, device))
    proto_embs = np.concatenate(proto_embs_parts, axis=0)
    proto_term_idx_np = np.array(proto_term_idx_list, dtype=np.int64)

    sorted_term_list = sorted(tts_valid_term_set)
    term_pos_map = {ti: pos for pos, ti in enumerate(sorted_term_list)}

    # ---- Encode speech chunks & search TTS bank ----
    _log(f"TTS model: encoding {len(chunks)} chunks -> TTS bank Top-{TOP_K} ...")
    results: Dict[str, TopKResult] = {}

    for start in range(0, len(chunks), TTS_EMB_BATCH_SIZE):
        batch = chunks[start: start + TTS_EMB_BATCH_SIZE]
        audios = [_load_audio_mono_16k(c.audio_path) for c in batch]
        speech_embs = _encode_audio_batch_xeus(xeus_model, audios, device)

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

    _log(f"TTS model retrieval done: {len(results)} chunks.")

    del xeus_model
    gc.collect()
    torch.cuda.empty_cache()

    return results, tts_bank_terms, tts_bank_prototypes


# ---------------------------------------------------------------------------
# Metrics
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
    no_term_m = CategoryMetrics(category="no_term")
    with_term_m = CategoryMetrics(category="with_term")
    overall_m = CategoryMetrics(category="overall")

    for chunk in chunks:
        cid = chunk.key.as_id()
        text_r = text_results[cid]
        tts_r = tts_results[cid]

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
            for target in (m, overall_m):
                target.total_gt += len(gt_indices)
                target.text_tp += len(text_set & gt_indices)
                target.text_pred_total += len(text_set)
                target.tts_tp += len(tts_set & gt_indices)
                target.tts_pred_total += len(tts_set)
                target.inter_tp += len(inter_set & gt_indices)
                target.inter_pred_total += len(inter_set)

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
    _add("QUALITATIVE SAMPLES: WITH-TERM CHUNKS")
    _add(f"  Text encoder: Qwen3-Omni ({TEXT_MODEL_NAME})")
    _add(f"  TTS encoder:  XEUS ({TTS_MODEL_NAME})")
    _add("=" * 90)
    sample_wt = np.random.choice(
        len(with_term_chunks),
        size=min(NUM_QUALITATIVE_SAMPLES_PER_CATEGORY, len(with_term_chunks)),
        replace=False,
    )
    for si, ci in enumerate(sorted(sample_wt)):
        chunk = with_term_chunks[ci]
        cid = chunk.key.as_id()
        text_r = text_results[cid]
        tts_r = tts_results[cid]

        gt_idx_set = {term_to_idx[t] for t in chunk.gt_terms if t in term_to_idx}
        inter_set = set(text_r.term_indices) & set(tts_r.term_indices)
        inter_gt = inter_set & gt_idx_set

        _add(f"\n--- Sample {si+1} (with-term) ---")
        _add(f"  chunk_id:       {cid}")
        _add(f'  chunk_src_text: "{chunk.chunk_src_text}"')
        _add(f"  gt_terms:       {chunk.gt_terms}")

        _add(f"  Text Top-{TOP_K} (Qwen3-Omni -> BGE-M3 FAISS, semantic):")
        for rank, (ti, sc) in enumerate(zip(text_r.term_indices, text_r.scores)):
            name = idx_to_term.get(ti, f"?idx={ti}")
            hit = " << GT" if ti in gt_idx_set else ""
            _add(f"    {rank+1:2d}. {name:<35s} score={sc:.4f}{hit}")

        _add(f"  TTS Top-{TOP_K} (XEUS -> XEUS TTS bank, acoustic):")
        for rank, (ti, sc) in enumerate(zip(tts_r.term_indices, tts_r.scores)):
            name = idx_to_term.get(ti, f"?idx={ti}")
            hit = " << GT" if ti in gt_idx_set else ""
            _add(f"    {rank+1:2d}. {name:<35s} score={sc:.4f}{hit}")

        _add(f"  Intersection ({len(inter_set)} terms):")
        if inter_set:
            for ti in sorted(inter_set):
                name = idx_to_term.get(ti, f"?idx={ti}")
                hit = " << GT" if ti in gt_idx_set else ""
                _add(f"    + {name}{hit}")
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
    for si, ci in enumerate(sorted(sample_nt)):
        chunk = no_term_chunks[ci]
        cid = chunk.key.as_id()
        text_r = text_results[cid]
        tts_r = tts_results[cid]
        inter_set = set(text_r.term_indices) & set(tts_r.term_indices)

        _add(f"\n--- Sample {si+1} (no-term) ---")
        _add(f"  chunk_id:       {cid}")
        _add(f'  chunk_src_text: "{chunk.chunk_src_text}"')
        _add(f"  gt_terms:       (none)")

        _add(f"  Text Top-{TOP_K} (Qwen3-Omni, semantic):")
        for rank, (ti, sc) in enumerate(zip(text_r.term_indices, text_r.scores)):
            name = idx_to_term.get(ti, f"?idx={ti}")
            marker = " [INTER]" if ti in inter_set else ""
            _add(f"    {rank+1:2d}. {name:<35s} score={sc:.4f}{marker}")

        _add(f"  TTS Top-{TOP_K} (XEUS, acoustic):")
        for rank, (ti, sc) in enumerate(zip(tts_r.term_indices, tts_r.scores)):
            name = idx_to_term.get(ti, f"?idx={ti}")
            marker = " [INTER]" if ti in inter_set else ""
            _add(f"    {rank+1:2d}. {name:<35s} score={sc:.4f}{marker}")

        _add(f"  Intersection ({len(inter_set)} noise terms):")
        if inter_set:
            for ti in sorted(inter_set):
                _add(f"    - {idx_to_term.get(ti, f'?idx={ti}')}")
        else:
            _add("    (empty) << intersection removed ALL noise")

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
    _add("ENCODER CONFIGURATION")
    _add("=" * 90)
    _add(f"  Text path (semantic):  Qwen3-Omni AuT ({TEXT_MODEL_NAME})")
    _add(f"  TTS path (acoustic):   XEUS E-Branchformer ({TTS_MODEL_NAME})")

    _add("\n" + "=" * 90)
    _add("TTS BANK STATISTICS")
    _add("=" * 90)
    _add(f"  Glossary size (unique terms):        {glossary_size}")
    _add(f"  TTS bank terms (with TTS audio):     {tts_bank_terms}")
    _add(f"  TTS bank prototypes (total audios):   {tts_bank_prototypes}")
    if tts_bank_terms > 0:
        _add(f"  Avg prototypes per term:              {tts_bank_prototypes/tts_bank_terms:.2f}")
        _add(f"  TTS coverage:                         {tts_bank_terms}/{glossary_size} = {tts_bank_terms/glossary_size*100:.1f}%")

    _add("\n" + "=" * 90)
    _add("NO-TERM CHUNKS: Noise Reduction")
    _add("=" * 90)
    _add(f"  Chunks: {no_term_m.num_chunks}")
    _add(f"  Text preds (Qwen3):  {no_term_m.total_text_preds}  (avg {no_term_m.avg_text_preds():.2f})")
    _add(f"  TTS preds (XEUS):    {no_term_m.total_tts_preds}  (avg {no_term_m.avg_tts_preds():.2f})")
    _add(f"  Inter preds:         {no_term_m.total_inter_preds}  (avg {no_term_m.avg_inter_preds():.2f})")
    _add(f"  Noise reduction vs text: {no_term_m.noise_reduction_vs_text()*100:.1f}%")
    _add(f"  Noise reduction vs TTS:  {no_term_m.noise_reduction_vs_tts()*100:.1f}%")

    _add("\n" + "=" * 90)
    _add("WITH-TERM CHUNKS: Recall / Precision / F1")
    _add("=" * 90)
    _add(f"  Chunks: {with_term_m.num_chunks}, GT positives: {with_term_m.total_gt}")
    _add("")
    _add(f"  {'Method':<35s} {'Recall':>10s} {'Precision':>10s} {'F1':>10s} {'TP':>6s} {'Pred':>6s}")
    _add(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*6}")
    for label, rec, prec, f1v, tp, pred in [
        ("Text (Qwen3-Omni, semantic)", with_term_m.text_recall(), with_term_m.text_precision(), with_term_m.text_f1(), with_term_m.text_tp, with_term_m.text_pred_total),
        ("TTS (XEUS, acoustic)", with_term_m.tts_recall(), with_term_m.tts_precision(), with_term_m.tts_f1(), with_term_m.tts_tp, with_term_m.tts_pred_total),
        ("Intersection", with_term_m.inter_recall(), with_term_m.inter_precision(), with_term_m.inter_f1(), with_term_m.inter_tp, with_term_m.inter_pred_total),
    ]:
        _add(f"  {label:<35s} {rec:>10.4f} {prec:>10.4f} {f1v:>10.4f} {tp:>6d} {pred:>6d}")

    _add(f"\n  Noise reduction (pred count): text={with_term_m.text_pred_total} -> inter={with_term_m.inter_pred_total} "
         f"(-{with_term_m.noise_reduction_vs_text()*100:.1f}%)")

    _add("\n" + "=" * 90)
    _add("OVERALL")
    _add("=" * 90)
    _add(f"  Chunks: {overall_m.num_chunks}, GT positives: {overall_m.total_gt}")
    _add(f"  Noise reduction vs text: {overall_m.noise_reduction_vs_text()*100:.1f}%")
    if overall_m.total_gt > 0:
        _add("")
        _add(f"  {'Method':<35s} {'Recall':>10s} {'Precision':>10s} {'F1':>10s}")
        _add(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
        for label, rec, prec, f1v in [
            ("Text (Qwen3-Omni, semantic)", overall_m.text_recall(), overall_m.text_precision(), overall_m.text_f1()),
            ("TTS (XEUS, acoustic)", overall_m.tts_recall(), overall_m.tts_precision(), overall_m.tts_f1()),
            ("Intersection", overall_m.inter_recall(), overall_m.inter_precision(), overall_m.inter_f1()),
        ]:
            _add(f"  {label:<35s} {rec:>10.4f} {prec:>10.4f} {f1v:>10.4f}")

    output = "\n".join(lines)
    print(output, flush=True)
    return output


def _write_tsv(path: Path, no_term_m: CategoryMetrics, with_term_m: CategoryMetrics, overall_m: CategoryMetrics) -> None:
    fieldnames = [
        "category", "num_chunks", "total_gt",
        "text_recall", "text_precision", "text_f1", "text_tp", "text_pred_total",
        "tts_recall", "tts_precision", "tts_f1", "tts_tp", "tts_pred_total",
        "inter_recall", "inter_precision", "inter_f1", "inter_tp", "inter_pred_total",
        "avg_text_preds", "avg_tts_preds", "avg_inter_preds",
        "noise_reduction_vs_text", "noise_reduction_vs_tts",
    ]

    def _row(m: CategoryMetrics) -> Dict[str, Any]:
        return {
            "category": m.category, "num_chunks": m.num_chunks, "total_gt": m.total_gt,
            "text_recall": _format_float(m.text_recall()), "text_precision": _format_float(m.text_precision()),
            "text_f1": _format_float(m.text_f1()), "text_tp": m.text_tp, "text_pred_total": m.text_pred_total,
            "tts_recall": _format_float(m.tts_recall()), "tts_precision": _format_float(m.tts_precision()),
            "tts_f1": _format_float(m.tts_f1()), "tts_tp": m.tts_tp, "tts_pred_total": m.tts_pred_total,
            "inter_recall": _format_float(m.inter_recall()), "inter_precision": _format_float(m.inter_precision()),
            "inter_f1": _format_float(m.inter_f1()), "inter_tp": m.inter_tp, "inter_pred_total": m.inter_pred_total,
            "avg_text_preds": _format_float(m.avg_text_preds()), "avg_tts_preds": _format_float(m.avg_tts_preds()),
            "avg_inter_preds": _format_float(m.avg_inter_preds()),
            "noise_reduction_vs_text": _format_float(m.noise_reduction_vs_text()),
            "noise_reduction_vs_tts": _format_float(m.noise_reduction_vs_tts()),
        }

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=CSV_DELIMITER, lineterminator=CSV_LINE_TERMINATOR)
        writer.writeheader()
        for row in [_row(no_term_m), _row(with_term_m), _row(overall_m)]:
            writer.writerow(row)
    _log(f"Wrote TSV: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global DEVICE, OUTPUT_DIR, TTS_ROOT_DIR, TOP_K

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

    result_tsv_name = f"xeus_tts_qwen3_text_intersection_metrics_top{TOP_K}.tsv"
    samples_txt_name = f"qualitative_samples_top{TOP_K}.txt"
    _log(f"TOP_K={TOP_K}  result_tsv={result_tsv_name}")

    import torch
    effective_device = DEVICE
    if not torch.cuda.is_available():
        _warn("CUDA not available, falling back to CPU.")
        effective_device = "cpu"
    _log(f"DEVICE: {effective_device}")

    out_dir = Path(OUTPUT_DIR)
    _ensure_dir(out_dir)

    # ---- Phase 1: Load data ----
    _log("=== Phase 1: Loading data ===")
    all_chunks = _load_full_dev_dataset(Path(DEV_JSONL))
    if MAX_CHUNKS > 0:
        all_chunks = all_chunks[:MAX_CHUNKS]
        _log(f"Applied MAX_CHUNKS={MAX_CHUNKS}, chunks={len(all_chunks)}")

    term_to_tts_paths = _load_tts_paths(Path(DEV_JSONL_WITH_TTS))

    unique_terms = sorted({term for chunk in all_chunks for term in chunk.gt_terms})
    glossary_size = len(unique_terms)
    _log(f"Unique terms: {glossary_size}")

    glossary_path = out_dir / GLOSSARY_JSON_NAME
    if not glossary_path.exists():
        _build_glossary_json(unique_terms, glossary_path)

    # Text index uses the Qwen3-Omni text model checkpoint (ttsw=0.0)
    text_index_path = out_dir / f"index_v4_tr{TEXT_LORA_R}_{_safe_name(TEXT_MODEL_NAME)}.pkl"
    assert Path(TEXT_MODEL_PATH).exists(), f"Text model not found: {TEXT_MODEL_PATH}"
    assert Path(TTS_MODEL_PATH).exists(), f"TTS model not found: {TTS_MODEL_PATH}"

    _maybe_build_index(glossary_path, text_index_path, TEXT_MODEL_PATH, effective_device)
    term_to_idx, idx_to_term = _load_index_data(text_index_path)

    # ---- Phase 2: Text retrieval — Qwen3-Omni (load, run, free) ----
    text_results = _run_text_model_retrieval(
        chunks=all_chunks,
        term_to_idx=term_to_idx,
        index_path=text_index_path,
        model_path=TEXT_MODEL_PATH,
        effective_device=effective_device,
    )

    # ---- Phase 3: TTS retrieval — XEUS (load, run, free) ----
    tts_results, tts_bank_terms, tts_bank_prototypes = _run_tts_model_retrieval(
        chunks=all_chunks,
        term_to_idx=term_to_idx,
        term_to_tts_paths_raw=term_to_tts_paths,
        effective_device=effective_device,
    )

    # ---- Phase 4: Analysis ----
    _log("=== Phase 4: Analysis ===")
    no_term_m, with_term_m, overall_m = _compute_metrics(all_chunks, text_results, tts_results, term_to_idx)

    samples_text = _print_qualitative_samples(
        all_chunks, text_results, tts_results, idx_to_term, term_to_idx, out_dir / samples_txt_name,
    )
    print(samples_text, flush=True)

    _print_metrics_table(no_term_m, with_term_m, overall_m, tts_bank_terms, tts_bank_prototypes, glossary_size)
    _write_tsv(out_dir / result_tsv_name, no_term_m, with_term_m, overall_m)

    _log(f"Done. Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit("Missing dependency: torch.")
    raise SystemExit(main())
