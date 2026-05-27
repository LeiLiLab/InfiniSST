#!/usr/bin/env python3
"""
Sweep TOP_K values for dual-encoder evaluation (Qwen3-Omni text + XEUS TTS).

Encodes embeddings ONCE, then sweeps K in SWEEP_K_VALUES to find the best K
for intersection recall/precision/F1 and noise reduction.

Output: a comparison table + TSV with metrics for all K values.
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
TEXT_AUDIO_BASE_MODEL_NAME = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_AUDIO_LORA_R = 32
TEXT_AUDIO_LORA_ALPHA = 64
TEXT_LORA_R = 16
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

# K values to sweep; override via OFFLINE_EVAL_SWEEP_K env var (comma-separated)
SWEEP_K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20]

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SAMPLES = 30720

EVAL_BATCH_SIZE = 32
TTS_EMB_BATCH_SIZE = 64
MAX_TTS_PROTOTYPES_PER_TERM = 0
MAX_CHUNKS = 0

OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/offline_eval_xeus_tts_qwen3_text_intersection"
GLOSSARY_JSON_NAME = "gigaspeech_dev_terms_glossary.json"
SWEEP_TSV_NAME = "sweep_topk_metrics.tsv"

VOTING_MIN_VOTES = 1
SCORE_THRESHOLD = 0.0

CSV_DELIMITER = "\t"
CSV_LINE_TERMINATOR = "\n"
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
class RawScores:
    """All FAISS indices/distances (text) and full per-term score array (TTS) for one chunk."""
    # Text: top MAX_K indices and distances from FAISS (sorted by score desc)
    text_indices: np.ndarray   # shape (MAX_K,), int
    text_scores: np.ndarray    # shape (MAX_K,), float32

    # TTS: dense score array over all bank terms (sorted by score desc)
    tts_sorted_term_indices: np.ndarray  # shape (bank_size,), int — sorted desc by score
    tts_sorted_scores: np.ndarray        # shape (bank_size,), float32 — sorted desc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_full_dev_dataset(dev_jsonl: Path) -> List[ChunkData]:
    _log(f"Loading DEV_JSONL: {dev_jsonl}")
    groups: Dict[str, ChunkData] = {}
    for obj in _read_jsonl(dev_jsonl):
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
    _log(f"unique_chunks={len(chunks)} (with_term={with_term}, no_term={len(chunks)-with_term})")
    assert chunks
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
    assert sr == EXPECTED_SAMPLE_RATE, f"Bad SR: {path} sr={sr}"
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
    assert term_to_idx
    return term_to_idx, idx_to_term


# ---------------------------------------------------------------------------
# Encode text embeddings — store top MAX_K from FAISS
# ---------------------------------------------------------------------------

def _encode_text_all(
    chunks: Sequence[ChunkData],
    index_path: Path,
    model_path: str,
    effective_device: str,
    max_k: int,
) -> Dict[str, RawScores]:
    """Run Qwen3-Omni text retrieval with max_k, store raw indices+scores."""
    _log(f"=== Text encoding (Qwen3-Omni) — fetching top {max_k} from FAISS ===")
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
        top_k=max_k,
        voting_k=max_k,
        voting_min_votes=VOTING_MIN_VOTES,
        target_lang=TARGET_LANG_CODE,
        score_threshold=SCORE_THRESHOLD,
        chunk_size=1.92,
        hop_size=1.92,
        aggregation_strategy="max_pool",
        sample_rate=EXPECTED_SAMPLE_RATE,
        debug_audio_dir=None,
        verbose=False,
    )

    raw_scores: Dict[str, RawScores] = {}
    for start in range(0, len(chunks), EVAL_BATCH_SIZE):
        batch = chunks[start: start + EVAL_BATCH_SIZE]
        audios = [_load_audio_mono_16k(c.audio_path) for c in batch]
        inputs = retriever.feature_extractor(
            audios, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False
        )
        features = inputs.input_features
        bsz, channels, mel_len = features.shape
        inp = features.transpose(0, 1).reshape(channels, -1).to(retriever.device).to(feature_dtype)
        lens = torch.full((bsz,), mel_len, dtype=torch.long, device=retriever.device)

        with torch.no_grad():
            if use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    audio_embs = retriever.model(inp, lens)
            else:
                audio_embs = retriever.model(inp, lens)
            audio_embs = audio_embs.detach().cpu().float().numpy()

        faiss.normalize_L2(audio_embs)
        dists, indices = retriever.index.search(audio_embs, max_k)

        for i, chunk in enumerate(batch):
            cid = chunk.key.as_id()
            # Store as-is; FAISS returns sorted by score desc already
            raw_scores[cid] = RawScores(
                text_indices=indices[i].copy(),
                text_scores=dists[i].copy(),
                tts_sorted_term_indices=np.empty(0, dtype=np.int64),
                tts_sorted_scores=np.empty(0, dtype=np.float32),
            )

    _log(f"Text encoding done: {len(raw_scores)} chunks.")
    del retriever
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return raw_scores


# ---------------------------------------------------------------------------
# Encode TTS embeddings — store full sorted score array per chunk
# ---------------------------------------------------------------------------

def _load_xeus_audio_encoder(device_str: str):
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
    device = torch.device(device_str)
    ckpt = torch.load(TTS_MODEL_PATH, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    _log("XeusRetriever loaded (strict=True).")
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model, device


def _encode_audio_batch_xeus(model, audio_arrays: Sequence[np.ndarray], device) -> np.ndarray:
    import torch
    import faiss
    wavs = torch.from_numpy(np.stack(audio_arrays, axis=0)).float().to(device)
    wav_lens = torch.tensor([a.shape[0] for a in audio_arrays], dtype=torch.long, device=device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = model(wavs, wav_lens)
        embs = embs.detach().cpu().float().numpy()
    faiss.normalize_L2(embs)
    return embs.astype(np.float32, copy=False)


def _encode_tts_all(
    chunks: Sequence[ChunkData],
    term_to_idx: Dict[str, int],
    term_to_tts_paths_raw: Dict[str, List[str]],
    raw_scores: Dict[str, RawScores],
    effective_device: str,
) -> Tuple[int, int]:
    """Fill tts_sorted_term_indices/scores into raw_scores in-place. Returns (bank_terms, bank_protos)."""
    _log(f"=== TTS encoding (XEUS) — building proto bank + encoding chunks ===")
    import torch

    xeus_model, device = _load_xeus_audio_encoder(effective_device)

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

    sorted_term_list = np.array(sorted(tts_valid_term_set), dtype=np.int64)
    term_pos_map = {int(ti): pos for pos, ti in enumerate(sorted_term_list)}

    _log(f"Encoding {len(chunks)} speech chunks ...")
    for start in range(0, len(chunks), TTS_EMB_BATCH_SIZE):
        batch = chunks[start: start + TTS_EMB_BATCH_SIZE]
        audios = [_load_audio_mono_16k(c.audio_path) for c in batch]
        speech_embs = _encode_audio_batch_xeus(xeus_model, audios, device)

        for i, chunk in enumerate(batch):
            cid = chunk.key.as_id()
            scores_raw = proto_embs @ speech_embs[i]

            term_scores = np.full(tts_bank_terms, -np.inf, dtype=np.float32)
            for pi in range(scores_raw.shape[0]):
                ti = int(proto_term_idx_np[pi])
                pos = term_pos_map.get(ti)
                if pos is not None and scores_raw[pi] > term_scores[pos]:
                    term_scores[pos] = float(scores_raw[pi])

            # Sort all bank terms desc by score — store full sorted list
            valid_mask = np.isfinite(term_scores)
            assert np.any(valid_mask), f"No finite TTS scores for chunk {cid}"
            valid_pos = np.where(valid_mask)[0]
            valid_sc = term_scores[valid_pos]
            sort_order = np.argsort(-valid_sc)

            rs = raw_scores[cid]
            raw_scores[cid] = RawScores(
                text_indices=rs.text_indices,
                text_scores=rs.text_scores,
                tts_sorted_term_indices=sorted_term_list[valid_pos[sort_order]],
                tts_sorted_scores=valid_sc[sort_order],
            )

    _log("TTS encoding done.")
    del xeus_model
    gc.collect()
    torch.cuda.empty_cache()
    return tts_bank_terms, tts_bank_prototypes


# ---------------------------------------------------------------------------
# Compute metrics at a given K (pure numpy, no model needed)
# ---------------------------------------------------------------------------

@dataclass
class SweepRow:
    k: int
    # with-term
    wt_chunks: int = 0
    total_gt: int = 0
    text_recall: float = 0.0
    text_precision: float = 0.0
    text_f1: float = 0.0
    tts_recall: float = 0.0
    tts_precision: float = 0.0
    tts_f1: float = 0.0
    inter_recall: float = 0.0
    inter_precision: float = 0.0
    inter_f1: float = 0.0
    # no-term noise
    nt_chunks: int = 0
    avg_text_preds: float = 0.0
    avg_tts_preds: float = 0.0
    avg_inter_preds: float = 0.0
    noise_reduction_vs_text: float = 0.0


def _metrics_at_k(
    chunks: Sequence[ChunkData],
    raw_scores: Dict[str, RawScores],
    term_to_idx: Dict[str, int],
    k: int,
) -> SweepRow:
    wt = SweepRow(k=k)
    wt_chunks = 0
    total_gt = 0
    text_tp = text_pred = tts_tp = tts_pred = inter_tp = inter_pred = 0
    nt_chunks = 0
    nt_text_preds = nt_tts_preds = nt_inter_preds = 0

    for chunk in chunks:
        cid = chunk.key.as_id()
        rs = raw_scores[cid]

        # Text top-k: FAISS already sorted desc, just slice k valid entries
        text_set: Set[int] = set()
        for idx in rs.text_indices[:k]:
            if int(idx) >= 0:
                text_set.add(int(idx))

        # TTS top-k: already sorted desc, take first k
        tts_set: Set[int] = set(int(x) for x in rs.tts_sorted_term_indices[:k])

        inter_set = text_set & tts_set

        gt_indices: Set[int] = set()
        for term in chunk.gt_terms:
            idx = term_to_idx.get(term)
            if idx is not None:
                gt_indices.add(idx)

        if chunk.has_term:
            wt_chunks += 1
            total_gt += len(gt_indices)
            text_tp += len(text_set & gt_indices)
            text_pred += len(text_set)
            tts_tp += len(tts_set & gt_indices)
            tts_pred += len(tts_set)
            inter_tp += len(inter_set & gt_indices)
            inter_pred += len(inter_set)
        else:
            nt_chunks += 1
            nt_text_preds += len(text_set)
            nt_tts_preds += len(tts_set)
            nt_inter_preds += len(inter_set)

    def _rec(tp: int, gt: int) -> float:
        return tp / gt if gt > 0 else 0.0

    def _prec(tp: int, pred: int) -> float:
        return tp / pred if pred > 0 else 0.0

    def _f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    tr = _rec(text_tp, total_gt)
    tp_ = _prec(text_tp, text_pred)
    rr = _rec(tts_tp, total_gt)
    rp = _prec(tts_tp, tts_pred)
    ir = _rec(inter_tp, total_gt)
    ip = _prec(inter_tp, inter_pred)

    return SweepRow(
        k=k,
        wt_chunks=wt_chunks,
        total_gt=total_gt,
        text_recall=tr, text_precision=tp_, text_f1=_f1(tp_, tr),
        tts_recall=rr, tts_precision=rp, tts_f1=_f1(rp, rr),
        inter_recall=ir, inter_precision=ip, inter_f1=_f1(ip, ir),
        nt_chunks=nt_chunks,
        avg_text_preds=nt_text_preds / nt_chunks if nt_chunks > 0 else 0.0,
        avg_tts_preds=nt_tts_preds / nt_chunks if nt_chunks > 0 else 0.0,
        avg_inter_preds=nt_inter_preds / nt_chunks if nt_chunks > 0 else 0.0,
        noise_reduction_vs_text=1.0 - nt_inter_preds / nt_text_preds if nt_text_preds > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_sweep_table(rows: List[SweepRow]) -> None:
    print(flush=True)
    print("=" * 116, flush=True)
    print("TOP-K SWEEP: Qwen3-Omni (text) + XEUS (TTS) intersection", flush=True)
    print("  avg_inter_noise = avg #false-positive terms per no-term chunk after intersection (lower=better)", flush=True)
    print("=" * 116, flush=True)

    hdr = (
        f"  {'K':>3s}  "
        f"{'text_R':>7s} {'text_P':>7s} {'text_F1':>8s}  "
        f"{'tts_R':>7s} {'tts_P':>7s} {'tts_F1':>8s}  "
        f"{'inter_R':>8s} {'inter_P':>8s} {'inter_F1':>9s}  "
        f"{'noise_red':>9s} {'avg_inter_noise':>15s}"
    )
    print(hdr, flush=True)
    print("  " + "-" * 112, flush=True)

    best_f1_k = max(rows, key=lambda r: r.inter_f1)
    best_rec_k = max(rows, key=lambda r: r.inter_recall)
    best_noise_k = max(rows, key=lambda r: r.noise_reduction_vs_text)

    for r in rows:
        flags = []
        if r.k == best_f1_k.k:
            flags.append("best_inter_F1")
        if r.k == best_rec_k.k and r.k != best_f1_k.k:
            flags.append("best_inter_recall")
        if r.k == best_noise_k.k and r.k != best_f1_k.k:
            flags.append("best_noise_red")
        flag_str = f"  << {', '.join(flags)}" if flags else ""

        print(
            f"  {r.k:>3d}  "
            f"{r.text_recall:>7.4f} {r.text_precision:>7.4f} {r.text_f1:>8.4f}  "
            f"{r.tts_recall:>7.4f} {r.tts_precision:>7.4f} {r.tts_f1:>8.4f}  "
            f"{r.inter_recall:>8.4f} {r.inter_precision:>8.4f} {r.inter_f1:>9.4f}  "
            f"{r.noise_reduction_vs_text:>9.4f} {r.avg_inter_preds:>15.2f}"
            f"{flag_str}",
            flush=True,
        )

    print("=" * 116, flush=True)
    print(f"\n  Best inter_F1:     K={best_f1_k.k}  inter_F1={best_f1_k.inter_f1:.4f}  "
          f"recall={best_f1_k.inter_recall:.4f}  precision={best_f1_k.inter_precision:.4f}", flush=True)
    print(f"  Best inter_recall: K={best_rec_k.k}  inter_recall={best_rec_k.inter_recall:.4f}", flush=True)
    print(f"  Best noise_red:    K={best_noise_k.k}  noise_reduction={best_noise_k.noise_reduction_vs_text:.4f}", flush=True)


def _write_sweep_tsv(path: Path, rows: List[SweepRow]) -> None:
    fieldnames = [
        "k", "wt_chunks", "total_gt",
        "text_recall", "text_precision", "text_f1",
        "tts_recall", "tts_precision", "tts_f1",
        "inter_recall", "inter_precision", "inter_f1",
        "nt_chunks", "avg_text_preds", "avg_tts_preds", "avg_inter_preds",
        "noise_reduction_vs_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=CSV_DELIMITER, lineterminator=CSV_LINE_TERMINATOR)
        w.writeheader()
        for r in rows:
            w.writerow({
                "k": r.k, "wt_chunks": r.wt_chunks, "total_gt": r.total_gt,
                "text_recall": f"{r.text_recall:.6f}", "text_precision": f"{r.text_precision:.6f}", "text_f1": f"{r.text_f1:.6f}",
                "tts_recall": f"{r.tts_recall:.6f}", "tts_precision": f"{r.tts_precision:.6f}", "tts_f1": f"{r.tts_f1:.6f}",
                "inter_recall": f"{r.inter_recall:.6f}", "inter_precision": f"{r.inter_precision:.6f}", "inter_f1": f"{r.inter_f1:.6f}",
                "nt_chunks": r.nt_chunks,
                "avg_text_preds": f"{r.avg_text_preds:.4f}", "avg_tts_preds": f"{r.avg_tts_preds:.4f}",
                "avg_inter_preds": f"{r.avg_inter_preds:.4f}",
                "noise_reduction_vs_text": f"{r.noise_reduction_vs_text:.6f}",
            })
    _log(f"Wrote sweep TSV: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global DEVICE, OUTPUT_DIR, TTS_ROOT_DIR, SWEEP_K_VALUES

    env_device = os.environ.get("OFFLINE_EVAL_DEVICE", "").strip()
    if env_device:
        DEVICE = env_device
    env_out = os.environ.get("OFFLINE_EVAL_OUTPUT_DIR", "").strip()
    if env_out:
        OUTPUT_DIR = env_out
    env_tts_root = os.environ.get("OFFLINE_EVAL_TTS_ROOT_DIR", "").strip()
    if env_tts_root:
        TTS_ROOT_DIR = env_tts_root
    env_sweep = os.environ.get("OFFLINE_EVAL_SWEEP_K", "").strip()
    if env_sweep:
        SWEEP_K_VALUES = [int(x.strip()) for x in env_sweep.split(",") if x.strip()]
        assert all(k > 0 for k in SWEEP_K_VALUES), f"All K values must be > 0: {SWEEP_K_VALUES}"

    SWEEP_K_VALUES = sorted(set(SWEEP_K_VALUES))
    max_k = max(SWEEP_K_VALUES)
    _log(f"Sweep K values: {SWEEP_K_VALUES}  (max_k={max_k})")

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
        _log(f"MAX_CHUNKS={MAX_CHUNKS}, chunks={len(all_chunks)}")

    term_to_tts_paths = _load_tts_paths(Path(DEV_JSONL_WITH_TTS))

    unique_terms = sorted({t for c in all_chunks for t in c.gt_terms})
    glossary_size = len(unique_terms)
    _log(f"Unique terms: {glossary_size}")

    glossary_path = out_dir / GLOSSARY_JSON_NAME
    if not glossary_path.exists():
        _build_glossary_json(unique_terms, glossary_path)

    text_index_path = out_dir / f"index_v4_tr{TEXT_LORA_R}_{_safe_name(TEXT_MODEL_NAME)}.pkl"
    assert Path(TEXT_MODEL_PATH).exists(), f"Text model not found: {TEXT_MODEL_PATH}"
    assert Path(TTS_MODEL_PATH).exists(), f"TTS model not found: {TTS_MODEL_PATH}"

    _maybe_build_index(glossary_path, text_index_path, TEXT_MODEL_PATH, effective_device)
    term_to_idx, idx_to_term = _load_index_data(text_index_path)

    # ---- Phase 2: Encode text — once with max_k ----
    raw_scores = _encode_text_all(
        all_chunks, text_index_path, TEXT_MODEL_PATH, effective_device, max_k=max_k,
    )

    # ---- Phase 3: Encode TTS — once, store full sorted arrays ----
    tts_bank_terms, tts_bank_prototypes = _encode_tts_all(
        all_chunks, term_to_idx, term_to_tts_paths, raw_scores, effective_device,
    )
    _log(f"TTS bank: {tts_bank_terms} terms, {tts_bank_prototypes} prototypes")

    # ---- Phase 4: Sweep K — pure numpy, no GPU needed ----
    _log(f"=== Phase 4: Sweeping K = {SWEEP_K_VALUES} ===")
    sweep_rows: List[SweepRow] = []
    for k in SWEEP_K_VALUES:
        row = _metrics_at_k(all_chunks, raw_scores, term_to_idx, k)
        sweep_rows.append(row)
        _log(f"  K={k:>2d}  inter_F1={row.inter_f1:.4f}  inter_R={row.inter_recall:.4f}  inter_P={row.inter_precision:.4f}  noise_red={row.noise_reduction_vs_text:.4f}")

    _print_sweep_table(sweep_rows)
    _write_sweep_tsv(out_dir / SWEEP_TSV_NAME, sweep_rows)

    _log(f"Done. Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit("Missing dependency: torch.")
    raise SystemExit(main())
