#!/usr/bin/env python3

"""
Offline evaluation: score sweep for F1/F2/F3 on GigaSpeech dev term dataset.

Protocol:
  - Filter out rows with empty term.
  - Group by (utter_id, chunk_idx) to build multi-positive ground-truth terms per audio chunk.
  - Build a glossary JSON from the unique terms in the dataset.
  - Build a FAISS index for the glossary using the tuned V4 text encoder (build_index_v4.py format).
  - Encode each audio chunk with the V4 audio retriever, retrieve Top-K1 candidates,
    and compute micro-averaged precision/recall and F1/F2/F3 for a sweep:
      * absolute threshold: score >= threshold
      * relative margin: score >= (chunk_max_score - margin)

All log messages are in English.
"""

from __future__ import annotations

# ======Configuration=====
DEV_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_final.jsonl"

# Model checkpoint that contains both audio retriever weights and (optionally) tuned text encoder weights.
MODEL_PATH = "/mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_epoch_16.pt"

# Index build settings (match build_index_v4.py expectations).
TEXT_LORA_R = 16
TARGET_LANG_CODE = "zh"

# Audio encoder settings (match StreamingQwen3RAGRetrieverV4 defaults).
DEVICE = "cuda:0"
AUDIO_BASE_MODEL_NAME = "Atotti/Qwen3-Omni-AudioTransformer"
AUDIO_LORA_R = 32
AUDIO_LORA_ALPHA = 64
TTS_ROOT_DIR = "/mnt/gemini/data/siqiouyang/term_train_tts"

# Audio chunk assumptions for this dataset.
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SECONDS = 1.92
EXPECTED_CHUNK_SAMPLES = 30720  # 1.92s * 16kHz

# Fixed Top-K1 for threshold ablation.
K1 = 10
EVAL_MODE = "intersection"  # one of: text, tts, intersection

# Sweep settings.
EVAL_SWEEP_TYPE = "absolute"  # one of: absolute, relative_margin

# Absolute threshold sweep settings.
ABS_THRESHOLD_MIN = 0.0
ABS_THRESHOLD_MAX = 1.0
ABS_THRESHOLD_STEPS = 51
ABS_THRESHOLD_USE_DATA_RANGE = True

# Relative margin sweep settings.
RELATIVE_MARGIN_MIN = 0.0
RELATIVE_MARGIN_MAX = 0.5
RELATIVE_MARGIN_STEPS = 51

# Evaluation limits (0 means no limit).
MAX_CHUNKS = 0

# Output paths
OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/offline_eval_threshold_ablation_k1_10_gigaspeech_dev"
GLOSSARY_JSON_NAME = "gigaspeech_dev_terms_glossary.json"
INDEX_PKL_NAME = f"gigaspeech_dev_terms_index_v4_tr{TEXT_LORA_R}.pkl"
RESULT_TSV_NAME = "threshold_f1_f2_f3_k1_10.tsv"
PLOT_PNG_NAME = "threshold_f1_f2_f3_k1_10.png"

# Execution
INDEX_BUILD_BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 32
TTS_EMB_BATCH_SIZE = 256

# Plot settings
PLOT_DPI = 180
PLOT_FIGSIZE_W = 8
PLOT_FIGSIZE_H = 4.8

# Misc
CSV_DELIMITER = "\t"
CSV_LINE_TERMINATOR = "\n"
FLOAT_DECIMALS = 6
# ======Configuration=====

import csv
import json
import os
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# Ensure repository root is importable (so `import agents` works in Slurm without PYTHONPATH).
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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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
    gt_terms: Set[str]  # canonical lower-cased terms


@dataclass
class ChunkPrediction:
    chunk_id: str
    audio_path: str
    pos_indices: Set[int]
    candidates: List[Tuple[int, float]]  # (term_index, score)


def _build_tts_audio_path(utter_id: str, chunk_idx: str, tts_root_dir: str) -> Optional[str]:
    if not utter_id:
        return None
    parts = utter_id.rsplit("_", 1)
    if len(parts) != 2:
        return None
    utt_prefix, segment_id = parts
    return str(Path(tts_root_dir) / utt_prefix / segment_id / f"chunk_{chunk_idx}.wav")


def _map_term_to_tts_paths(
    chunk_pos_indices: Sequence[Tuple[str, str, Set[int], str, str]],
    tts_root_dir: str,
) -> Dict[int, str]:
    term_to_tts_path: Dict[int, str] = {}
    for _, _, pos, utter_id, chunk_idx in chunk_pos_indices:
        tts_path = _build_tts_audio_path(utter_id, chunk_idx, tts_root_dir)
        if not tts_path or not Path(tts_path).exists():
            continue
        for term_idx in pos:
            if term_idx not in term_to_tts_path:
                term_to_tts_path[term_idx] = tts_path
    return term_to_tts_path


def _load_and_group_dev_jsonl(dev_jsonl: Path) -> List[ChunkExample]:
    _log(f"Loading DEV_JSONL: {dev_jsonl}")

    groups: Dict[str, ChunkExample] = {}
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

    examples = list(groups.values())
    examples.sort(key=lambda x: (x.key.utter_id, int(x.key.chunk_idx) if x.key.chunk_idx.isdigit() else x.key.chunk_idx))

    _log(f"Loaded rows={total_rows}, kept_rows_with_term={kept_rows}, unique_chunks={len(examples)}")
    if not examples:
        _err("No valid chunks after filtering. Check DEV_JSONL path and 'term' field.")
    return examples


def _build_glossary_json(unique_terms: Sequence[str], glossary_path: Path) -> None:
    # build_index_v4.py accepts a dict of {term: payload}. payload can be str or dict.
    glossary: Dict[str, str] = {t: "" for t in unique_terms}
    with glossary_path.open("w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)


def _maybe_build_index(glossary_path: Path, index_path: Path) -> None:
    if index_path.exists():
        _log(f"Using existing index: {index_path}")
        return

    # Build index with the repository script (same output format).
    script_path = _REPO_ROOT / "retriever" / "gigaspeech" / "build_index_v4.py"
    if not script_path.exists():
        _err(f"build_index_v4.py not found: {script_path}")

    _log(f"Building index: {index_path}")
    cmd = [
        sys.executable,
        str(script_path),
        "--glossary_path",
        str(glossary_path),
        "--model_path",
        str(MODEL_PATH),
        "--output_path",
        str(index_path),
        "--text_lora_r",
        str(TEXT_LORA_R),
        "--device",
        str(DEVICE),
        "--batch_size",
        str(INDEX_BUILD_BATCH_SIZE),
        "--target_lang_code",
        str(TARGET_LANG_CODE),
    ]

    # Use subprocess to keep this script self-contained.
    import subprocess

    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        _err(f"Index build failed (rc={proc.returncode}). stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    _log("Index build finished.")


def _load_index_term_map(index_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    import pickle
    import faiss  # type: ignore

    with index_path.open("rb") as f:
        data = pickle.load(f)

    term_list = data["term_list"]
    # Ensure the FAISS index can be deserialized (sanity check).
    _ = faiss.deserialize_index(data["faiss_index"])

    term_to_idx: Dict[str, int] = {}
    idx_to_term: Dict[int, str] = {}
    for i, item in enumerate(term_list):
        key = str(item.get("key", "")).strip().lower()
        if key:
            term_to_idx[key] = i
            idx_to_term[i] = key
    if not term_to_idx:
        _err(f"Empty term_list in index: {index_path}")
    return term_to_idx, idx_to_term


def _load_audio_mono_16k(path: str) -> np.ndarray:
    import soundfile as sf  # type: ignore

    audio, sr = sf.read(path)
    if sr != EXPECTED_SAMPLE_RATE:
        _warn(f"Unexpected sample rate: path={path} sr={sr} expected={EXPECTED_SAMPLE_RATE}")

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).flatten()

    max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_val > 0:
        audio = audio / max_val

    if audio.shape[0] < EXPECTED_CHUNK_SAMPLES:
        audio = np.pad(audio, (0, EXPECTED_CHUNK_SAMPLES - audio.shape[0]), mode="constant")
    elif audio.shape[0] > EXPECTED_CHUNK_SAMPLES:
        audio = audio[:EXPECTED_CHUNK_SAMPLES]
    return audio


def _format_float(x: float) -> str:
    return f"{x:.{FLOAT_DECIMALS}f}"


def _compute_f_beta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return (1.0 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def _cosine_to_l2_score(cosine_sim: float) -> float:
    # For L2-normalized vectors: L2_dist^2 = 2 - 2 * cosine_similarity
    l2_dist = math.sqrt(max(0.0, 2.0 - 2.0 * float(cosine_sim)))
    return 1.0 / (1.0 + l2_dist)


def _build_sweep_values(all_scores: List[float]) -> List[float]:
    if EVAL_SWEEP_TYPE == "relative_margin":
        min_score = RELATIVE_MARGIN_MIN
        max_score = RELATIVE_MARGIN_MAX
        num_steps = RELATIVE_MARGIN_STEPS
    else:
        min_score = ABS_THRESHOLD_MIN
        max_score = ABS_THRESHOLD_MAX
        num_steps = ABS_THRESHOLD_STEPS
        if ABS_THRESHOLD_USE_DATA_RANGE and all_scores:
            min_score = float(min(all_scores))
            max_score = float(max(all_scores))

    if max_score < min_score:
        min_score, max_score = max_score, min_score
    if num_steps <= 1:
        return [min_score]
    values = np.linspace(min_score, max_score, num_steps, dtype=np.float64).tolist()
    return [float(x) for x in values]


def _pred_set_by_absolute_threshold(score_map: Dict[int, float], threshold: float) -> Set[int]:
    return {idx for idx, score in score_map.items() if float(score) >= threshold}


def _pred_set_by_relative_margin(score_map: Dict[int, float], margin: float) -> Set[int]:
    if not score_map:
        return set()
    max_score = max(float(score) for score in score_map.values())
    dynamic_threshold = max_score - margin
    return {idx for idx, score in score_map.items() if float(score) >= dynamic_threshold}


def _topk_candidates_from_scores(
    term_indices: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> List[Tuple[int, float]]:
    if scores.size == 0:
        return []
    k = min(top_k, scores.shape[0])
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(term_indices[i]), float(scores[i])) for i in top_idx]


def main() -> int:
    # Allow lightweight overrides via environment variables (useful for Slurm jobs).
    # Examples:
    #   OFFLINE_EVAL_DEVICE=cuda:0 OFFLINE_EVAL_OUTPUT_DIR=... bash run_*.sh
    global DEVICE, OUTPUT_DIR, MODEL_PATH, ABS_THRESHOLD_MIN, ABS_THRESHOLD_MAX, ABS_THRESHOLD_STEPS, ABS_THRESHOLD_USE_DATA_RANGE, EVAL_MODE, TTS_ROOT_DIR, EVAL_SWEEP_TYPE, RELATIVE_MARGIN_MIN, RELATIVE_MARGIN_MAX, RELATIVE_MARGIN_STEPS
    env_device = os.environ.get("OFFLINE_EVAL_DEVICE", "").strip()
    if env_device:
        DEVICE = env_device
    env_out = os.environ.get("OFFLINE_EVAL_OUTPUT_DIR", "").strip()
    if env_out:
        OUTPUT_DIR = env_out
    env_model = os.environ.get("OFFLINE_EVAL_MODEL_PATH", "").strip()
    if env_model:
        MODEL_PATH = env_model
    env_tts_root = os.environ.get("OFFLINE_EVAL_TTS_ROOT_DIR", "").strip()
    if env_tts_root:
        TTS_ROOT_DIR = env_tts_root
    env_sweep_type = os.environ.get("OFFLINE_EVAL_SWEEP_TYPE", "").strip().lower()
    if env_sweep_type:
        EVAL_SWEEP_TYPE = env_sweep_type
    if EVAL_SWEEP_TYPE not in {"absolute", "relative_margin"}:
        _err(
            f"Invalid EVAL_SWEEP_TYPE={EVAL_SWEEP_TYPE!r}, "
            "expected one of: absolute, relative_margin."
        )

    env_thr_min = os.environ.get("OFFLINE_EVAL_THRESHOLD_MIN", "").strip()
    if env_thr_min:
        ABS_THRESHOLD_MIN = float(env_thr_min)
    env_thr_max = os.environ.get("OFFLINE_EVAL_THRESHOLD_MAX", "").strip()
    if env_thr_max:
        ABS_THRESHOLD_MAX = float(env_thr_max)
    env_thr_steps = os.environ.get("OFFLINE_EVAL_THRESHOLD_STEPS", "").strip()
    if env_thr_steps:
        ABS_THRESHOLD_STEPS = int(env_thr_steps)
    env_thr_data = os.environ.get("OFFLINE_EVAL_THRESHOLD_USE_DATA_RANGE", "").strip().lower()
    if env_thr_data in {"0", "false", "no"}:
        ABS_THRESHOLD_USE_DATA_RANGE = False
    elif env_thr_data in {"1", "true", "yes"}:
        ABS_THRESHOLD_USE_DATA_RANGE = True

    env_margin_min = os.environ.get("OFFLINE_EVAL_MARGIN_MIN", "").strip()
    if env_margin_min:
        RELATIVE_MARGIN_MIN = float(env_margin_min)
    env_margin_max = os.environ.get("OFFLINE_EVAL_MARGIN_MAX", "").strip()
    if env_margin_max:
        RELATIVE_MARGIN_MAX = float(env_margin_max)
    env_margin_steps = os.environ.get("OFFLINE_EVAL_MARGIN_STEPS", "").strip()
    if env_margin_steps:
        RELATIVE_MARGIN_STEPS = int(env_margin_steps)
    env_eval_mode = os.environ.get("OFFLINE_EVAL_MODE", "").strip().lower()
    if env_eval_mode:
        EVAL_MODE = env_eval_mode
    if EVAL_MODE not in {"text", "tts", "intersection"}:
        _err(f"Invalid EVAL_MODE={EVAL_MODE!r}, expected one of: text, tts, intersection.")

    out_dir = Path(OUTPUT_DIR)
    _ensure_dir(out_dir)

    dev_jsonl = Path(DEV_JSONL)
    if not dev_jsonl.exists():
        _err(f"DEV_JSONL not found: {dev_jsonl}")
    if not Path(MODEL_PATH).exists():
        _err(f"MODEL_PATH not found: {MODEL_PATH}")

    examples = _load_and_group_dev_jsonl(dev_jsonl)
    if MAX_CHUNKS > 0:
        examples = examples[:MAX_CHUNKS]
        _log(f"Applied MAX_CHUNKS={MAX_CHUNKS}, evaluating chunks={len(examples)}")

    # Build glossary from unique terms (from the filtered dataset).
    unique_terms = sorted({t for ex in examples for t in ex.gt_terms})
    _log(f"Unique terms from dataset: {len(unique_terms)}")

    glossary_path = out_dir / GLOSSARY_JSON_NAME
    index_path = out_dir / INDEX_PKL_NAME

    if not glossary_path.exists():
        _log(f"Writing glossary JSON: {glossary_path}")
        _build_glossary_json(unique_terms, glossary_path)
    else:
        _log(f"Using existing glossary JSON: {glossary_path}")

    _maybe_build_index(glossary_path, index_path)

    # Load index + term mapping
    term_to_idx, idx_to_term = _load_index_term_map(index_path)
    if K1 <= 0:
        _err("K1 must be positive.")

    # Prepare positives per chunk in index space.
    chunk_pos_indices: List[Tuple[str, str, Set[int], str, str]] = []
    total_valid_pos = 0
    missing_terms = 0
    for ex in examples:
        pos: Set[int] = set()
        for t in ex.gt_terms:
            idx = term_to_idx.get(t)
            if idx is not None:
                pos.add(idx)
            else:
                missing_terms += 1
        if not pos:
            continue
        total_valid_pos += len(pos)
        chunk_pos_indices.append((ex.key.as_id(), ex.audio_path, pos, ex.key.utter_id, ex.key.chunk_idx))

    _log(f"Chunks with >=1 mapped positive term: {len(chunk_pos_indices)}")
    _log(f"Total positives (strict denominator): {total_valid_pos}")
    if missing_terms:
        _warn(f"Missing terms not found in index (skipped in positives): {missing_terms}")
    if not chunk_pos_indices or total_valid_pos <= 0:
        _err("No positives to evaluate after mapping terms to index. Check glossary build/mapping.")

    # Load retriever + index through the existing agent class (consistent feature extraction + audio encoder).
    from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4

    _log(f"Initializing retriever (K1={K1}) on device={DEVICE}")
    retriever = StreamingQwen3RAGRetrieverV4(
        index_path=str(index_path),
        model_path=str(MODEL_PATH),
        base_model_name=AUDIO_BASE_MODEL_NAME,
        device=DEVICE,
        lora_r=AUDIO_LORA_R,
        lora_alpha=AUDIO_LORA_ALPHA,
        text_lora_r=TEXT_LORA_R,
        top_k=K1,
        voting_k=K1,
        voting_min_votes=1,
        target_lang=TARGET_LANG_CODE,
        score_threshold=0.0,
        chunk_size=EXPECTED_CHUNK_SECONDS,
        hop_size=EXPECTED_CHUNK_SECONDS,
        aggregation_strategy="max_pool",
        sample_rate=EXPECTED_SAMPLE_RATE,
        debug_audio_dir=None,
        verbose=False,
    )

    import faiss  # type: ignore
    import torch

    predictions: List[ChunkPrediction] = []
    speech_emb_by_chunk: Dict[str, np.ndarray] = {}
    text_score_pool: List[float] = []

    _log(f"Encoding audio and searching FAISS (batch_size={EVAL_BATCH_SIZE}) ...")
    for start in range(0, len(chunk_pos_indices), EVAL_BATCH_SIZE):
        batch = chunk_pos_indices[start : start + EVAL_BATCH_SIZE]
        audios = [_load_audio_mono_16k(item[1]) for item in batch]

        inputs = retriever.feature_extractor(audios, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False)
        features = inputs.input_features  # [B, C, T]
        B, C, T_mel = features.shape

        input_features = features.transpose(0, 1).reshape(C, -1).to(retriever.device).to(torch.bfloat16)
        feature_lens = torch.full((B,), T_mel, dtype=torch.long, device=retriever.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = retriever.model(input_features, feature_lens)
            audio_embs = audio_embs.detach().cpu().float().numpy()

        faiss.normalize_L2(audio_embs)
        D, I = retriever.index.search(audio_embs, K1)

        for i in range(B):
            chunk_id = batch[i][0]
            pos = batch[i][2]
            cand: List[Tuple[int, float]] = []
            for dist, idx in zip(D[i], I[i]):
                if idx < 0:
                    continue
                # FAISS index uses inner product on L2-normalized vectors, so dist is cosine similarity.
                score = float(dist)
                cand.append((int(idx), score))
                text_score_pool.append(score)
            predictions.append(ChunkPrediction(chunk_id, batch[i][1], pos, cand))
            speech_emb_by_chunk[chunk_id] = audio_embs[i].astype(np.float32, copy=False)

    tts_candidates_by_chunk: Dict[str, List[Tuple[int, float]]] = {}
    tts_valid_term_indices: Set[int] = set()
    tts_score_pool: List[float] = []
    if EVAL_MODE in {"tts", "intersection"}:
        _log("Preparing TTS-term embedding bank...")
        term_to_tts_path = _map_term_to_tts_paths(chunk_pos_indices, TTS_ROOT_DIR)
        if not term_to_tts_path:
            alt_root_dir = str(Path(TTS_ROOT_DIR).with_name("term_dev_tts"))
            if Path(alt_root_dir).exists() and alt_root_dir != TTS_ROOT_DIR:
                _warn(
                    f"No TTS paths found under TTS_ROOT_DIR={TTS_ROOT_DIR}. "
                    f"Retry with fallback root={alt_root_dir}."
                )
                term_to_tts_path = _map_term_to_tts_paths(chunk_pos_indices, alt_root_dir)
                if term_to_tts_path:
                    _log(f"Using fallback TTS root: {alt_root_dir}")
                    TTS_ROOT_DIR = alt_root_dir

        if not term_to_tts_path:
            _err("No valid TTS paths mapped for TTS/intersection mode.")

        sorted_term_indices = sorted(term_to_tts_path.keys())
        term_embeddings: Dict[int, np.ndarray] = {}
        for start in range(0, len(sorted_term_indices), TTS_EMB_BATCH_SIZE):
            idx_batch = sorted_term_indices[start : start + TTS_EMB_BATCH_SIZE]
            audios = [_load_audio_mono_16k(term_to_tts_path[i]) for i in idx_batch]

            inputs = retriever.feature_extractor(audios, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False)
            features = inputs.input_features  # [B, C, T]
            bsz, channels, mel_len = features.shape
            input_features = features.transpose(0, 1).reshape(channels, -1).to(retriever.device).to(torch.bfloat16)
            feature_lens = torch.full((bsz,), mel_len, dtype=torch.long, device=retriever.device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    emb_batch = retriever.model(input_features, feature_lens)
                emb_batch = emb_batch.detach().cpu().float().numpy()
            faiss.normalize_L2(emb_batch)
            for i, term_idx in enumerate(idx_batch):
                term_embeddings[term_idx] = emb_batch[i].astype(np.float32, copy=False)
                tts_valid_term_indices.add(term_idx)

        if not tts_valid_term_indices:
            _err("No valid TTS term embeddings encoded for TTS/intersection mode.")

        tts_term_indices_np = np.array(sorted(tts_valid_term_indices), dtype=np.int64)
        tts_term_embs_np = np.stack([term_embeddings[idx] for idx in tts_term_indices_np], axis=0).astype(np.float32, copy=False)
        for pred in predictions:
            speech_vec = speech_emb_by_chunk[pred.chunk_id]
            scores = tts_term_embs_np @ speech_vec
            cand = _topk_candidates_from_scores(tts_term_indices_np, scores, K1)
            tts_candidates_by_chunk[pred.chunk_id] = cand
            for _, score in cand:
                tts_score_pool.append(float(score))

    intersection_score_pool: List[float] = []
    if EVAL_MODE == "intersection":
        for pred in predictions:
            text_score_map = {idx: score for idx, score in pred.candidates}
            tts_score_map = {idx: score for idx, score in tts_candidates_by_chunk.get(pred.chunk_id, [])}
            overlap_ids = set(text_score_map.keys()) & set(tts_score_map.keys())
            for idx in overlap_ids:
                # Conservative fused score: candidate must be strong in both databases.
                intersection_score_pool.append(min(float(text_score_map[idx]), float(tts_score_map[idx])))

    if EVAL_MODE == "text":
        score_pool = text_score_pool
    elif EVAL_MODE == "tts":
        score_pool = tts_score_pool
    else:
        score_pool = intersection_score_pool

    sweep_values = _build_sweep_values(score_pool)
    if not sweep_values:
        _err("No sweep values generated. Check sweep settings and score collection.")

    if EVAL_MODE == "text":
        mode_total_valid_pos = total_valid_pos
        mode_num_chunks = len(predictions)
    else:
        mode_total_valid_pos = 0
        mode_num_chunks = 0
        for pred in predictions:
            pos_tts = pred.pos_indices & tts_valid_term_indices
            if pos_tts:
                mode_total_valid_pos += len(pos_tts)
                mode_num_chunks += 1
        if mode_total_valid_pos <= 0:
            _err("No valid positives for TTS/intersection mode after applying TTS availability filter.")

    rows: List[Dict[str, Any]] = []
    for sweep_value in sweep_values:
        tp = 0
        pred_total = 0
        for pred in predictions:
            text_score_map = {idx: score for idx, score in pred.candidates}
            tts_score_map = {idx: score for idx, score in tts_candidates_by_chunk.get(pred.chunk_id, [])}
            if EVAL_SWEEP_TYPE == "relative_margin":
                text_pred_set = _pred_set_by_relative_margin(text_score_map, sweep_value)
                tts_pred_set = _pred_set_by_relative_margin(tts_score_map, sweep_value)
            else:
                text_pred_set = _pred_set_by_absolute_threshold(text_score_map, sweep_value)
                tts_pred_set = _pred_set_by_absolute_threshold(tts_score_map, sweep_value)
            if EVAL_MODE == "text":
                pred_set = text_pred_set
                pos_set = pred.pos_indices
            elif EVAL_MODE == "tts":
                pred_set = tts_pred_set
                pos_set = pred.pos_indices & tts_valid_term_indices
            else:
                overlap_ids = set(text_score_map.keys()) & set(tts_score_map.keys())
                overlap_score_map = {
                    idx: min(float(text_score_map[idx]), float(tts_score_map[idx])) for idx in overlap_ids
                }
                if EVAL_SWEEP_TYPE == "relative_margin":
                    pred_set = _pred_set_by_relative_margin(overlap_score_map, sweep_value)
                else:
                    pred_set = _pred_set_by_absolute_threshold(overlap_score_map, sweep_value)
                pos_set = pred.pos_indices & tts_valid_term_indices

            if not pos_set:
                continue
            hits = len(pred_set & pos_set)
            tp += hits
            pred_total += len(pred_set)

        fp = pred_total - tp
        fn = mode_total_valid_pos - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = _compute_f_beta(precision, recall, 1.0)
        f2 = _compute_f_beta(precision, recall, 2.0)
        f3 = _compute_f_beta(precision, recall, 3.0)

        rows.append(
            {
                "sweep_type": EVAL_SWEEP_TYPE,
                "sweep_value": _format_float(sweep_value),
                "threshold_cosine": _format_float(sweep_value) if EVAL_SWEEP_TYPE == "absolute" else "",
                "threshold_l2_score": _format_float(_cosine_to_l2_score(sweep_value)) if EVAL_SWEEP_TYPE == "absolute" else "",
                "relative_margin": _format_float(sweep_value) if EVAL_SWEEP_TYPE == "relative_margin" else "",
                "precision": _format_float(precision),
                "recall": _format_float(recall),
                "f1": _format_float(f1),
                "f2": _format_float(f2),
                "f3": _format_float(f3),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mode": EVAL_MODE,
                "pred_total": pred_total,
                "total_pos": mode_total_valid_pos,
                "num_chunks": mode_num_chunks,
                "unique_terms": len(unique_terms),
            }
        )

    # Write results
    tsv_path = out_dir / RESULT_TSV_NAME
    _log(f"Writing TSV: {tsv_path}")
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sweep_type",
                "sweep_value",
                "threshold_cosine",
                "threshold_l2_score",
                "relative_margin",
                "precision",
                "recall",
                "f1",
                "f2",
                "f3",
                "tp",
                "fp",
                "fn",
                "mode",
                "pred_total",
                "total_pos",
                "num_chunks",
                "unique_terms",
            ],
            delimiter=CSV_DELIMITER,
            lineterminator=CSV_LINE_TERMINATOR,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Report best thresholds
    def _best_row(metric_key: str) -> Optional[Dict[str, Any]]:
        if not rows:
            return None
        return max(rows, key=lambda x: float(x[metric_key]))

    best_f1 = _best_row("f1")
    best_f2 = _best_row("f2")
    best_f3 = _best_row("f3")
    if best_f1:
        _log(
            f"Best F1 [{EVAL_MODE}, {EVAL_SWEEP_TYPE}]: "
            f"sweep_value={best_f1['sweep_value']} f1={best_f1['f1']}"
        )
    if best_f2:
        _log(
            f"Best F2 [{EVAL_MODE}, {EVAL_SWEEP_TYPE}]: "
            f"sweep_value={best_f2['sweep_value']} f2={best_f2['f2']}"
        )
    if best_f3:
        _log(
            f"Best F3 [{EVAL_MODE}, {EVAL_SWEEP_TYPE}]: "
            f"sweep_value={best_f3['sweep_value']} f3={best_f3['f3']}"
        )

    # Plot (optional)
    png_path = out_dir / PLOT_PNG_NAME
    try:
        import matplotlib.pyplot as plt  # type: ignore

        xs = [float(r["sweep_value"]) for r in rows]
        f1s = [float(r["f1"]) for r in rows]
        f2s = [float(r["f2"]) for r in rows]
        f3s = [float(r["f3"]) for r in rows]

        plt.figure(figsize=(PLOT_FIGSIZE_W, PLOT_FIGSIZE_H))
        plt.plot(xs, f1s, marker="o", linewidth=2, label="F1")
        plt.plot(xs, f2s, marker="o", linewidth=2, label="F2")
        plt.plot(xs, f3s, marker="o", linewidth=2, label="F3")
        plt.xlabel("Relative margin" if EVAL_SWEEP_TYPE == "relative_margin" else "Score threshold")
        plt.ylabel("F-score")
        plt.title(f"GigaSpeech dev: Sweep (K1={K1}, mode={EVAL_MODE}, type={EVAL_SWEEP_TYPE})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=PLOT_DPI)
        plt.close()
        _log(f"Wrote plot: {png_path}")
    except Exception as e:
        _warn(f"Plot skipped (matplotlib not available or failed): {e}")

    _log("Done.")
    _log(f"TSV: {tsv_path}")
    _log(f"Glossary: {glossary_path}")
    _log(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    # Fast-fail with a clear message when this script is executed outside the intended environment.
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit(
            "Missing dependency: torch. Run this script in an environment that has PyTorch installed "
            "(e.g., activate your conda env and re-run)."
        )
    raise SystemExit(main())

