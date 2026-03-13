#!/usr/bin/env python3

"""
Offline analysis on GigaSpeech dev:
Compare cosine similarities of Top-K terms vs non-Top-K terms for each speech chunk.

Outputs:
- summary JSON
- per-chunk TSV
- plot PNG
"""

from __future__ import annotations

# ======Configuration=====
DEV_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_final.jsonl"
MODEL_PATH = "/mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_v2_step_4000.pt"
OUTPUT_DIR = "/mnt/gemini/data/jiaxuanluo/offline_eval_tts_top10_vs_rest"

TEXT_LORA_R = 16
TARGET_LANG_CODE = "zh"

DEVICE = "cuda:0"
AUDIO_BASE_MODEL_NAME = "Atotti/Qwen3-Omni-AudioTransformer"
AUDIO_LORA_R = 32
AUDIO_LORA_ALPHA = 64

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SECONDS = 1.92
EXPECTED_CHUNK_SAMPLES = 30720

TOP_K = 10
MAX_CHUNKS = 0

INDEX_BUILD_BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 32

GLOSSARY_JSON_NAME = "gigaspeech_dev_terms_glossary.json"
INDEX_PKL_NAME = f"gigaspeech_dev_terms_index_v4_tr{TEXT_LORA_R}.pkl"
SUMMARY_JSON_NAME = "top10_vs_rest_summary.json"
PER_CHUNK_TSV_NAME = "top10_vs_rest_per_chunk.tsv"
PLOT_PNG_NAME = "top10_vs_rest_similarity.png"

PLOT_DPI = 180
PLOT_FIGSIZE_W = 10
PLOT_FIGSIZE_H = 5

FLOAT_DECIMALS = 6
# =======================

import csv
import json
import os
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


def _load_and_group_dev_jsonl(dev_jsonl: Path) -> List[ChunkExample]:
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
            continue

        key = ChunkKey(utter_id=utter_id, chunk_idx=chunk_idx)
        chunk_id = key.as_id()
        if chunk_id not in groups:
            groups[chunk_id] = ChunkExample(key=key, audio_path=audio_path, gt_terms=set())
        groups[chunk_id].gt_terms.add(term)

    examples = list(groups.values())
    examples.sort(key=lambda x: (x.key.utter_id, int(x.key.chunk_idx) if x.key.chunk_idx.isdigit() else x.key.chunk_idx))
    _log(f"Loaded rows={total_rows}, kept_rows_with_term={kept_rows}, unique_chunks={len(examples)}")
    if not examples:
        _err("No valid chunks found after filtering non-empty terms.")
    return examples


def _build_glossary_json(unique_terms: Sequence[str], glossary_path: Path) -> None:
    glossary = {term: "" for term in unique_terms}
    with glossary_path.open("w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)


def _maybe_build_index(glossary_path: Path, index_path: Path) -> None:
    if index_path.exists():
        _log(f"Using existing index: {index_path}")
        return

    script_path = _REPO_ROOT / "retriever" / "gigaspeech" / "build_index_v4.py"
    if not script_path.exists():
        _err(f"build_index_v4.py not found: {script_path}")

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

    import subprocess

    _log(f"Building index: {index_path}")
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        _err(f"Index build failed (rc={proc.returncode}). stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    _log("Index build finished.")


def _load_index(index_path: Path):
    import pickle
    import faiss  # type: ignore

    with index_path.open("rb") as f:
        data = pickle.load(f)
    index = faiss.deserialize_index(data["faiss_index"])
    term_list = data["term_list"]
    if index.ntotal <= 0:
        _err(f"Loaded index has no vectors: {index_path}")
    return index, term_list


def _extract_term_embeddings(index) -> np.ndarray:
    # IndexFlatIP supports reconstruct_n; embeddings should already be L2-normalized by build_index_v4.
    term_embeddings = index.reconstruct_n(0, index.ntotal).astype(np.float32, copy=False)
    if term_embeddings.ndim != 2:
        _err("Invalid term embedding matrix shape.")
    return term_embeddings


def _load_audio_mono_16k(path: str) -> np.ndarray:
    import soundfile as sf  # type: ignore

    audio, sr = sf.read(path)
    if sr != EXPECTED_SAMPLE_RATE:
        _log(f"[WARN] Unexpected sample rate path={path} sr={sr} expected={EXPECTED_SAMPLE_RATE}")

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


def main() -> int:
    global DEVICE, OUTPUT_DIR, MODEL_PATH

    env_device = os.environ.get("OFFLINE_EVAL_DEVICE", "").strip()
    if env_device:
        DEVICE = env_device
    env_out = os.environ.get("OFFLINE_EVAL_OUTPUT_DIR", "").strip()
    if env_out:
        OUTPUT_DIR = env_out
    env_model = os.environ.get("OFFLINE_EVAL_MODEL_PATH", "").strip()
    if env_model:
        MODEL_PATH = env_model

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

    unique_terms = sorted({t for ex in examples for t in ex.gt_terms})
    _log(f"Unique terms in dev: {len(unique_terms)}")

    glossary_path = out_dir / GLOSSARY_JSON_NAME
    index_path = out_dir / INDEX_PKL_NAME
    if not glossary_path.exists():
        _build_glossary_json(unique_terms, glossary_path)
    _maybe_build_index(glossary_path, index_path)

    index, term_list = _load_index(index_path)
    term_embeddings = _extract_term_embeddings(index)
    num_terms = term_embeddings.shape[0]
    if num_terms <= TOP_K:
        _err(f"num_terms={num_terms} must be greater than TOP_K={TOP_K}")

    from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4
    import torch

    retriever = StreamingQwen3RAGRetrieverV4(
        index_path=str(index_path),
        model_path=str(MODEL_PATH),
        base_model_name=AUDIO_BASE_MODEL_NAME,
        device=DEVICE,
        lora_r=AUDIO_LORA_R,
        lora_alpha=AUDIO_LORA_ALPHA,
        text_lora_r=TEXT_LORA_R,
        top_k=TOP_K,
        voting_k=TOP_K,
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

    per_chunk_rows: List[Dict[str, Any]] = []
    topk_values_all: List[float] = []
    rest_values_all: List[float] = []
    top1_minus_top11_all: List[float] = []
    top10_minus_rest_mean_all: List[float] = []

    _log(f"Encoding and scoring chunks (batch_size={EVAL_BATCH_SIZE}) ...")
    for start in range(0, len(examples), EVAL_BATCH_SIZE):
        batch_examples = examples[start : start + EVAL_BATCH_SIZE]
        audios = [_load_audio_mono_16k(ex.audio_path) for ex in batch_examples]

        inputs = retriever.feature_extractor(audios, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False)
        features = inputs.input_features
        batch_size, channels, mel_len = features.shape
        input_features = features.transpose(0, 1).reshape(channels, -1).to(retriever.device).to(torch.bfloat16)
        feature_lens = torch.full((batch_size,), mel_len, dtype=torch.long, device=retriever.device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                audio_embs = retriever.model(input_features, feature_lens)
            audio_embs = audio_embs.detach().cpu().float().numpy().astype(np.float32, copy=False)

        # score_matrix: [B, num_terms]
        score_matrix = audio_embs @ term_embeddings.T
        for i, ex in enumerate(batch_examples):
            scores = score_matrix[i]
            sorted_indices = np.argsort(-scores)
            topk_idx = sorted_indices[:TOP_K]
            rest_idx = sorted_indices[TOP_K:]

            topk_scores = scores[topk_idx]
            rest_scores = scores[rest_idx]
            top1 = float(topk_scores[0])
            top11 = float(scores[sorted_indices[TOP_K]])
            top10_mean = float(topk_scores.mean())
            rest_mean = float(rest_scores.mean())
            gap_mean = top10_mean - rest_mean
            margin_top1_top11 = top1 - top11

            topk_values_all.extend(topk_scores.tolist())
            rest_values_all.extend(rest_scores.tolist())
            top1_minus_top11_all.append(margin_top1_top11)
            top10_minus_rest_mean_all.append(gap_mean)

            per_chunk_rows.append(
                {
                    "chunk_id": ex.key.as_id(),
                    "audio_path": ex.audio_path,
                    "num_gt_terms": len(ex.gt_terms),
                    "top1_cosine": top1,
                    "top10_mean_cosine": top10_mean,
                    "rest_mean_cosine": rest_mean,
                    "top10_minus_rest_mean": gap_mean,
                    "top1_minus_top11": margin_top1_top11,
                }
            )

    def _quantiles(values: List[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }

    summary = {
        "model_path": MODEL_PATH,
        "dev_jsonl": str(dev_jsonl),
        "num_chunks": len(per_chunk_rows),
        "num_terms_in_index": int(num_terms),
        "top_k": TOP_K,
        "top10_cosine_mean": float(np.mean(topk_values_all)),
        "rest_cosine_mean": float(np.mean(rest_values_all)),
        "top10_minus_rest_mean": float(np.mean(top10_minus_rest_mean_all)),
        "top1_minus_top11_mean": float(np.mean(top1_minus_top11_all)),
        "top10_cosine_quantiles": _quantiles(topk_values_all),
        "rest_cosine_quantiles": _quantiles(rest_values_all),
        "top10_minus_rest_quantiles": _quantiles(top10_minus_rest_mean_all),
        "top1_minus_top11_quantiles": _quantiles(top1_minus_top11_all),
    }

    summary_json_path = out_dir / SUMMARY_JSON_NAME
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    per_chunk_tsv_path = out_dir / PER_CHUNK_TSV_NAME
    with per_chunk_tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_id",
                "audio_path",
                "num_gt_terms",
                "top1_cosine",
                "top10_mean_cosine",
                "rest_mean_cosine",
                "top10_minus_rest_mean",
                "top1_minus_top11",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in per_chunk_rows:
            writer.writerow(
                {
                    "chunk_id": row["chunk_id"],
                    "audio_path": row["audio_path"],
                    "num_gt_terms": row["num_gt_terms"],
                    "top1_cosine": _format_float(row["top1_cosine"]),
                    "top10_mean_cosine": _format_float(row["top10_mean_cosine"]),
                    "rest_mean_cosine": _format_float(row["rest_mean_cosine"]),
                    "top10_minus_rest_mean": _format_float(row["top10_minus_rest_mean"]),
                    "top1_minus_top11": _format_float(row["top1_minus_top11"]),
                }
            )

    plot_path = out_dir / PLOT_PNG_NAME
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, axes = plt.subplots(1, 2, figsize=(PLOT_FIGSIZE_W, PLOT_FIGSIZE_H))
        axes[0].hist(topk_values_all, bins=60, alpha=0.7, label="Top-10 cosine")
        axes[0].hist(rest_values_all, bins=60, alpha=0.7, label="Non-Top-10 cosine")
        axes[0].set_title("Cosine Distribution")
        axes[0].set_xlabel("Cosine similarity")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].hist(top10_minus_rest_mean_all, bins=60, alpha=0.8, color="tab:green")
        axes[1].set_title("Per-chunk Gap: Top10 mean - Rest mean")
        axes[1].set_xlabel("Gap")
        axes[1].set_ylabel("Count")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("GigaSpeech dev: Top-10 vs Non-Top-10 Cosine Analysis")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=PLOT_DPI)
        plt.close()
    except Exception as exc:
        _log(f"[WARN] Plot skipped due to error: {exc}")

    _log(f"Summary JSON: {summary_json_path}")
    _log(f"Per-chunk TSV: {per_chunk_tsv_path}")
    _log(f"Plot PNG: {plot_path}")
    _log(f"Top10 mean cosine: {_format_float(summary['top10_cosine_mean'])}")
    _log(f"Non-Top10 mean cosine: {_format_float(summary['rest_cosine_mean'])}")
    _log(f"Mean gap (Top10 - Rest): {_format_float(summary['top10_minus_rest_mean'])}")
    _log(f"Mean margin (Top1 - Top11): {_format_float(summary['top1_minus_top11_mean'])}")

    return 0


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit("Missing dependency: torch.")
    raise SystemExit(main())

