#!/usr/bin/env python3

"""
Offline evaluation: Recall@K1 saturation curve on GigaSpeech dev term dataset.

Dataset:
  DEV_JSONL contains rows with:
    - chunk_audio_path
    - utter_id
    - chunk_idx
    - term (may be empty)

Protocol:
  - Filter out rows with empty term.
  - Group by (utter_id, chunk_idx) to build multi-positive ground-truth terms per audio chunk.
  - Build a glossary JSON from the unique terms in the dataset.
  - Build a FAISS index for the glossary using the tuned V4 text encoder (build_index_v4.py format).
  - Encode each audio chunk with the V4 audio retriever, retrieve Top-(max K1), and compute strict multi-positive recall@K1.
  - Output a TSV and a PNG plot (if matplotlib is available).

All log messages are in English.
"""

from __future__ import annotations

# ======Configuration=====
DEV_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_final.jsonl"

# Model checkpoint that contains both audio retriever weights and (optionally) tuned text encoder weights.
MODEL_PATH = "/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"

# Index build settings (match build_index_v4.py expectations).
TEXT_LORA_R = 16
TARGET_LANG_CODE = "zh"

# Audio encoder settings (match StreamingQwen3RAGRetrieverV4 defaults).
DEVICE = "cuda:0"
AUDIO_BASE_MODEL_NAME = "Atotti/Qwen3-Omni-AudioTransformer"
AUDIO_LORA_R = 32
AUDIO_LORA_ALPHA = 64

# Audio chunk assumptions for this dataset.
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SECONDS = 1.92
EXPECTED_CHUNK_SAMPLES = 30720  # 1.92s * 16kHz

# K1 sweep (recall@K1).
K1_LIST = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 80, 100]

# Evaluation limits (0 means no limit).
MAX_CHUNKS = 0

# Output paths
OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/offline_eval_k1_saturation_gigaspeech_dev"
GLOSSARY_JSON_NAME = "gigaspeech_dev_terms_glossary.json"
INDEX_PKL_NAME = f"gigaspeech_dev_terms_index_v4_tr{TEXT_LORA_R}.pkl"
RESULT_TSV_NAME = "recall_k1_saturation.tsv"
PLOT_PNG_NAME = "recall_k1_saturation.png"
PLOT_LATENCY_PNG_NAME = "recall_k1_saturation_with_latency.png"

# Execution
INDEX_BUILD_BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 32

# Retriever behavior (avoid magic numbers in constructor)
VOTING_MIN_VOTES = 1
SCORE_THRESHOLD = 0.0

# Timing / latency metrics
ENABLE_TIMING_METRICS = True
WARMUP_BATCHES_FOR_TIMING = 1
CUDA_SYNC_FOR_TIMING = True
LATENCY_PERCENTILES = [50, 90]
LATENCY_UNIT_SECONDS = True

# Plot settings
PLOT_DPI = 180
PLOT_FIGSIZE = (8.0, 4.8)
PLOT_LINEWIDTH = 2.0
PLOT_GRID_ALPHA = 0.3
PLOT_RECALL_MARKER = "o"
PLOT_SCORE_MARKER = "s"
PLOT_RECALL_COLOR = "tab:blue"
PLOT_SCORE_COLOR = "tab:red"

# Misc
CSV_DELIMITER = "\t"
CSV_LINE_TERMINATOR = "\n"
FLOAT_DECIMALS = 6
# ======Configuration=====

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# Ensure repository root is importable (so `import agents` works in Slurm without PYTHONPATH).
_REPO_ROOT = Path(__file__).resolve().parents[3]
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
            _warn(f"Skip row with missing fields: utter_id={utter_id!r} chunk_idx={chunk_idx!r} audio_path={audio_path!r}")
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


def _load_index_term_map(index_path: Path) -> Dict[str, int]:
    import pickle
    import faiss  # type: ignore

    with index_path.open("rb") as f:
        data = pickle.load(f)

    term_list = data["term_list"]
    # Ensure the FAISS index can be deserialized (sanity check).
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
    if max_val > 0:
        audio = audio / max_val

    if audio.shape[0] < EXPECTED_CHUNK_SAMPLES:
        audio = np.pad(audio, (0, EXPECTED_CHUNK_SAMPLES - audio.shape[0]), mode="constant")
    elif audio.shape[0] > EXPECTED_CHUNK_SAMPLES:
        audio = audio[:EXPECTED_CHUNK_SAMPLES]
    return audio


def _format_float(x: float) -> str:
    return f"{x:.{FLOAT_DECIMALS}f}"


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _maybe_cuda_sync(torch_module: Any, enabled: bool, device_str: str) -> None:
    if not enabled:
        return
    if "cuda" not in device_str:
        return
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
    except Exception:
        # Best-effort sync only; do not fail evaluation.
        return


def main() -> int:
    # Allow lightweight overrides via environment variables (useful for Slurm jobs).
    # Examples:
    #   OFFLINE_EVAL_DEVICE=cuda:0 OFFLINE_EVAL_OUTPUT_DIR=... bash run_*.sh
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
    term_to_idx = _load_index_term_map(index_path)
    max_k1 = max(K1_LIST) if K1_LIST else 0
    if max_k1 <= 0:
        _err("K1_LIST is empty or invalid.")

    # Prepare positives per chunk in index space.
    chunk_pos_indices: List[Tuple[str, str, Set[int]]] = []
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
        chunk_pos_indices.append((ex.key.as_id(), ex.audio_path, pos))

    _log(f"Chunks with >=1 mapped positive term: {len(chunk_pos_indices)}")
    _log(f"Total positives (strict denominator): {total_valid_pos}")
    if missing_terms:
        _warn(f"Missing terms not found in index (skipped in positives): {missing_terms}")
    if not chunk_pos_indices or total_valid_pos <= 0:
        _err("No positives to evaluate after mapping terms to index. Check glossary build/mapping.")

    # Load retriever + index through the existing agent class (consistent feature extraction + audio encoder).
    from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4

    # Use voting_k=max_k1 to retrieve enough candidates; we will compute recall for all k in K1_LIST.
    _log(f"Initializing retriever (voting_k={max_k1}) on device={DEVICE}")
    retriever = StreamingQwen3RAGRetrieverV4(
        index_path=str(index_path),
        model_path=str(MODEL_PATH),
        base_model_name=AUDIO_BASE_MODEL_NAME,
        device=DEVICE,
        lora_r=AUDIO_LORA_R,
        lora_alpha=AUDIO_LORA_ALPHA,
        text_lora_r=TEXT_LORA_R,
        top_k=max_k1,
        voting_k=max_k1,
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

    import faiss  # type: ignore
    import torch

    # Encode + search in batches.
    pos_hits: Dict[int, int] = {k: 0 for k in K1_LIST}
    # Cosine similarity score distribution when a positive is retrieved.
    # Index is IndexFlatIP with L2-normalized vectors => FAISS returns cosine similarity in D.
    k1_sorted = sorted(set(int(k) for k in K1_LIST))
    hit_best_cos_sim_by_k: Dict[int, List[float]] = {k: [] for k in k1_sorted}
    hit_chunks_by_k: Dict[int, int] = {k: 0 for k in k1_sorted}

    # Latency metrics (per-chunk seconds)
    search_latency_s_by_k: Dict[int, List[float]] = {k: [] for k in k1_sorted}
    retrieval_latency_s_by_k: Dict[int, List[float]] = {k: [] for k in k1_sorted}
    feature_latency_s: List[float] = []
    encode_latency_s: List[float] = []
    audio_load_latency_s: List[float] = []

    _log(f"Encoding audio and searching FAISS (batch_size={EVAL_BATCH_SIZE}) ...")
    for batch_idx, start in enumerate(range(0, len(chunk_pos_indices), EVAL_BATCH_SIZE)):
        batch = chunk_pos_indices[start : start + EVAL_BATCH_SIZE]
        B = len(batch)

        # Audio load time (I/O + normalize/pad/truncate); often excluded from "retriever" time in practice.
        t0_load = time.perf_counter()
        audios = [_load_audio_mono_16k(p) for _, p, _ in batch]
        t1_load = time.perf_counter()

        # Feature extractor time (CPU)
        t0_feat = time.perf_counter()
        inputs = retriever.feature_extractor(
            audios, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False
        )
        features = inputs.input_features  # [B, C, T]
        t1_feat = time.perf_counter()

        _, C, T_mel = features.shape

        input_features = features.transpose(0, 1).reshape(C, -1).to(retriever.device).to(torch.bfloat16)
        feature_lens = torch.full((B,), T_mel, dtype=torch.long, device=retriever.device)

        # Model encode time (GPU/CPU)
        device_str = str(retriever.device)
        _maybe_cuda_sync(torch, CUDA_SYNC_FOR_TIMING, device_str)
        t0_enc = time.perf_counter()
        with torch.no_grad():
            if "cuda" in device_str:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    audio_embs = retriever.model(input_features, feature_lens)
            else:
                audio_embs = retriever.model(input_features, feature_lens)
            audio_embs = audio_embs.detach().cpu().float().numpy()
        _maybe_cuda_sync(torch, CUDA_SYNC_FOR_TIMING, device_str)
        t1_enc = time.perf_counter()

        faiss.normalize_L2(audio_embs)

        # Search time varies with K; measure each K on identical embeddings (fair comparison).
        # NOTE: This repeats FAISS search per K, but avoids re-encoding audio for each K.
        for k in k1_sorted:
            t0_search = time.perf_counter()
            D_k, I_k = retriever.index.search(audio_embs, int(k))
            t1_search = time.perf_counter()

            # Update recall hits for this K
            for i in range(B):
                pos = batch[i][2]
                # This should not happen because we filtered chunks with >=1 positive.
                if not pos:
                    continue

                hit_scores: List[float] = []
                for j in range(int(k)):
                    idx = int(I_k[i][j])
                    if idx in pos:
                        hit_scores.append(float(D_k[i][j]))

                pos_in_topk = len(hit_scores)
                # Micro recall numerator (over positives)
                if pos_in_topk > 0:
                    pos_hits[k] += pos_in_topk
                    # Best cosine similarity among the retrieved positives within top-K.
                    hit_best_cos_sim_by_k[k].append(float(max(hit_scores)))
                    hit_chunks_by_k[k] += 1

            # Timing metrics (per-chunk seconds), optionally skipping warmup batches.
            if ENABLE_TIMING_METRICS and batch_idx >= WARMUP_BATCHES_FOR_TIMING:
                search_s_per_chunk = (t1_search - t0_search) / float(B)
                feat_s_per_chunk = (t1_feat - t0_feat) / float(B)
                enc_s_per_chunk = (t1_enc - t0_enc) / float(B)
                retrieval_s_per_chunk = feat_s_per_chunk + enc_s_per_chunk + search_s_per_chunk
                search_latency_s_by_k[k].extend([search_s_per_chunk] * B)
                retrieval_latency_s_by_k[k].extend([retrieval_s_per_chunk] * B)

        if ENABLE_TIMING_METRICS and batch_idx >= WARMUP_BATCHES_FOR_TIMING:
            audio_load_latency_s.extend([((t1_load - t0_load) / float(B))] * B)
            feature_latency_s.extend([((t1_feat - t0_feat) / float(B))] * B)
            encode_latency_s.extend([((t1_enc - t0_enc) / float(B))] * B)

    # Write results
    tsv_path = out_dir / RESULT_TSV_NAME
    _log(f"Writing TSV: {tsv_path}")

    rows: List[Dict[str, Any]] = []
    for k in k1_sorted:
        recall = (pos_hits[k] / total_valid_pos) if total_valid_pos > 0 else 0.0
        hit_scores = hit_best_cos_sim_by_k.get(k, [])
        hit_score_mean = _mean(hit_scores)
        hit_score_min = float(min(hit_scores)) if hit_scores else 0.0
        hit_chunks = int(hit_chunks_by_k.get(k, 0))
        timed_chunks = len(retrieval_latency_s_by_k.get(k, []))
        retrieval_mean = _mean(retrieval_latency_s_by_k.get(k, []))
        search_mean = _mean(search_latency_s_by_k.get(k, []))

        p_metrics: Dict[str, Any] = {}
        for p in LATENCY_PERCENTILES:
            p_metrics[f"retrieval_p{int(p)}_s"] = _format_float(_percentile(retrieval_latency_s_by_k.get(k, []), float(p)))
            p_metrics[f"search_p{int(p)}_s"] = _format_float(_percentile(search_latency_s_by_k.get(k, []), float(p)))

        throughput = (1.0 / retrieval_mean) if retrieval_mean > 0 else 0.0
        rows.append(
            {
                "k1": k,
                "recall": _format_float(recall),
                "hit_chunks": hit_chunks,
                "hit_score_mean": _format_float(hit_score_mean),
                "hit_score_min": _format_float(hit_score_min),
                "pos_hits": pos_hits[k],
                "total_pos": total_valid_pos,
                "num_chunks": len(chunk_pos_indices),
                "unique_terms": len(unique_terms),
                "timed_chunks": timed_chunks,
                "retrieval_mean_s": _format_float(retrieval_mean),
                "search_mean_s": _format_float(search_mean),
                "throughput_chunks_per_s": _format_float(throughput),
                **p_metrics,
            }
        )

    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "k1",
            "recall",
            "hit_chunks",
            "hit_score_mean",
            "hit_score_min",
            "pos_hits",
            "total_pos",
            "num_chunks",
            "unique_terms",
            "timed_chunks",
            "retrieval_mean_s",
            "search_mean_s",
            "throughput_chunks_per_s",
        ]
        for p in LATENCY_PERCENTILES:
            fieldnames.append(f"retrieval_p{int(p)}_s")
        for p in LATENCY_PERCENTILES:
            fieldnames.append(f"search_p{int(p)}_s")

        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=CSV_DELIMITER, lineterminator=CSV_LINE_TERMINATOR)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Plot (optional)
    png_path = out_dir / PLOT_PNG_NAME
    try:
        import matplotlib.pyplot as plt  # type: ignore

        xs = [int(r["k1"]) for r in rows]
        ys = [float(r["recall"]) for r in rows]

        plt.figure(figsize=PLOT_FIGSIZE)
        plt.plot(xs, ys, marker=PLOT_RECALL_MARKER, linewidth=PLOT_LINEWIDTH)
        plt.xlabel("K1")
        plt.ylabel("Recall@K1")
        plt.title("GigaSpeech dev: Recall@K1 saturation (chunk=1.92s)")
        plt.grid(True, alpha=PLOT_GRID_ALPHA)
        plt.tight_layout()
        plt.savefig(png_path, dpi=PLOT_DPI)
        plt.close()
        _log(f"Wrote plot: {png_path}")

        # Optional plot with hit score (secondary y-axis)
        if ENABLE_TIMING_METRICS:
            png_latency_path = out_dir / PLOT_LATENCY_PNG_NAME
            hit_score_min = [float(r["hit_score_min"]) for r in rows]
            plt.figure(figsize=PLOT_FIGSIZE)
            ax1 = plt.gca()
            ax1.plot(xs, ys, marker=PLOT_RECALL_MARKER, linewidth=PLOT_LINEWIDTH, color=PLOT_RECALL_COLOR)
            ax1.set_xlabel("K1")
            ax1.set_ylabel("Recall@K1", color=PLOT_RECALL_COLOR)
            ax1.tick_params(axis="y", labelcolor=PLOT_RECALL_COLOR)
            ax1.grid(True, alpha=PLOT_GRID_ALPHA)

            ax2 = ax1.twinx()
            ax2.plot(xs, hit_score_min, marker=PLOT_SCORE_MARKER, linewidth=PLOT_LINEWIDTH, color=PLOT_SCORE_COLOR)
            ax2.set_ylabel("Hit score min (cosine similarity)", color=PLOT_SCORE_COLOR)
            ax2.tick_params(axis="y", labelcolor=PLOT_SCORE_COLOR)

            plt.title("GigaSpeech dev: Recall@K1 vs hit_score_min")
            plt.tight_layout()
            plt.savefig(png_latency_path, dpi=PLOT_DPI)
            plt.close()
            _log(f"Wrote plot: {png_latency_path}")
    except Exception as e:
        _warn(f"Plot skipped (matplotlib not available or failed): {e}")

    if ENABLE_TIMING_METRICS and feature_latency_s and encode_latency_s:
        _log(
            "Timing summary (per-chunk seconds, excluding warmup batches): "
            f"audio_load_mean={_format_float(_mean(audio_load_latency_s))} "
            f"feature_mean={_format_float(_mean(feature_latency_s))} "
            f"encode_mean={_format_float(_mean(encode_latency_s))}"
        )

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


