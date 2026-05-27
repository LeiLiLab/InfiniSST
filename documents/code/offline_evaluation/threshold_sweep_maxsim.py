#!/usr/bin/env python3
"""
Threshold sweep for MaxSim retriever.

Three filtering strategies:
  A. Absolute:  emit term if score >= tau
  B. Gap:       emit term if score >= top10_mean + delta
  C. Hybrid:    emit term if score >= tau AND score >= top10_mean + delta

Also reports chunk-level has_term detection (top1 >= tau_chunk).

Phase 1: sweep on dev data  -> select best params
Phase 2: validate on ACL 6060
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ======Configuration=====
AUDIO_MODEL_ID = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_MODEL_ID = "BAAI/bge-m3"
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SAMPLES = 30720  # 1.92s * 16kHz
TEXT_ENCODE_BATCH = 256
AUDIO_ENCODE_BATCH = 32
FLOAT_DECIMALS = 6

K_CANDIDATES = 10

# Strategy A: absolute threshold sweep (overridable via --abs_tau_values)
ABS_MIN = 0.10
ABS_MAX = 0.70
ABS_STEPS = 61
# When --abs_tau_values is provided, these override the (min/max/steps)
# defaults above and `HIST_DEFAULT_TAUS` is still used for per-chunk topk
# retention plots.
HIST_DEFAULT_TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Strategy B: gap above top10_mean sweep
GAP_MIN = 0.00
GAP_MAX = 0.25
GAP_STEPS = 26

# Strategy C: hybrid 2D grid (coarser for tractability)
HYBRID_ABS_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
HYBRID_GAP_VALUES = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]

# Chunk-level has_term detection sweep (based on top1 score)
CHUNK_DET_MIN = 0.15
CHUNK_DET_MAX = 0.65
CHUNK_DET_STEPS = 51

# Selection constraint: only consider configs with recall >= this
MIN_RECALL_CONSTRAINT = 0.90
# ======Configuration=====


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _fmt(x: float) -> str:
    return f"{x:.{FLOAT_DECIMALS}f}"


def _fbeta(p: float, r: float, beta: float = 1.0) -> float:
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


@dataclass
class Chunk:
    chunk_id: str
    audio_path: str
    gt_terms: Set[str] = field(default_factory=set)
    has_term: bool = False


def load_chunks(jsonl_path: str) -> Tuple[List[Chunk], List[Chunk]]:
    groups: Dict[str, Chunk] = {}
    for line in open(jsonl_path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        uid = obj.get("utter_id", "")
        cidx = str(obj.get("chunk_idx", ""))
        apath = obj.get("chunk_audio_path", "")
        term = (obj.get("term_key", "") or obj.get("term", "") or "").strip().lower()

        cid = f"{uid}::{cidx}"
        if cid not in groups:
            groups[cid] = Chunk(chunk_id=cid, audio_path=apath)
        if term:
            groups[cid].gt_terms.add(term)
            groups[cid].has_term = True

    with_term = sorted([c for c in groups.values() if c.has_term], key=lambda x: x.chunk_id)
    no_term = sorted([c for c in groups.values() if not c.has_term], key=lambda x: x.chunk_id)
    return with_term, no_term


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(path)
    assert sr == EXPECTED_SAMPLE_RATE, f"Unexpected SR {sr} for {path}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    mx = float(np.max(np.abs(audio))) if audio.size else 0.0
    if mx > 0:
        audio = audio / mx
    if audio.shape[0] < EXPECTED_CHUNK_SAMPLES:
        audio = np.pad(audio, (0, EXPECTED_CHUNK_SAMPLES - audio.shape[0]))
    elif audio.shape[0] > EXPECTED_CHUNK_SAMPLES:
        audio = audio[:EXPECTED_CHUNK_SAMPLES]
    return audio


def build_model(args):
    sys.path.insert(0, str(_REPO_ROOT / "documents" / "code" / "train" / "term_train"))
    from qwen3_glossary_neg_train import (
        BgeM3TextEncoder,
        Qwen3OmniRetriever,
        _maxsim_score,
    )
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    device = torch.device(args.device)

    retriever = Qwen3OmniRetriever(
        model_id=AUDIO_MODEL_ID,
        target_dim=args.target_dim,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules.split(),
        temperature=args.temperature,
        learn_temp=False,
        pooling_type=args.pooling_type,
        use_maxsim=args.use_maxsim,
        maxsim_windows=[int(x) for x in args.maxsim_windows.split()] if args.maxsim_windows else None,
        maxsim_stride=args.maxsim_stride,
    ).to(device)

    text_encoder = BgeM3TextEncoder(
        model_id=TEXT_MODEL_ID,
        lora_rank=args.text_lora_rank,
        lora_alpha=args.text_lora_alpha,
        target_modules=args.text_lora_target_modules.split(),
        full_finetune=False,
        sparse_weight=args.sparse_weight,
        text_pooling=args.text_pooling,
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device)

    def _strip(sd):
        return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    retriever.load_state_dict(_strip(ckpt.get("model_state_dict", {})), strict=False)
    if "text_model_state_dict" in ckpt:
        text_encoder.load_state_dict(_strip(ckpt["text_model_state_dict"]), strict=False)

    retriever.eval()
    text_encoder.eval()

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    feat_ext = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    return retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device


@torch.no_grad()
def encode_terms(terms: List[str], text_encoder, tokenizer, device) -> torch.Tensor:
    all_embs = []
    for start in range(0, len(terms), TEXT_ENCODE_BATCH):
        batch = terms[start:start + TEXT_ENCODE_BATCH]
        tok = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = text_encoder(tok.input_ids, tok.attention_mask)
        all_embs.append(embs.float())
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_audio_chunks(chunks: List[Chunk], retriever, feat_ext, device) -> torch.Tensor:
    all_embs = []
    for start in range(0, len(chunks), AUDIO_ENCODE_BATCH):
        batch = chunks[start:start + AUDIO_ENCODE_BATCH]
        audios = [load_audio(c.audio_path) for c in batch]
        inputs = feat_ext(audios, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False)
        features = inputs.input_features
        B, C, T_mel = features.shape
        input_features = features.transpose(0, 1).reshape(C, -1).to(device).to(torch.bfloat16)
        feature_lens = torch.full((B,), T_mel, dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = retriever(input_features, feature_lens)
        all_embs.append(embs.float())
        done = start + len(batch)
        if (start // AUDIO_ENCODE_BATCH) % 20 == 0:
            _log(f"  encoded {done}/{len(chunks)} audio chunks")
    return torch.cat(all_embs, dim=0)


def compute_sim(speech_embs: torch.Tensor, text_embs: torch.Tensor, _maxsim_score) -> torch.Tensor:
    if speech_embs.ndim == 3:
        return _maxsim_score(speech_embs, text_embs)
    return speech_embs @ text_embs.T


# ---------------------------------------------------------------------------
# Pre-compute per-chunk top-K info once, then reuse across all sweeps
# ---------------------------------------------------------------------------

@dataclass
class ChunkTopK:
    idx: int
    has_term: bool
    pos_indices: Set[int]
    top_k_indices: np.ndarray   # [K]
    top_k_scores: np.ndarray    # [K]
    top1_score: float
    top10_mean: float


def precompute_topk(
    sim_matrix: torch.Tensor,
    chunks: List[Chunk],
    term_to_idx: Dict[str, int],
) -> Tuple[List[ChunkTopK], int]:
    """Extract sorted top-K indices/scores for every chunk."""
    sim_np = sim_matrix.cpu().numpy()
    N_text = sim_np.shape[1]
    k = min(K_CANDIDATES, N_text)

    top_k_idx = np.argpartition(-sim_np, k, axis=1)[:, :k]
    top_k_sco = np.take_along_axis(sim_np, top_k_idx, axis=1)
    order = np.argsort(-top_k_sco, axis=1)
    top_k_idx = np.take_along_axis(top_k_idx, order, axis=1)
    top_k_sco = np.take_along_axis(top_k_sco, order, axis=1)

    results = []
    total_valid_pos = 0
    for i, c in enumerate(chunks):
        mapped = {term_to_idx[t] for t in c.gt_terms if t in term_to_idx}
        if c.has_term:
            total_valid_pos += len(mapped)
        results.append(ChunkTopK(
            idx=i,
            has_term=c.has_term,
            pos_indices=mapped,
            top_k_indices=top_k_idx[i],
            top_k_scores=top_k_sco[i],
            top1_score=float(top_k_sco[i, 0]),
            top10_mean=float(np.mean(top_k_sco[i, :k])),
        ))
    return results, total_valid_pos


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def _eval_filter(
    chunk_data: List[ChunkTopK],
    total_valid_pos: int,
    emit_fn,
) -> Dict[str, Any]:
    """
    Apply emit_fn(score, top10_mean) -> bool to each candidate.
    Compute term-level P/R/F1 on with_term chunks, avg_noise on no_term chunks.
    """
    tp = 0
    pred_total = 0
    noterm_recalled = []

    for cd in chunk_data:
        pred_set = set()
        for j in range(len(cd.top_k_scores)):
            if emit_fn(cd.top_k_scores[j], cd.top10_mean):
                pred_set.add(int(cd.top_k_indices[j]))

        if cd.has_term:
            tp += len(pred_set & cd.pos_indices)
            pred_total += len(pred_set)
        else:
            noterm_recalled.append(len(pred_set))

    fp = pred_total - tp
    fn = total_valid_pos - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _fbeta(precision, recall, beta=1.0)
    f2 = _fbeta(precision, recall, beta=2.0)
    avg_noise = float(np.mean(noterm_recalled)) if noterm_recalled else 0.0

    n_with = sum(1 for cd in chunk_data if cd.has_term)
    n_no = sum(1 for cd in chunk_data if not cd.has_term)

    return {
        "precision": _fmt(precision),
        "recall": _fmt(recall),
        "f1": _fmt(f1),
        "f2": _fmt(f2),
        "tp": tp, "fp": fp, "fn": fn,
        "pred_total": pred_total,
        "total_pos": total_valid_pos,
        "with_term_chunks": n_with,
        "no_term_chunks": n_no,
        "avg_noise_terms": _fmt(avg_noise),
    }


def _eval_chunk_detection(
    chunk_data: List[ChunkTopK],
    tau_chunk: float,
) -> Dict[str, Any]:
    """
    Chunk-level has_term detection: predict has_term if top1 >= tau_chunk.
    """
    tp = fp = fn = tn = 0
    for cd in chunk_data:
        pred_has = cd.top1_score >= tau_chunk
        if cd.has_term and pred_has:
            tp += 1
        elif cd.has_term and not pred_has:
            fn += 1
        elif not cd.has_term and pred_has:
            fp += 1
        else:
            tn += 1

    chunk_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    chunk_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    chunk_f1 = _fbeta(chunk_p, chunk_r, beta=1.0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tau_chunk": _fmt(tau_chunk),
        "chunk_precision": _fmt(chunk_p),
        "chunk_recall": _fmt(chunk_r),
        "chunk_f1": _fmt(chunk_f1),
        "chunk_fpr": _fmt(fpr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Main sweep per dataset
# ---------------------------------------------------------------------------

def run_sweep_on_dataset(
    dataset_name: str,
    jsonl_path: str,
    wiki_glossary_path: str,
    glossary_sizes: List[int],
    retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
    output_dir: str,
    abs_tau_values: List[float] = None,
    plot_histograms: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    _log(f"=== {dataset_name} ===")

    with_term, no_term = load_chunks(jsonl_path)
    all_chunks = with_term + no_term
    _log(f"  with_term={len(with_term)}, no_term={len(no_term)}, total={len(all_chunks)}")

    gt_terms = sorted({t for c in with_term for t in c.gt_terms})
    _log(f"  unique GT terms: {len(gt_terms)}")

    with open(wiki_glossary_path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        wiki_all = [item["term"] if isinstance(item, dict) else str(item) for item in raw]
    else:
        assert isinstance(raw, dict), f"Unexpected glossary format: {type(raw)}"
        wiki_all = list(raw.keys())
    gt_set = set(gt_terms)
    wiki_filtered = [t for t in wiki_all if t.lower() not in gt_set]
    _log(f"  wiki distractors available: {len(wiki_filtered)}")

    _log("  Encoding audio...")
    speech_embs = encode_audio_chunks(all_chunks, retriever, feat_ext, device)
    _log(f"  speech_embs: {speech_embs.shape}")

    os.makedirs(output_dir, exist_ok=True)
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for gs in glossary_sizes:
        # gs=0 is a special "raw" marker: use only GT terms, no wiki distractors.
        if gs == 0:
            gs_label = "raw"
            wiki_subset = []
            term_list = list(gt_terms)
            _log(f"  {gs_label}: bank={len(term_list)} (GT={len(gt_terms)} only, no distractors)")
        else:
            gs_label = f"gs{gs}"
            n_extra = gs - len(gt_terms)
            if n_extra < 0:
                _log(f"  {gs_label}: skipped (GT={len(gt_terms)} >= gs)")
                continue
            wiki_subset = wiki_filtered[:n_extra]
            term_list = gt_terms + [t.lower() for t in wiki_subset]
            _log(f"  {gs_label}: bank={len(term_list)} (GT={len(gt_terms)} + wiki={len(wiki_subset)})")

        text_embs = encode_terms(term_list, text_encoder, tokenizer, device)
        sim = compute_sim(speech_embs, text_embs, _maxsim_score)
        _log(f"  sim matrix: {sim.shape}")

        term_to_idx = {t: i for i, t in enumerate(term_list)}
        chunk_data, total_valid_pos = precompute_topk(sim, all_chunks, term_to_idx)

        # ---- Chunk-level has_term detection sweep ----
        chunk_det_rows = []
        for tau in np.linspace(CHUNK_DET_MIN, CHUNK_DET_MAX, CHUNK_DET_STEPS):
            row = _eval_chunk_detection(chunk_data, float(tau))
            row["gs"] = gs_label
            chunk_det_rows.append(row)

        best_cd = max(chunk_det_rows, key=lambda x: float(x["chunk_f1"]))
        _log(
            f"  [{gs_label}] Chunk detection best: tau={best_cd['tau_chunk']} "
            f"P={best_cd['chunk_precision']} R={best_cd['chunk_recall']} "
            f"F1={best_cd['chunk_f1']} FPR={best_cd['chunk_fpr']}"
        )

        # ---- Strategy A: absolute threshold ----
        if abs_tau_values is not None and len(abs_tau_values) > 0:
            tau_grid = list(abs_tau_values)
        else:
            tau_grid = [float(x) for x in np.linspace(ABS_MIN, ABS_MAX, ABS_STEPS)]
        abs_rows = []
        for tau_f in tau_grid:
            tau_f = float(tau_f)
            r = _eval_filter(chunk_data, total_valid_pos, lambda s, m, t=tau_f: s >= t)
            r["strategy"] = "absolute"
            r["param_tau"] = _fmt(tau_f)
            r["param_delta"] = ""
            r["gs"] = gs_label
            abs_rows.append(r)

        best_a = _select_best(abs_rows)
        _log(
            f"  [{gs_label}] Absolute best: tau={best_a['param_tau']} "
            f"P={best_a['precision']} R={best_a['recall']} F1={best_a['f1']} F2={best_a['f2']} "
            f"noise={best_a['avg_noise_terms']}"
        )

        # ---- Strategy B: gap above top10_mean ----
        gap_rows = []
        for delta in np.linspace(GAP_MIN, GAP_MAX, GAP_STEPS):
            delta_f = float(delta)
            r = _eval_filter(chunk_data, total_valid_pos, lambda s, m, d=delta_f: s >= m + d)
            r["strategy"] = "gap_top10mean"
            r["param_tau"] = ""
            r["param_delta"] = _fmt(delta_f)
            r["gs"] = gs_label
            gap_rows.append(r)

        best_g = _select_best(gap_rows)
        _log(
            f"  [{gs_label}] Gap best: delta={best_g['param_delta']} "
            f"P={best_g['precision']} R={best_g['recall']} F1={best_g['f1']} F2={best_g['f2']} "
            f"noise={best_g['avg_noise_terms']}"
        )

        # ---- Strategy C: hybrid (tau AND gap) ----
        hybrid_rows = []
        for tau_f in HYBRID_ABS_VALUES:
            for delta_f in HYBRID_GAP_VALUES:
                r = _eval_filter(
                    chunk_data, total_valid_pos,
                    lambda s, m, t=tau_f, d=delta_f: s >= t and s >= m + d,
                )
                r["strategy"] = "hybrid"
                r["param_tau"] = _fmt(tau_f)
                r["param_delta"] = _fmt(delta_f)
                r["gs"] = gs_label
                hybrid_rows.append(r)

        best_h = _select_best(hybrid_rows)
        _log(
            f"  [{gs_label}] Hybrid best: tau={best_h['param_tau']} delta={best_h['param_delta']} "
            f"P={best_h['precision']} R={best_h['recall']} F1={best_h['f1']} F2={best_h['f2']} "
            f"noise={best_h['avg_noise_terms']}"
        )

        # ---- Write TSVs ----
        term_rows = abs_rows + gap_rows + hybrid_rows
        term_tsv = os.path.join(output_dir, f"{dataset_name}_{gs_label}_term_filter.tsv")
        _write_tsv(term_tsv, term_rows, [
            "strategy", "param_tau", "param_delta", "gs",
            "precision", "recall", "f1", "f2",
            "tp", "fp", "fn", "pred_total", "total_pos",
            "with_term_chunks", "no_term_chunks", "avg_noise_terms",
        ])

        chunk_tsv = os.path.join(output_dir, f"{dataset_name}_{gs_label}_chunk_detection.tsv")
        _write_tsv(chunk_tsv, chunk_det_rows, [
            "tau_chunk", "gs",
            "chunk_precision", "chunk_recall", "chunk_f1", "chunk_fpr",
            "tp", "fp", "fn", "tn",
        ])

        all_results[gs_label] = term_rows

        if plot_histograms:
            _plot_score_histogram(
                chunk_data=chunk_data,
                dataset_name=dataset_name,
                gs_label=gs_label,
                output_dir=output_dir,
                tau_marks=HIST_DEFAULT_TAUS,
            )
            _plot_threshold_curve(
                abs_rows=abs_rows,
                dataset_name=dataset_name,
                gs_label=gs_label,
                output_dir=output_dir,
            )

    # ---- Score distribution stats ----
    _log_score_stats(chunk_data)

    return all_results


def _plot_score_histogram(
    chunk_data,
    dataset_name: str,
    gs_label: str,
    output_dir: str,
    tau_marks: List[float],
) -> None:
    """Plot top-k candidate score distribution split by positive/negative."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _log("  matplotlib unavailable, skipping histogram")
        return

    pos_scores: List[float] = []
    neg_scores: List[float] = []
    for cd in chunk_data:
        pos_set = cd.pos_indices
        for idx, sc in zip(cd.top_k_indices, cd.top_k_scores):
            target = pos_scores if int(idx) in pos_set else neg_scores
            target.append(float(sc))

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-0.2, 1.0, 61)
    if pos_scores:
        ax.hist(pos_scores, bins=bins, alpha=0.55, label=f"positive (n={len(pos_scores)})", color="#2a9d8f")
    if neg_scores:
        ax.hist(neg_scores, bins=bins, alpha=0.55, label=f"negative (n={len(neg_scores)})", color="#e76f51")
    for tau in tau_marks:
        ax.axvline(tau, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("MaxSim score (top-10 candidates)")
    ax.set_ylabel("count")
    ax.set_title(f"{dataset_name} / {gs_label}: top-10 score dist (pos vs neg)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_png = os.path.join(output_dir, f"hist_{dataset_name}_{gs_label}.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    _log(f"  Saved histogram: {out_png}")


def _plot_threshold_curve(
    abs_rows: List[Dict[str, Any]],
    dataset_name: str,
    gs_label: str,
    output_dir: str,
) -> None:
    """Plot P/R/F1 and avg_noise vs. absolute threshold."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    taus = [float(r["param_tau"]) for r in abs_rows]
    prec = [float(r["precision"]) for r in abs_rows]
    rec = [float(r["recall"]) for r in abs_rows]
    f1 = [float(r["f1"]) for r in abs_rows]
    noise = [float(r["avg_noise_terms"]) for r in abs_rows]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_l.plot(taus, prec, marker="o", label="precision")
    ax_l.plot(taus, rec, marker="o", label="recall")
    ax_l.plot(taus, f1, marker="o", label="F1")
    ax_l.set_xlabel("absolute threshold tau")
    ax_l.set_ylabel("term-level metric")
    ax_l.set_title(f"{dataset_name}/{gs_label}: P/R/F1 vs tau")
    ax_l.set_ylim(0, 1.02)
    ax_l.grid(True, alpha=0.3)
    ax_l.legend()

    ax_r.plot(taus, noise, marker="s", color="#e76f51", label="no_term chunks: avg kept terms")
    ax_r.set_xlabel("absolute threshold tau")
    ax_r.set_ylabel("avg kept terms (noise)")
    ax_r.set_title(f"{dataset_name}/{gs_label}: no-term chunk noise")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend()

    fig.tight_layout()
    out_png = os.path.join(output_dir, f"curve_{dataset_name}_{gs_label}.png")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    _log(f"  Saved curve: {out_png}")


def _select_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select best config: among those with recall >= MIN_RECALL_CONSTRAINT,
    pick highest precision. If none meets the constraint, pick highest recall."""
    feasible = [r for r in rows if float(r["recall"]) >= MIN_RECALL_CONSTRAINT]
    if feasible:
        return max(feasible, key=lambda x: float(x["precision"]))
    return max(rows, key=lambda x: float(x["recall"]))


def _write_tsv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _log(f"  Written: {path}")


def _log_score_stats(chunk_data: List[ChunkTopK]) -> None:
    wt_top1 = [cd.top1_score for cd in chunk_data if cd.has_term]
    nt_top1 = [cd.top1_score for cd in chunk_data if not cd.has_term]
    wt_mean10 = [cd.top10_mean for cd in chunk_data if cd.has_term]
    nt_mean10 = [cd.top10_mean for cd in chunk_data if not cd.has_term]

    def _stats(arr, label):
        if not arr:
            return
        a = np.array(arr)
        _log(
            f"  {label}: n={len(a)} mean={a.mean():.4f} std={a.std():.4f} "
            f"min={a.min():.4f} p25={np.percentile(a,25):.4f} "
            f"median={np.median(a):.4f} p75={np.percentile(a,75):.4f} max={a.max():.4f}"
        )

    _log("  --- Score distribution ---")
    _stats(wt_top1, "with_term top1")
    _stats(nt_top1, "no_term  top1")
    _stats(wt_mean10, "with_term top10_mean")
    _stats(nt_mean10, "no_term  top10_mean")
    if wt_top1 and nt_top1:
        wt_gap = [cd.top1_score - cd.top10_mean for cd in chunk_data if cd.has_term]
        nt_gap = [cd.top1_score - cd.top10_mean for cd in chunk_data if not cd.has_term]
        _stats(wt_gap, "with_term top1-top10mean gap")
        _stats(nt_gap, "no_term  top1-top10mean gap")


def print_summary(dataset_name: str, results: Dict[str, List[Dict[str, Any]]]) -> None:
    _log(f"\n{'='*100}")
    _log(f"SUMMARY: {dataset_name}  (selection: best precision where recall >= {MIN_RECALL_CONSTRAINT})")
    _log(f"{'='*100}")
    header = (
        f"{'strategy':<16} {'tau':>8} {'delta':>8} {'gs':>8} "
        f"{'P':>8} {'R':>8} {'F1':>8} {'F2':>8} {'noise':>8}"
    )
    _log(header)
    _log("-" * len(header))

    for gs_label in sorted(results.keys()):
        rows = results[gs_label]
        for strat in ["absolute", "gap_top10mean", "hybrid"]:
            subset = [r for r in rows if r["strategy"] == strat]
            if not subset:
                continue
            best = _select_best(subset)
            tau_s = best.get("param_tau", "")
            delta_s = best.get("param_delta", "")
            met = "Y" if float(best["recall"]) >= MIN_RECALL_CONSTRAINT else "N"
            _log(
                f"{strat:<16} {tau_s:>8} {delta_s:>8} {gs_label:>8} "
                f"{best['precision']:>8} {best['recall']:>8} "
                f"{best['f1']:>8} {best['f2']:>8} {best['avg_noise_terms']:>8}"
                f"  R>={MIN_RECALL_CONSTRAINT}:{met}"
            )
    _log(f"{'='*100}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dev_jsonl", type=str, required=True)
    p.add_argument("--acl_jsonl", type=str, required=True)
    p.add_argument("--wiki_glossary", type=str, required=True)
    p.add_argument(
        "--glossary_sizes", type=int, nargs="+", default=[1000, 10000],
        help="Glossary bank sizes. Use 0 as a special marker for 'raw' (GT terms only, no wiki distractors).",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--abs_tau_values", type=float, nargs="+", default=None,
        help="Explicit list of absolute thresholds for Strategy A sweep. Overrides ABS_MIN/MAX/STEPS defaults.",
    )
    p.add_argument(
        "--plot_histograms", action="store_true", default=False,
        help="If set, dump per-(dataset, gs) score histograms and P/R/F1 curves as PNG into output_dir.",
    )

    p.add_argument("--target_dim", type=int, default=1024)
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument("--lora_target_modules", type=str, default="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2")
    p.add_argument("--pooling_type", type=str, default="transformer")
    p.add_argument("--temperature", type=float, default=0.03)

    p.add_argument("--use_maxsim", action="store_true", default=False)
    p.add_argument("--maxsim_windows", type=str, default="6 10 16 24")
    p.add_argument("--maxsim_stride", type=int, default=2)

    p.add_argument("--text_lora_rank", type=int, default=128)
    p.add_argument("--text_lora_alpha", type=int, default=256)
    p.add_argument("--text_lora_target_modules", type=str, default="query key value dense")
    p.add_argument("--text_pooling", type=str, default="cls")
    p.add_argument("--sparse_weight", type=float, default=0.7)

    args = p.parse_args()

    retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device = build_model(args)

    _log("Phase 1: Dev data (parameter selection)")
    dev_results = run_sweep_on_dataset(
        "dev", args.dev_jsonl, args.wiki_glossary, args.glossary_sizes,
        retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
        args.output_dir,
        abs_tau_values=args.abs_tau_values,
        plot_histograms=args.plot_histograms,
    )
    print_summary("dev", dev_results)

    _log("Phase 2: ACL 6060 (validation)")
    acl_results = run_sweep_on_dataset(
        "acl6060", args.acl_jsonl, args.wiki_glossary, args.glossary_sizes,
        retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
        args.output_dir,
        abs_tau_values=args.abs_tau_values,
        plot_histograms=args.plot_histograms,
    )
    print_summary("acl6060", acl_results)

    _log("All done.")


if __name__ == "__main__":
    main()