#!/usr/bin/env python3
"""
Chunk-level has_term detection sweep.
Three strategies: absolute top1, gap-based (top1 - top10_mean), hybrid.
Select best tau/delta on dev, validate on ACL 6060.
Reports results across multiple glossary sizes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_sweep_maxsim import (
    build_model,
    load_chunks,
    encode_audio_chunks,
    encode_terms,
    compute_sim,
    _log,
    _fmt,
    K_CANDIDATES,
    Chunk,
)


# ======Configuration=====
GLOSSARY_SIZES = [100, 1000, 10000]

# Absolute: has_term if top1 >= tau
ABS_MIN, ABS_MAX, ABS_STEPS = 0.30, 0.85, 56

# Gap: has_term if (top1 - top10_mean) >= delta
GAP_MIN, GAP_MAX, GAP_STEPS = 0.00, 0.25, 51

# Hybrid: has_term if top1 >= tau AND gap >= delta
HYBRID_TAU_VALS = np.arange(0.30, 0.75, 0.05)
HYBRID_DELTA_VALS = np.arange(0.00, 0.20, 0.02)

MIN_RECALL_CONSTRAINT = 0.90
# ======Configuration=====


@dataclass
class ChunkStats:
    has_term: bool
    top1_score: float
    top10_mean: float
    gap: float  # top1 - top10_mean


def _fbeta(p: float, r: float, beta: float = 1.0) -> float:
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def compute_chunk_stats(
    sim: torch.Tensor,
    all_chunks: List[Chunk],
) -> List[ChunkStats]:
    """Compute top1_score, top10_mean, gap for each chunk."""
    sim_np = sim.cpu().numpy()
    result = []
    for i, chunk in enumerate(all_chunks):
        scores_sorted = np.sort(sim_np[i])[::-1]
        top1 = float(scores_sorted[0])
        top10_mean = float(scores_sorted[:min(10, len(scores_sorted))].mean())
        result.append(ChunkStats(
            has_term=chunk.has_term,
            top1_score=top1,
            top10_mean=top10_mean,
            gap=top1 - top10_mean,
        ))
    return result


def evaluate_detection(
    stats: List[ChunkStats],
    predict_fn,
) -> Dict[str, Any]:
    """Binary classification: predict has_term using predict_fn(ChunkStats) -> bool."""
    tp = fp = fn = tn = 0
    for cs in stats:
        pred = predict_fn(cs)
        if cs.has_term and pred:
            tp += 1
        elif cs.has_term and not pred:
            fn += 1
        elif not cs.has_term and pred:
            fp += 1
        else:
            tn += 1

    n_pos = tp + fn
    n_neg = fp + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _fbeta(precision, recall, beta=1.0)
    f2 = _fbeta(precision, recall, beta=2.0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "fpr": fpr,
        "accuracy": acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_pos": n_pos, "n_neg": n_neg,
    }


def sweep_one_gs(
    stats: List[ChunkStats],
    gs_label: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run all three strategy sweeps for one glossary size."""

    results = {"absolute": [], "gap": [], "hybrid": []}

    # Strategy A: absolute top1
    for tau in np.linspace(ABS_MIN, ABS_MAX, ABS_STEPS):
        tau_f = float(tau)
        r = evaluate_detection(stats, lambda cs, t=tau_f: cs.top1_score >= t)
        r["strategy"] = "absolute"
        r["tau"] = tau_f
        r["delta"] = None
        r["gs"] = gs_label
        results["absolute"].append(r)

    # Strategy B: gap-based
    for delta in np.linspace(GAP_MIN, GAP_MAX, GAP_STEPS):
        delta_f = float(delta)
        r = evaluate_detection(stats, lambda cs, d=delta_f: cs.gap >= d)
        r["strategy"] = "gap"
        r["tau"] = None
        r["delta"] = delta_f
        r["gs"] = gs_label
        results["gap"].append(r)

    # Strategy C: hybrid
    for tau in HYBRID_TAU_VALS:
        for delta in HYBRID_DELTA_VALS:
            tau_f, delta_f = float(tau), float(delta)
            r = evaluate_detection(
                stats,
                lambda cs, t=tau_f, d=delta_f: cs.top1_score >= t and cs.gap >= d,
            )
            r["strategy"] = "hybrid"
            r["tau"] = tau_f
            r["delta"] = delta_f
            r["gs"] = gs_label
            results["hybrid"].append(r)

    return results


def select_best(rows: List[Dict[str, Any]], metric: str = "f2") -> Dict[str, Any]:
    """Select best config: highest precision among those with recall >= constraint,
    fallback to highest recall if none meet constraint."""
    meet = [r for r in rows if r["recall"] >= MIN_RECALL_CONSTRAINT]
    if meet:
        return max(meet, key=lambda r: r["precision"])
    return max(rows, key=lambda r: r["recall"])


def format_row(r: Dict[str, Any]) -> str:
    tau_s = f"{r['tau']:.2f}" if r["tau"] is not None else "   -  "
    delta_s = f"{r['delta']:.2f}" if r["delta"] is not None else "   -  "
    meets = "Y" if r["recall"] >= MIN_RECALL_CONSTRAINT else "N"
    return (
        f"{r['strategy']:<10} {tau_s:>7} {delta_s:>7} "
        f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
        f"{r['f1']:>7.4f} {r['f2']:>7.4f} "
        f"{r['fpr']:>7.4f} {r['accuracy']:>7.4f} "
        f"{r['tp']:>5}/{r['n_pos']:<5} {r['fp']:>5}/{r['n_neg']:<5}  R>={MIN_RECALL_CONSTRAINT}:{meets}"
    )


HEADER = (
    f"{'strategy':<10} {'tau':>7} {'delta':>7} "
    f"{'P':>7} {'R':>7} {'F1':>7} {'F2':>7} "
    f"{'FPR':>7} {'Acc':>7} "
    f"{'TP/pos':>11} {'FP/neg':>11}"
)


def run_dataset(
    name: str,
    jsonl_path: str,
    wiki_glossary_path: str,
    retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Returns {gs_label: {strategy_name: best_row}}"""
    _log(f"\n{'='*120}")
    _log(f"Dataset: {name}")
    _log(f"{'='*120}")

    with_term, no_term = load_chunks(jsonl_path)
    all_chunks = with_term + no_term
    _log(f"  with_term={len(with_term)}, no_term={len(no_term)}")

    gt_terms = sorted({t for c in with_term for t in c.gt_terms})
    _log(f"  unique GT terms: {len(gt_terms)}")

    with open(wiki_glossary_path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        wiki_all = [item["term"] if isinstance(item, dict) else str(item) for item in raw]
    else:
        wiki_all = list(raw.keys())
    gt_set = set(gt_terms)
    wiki_filtered = [t for t in wiki_all if t.lower() not in gt_set]

    _log("  Encoding audio...")
    speech_embs = encode_audio_chunks(all_chunks, retriever, feat_ext, device)
    _log(f"  speech_embs: {speech_embs.shape}")

    best_per_gs = {}

    for gs in GLOSSARY_SIZES:
        gs_label = f"gs{gs}"
        n_extra = gs - len(gt_terms)
        if n_extra < 0:
            _log(f"  {gs_label}: skipped (GT={len(gt_terms)} >= gs)")
            continue
        wiki_subset = wiki_filtered[:n_extra]
        term_list = gt_terms + [t.lower() for t in wiki_subset]
        _log(f"\n  --- {gs_label}: {len(term_list)} terms ---")

        text_embs = encode_terms(term_list, text_encoder, tokenizer, device)
        sim = compute_sim(speech_embs, text_embs, _maxsim_score)

        stats = compute_chunk_stats(sim, all_chunks)

        # Log score distributions
        wt = [s for s in stats if s.has_term]
        nt = [s for s in stats if not s.has_term]
        wt_top1 = np.array([s.top1_score for s in wt])
        nt_top1 = np.array([s.top1_score for s in nt])
        wt_gap = np.array([s.gap for s in wt])
        nt_gap = np.array([s.gap for s in nt])

        _log(f"  Score distributions ({gs_label}):")
        _log(f"    has_term  top1: mean={wt_top1.mean():.4f} median={np.median(wt_top1):.4f} std={wt_top1.std():.4f}")
        _log(f"    no_term   top1: mean={nt_top1.mean():.4f} median={np.median(nt_top1):.4f} std={nt_top1.std():.4f}")
        _log(f"    has_term  gap:  mean={wt_gap.mean():.4f} median={np.median(wt_gap):.4f} std={wt_gap.std():.4f}")
        _log(f"    no_term   gap:  mean={nt_gap.mean():.4f} median={np.median(nt_gap):.4f} std={nt_gap.std():.4f}")

        all_results = sweep_one_gs(stats, gs_label)

        _log(f"\n  Best configs ({gs_label}, recall >= {MIN_RECALL_CONSTRAINT}):")
        _log(f"  {HEADER}")
        _log(f"  {'-'*len(HEADER)}")

        best_per_gs[gs_label] = {}
        for strat_name in ["absolute", "gap", "hybrid"]:
            best = select_best(all_results[strat_name])
            best_per_gs[gs_label][strat_name] = best
            _log(f"  {format_row(best)}")

    return best_per_gs


def validate_with_dev_params(
    name: str,
    jsonl_path: str,
    wiki_glossary_path: str,
    dev_best: Dict[str, Dict[str, Dict[str, Any]]],
    retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
):
    """Apply dev-selected params on ACL without re-sweeping."""
    _log(f"\n{'='*120}")
    _log(f"VALIDATION: {name} (using dev-selected params, NO re-sweep)")
    _log(f"{'='*120}")

    with_term, no_term = load_chunks(jsonl_path)
    all_chunks = with_term + no_term
    _log(f"  with_term={len(with_term)}, no_term={len(no_term)}")

    gt_terms = sorted({t for c in with_term for t in c.gt_terms})

    with open(wiki_glossary_path, "r") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        wiki_all = [item["term"] if isinstance(item, dict) else str(item) for item in raw]
    else:
        wiki_all = list(raw.keys())
    gt_set = set(gt_terms)
    wiki_filtered = [t for t in wiki_all if t.lower() not in gt_set]

    _log("  Encoding audio...")
    speech_embs = encode_audio_chunks(all_chunks, retriever, feat_ext, device)

    _log(f"\n  {HEADER}")
    _log(f"  {'-'*len(HEADER)}")

    for gs in GLOSSARY_SIZES:
        gs_label = f"gs{gs}"
        if gs_label not in dev_best:
            continue
        n_extra = gs - len(gt_terms)
        if n_extra < 0:
            continue
        wiki_subset = wiki_filtered[:n_extra]
        term_list = gt_terms + [t.lower() for t in wiki_subset]

        text_embs = encode_terms(term_list, text_encoder, tokenizer, device)
        sim = compute_sim(speech_embs, text_embs, _maxsim_score)
        stats = compute_chunk_stats(sim, all_chunks)

        for strat_name in ["absolute", "gap", "hybrid"]:
            if strat_name not in dev_best[gs_label]:
                continue
            dev_row = dev_best[gs_label][strat_name]
            tau = dev_row["tau"]
            delta = dev_row["delta"]

            if strat_name == "absolute":
                pred_fn = lambda cs, t=tau: cs.top1_score >= t
            elif strat_name == "gap":
                pred_fn = lambda cs, d=delta: cs.gap >= d
            else:
                pred_fn = lambda cs, t=tau, d=delta: cs.top1_score >= t and cs.gap >= d

            r = evaluate_detection(stats, pred_fn)
            r["strategy"] = strat_name
            r["tau"] = tau
            r["delta"] = delta
            r["gs"] = gs_label
            _log(f"  {format_row(r)}")

        _log(f"  {'-'*len(HEADER)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dev_jsonl", type=str, required=True)
    p.add_argument("--acl_jsonl", type=str, required=True)
    p.add_argument("--wiki_glossary", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")

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

    # Phase 1: sweep on dev
    dev_best = run_dataset(
        "Dev", args.dev_jsonl, args.wiki_glossary,
        retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
    )

    # Phase 2: validate on ACL with dev params (NO re-sweep)
    validate_with_dev_params(
        "ACL 6060", args.acl_jsonl, args.wiki_glossary, dev_best,
        retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
    )

    _log("\nAll done.")


if __name__ == "__main__":
    main()
