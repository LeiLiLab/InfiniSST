#!/usr/bin/env python3
"""
Validate dev-selected threshold strategies on ACL 6060 with FIXED parameters.
No re-sweeping - just evaluate three strategies at the dev-chosen operating points.

Reports gs100, gs1000, gs10000.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse infrastructure from the sweep script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_sweep_maxsim import (
    build_model,
    load_chunks,
    encode_audio_chunks,
    encode_terms,
    compute_sim,
    precompute_topk,
    _log,
    _fmt,
    _fbeta,
    K_CANDIDATES,
)

# ======Configuration=====
ACL_JSONL = "/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
WIKI_GLOSSARY = "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
GLOSSARY_SIZES = [100, 1000, 10000]

# Dev-selected parameters (FIXED, no re-sweeping)
STRATEGIES = [
    {"name": "Absolute",  "tau": 0.65,  "delta": None},
    {"name": "Gap",       "tau": None,  "delta": 0.09},
    {"name": "Hybrid",    "tau": 0.20,  "delta": 0.08},
]
# ======Configuration=====


def evaluate_fixed(chunk_data, total_valid_pos, tau, delta):
    """Evaluate with fixed tau and/or delta."""
    k = K_CANDIDATES
    tp = 0
    pred_total = 0
    noterm_recalled = []

    for cd in chunk_data:
        pred_set = set()
        for j in range(min(k, len(cd.top_k_scores))):
            score = cd.top_k_scores[j]
            keep = True
            if tau is not None:
                keep = keep and (score >= tau)
            if delta is not None:
                keep = keep and (score >= cd.top10_mean + delta)
            if keep:
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
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "avg_noise": avg_noise,
        "tp": tp, "fp": fp, "fn": fn,
        "with_term_chunks": n_with,
        "no_term_chunks": n_no,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
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

    _log("Loading ACL 6060 data...")
    with_term, no_term = load_chunks(ACL_JSONL)
    all_chunks = with_term + no_term
    _log(f"with_term={len(with_term)}, no_term={len(no_term)}")

    gt_terms = sorted({t for c in with_term for t in c.gt_terms})
    _log(f"unique GT terms: {len(gt_terms)}")

    with open(WIKI_GLOSSARY, "r") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        wiki_all = [item["term"] if isinstance(item, dict) else str(item) for item in raw]
    else:
        wiki_all = list(raw.keys())
    gt_set = set(gt_terms)
    wiki_filtered = [t for t in wiki_all if t.lower() not in gt_set]

    _log("Encoding audio...")
    speech_embs = encode_audio_chunks(all_chunks, retriever, feat_ext, device)
    _log(f"speech_embs: {speech_embs.shape}")

    # Table header
    header = (
        f"{'gs':>8} | {'strategy':<12} | {'tau':>6} | {'delta':>6} | "
        f"{'P':>8} | {'R':>8} | {'F1':>8} | {'F2':>8} | "
        f"{'noise':>6} | {'TP':>5} | {'FP':>5} | {'FN':>5}"
    )
    sep = "-" * len(header)

    _log(f"\n{sep}")
    _log("ACL 6060 Validation (dev-selected fixed parameters)")
    _log(sep)
    _log(header)
    _log(sep)

    for gs in GLOSSARY_SIZES:
        n_extra = gs - len(gt_terms)
        if n_extra < 0:
            _log(f"gs{gs}: skipped (GT={len(gt_terms)} >= gs)")
            continue

        wiki_subset = wiki_filtered[:n_extra]
        term_list = gt_terms + [t.lower() for t in wiki_subset]

        text_embs = encode_terms(term_list, text_encoder, tokenizer, device)
        sim = compute_sim(speech_embs, text_embs, _maxsim_score)

        term_to_idx = {t: i for i, t in enumerate(term_list)}
        chunk_data, total_valid_pos = precompute_topk(sim, all_chunks, term_to_idx)

        for strat in STRATEGIES:
            r = evaluate_fixed(chunk_data, total_valid_pos, strat["tau"], strat["delta"])
            tau_s = f"{strat['tau']:.2f}" if strat["tau"] is not None else "  -   "
            delta_s = f"{strat['delta']:.2f}" if strat["delta"] is not None else "  -   "
            _log(
                f"  gs{gs:>5} | {strat['name']:<12} | {tau_s:>6} | {delta_s:>6} | "
                f"{r['precision']:>8.4f} | {r['recall']:>8.4f} | {r['f1']:>8.4f} | {r['f2']:>8.4f} | "
                f"{r['avg_noise']:>6.2f} | {r['tp']:>5} | {r['fp']:>5} | {r['fn']:>5}"
            )
        _log(sep)

    _log("Done.")


if __name__ == "__main__":
    main()
