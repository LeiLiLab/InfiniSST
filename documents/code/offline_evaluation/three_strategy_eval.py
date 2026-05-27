#!/usr/bin/env python3
"""
Evaluate three retrieval strategies on ACL 6060:
  1. Baseline: always inject top-K terms (fixed K)
  2. Chunk gate (Gap): inject top-K only if gap >= delta, else inject nothing
  3. Term filter (Absolute): inject terms with sim >= tau (variable length)

Also runs strategy selection on Dev for strategies 2 & 3.
Reports: precision, recall, F1, F2, avg_terms_injected, noise (no-term avg)
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
    Chunk,
)

# ======Configuration=====
GLOSSARY_SIZES = [100, 1000, 10000]
TOP_K = 10

# Gap sweep range (for chunk gate)
GAP_MIN, GAP_MAX, GAP_STEPS = 0.02, 0.25, 47

# Absolute sweep range (for term filter)
ABS_MIN, ABS_MAX, ABS_STEPS = 0.30, 0.85, 56

# Selection criteria
MIN_RECALL_CONSTRAINT = 0.85
# ======Configuration=====


def _fbeta(p: float, r: float, beta: float = 1.0) -> float:
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


@dataclass
class ChunkEvalData:
    has_term: bool
    pos_indices: Set[int]
    top_k_indices: np.ndarray
    top_k_scores: np.ndarray
    top1_score: float
    top10_mean: float
    gap: float


def precompute(
    sim: torch.Tensor,
    all_chunks: List[Chunk],
    term_to_idx: Dict[str, int],
) -> Tuple[List[ChunkEvalData], int]:
    sim_np = sim.cpu().numpy()
    result = []
    total_valid_pos = 0

    for i, chunk in enumerate(all_chunks):
        scores = sim_np[i]
        top_indices = np.argsort(scores)[::-1][:TOP_K]
        top_scores = scores[top_indices]

        top1 = float(top_scores[0])
        top10_mean = float(top_scores[:min(10, len(top_scores))].mean())

        pos_idx = set()
        if chunk.has_term:
            for gt in chunk.gt_terms:
                gt_lower = gt.lower()
                if gt_lower in term_to_idx:
                    pos_idx.add(term_to_idx[gt_lower])
            total_valid_pos += len(pos_idx)

        result.append(ChunkEvalData(
            has_term=chunk.has_term,
            pos_indices=pos_idx,
            top_k_indices=top_indices,
            top_k_scores=top_scores,
            top1_score=top1,
            top10_mean=top10_mean,
            gap=top1 - top10_mean,
        ))

    return result, total_valid_pos


def eval_strategy(
    chunks: List[ChunkEvalData],
    total_valid_pos: int,
    emit_fn,
) -> Dict[str, Any]:
    """
    emit_fn(chunk: ChunkEvalData) -> Set[int]: returns set of term indices to inject
    """
    tp = fp = 0
    terms_injected_has = []
    terms_injected_no = []

    for cd in chunks:
        emitted = emit_fn(cd)

        if cd.has_term:
            tp += len(emitted & cd.pos_indices)
            fp += len(emitted - cd.pos_indices)
            terms_injected_has.append(len(emitted))
        else:
            terms_injected_no.append(len(emitted))

    fn = total_valid_pos - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _fbeta(precision, recall, 1.0)
    f2 = _fbeta(precision, recall, 2.0)

    avg_inject_has = float(np.mean(terms_injected_has)) if terms_injected_has else 0.0
    avg_inject_no = float(np.mean(terms_injected_no)) if terms_injected_no else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "tp": tp, "fp": fp, "fn": fn,
        "avg_terms_has": avg_inject_has,
        "avg_terms_no": avg_inject_no,
    }


def baseline_emit(cd: ChunkEvalData) -> Set[int]:
    return set(int(x) for x in cd.top_k_indices[:TOP_K])


def make_chunk_gate_emit(delta: float):
    def emit(cd: ChunkEvalData) -> Set[int]:
        if cd.gap >= delta:
            return set(int(x) for x in cd.top_k_indices[:TOP_K])
        return set()
    return emit


def make_term_filter_emit(tau: float):
    def emit(cd: ChunkEvalData) -> Set[int]:
        result = set()
        for j in range(min(TOP_K, len(cd.top_k_scores))):
            if cd.top_k_scores[j] >= tau:
                result.add(int(cd.top_k_indices[j]))
        return result
    return emit


def sweep_gap(chunks, total_valid_pos):
    """Sweep gap delta, return all results."""
    results = []
    for delta in np.linspace(GAP_MIN, GAP_MAX, GAP_STEPS):
        delta_f = float(delta)
        r = eval_strategy(chunks, total_valid_pos, make_chunk_gate_emit(delta_f))
        r["delta"] = delta_f
        results.append(r)
    return results


def sweep_abs(chunks, total_valid_pos):
    """Sweep absolute tau, return all results."""
    results = []
    for tau in np.linspace(ABS_MIN, ABS_MAX, ABS_STEPS):
        tau_f = float(tau)
        r = eval_strategy(chunks, total_valid_pos, make_term_filter_emit(tau_f))
        r["tau"] = tau_f
        results.append(r)
    return results


def select_best(rows, key_name):
    """Best precision where recall >= constraint; fallback: highest recall."""
    meet = [r for r in rows if r["recall"] >= MIN_RECALL_CONSTRAINT]
    if meet:
        return max(meet, key=lambda r: r["precision"])
    return max(rows, key=lambda r: r["recall"])


def fmt(x):
    return f"{x:.4f}"


HEADER = (
    f"{'strategy':<20} | {'param':>12} | {'P':>7} | {'R':>7} | "
    f"{'F1':>7} | {'F2':>7} | {'avg_inj_has':>11} | {'avg_inj_no':>10}"
)


def print_row(name, param_str, r):
    _log(
        f"{name:<20} | {param_str:>12} | {fmt(r['precision']):>7} | {fmt(r['recall']):>7} | "
        f"{fmt(r['f1']):>7} | {fmt(r['f2']):>7} | "
        f"{r['avg_terms_has']:>11.2f} | {r['avg_terms_no']:>10.2f}"
    )


def run_on_dataset(
    name, jsonl_path, wiki_glossary_path,
    retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
    fixed_gap_delta=None, fixed_abs_tau=None,
):
    """
    If fixed_gap_delta/fixed_abs_tau are None: sweep and select best (dev mode).
    If provided: just evaluate with those params (validation mode).
    Returns (best_gap_delta, best_abs_tau) for dev mode.
    """
    _log(f"\n{'='*110}")
    _log(f"  {name}")
    _log(f"{'='*110}")

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

    best_deltas = {}
    best_taus = {}

    for gs in GLOSSARY_SIZES:
        gs_label = f"gs{gs}"
        n_extra = gs - len(gt_terms)
        if n_extra < 0:
            _log(f"\n  {gs_label}: skipped (GT={len(gt_terms)} > gs)")
            continue

        wiki_subset = wiki_filtered[:n_extra]
        term_list = gt_terms + [t.lower() for t in wiki_subset]
        _log(f"\n  --- {gs_label}: {len(term_list)} terms ---")

        text_embs = encode_terms(term_list, text_encoder, tokenizer, device)
        sim = compute_sim(speech_embs, text_embs, _maxsim_score)
        chunks, total_vp = precompute(sim, all_chunks, {t: i for i, t in enumerate(term_list)})

        # Score distribution
        wt_gap = np.array([c.gap for c in chunks if c.has_term])
        nt_gap = np.array([c.gap for c in chunks if not c.has_term])
        wt_top1 = np.array([c.top1_score for c in chunks if c.has_term])
        nt_top1 = np.array([c.top1_score for c in chunks if not c.has_term])
        _log(f"  has_term top1: mean={wt_top1.mean():.4f}, gap: mean={wt_gap.mean():.4f}")
        _log(f"  no_term  top1: mean={nt_top1.mean():.4f}, gap: mean={nt_gap.mean():.4f}")

        _log(f"\n  {HEADER}")
        _log(f"  {'-'*len(HEADER)}")

        # 1. Baseline
        bl = eval_strategy(chunks, total_vp, baseline_emit)
        print_row("1.Baseline(top10)", "K=10", bl)

        # 2. Chunk gate (Gap)
        if fixed_gap_delta is not None:
            delta = fixed_gap_delta.get(gs_label, fixed_gap_delta.get("default"))
            cg = eval_strategy(chunks, total_vp, make_chunk_gate_emit(delta))
            print_row("2.ChunkGate(gap)", f"d={delta:.2f}", cg)
        else:
            gap_results = sweep_gap(chunks, total_vp)
            best_g = select_best(gap_results, "delta")
            delta = best_g["delta"]
            best_deltas[gs_label] = delta
            print_row("2.ChunkGate(gap)", f"d={delta:.2f}", best_g)

        # 3. Term filter (Absolute)
        if fixed_abs_tau is not None:
            tau = fixed_abs_tau.get(gs_label, fixed_abs_tau.get("default"))
            tf = eval_strategy(chunks, total_vp, make_term_filter_emit(tau))
            print_row("3.TermFilter(abs)", f"t={tau:.2f}", tf)
        else:
            abs_results = sweep_abs(chunks, total_vp)
            best_a = select_best(abs_results, "tau")
            tau = best_a["tau"]
            best_taus[gs_label] = tau
            print_row("3.TermFilter(abs)", f"t={tau:.2f}", best_a)

        _log(f"  {'-'*len(HEADER)}")

    return best_deltas, best_taus


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

    # Phase 1: Dev sweep (parameter selection)
    _log("Phase 1: Dev (parameter selection)")
    best_deltas, best_taus = run_on_dataset(
        "Dev (sweep)", args.dev_jsonl, args.wiki_glossary,
        retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
    )

    _log(f"\n  Dev-selected parameters:")
    _log(f"    Chunk gate (gap) deltas: {best_deltas}")
    _log(f"    Term filter (abs) taus:  {best_taus}")

    # Phase 2: ACL validation (fixed params from dev)
    # For gs sizes not available in dev, use gs10000 params as default
    default_delta = best_deltas.get("gs10000", list(best_deltas.values())[0] if best_deltas else 0.10)
    default_tau = best_taus.get("gs10000", list(best_taus.values())[0] if best_taus else 0.65)

    gap_params = {**best_deltas, "default": default_delta}
    abs_params = {**best_taus, "default": default_tau}

    _log("\nPhase 2: ACL 6060 (validation with dev params)")
    run_on_dataset(
        "ACL 6060 (validate)", args.acl_jsonl, args.wiki_glossary,
        retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
        fixed_gap_delta=gap_params,
        fixed_abs_tau=abs_params,
    )

    _log("\nDone.")


if __name__ == "__main__":
    main()
