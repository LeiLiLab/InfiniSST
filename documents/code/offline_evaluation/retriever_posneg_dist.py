#!/usr/bin/env python3
"""Pos/Neg similarity-distribution diagnostic for TCM threshold tuning.

Unlike `threshold_sweep_maxsim.py`, this script's SINGLE purpose is to
produce an honest pos-vs-neg sim distribution for a given retriever
checkpoint, so we can pick TCM_NEG_THRESHOLD (alpha) and
TCM_POS_THRESHOLD (beta) off the data instead of guessing.

Two key differences from the hist in threshold_sweep_maxsim.py:

1. S_pos is UNBIASED: we compute cos-sim(chunk, GT term) directly for
   every (has_term chunk, GT term) pair. The threshold_sweep hist only
   collects positives that made it into top-10, so it silently drops
   the (1 - recall@10)-fraction of hardest positives -- exactly the
   ones TCM beta is supposed to penalize.

2. S_neg is collected up to `--topk_neg` (default 50), not just 10.
   The right tail of S_neg is where OOD noise lives. 10 is enough to
   see the max, but 50 gives the full 90-99th-percentile shape we need
   to pick alpha.

We reuse the model/encoding helpers from threshold_sweep_maxsim.py so
this stays a thin wrapper.

Outputs (per domain, `gs_pod`/`gs_you`/`gs_aud`/`wiki_synth`/`acl6060`):

  <out_dir>/raw_sims.npz         -- S_pos, S_neg_top{1,5,10,K}, domain tags
  <out_dir>/posneg_percentiles.tsv
  <out_dir>/hist_posneg_<domain>.png
  <out_dir>/hist_posneg_global.png
  <out_dir>/alpha_beta_suggest.txt  -- actionable TCM thresholds from percentiles
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from threshold_sweep_maxsim import (
    Chunk,
    build_model,
    compute_sim,
    encode_audio_chunks,
    encode_terms,
    load_chunks,
)


def _log(msg: str) -> None:
    print(f"[POSNEG] {msg}", flush=True)


def _domain_of(chunk_id: str, source: str) -> str:
    if source == "acl":
        return "acl6060"
    uid = (chunk_id or "").split("::")[0]
    if uid.startswith("wiki_synth_"):
        return "wiki_synth"
    if uid.startswith("POD"):
        return "gs_pod"
    if uid.startswith("YOU"):
        return "gs_you"
    if uid.startswith("AUD"):
        return "gs_aud"
    return "gs_other"


def _load_wiki_terms(path: str) -> List[str]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        terms = [item["term"] if isinstance(item, dict) else str(item) for item in raw]
    else:
        assert isinstance(raw, dict), f"Unexpected glossary format: {type(raw)}"
        terms = list(raw.keys())
    seen = set()
    out: List[str] = []
    for t in terms:
        k = t.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


@dataclass
class PerChunkSims:
    chunk_idx: int
    domain: str
    has_term: bool
    s_pos_list: List[float]              # len = # GT terms this chunk has (usually 1)
    s_neg_top_k: np.ndarray              # [topk_neg]


def _collect_sims(
    sim_matrix: torch.Tensor,           # [N_chunks, N_terms]
    chunks: List[Chunk],
    gt_idx_per_chunk: List[List[int]],  # mapped GT term indices per chunk
    topk_neg: int,
    source: str,
) -> List[PerChunkSims]:
    """Compute unbiased S_pos and top-K S_neg per chunk."""
    sim_np = sim_matrix.cpu().numpy()
    N_chunks, N_terms = sim_np.shape
    k_neg = min(topk_neg, N_terms)

    # Masked copy for negative top-K: mask out GT indices per chunk with -inf
    sim_neg = sim_np.copy()
    for i, gt_idxs in enumerate(gt_idx_per_chunk):
        for g in gt_idxs:
            sim_neg[i, g] = -np.inf

    top_k_idx = np.argpartition(-sim_neg, k_neg, axis=1)[:, :k_neg]
    top_k_sco = np.take_along_axis(sim_neg, top_k_idx, axis=1)
    order = np.argsort(-top_k_sco, axis=1)
    top_k_sco = np.take_along_axis(top_k_sco, order, axis=1)

    out: List[PerChunkSims] = []
    for i, c in enumerate(chunks):
        gt_idxs = gt_idx_per_chunk[i]
        s_pos = [float(sim_np[i, g]) for g in gt_idxs]
        out.append(PerChunkSims(
            chunk_idx=i,
            domain=_domain_of(c.chunk_id, source),
            has_term=c.has_term,
            s_pos_list=s_pos,
            s_neg_top_k=top_k_sco[i].astype(np.float32),
        ))
    return out


def _run_on_dataset(
    dataset_name: str,
    jsonl_path: str,
    gt_terms_extra: List[str],
    wiki_terms: List[str],
    gs_size: int,
    retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
    topk_neg: int,
    source_tag: str,
) -> Tuple[List[PerChunkSims], int]:
    _log(f"=== {dataset_name} ({source_tag}) ===")
    with_term, no_term = load_chunks(jsonl_path)
    all_chunks = with_term + no_term
    _log(f"  with_term={len(with_term)}  no_term={len(no_term)}  total={len(all_chunks)}")

    gt_terms = sorted({t for c in with_term for t in c.gt_terms})
    gt_set_lower = set(t.lower() for t in gt_terms)

    # Bank: GT ∪ wiki filler (up to gs_size)
    bank = list(gt_terms)
    if gs_size > 0:
        already_in = set(t.lower() for t in bank)
        for t in wiki_terms:
            if t.lower() in already_in or t.lower() in gt_set_lower:
                continue
            bank.append(t.lower())
            already_in.add(t.lower())
            if len(bank) >= gs_size:
                break
    _log(f"  bank size = {len(bank)}  (GT={len(gt_terms)})")

    t0 = time.time()
    _log("  Encoding audio ...")
    speech_embs = encode_audio_chunks(all_chunks, retriever, feat_ext, device)
    _log(f"  speech_embs={tuple(speech_embs.shape)} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    _log("  Encoding bank ...")
    text_embs = encode_terms(bank, text_encoder, tokenizer, device)
    _log(f"  text_embs={tuple(text_embs.shape)} ({time.time() - t0:.1f}s)")

    sim = compute_sim(speech_embs, text_embs, _maxsim_score)
    _log(f"  sim matrix = {tuple(sim.shape)}")

    term_to_idx = {t: i for i, t in enumerate(bank)}
    gt_idx_per_chunk: List[List[int]] = []
    for c in all_chunks:
        idxs = [term_to_idx[t] for t in c.gt_terms if t in term_to_idx]
        gt_idx_per_chunk.append(idxs)

    per_chunk = _collect_sims(sim, all_chunks, gt_idx_per_chunk, topk_neg, source_tag)
    return per_chunk, len(bank)


def _aggregate(
    per_chunk_all: List[PerChunkSims],
    topk_neg: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Bucket by domain. Keys: 'S_pos', 'S_neg_top1', 'S_neg_top5', 'S_neg_top10',
    'S_neg_pure_top1' (from no_term chunks), 'S_neg_all_topk' (flat top-K)."""
    buckets: Dict[str, Dict[str, list]] = {}

    def _b(d: str) -> Dict[str, list]:
        if d not in buckets:
            buckets[d] = {
                "S_pos": [],
                "S_neg_top1": [],
                "S_neg_top5": [],
                "S_neg_top10": [],
                "S_neg_pure_top1": [],
                "S_neg_all_topk": [],
            }
        return buckets[d]

    for rec in per_chunk_all:
        d = rec.domain
        b = _b(d)
        kneg = rec.s_neg_top_k
        k = kneg.shape[0]
        if rec.has_term:
            b["S_pos"].extend(rec.s_pos_list)
            b["S_neg_top1"].append(float(kneg[0]))
            if k >= 5:
                b["S_neg_top5"].append(float(kneg[4]))
            if k >= 10:
                b["S_neg_top10"].append(float(kneg[9]))
        else:
            b["S_neg_pure_top1"].append(float(kneg[0]))
        b["S_neg_all_topk"].extend([float(x) for x in kneg])

    return {
        d: {k: np.asarray(v, dtype=np.float32) for k, v in kinds.items()}
        for d, kinds in buckets.items()
    }


def _percentile_block(x: np.ndarray, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> Dict[str, float]:
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                **{f"p{p}": float("nan") for p in ps}}
    out: Dict[str, float] = {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }
    for p in ps:
        out[f"p{p}"] = float(np.percentile(x, p))
    return out


def _write_percentiles_tsv(
    out_path: str, records: Dict[str, Dict[str, np.ndarray]]
) -> None:
    ps = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    cols = ["domain", "kind", "n", "mean", "std"] + [f"p{p}" for p in ps]
    lines = ["\t".join(cols)]
    for dom in sorted(records.keys()):
        for kind in ["S_pos", "S_neg_top1", "S_neg_top5", "S_neg_top10",
                     "S_neg_pure_top1", "S_neg_all_topk"]:
            arr = records[dom].get(kind, np.asarray([], dtype=np.float32))
            st = _percentile_block(arr, ps)
            row = [dom, kind, str(st["n"]), f"{st['mean']:.4f}", f"{st['std']:.4f}"]
            row += [f"{st[f'p{p}']:.4f}" for p in ps]
            lines.append("\t".join(row))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _log(f"  Wrote: {out_path}")


def _plot_hist_posneg(
    out_dir: str, records: Dict[str, Dict[str, np.ndarray]],
    current_alpha: float, current_beta: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _log("  matplotlib unavailable; skipping plots.")
        return

    bins = np.linspace(-0.05, 1.0, 64)

    for dom in sorted(records.keys()):
        arrs = records[dom]
        fig, ax = plt.subplots(figsize=(10, 5))
        pos = arrs.get("S_pos", np.asarray([]))
        neg1 = arrs.get("S_neg_top1", np.asarray([]))
        negp = arrs.get("S_neg_pure_top1", np.asarray([]))
        if pos.size:
            ax.hist(pos, bins=bins, alpha=0.55, label=f"S_pos (n={pos.size})",
                    color="#2a9d8f", density=True)
            p10_pos = float(np.percentile(pos, 10))
            ax.axvline(p10_pos, color="#2a9d8f", linestyle=":", linewidth=1.2,
                       label=f"S_pos p10 = {p10_pos:.3f}")
        if neg1.size:
            ax.hist(neg1, bins=bins, alpha=0.55,
                    label=f"S_neg_top1 in-term (n={neg1.size})",
                    color="#e76f51", density=True)
            p90_neg = float(np.percentile(neg1, 90))
            p95_neg = float(np.percentile(neg1, 95))
            ax.axvline(p90_neg, color="#e76f51", linestyle=":", linewidth=1.2,
                       label=f"S_neg_top1 p90 = {p90_neg:.3f}")
            ax.axvline(p95_neg, color="#e76f51", linestyle="--", linewidth=1.0,
                       label=f"S_neg_top1 p95 = {p95_neg:.3f}")
        if negp.size:
            ax.hist(negp, bins=bins, alpha=0.35,
                    label=f"S_neg_pure_top1 no-term (n={negp.size})",
                    color="#f4a261", density=True)
        # Current TCM thresholds (from A2 / launcher config)
        ax.axvline(current_alpha, color="black", linestyle="-", linewidth=0.9, alpha=0.55,
                   label=f"current alpha = {current_alpha:.2f}")
        ax.axvline(current_beta, color="gray", linestyle="-", linewidth=0.9, alpha=0.55,
                   label=f"current beta = {current_beta:.2f}")
        ax.set_xlabel("cos-sim")
        ax.set_ylabel("density")
        ax.set_title(f"{dom}: S_pos vs S_neg (TCM threshold diagnostic)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"hist_posneg_{dom}.png")
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        _log(f"  Wrote: {out_png}")

    # Global 3-panel: S_pos / S_neg_top1 / S_neg_pure_top1
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    kinds = [("S_pos", "green"), ("S_neg_top1", "red"), ("S_neg_pure_top1", "orange")]
    for ax, (kind, _color) in zip(axes, kinds):
        for dom in sorted(records.keys()):
            arr = records[dom].get(kind)
            if arr is None or arr.size == 0:
                continue
            ax.hist(arr, bins=bins, alpha=0.5,
                    label=f"{dom} (n={arr.size})", density=True)
        ax.set_title(kind)
        ax.set_xlabel("cos-sim")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("density")
    fig.suptitle("Pos/Neg sim distribution across domains")
    fig.tight_layout()
    out_png = os.path.join(out_dir, "hist_posneg_global.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    _log(f"  Wrote: {out_png}")


def _suggest_alpha_beta(
    out_dir: str,
    records: Dict[str, Dict[str, np.ndarray]],
    current_alpha: float, current_beta: float,
) -> None:
    """Pick OOD-robust alpha/beta: conservative across all domains."""
    lines: List[str] = []
    lines.append(f"Current TCM config: alpha={current_alpha:.3f}  beta={current_beta:.3f}")
    lines.append("")

    ps_neg = (75, 90, 95, 99)
    ps_pos = (1, 5, 10, 25)

    per_dom: List[Tuple[str, Dict[str, float], Dict[str, float]]] = []
    for dom in sorted(records.keys()):
        pos = records[dom].get("S_pos", np.asarray([]))
        neg1 = records[dom].get("S_neg_top1", np.asarray([]))
        if pos.size == 0 and neg1.size == 0:
            continue
        pos_p = {f"p{p}": float(np.percentile(pos, p)) if pos.size else float("nan") for p in ps_pos}
        neg_p = {f"p{p}": float(np.percentile(neg1, p)) if neg1.size else float("nan") for p in ps_neg}
        per_dom.append((dom, pos_p, neg_p))

    lines.append("Per-domain percentiles (S_pos left tail, S_neg_top1 right tail):")
    lines.append("-" * 90)
    pos_hdr = "  ".join(f"pos.p{p:<2}".rjust(7) for p in ps_pos)
    neg_hdr = "  ".join(f"neg.p{p:<2}".rjust(7) for p in ps_neg)
    lines.append(f"{'domain':<14} {pos_hdr}  || {neg_hdr}")
    for dom, pos_p, neg_p in per_dom:
        pos_str = "  ".join(f"{pos_p[f'p{p}']:>7.3f}" for p in ps_pos)
        neg_str = "  ".join(f"{neg_p[f'p{p}']:>7.3f}" for p in ps_neg)
        lines.append(f"{dom:<14} {pos_str}  || {neg_str}")
    lines.append("")

    # OOD-robust picks: worst-domain p10(S_pos) for beta, worst-domain p90/p95(S_neg) for alpha
    def _agg(kind_ps: List[Tuple[str, Dict[str, float]]], p: int, op) -> float:
        vals = [d[f"p{p}"] for _name, d in kind_ps if not np.isnan(d[f"p{p}"])]
        return op(vals) if vals else float("nan")

    pos_by_dom = [(n, d) for n, d, _ in per_dom]
    neg_by_dom = [(n, d) for n, _, d in per_dom]

    beta_p10_min = _agg(pos_by_dom, 10, min)
    beta_p05_min = _agg(pos_by_dom, 5, min)
    alpha_p90_max = _agg(neg_by_dom, 90, max)
    alpha_p95_max = _agg(neg_by_dom, 95, max)

    lines.append("OOD-robust suggestions (worst domain dominates):")
    lines.append(f"  beta  (TCM_POS_THRESHOLD): ")
    lines.append(f"    beta=min_domain(p10 S_pos)  -> penalizes bottom ~10% positives per domain: {beta_p10_min:.3f}")
    lines.append(f"    beta=min_domain(p05 S_pos)  -> penalizes bottom ~5%  positives per domain: {beta_p05_min:.3f}")
    lines.append(f"  alpha (TCM_NEG_THRESHOLD): ")
    lines.append(f"    alpha=max_domain(p90 S_neg_top1) -> penalizes top ~10% hardest negs per domain: {alpha_p90_max:.3f}")
    lines.append(f"    alpha=max_domain(p95 S_neg_top1) -> penalizes top ~5%  hardest negs per domain: {alpha_p95_max:.3f}")
    lines.append("")
    lines.append("Reading guide:")
    lines.append("  - If current alpha << max_domain(p90 S_neg), TCM is penalizing >10% of negs = strong constant pressure.")
    lines.append("  - If current alpha >> max_domain(p95 S_neg), TCM barely touches negs = ornamental.")
    lines.append("  - Symmetric reasoning for beta vs p10 S_pos.")

    out_txt = os.path.join(out_dir, "alpha_beta_suggest.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _log(f"  Wrote: {out_txt}")
    _log("  --- alpha_beta_suggest.txt ---")
    for L in lines:
        _log(f"    {L}")


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device = build_model(args)

    wiki_terms = _load_wiki_terms(args.wiki_glossary) if args.wiki_glossary else []
    _log(f"wiki glossary terms: {len(wiki_terms)}")

    per_chunk_all: List[PerChunkSims] = []

    per_chunk_dev, bank_dev = _run_on_dataset(
        "dev", args.dev_jsonl, [], wiki_terms, args.gs_size,
        retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
        args.topk_neg, source_tag="gs",
    )
    per_chunk_all.extend(per_chunk_dev)
    _log(f"dev bank={bank_dev}  collected={len(per_chunk_dev)} chunks")

    if args.acl_jsonl and os.path.isfile(args.acl_jsonl):
        per_chunk_acl, bank_acl = _run_on_dataset(
            "acl6060", args.acl_jsonl, [], wiki_terms, args.gs_size,
            retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device,
            args.topk_neg, source_tag="acl",
        )
        per_chunk_all.extend(per_chunk_acl)
        _log(f"acl bank={bank_acl}  collected={len(per_chunk_acl)} chunks")

    records = _aggregate(per_chunk_all, args.topk_neg)
    _log(f"domains: {sorted(records.keys())}")

    npz_path = os.path.join(args.output_dir, "raw_sims.npz")
    flat: Dict[str, np.ndarray] = {}
    for dom, kinds in records.items():
        for k, v in kinds.items():
            flat[f"{dom}__{k}"] = v
    np.savez_compressed(npz_path, **flat)
    _log(f"Saved raw npz: {npz_path}")

    _write_percentiles_tsv(
        os.path.join(args.output_dir, "posneg_percentiles.tsv"),
        records,
    )
    _plot_hist_posneg(
        args.output_dir, records,
        current_alpha=args.current_alpha, current_beta=args.current_beta,
    )
    _suggest_alpha_beta(
        args.output_dir, records,
        current_alpha=args.current_alpha, current_beta=args.current_beta,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dev_jsonl", type=str, required=True)
    p.add_argument("--acl_jsonl", type=str, default="",
                   help="Optional: ACL6060 dev jsonl for OOD view.")
    p.add_argument("--wiki_glossary", type=str, default="",
                   help="Wiki glossary JSON for distractor padding.")
    p.add_argument("--gs_size", type=int, default=10000,
                   help="Max bank size (GT ∪ wiki, up to this).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--topk_neg", type=int, default=50,
                   help="Top-K negatives to record per chunk for right-tail stats.")
    p.add_argument("--current_alpha", type=float, default=0.40,
                   help="TCM_NEG_THRESHOLD used in current training (for plot reference).")
    p.add_argument("--current_beta", type=float, default=0.70,
                   help="TCM_POS_THRESHOLD used in current training (for plot reference).")

    # Model / encoder hparams (must match the checkpoint's training recipe).
    p.add_argument("--target_dim", type=int, default=1024)
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2")
    p.add_argument("--pooling_type", type=str, default="transformer")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--use_maxsim", action="store_true", default=False)
    p.add_argument("--maxsim_windows", type=str, default="2 3 4 5 6 7 8 10 12 16 20 24")
    p.add_argument("--maxsim_stride", type=int, default=2)
    p.add_argument("--text_lora_rank", type=int, default=128)
    p.add_argument("--text_lora_alpha", type=int, default=256)
    p.add_argument("--text_lora_target_modules", type=str, default="query key value dense")
    p.add_argument("--text_pooling", type=str, default="cls")
    p.add_argument("--sparse_weight", type=float, default=0.0)

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
