#!/usr/bin/env python3
"""
Analyze near-variant collision rate in the hard-negative (HN) bank.

Context:
  mine_hard_negatives_per_sample masks HN candidates only by strict term_id
  equality (blake2b-8 hash of the already-lowered term_key).  Terms that
  are string-variants of the anchor GT (plural/singular, punctuation,
  hyphenation, "A " prefix, substring containment) get different hashes
  and can be surfaced as "hard negatives", which under InfoNCE push the
  model to separate clean GT from a near-synonym — a destructive signal
  that scales with K.

  This script quantifies the upper bound on that collision rate by doing
  a string-only analysis on the same train-term bank that
  qwen3_glossary_neg_train.py builds.  No model, no GPU.

Output:
  TSV:  per-anchor top-K stats (K in {5, 64, 256, 1024})
  Summary TSV: collision rates by K / similarity definition
  PNG:  histograms of max-bank-sim for each anchor under each definition

Smoke:
  python analyze_hn_variant_collision.py --n_anchors 10 --out_dir /tmp/smoke_hn_coll

Full:
  python analyze_hn_variant_collision.py --n_anchors 200 \
      --out_dir /mnt/gemini/data2/jiaxuanluo/hn_variant_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("hn_variant_collision")

# ---- Paths (Tier 3 consistent; taurus absolute) ----
DEFAULT_TRAIN_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
DEFAULT_OUT_DIR = "/mnt/gemini/data2/jiaxuanluo/hn_variant_analysis"

K_VALUES = (5, 64, 256, 1024)
# Similarity definitions to audit
NORM_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")


# ---------- Normalization helpers ----------

def norm_strip_lower(t: str) -> str:
    return t.strip().lower()


def norm_punct(t: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    t = t.strip().lower()
    t = NORM_PUNCT_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


def norm_aggressive(t: str) -> str:
    """norm_punct + drop trailing 's' / 'es' / 'ies' on each token."""
    t = norm_punct(t)
    toks = []
    for w in t.split(" "):
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("es") and not w.endswith(("ses", "xes")):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        toks.append(w)
    return " ".join(toks)


def token_set(t: str) -> Set[str]:
    return set(norm_punct(t).split())


def sm_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


# ---------- Loading ----------

@dataclass
class BankIndex:
    terms: List[str]                    # raw (already strip.lower())
    norm_punct_map: Dict[str, List[int]]
    norm_aggr_map: Dict[str, List[int]]
    tokens: List[Set[str]]


def load_train_terms(train_jsonl: str, limit: int, seed: int) -> List[Dict]:
    """Return a random sample of training rows with their term_key."""
    rng = random.Random(seed)
    rows: List[Dict] = []
    n_seen = 0
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            n_seen += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            tk = (d.get("term_key") or "").strip().lower()
            if not tk:
                continue
            rows.append({"term_key": tk, "utter_id": d.get("utter_id", "")})
            if limit > 0 and len(rows) >= max(limit * 50, 200_000):
                # Overshoot then subsample; keeps stream short.
                break
    logger.info(f"[DATA] streamed {n_seen} rows, kept {len(rows)} with non-empty term_key")
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    return rows


def build_bank(train_term_keys: List[str], hn_glossary: str = "") -> BankIndex:
    train_set = sorted(set(train_term_keys) - {""})
    assert train_set, "No train terms to build bank"
    extra_terms: List[str] = []
    if hn_glossary:
        with open(hn_glossary, "r", encoding="utf-8") as f:
            entries = json.load(f)
        assert isinstance(entries, list)
        train_lookup = set(train_set)
        extra_terms = sorted(
            {
                (e.get("term") or "").strip().lower()
                for e in entries
                if isinstance(e, dict)
            }
            - {""}
            - train_lookup
        )
    all_terms = train_set + extra_terms
    logger.info(
        f"[BANK] train_unique={len(train_set)} extra_terms={len(extra_terms)} "
        f"total={len(all_terms)}"
    )
    norm_punct_map: Dict[str, List[int]] = {}
    norm_aggr_map: Dict[str, List[int]] = {}
    tokens: List[Set[str]] = []
    for i, t in enumerate(all_terms):
        np_ = norm_punct(t)
        na_ = norm_aggressive(t)
        norm_punct_map.setdefault(np_, []).append(i)
        norm_aggr_map.setdefault(na_, []).append(i)
        tokens.append(token_set(t))
    return BankIndex(
        terms=all_terms,
        norm_punct_map=norm_punct_map,
        norm_aggr_map=norm_aggr_map,
        tokens=tokens,
    )


# ---------- Collision analysis per anchor ----------

@dataclass
class AnchorReport:
    anchor: str
    gt_bank_idx: int
    exact_collisions: List[int]       # norm_punct equal (excluding GT itself)
    aggr_collisions: List[int]        # norm_aggressive equal (excluding GT itself)
    substr_collisions: List[int]      # bank term contained-in or contains GT
    edit_top: List[Tuple[int, float]]  # (idx, SM ratio), top-1024 by ratio


def analyze_anchor(anchor_term_key: str, bank: BankIndex, edit_topk: int = 1024,
                   edit_scan_cap: int = 20000, rng: random.Random = None
                   ) -> AnchorReport:
    """
    Compute string-level collision classes for one anchor.

    For edit distance: scanning the full bank (O(bank_size) per anchor with
    SM.ratio which is O(len^2)) is ~10-40 ms per anchor for 50-60k bank.
    We cap at edit_scan_cap if bank is larger, uniformly subsampling to keep
    wallclock bounded for smoke runs.
    """
    if rng is None:
        rng = random.Random(0)

    np_anchor = norm_punct(anchor_term_key)
    na_anchor = norm_aggressive(anchor_term_key)
    tok_anchor = token_set(anchor_term_key)

    # Find GT bank index (strict terms equality after strip.lower already applied).
    # Multiple matches theoretically not possible (bank is unique set).
    gt_idx = -1
    # GT uniqueness: the bank keyed by raw term string; anchor is already .strip().lower()
    # bank.terms is sorted-unique, linear scan fine (we only do this once per anchor call
    # but scale is acceptable for <=1k anchors). For speed we could precompute a dict.
    # For smoke: precompute bank term -> idx map outside this function.
    # (Caller passes `bank` each time — we put a cached dict on it lazily.)
    cache = getattr(bank, "_term_to_idx", None)
    if cache is None:
        cache = {t: i for i, t in enumerate(bank.terms)}
        bank._term_to_idx = cache  # type: ignore[attr-defined]
    gt_idx = cache.get(anchor_term_key, -1)

    exact_cols = [
        i for i in bank.norm_punct_map.get(np_anchor, []) if i != gt_idx
    ]
    aggr_cols = [
        i for i in bank.norm_aggr_map.get(na_anchor, []) if i != gt_idx
    ]

    # Substring: O(bank_size), cheap string contain.
    substr_cols: List[int] = []
    for i, bt in enumerate(bank.terms):
        if i == gt_idx:
            continue
        if anchor_term_key in bt or (len(bt) > 3 and bt in anchor_term_key):
            substr_cols.append(i)

    # Edit distance: scan up to edit_scan_cap bank terms; for each compute SM ratio.
    n_bank = len(bank.terms)
    scan_idx = list(range(n_bank))
    if n_bank > edit_scan_cap:
        scan_idx = rng.sample(scan_idx, edit_scan_cap)
        # Always include gt_idx + known collisions so the top-K isn't biased.
        forced = {gt_idx, *exact_cols, *aggr_cols, *substr_cols}
        forced.discard(-1)
        scan_idx = sorted(set(scan_idx) | forced)
    edit_scored: List[Tuple[int, float]] = []
    for i in scan_idx:
        if i == gt_idx:
            continue
        r = sm_ratio(anchor_term_key, bank.terms[i])
        edit_scored.append((i, r))
    edit_scored.sort(key=lambda x: -x[1])
    edit_top = edit_scored[:edit_topk]

    return AnchorReport(
        anchor=anchor_term_key,
        gt_bank_idx=gt_idx,
        exact_collisions=exact_cols,
        aggr_collisions=aggr_cols,
        substr_collisions=substr_cols,
        edit_top=edit_top,
    )


# ---------- Aggregation ----------

def summarize(reports: List[AnchorReport], out_dir: str, bank: BankIndex) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Per-anchor detail TSV
    detail_path = os.path.join(out_dir, "per_anchor_detail.tsv")
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write(
            "anchor\tgt_in_bank\t"
            "n_punct_eq\tn_aggr_eq\tn_substr\t"
            "n_edit_top5_ge_0.80\tn_edit_top64_ge_0.80\t"
            "n_edit_top256_ge_0.80\tn_edit_top1024_ge_0.80\t"
            "n_edit_top5_ge_0.90\tn_edit_top64_ge_0.90\t"
            "n_edit_top256_ge_0.90\tn_edit_top1024_ge_0.90\t"
            "example_top5\n"
        )
        for r in reports:
            gt_in = int(r.gt_bank_idx >= 0)
            def cnt(k, th):
                return sum(1 for _, s in r.edit_top[:k] if s >= th)
            sample_top5 = [f"{bank.terms[i]}({s:.3f})" for i, s in r.edit_top[:5]]
            f.write(
                f"{r.anchor}\t{gt_in}\t"
                f"{len(r.exact_collisions)}\t{len(r.aggr_collisions)}\t"
                f"{len(r.substr_collisions)}\t"
                f"{cnt(5, 0.8)}\t{cnt(64, 0.8)}\t{cnt(256, 0.8)}\t{cnt(1024, 0.8)}\t"
                f"{cnt(5, 0.9)}\t{cnt(64, 0.9)}\t{cnt(256, 0.9)}\t{cnt(1024, 0.9)}\t"
                f"{' | '.join(sample_top5)}\n"
            )
    logger.info(f"[OUT] per-anchor detail -> {detail_path}")

    # Summary TSV
    n = len(reports)
    summary_path = os.path.join(out_dir, "collision_summary.tsv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("metric\tvalue\tn_anchors\tnote\n")
        f.write(f"n_anchors\t{n}\t{n}\tInput anchor count\n")
        gt_in = sum(1 for r in reports if r.gt_bank_idx >= 0)
        f.write(
            f"frac_gt_in_bank\t{gt_in / n:.4f}\t{n}\t"
            f"Anchors whose GT is inside the HN bank (false-neg mask base)\n"
        )
        for metric_name, attr in [
            ("frac_has_punct_eq_variant", "exact_collisions"),
            ("frac_has_aggr_eq_variant", "aggr_collisions"),
            ("frac_has_substr_variant", "substr_collisions"),
        ]:
            x = sum(1 for r in reports if len(getattr(r, attr)) > 0)
            f.write(f"{metric_name}\t{x / n:.4f}\t{n}\tAnchors with >=1 such variant in bank\n")

        # Expected # variants inside top-K given SM-ratio thresholds
        for K in K_VALUES:
            for th in (0.70, 0.80, 0.85, 0.90):
                vals = [
                    sum(1 for _, s in r.edit_top[:K] if s >= th) for r in reports
                ]
                avg = sum(vals) / n
                any_nonzero = sum(1 for v in vals if v > 0) / n
                f.write(
                    f"avg_n_variants_top{K}_sm>={th:.2f}\t{avg:.3f}\t{n}\t"
                    f"Mean count of bank terms within SM-ratio>=th in each anchor's top-{K}\n"
                )
                f.write(
                    f"frac_anchors_top{K}_sm>={th:.2f}_nonzero\t{any_nonzero:.4f}\t{n}\t"
                    f"Fraction of anchors with >=1 such variant in top-{K}\n"
                )
    logger.info(f"[OUT] summary -> {summary_path}")

    # Also emit a short readable report to stdout for the record.
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            print(line.rstrip())


# ---------- Main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", default=DEFAULT_TRAIN_JSONL)
    p.add_argument("--hn_glossary", default="")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--n_anchors", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--edit_scan_cap", type=int, default=20000,
                   help="Subsample bank for SM-ratio scan (per anchor). "
                        "20k anchors * 1 anchor ~ 40ms => 200 anchors ~ 8s.")
    args = p.parse_args()

    assert os.path.isfile(args.train_jsonl), f"Missing --train_jsonl: {args.train_jsonl}"
    if args.hn_glossary:
        assert os.path.isfile(args.hn_glossary), f"Missing --hn_glossary: {args.hn_glossary}"

    logger.info(f"[CFG] n_anchors={args.n_anchors} seed={args.seed} "
                f"edit_scan_cap={args.edit_scan_cap} out_dir={args.out_dir}")

    rows = load_train_terms(args.train_jsonl, args.n_anchors, args.seed)
    assert rows, "No anchor rows sampled"
    anchor_term_keys = [r["term_key"] for r in rows]

    # Bank construction: use ALL unique train term_keys (full corpus scan for completeness).
    # The old external HN glossary path is optional and no longer part of the
    # default training bank.
    logger.info("[BANK] streaming full train_jsonl for unique term_keys ...")
    all_train_terms: Set[str] = set()
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            tk = (d.get("term_key") or "").strip().lower()
            if tk:
                all_train_terms.add(tk)
    logger.info(f"[BANK] n_unique_train_terms={len(all_train_terms)}")

    bank = build_bank(sorted(all_train_terms), args.hn_glossary)

    rng = random.Random(args.seed)
    reports: List[AnchorReport] = []
    import time
    t0 = time.time()
    for i, tk in enumerate(anchor_term_keys):
        rep = analyze_anchor(tk, bank, edit_topk=1024,
                             edit_scan_cap=args.edit_scan_cap, rng=rng)
        reports.append(rep)
        if (i + 1) % max(1, len(anchor_term_keys) // 10) == 0:
            elapsed = time.time() - t0
            logger.info(
                f"[PROG] {i + 1}/{len(anchor_term_keys)} anchors "
                f"(avg {elapsed / (i + 1) * 1000:.1f} ms/anchor)"
            )

    summarize(reports, args.out_dir, bank)

    # Print a few concrete collision examples
    logger.info("[SAMPLE] first 5 anchors with an aggressive-equal collision:")
    shown = 0
    for r in reports:
        if r.aggr_collisions:
            logger.info(
                f"  anchor={r.anchor!r} -> "
                f"bank_variants={[bank.terms[i] for i in r.aggr_collisions[:3]]}"
            )
            shown += 1
            if shown >= 5:
                break
    if shown == 0:
        logger.info("  (none)")


if __name__ == "__main__":
    main()
