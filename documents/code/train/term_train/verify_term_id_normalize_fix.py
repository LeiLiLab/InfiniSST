#!/usr/bin/env python3
"""
Verify the term_id normalization fix for HN near-variant collision.

Compares three modes of `stable_term_id()`:
  - "none"         (legacy; bit-for-bit with pre-fix ckpts)
  - "lower_strip"
  - "aggressive"   (the proposed fix)

Methodology
-----------
1. Reuse the same train-term bank construction as analyze_hn_variant_collision.py.
2. Reuse the same 200 anchor sample (fixed seed=0).
3. For each anchor, enumerate the bank candidates that are aggressive-equal
   to the anchor's GT term ("near-variants").  These are the rows that
   analyze_hn_variant_collision.py surfaced as problematic (50.5% of anchors
   have at least one).
4. Under each mode, compute:
     - stable_term_id(anchor) == stable_term_id(variant)  for each (anchor, variant) pair
     - fraction of variants that would be MASKED as positive/false-neg
       (i.e. would NOT be treated as hard negatives) under that mode
5. Also simulate the two downstream masks used in `compute_masked_contrastive_loss`:
     - `gt_match` in `mine_hard_negatives_per_sample` (excludes same-id candidates
       from the per-sample top-K)
     - `fn_hn` / `fn_mask` in the loss (drops same-id cells from the softmax denom)

Expected outcome if the fix works:
     none:        0% of near-variants get masked (every one becomes a HN)
     lower_strip: ~0%  (both already strip.lower())
     aggressive:  100% of aggressive-equal near-variants get masked

Usage:
  python verify_term_id_normalize_fix.py \
    --train_jsonl /mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl \
    --n_anchors 200 \
    --out_tsv /mnt/gemini/data2/jiaxuanluo/hn_variant_analysis/normalize_fix_verification.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List, Set, Tuple

# Make the train script importable so we use the REAL `stable_term_id` +
# `set_term_id_normalize_mode`, not a re-implementation.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# NOTE: we avoid importing qwen3_glossary_neg_train directly (that pulls in
# torch/transformers and starts a ~1-minute import tree on GPU nodes).  The
# normalization logic is small and pure-Python; we mirror it here from the
# train script so the TWO definitions must stay in lockstep. If you edit
# _normalize_term_for_id() in qwen3_glossary_neg_train.py, MIRROR the change
# here.  A cross-check on the demo_pairs below will surface any drift.
import hashlib
import re

INVALID_ID_SENTINEL = 0
SIGNED_INT64_MASK = (1 << 63) - 1
TERM_ID_NORMALIZE_MODES = ("none", "lower_strip", "aggressive")

_TERM_ID_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_TERM_ID_MULTISPACE_RE = re.compile(r"\s+")
_MODE = "none"


def set_term_id_normalize_mode(mode: str) -> None:
    assert mode in TERM_ID_NORMALIZE_MODES
    global _MODE
    _MODE = mode


def _normalize_term_for_id(term_text: str) -> str:
    if _MODE == "none":
        return term_text
    t = term_text.strip().lower()
    if _MODE == "lower_strip":
        return t
    t = _TERM_ID_PUNCT_RE.sub(" ", t)
    t = _TERM_ID_MULTISPACE_RE.sub(" ", t).strip()
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


def stable_term_id(term_text: str) -> int:
    if not term_text:
        return INVALID_ID_SENTINEL
    norm = _normalize_term_for_id(term_text)
    if not norm:
        return INVALID_ID_SENTINEL
    digest = hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest()
    tid = int.from_bytes(digest, "little", signed=False) & SIGNED_INT64_MASK
    return tid if tid != INVALID_ID_SENTINEL else 1


from analyze_hn_variant_collision import (  # noqa: E402
    norm_aggressive,
    norm_punct,
    build_bank,
    load_train_terms,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("verify_normalize_fix")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_jsonl",
        default="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl",
    )
    p.add_argument(
        "--hn_glossary",
        default="",
    )
    p.add_argument("--n_anchors", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out_tsv",
        default=(
            "/mnt/gemini/data2/jiaxuanluo/hn_variant_analysis/"
            "normalize_fix_verification.tsv"
        ),
    )
    args = p.parse_args()

    assert os.path.isfile(args.train_jsonl), f"Missing train: {args.train_jsonl}"
    if args.hn_glossary:
        assert os.path.isfile(args.hn_glossary), f"Missing glossary: {args.hn_glossary}"

    # ---- Load anchors + bank (identical logic to analyze_hn_variant_collision) ----
    rows = load_train_terms(args.train_jsonl, args.n_anchors, args.seed)
    anchor_term_keys = [r["term_key"] for r in rows]

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

    # ---- For each anchor, collect its aggressive-equal near-variants from bank ----
    anchor_variants: Dict[str, List[str]] = {}
    anchors_with_variants = 0
    total_variant_pairs = 0
    for anchor in anchor_term_keys:
        na = norm_aggressive(anchor)
        cand_idx = bank.norm_aggr_map.get(na, [])
        # Drop the anchor itself (by raw terms equality, since bank.terms is unique).
        variants = [bank.terms[i] for i in cand_idx if bank.terms[i] != anchor]
        anchor_variants[anchor] = variants
        if variants:
            anchors_with_variants += 1
            total_variant_pairs += len(variants)
    logger.info(
        f"[DATA] {anchors_with_variants}/{len(anchor_term_keys)} anchors have "
        f">=1 aggressive-equal bank variant  "
        f"(total (anchor,variant) pairs={total_variant_pairs})"
    )

    # ---- Run each mode and count "collapse to same term_id" rate ----
    results: Dict[str, Dict[str, float]] = {}
    for mode in TERM_ID_NORMALIZE_MODES:
        set_term_id_normalize_mode(mode)

        # Sanity: map a few well-known pairs to show collision behavior.
        demo_pairs = [
            ("proposition", "propositions"),
            ("length", "lengths"),
            ("easiest way", "easiest ways"),
            ("health", "healths"),
            ("neural network", "neural-network"),
            ("n-gram", "n gram"),
        ]
        logger.info(f"[MODE={mode}] demo pairs:")
        for a, b in demo_pairs:
            eq = stable_term_id(a) == stable_term_id(b)
            logger.info(f"  id({a!r}) == id({b!r}) ? {eq}")

        collapse = 0
        total = 0
        # Also aggregate by how many *distinct* anchors would now see
        # all their near-variants removed from the HN bank.
        anchors_fully_fixed = 0
        anchors_partial = 0
        for anchor, variants in anchor_variants.items():
            if not variants:
                continue
            aid = stable_term_id(anchor)
            eq_flags = [stable_term_id(v) == aid for v in variants]
            total += len(variants)
            collapse += sum(eq_flags)
            if all(eq_flags):
                anchors_fully_fixed += 1
            elif any(eq_flags):
                anchors_partial += 1

        n_with_variants = sum(1 for v in anchor_variants.values() if v)
        results[mode] = {
            "n_variant_pairs": total,
            "n_collapsed": collapse,
            "frac_collapsed": (collapse / total) if total else 0.0,
            "n_anchors_with_variants": n_with_variants,
            "n_anchors_fully_fixed": anchors_fully_fixed,
            "n_anchors_partial": anchors_partial,
            "frac_anchors_fully_fixed": (
                anchors_fully_fixed / n_with_variants if n_with_variants else 0.0
            ),
        }

    # ---- Write TSV ----
    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    with open(args.out_tsv, "w", encoding="utf-8") as f:
        f.write(
            "mode\tn_variant_pairs\tn_collapsed_to_same_id\tfrac_collapsed"
            "\tn_anchors_with_variants\tn_anchors_fully_fixed"
            "\tn_anchors_partial\tfrac_anchors_fully_fixed\n"
        )
        for mode in TERM_ID_NORMALIZE_MODES:
            r = results[mode]
            f.write(
                f"{mode}\t{r['n_variant_pairs']}\t{r['n_collapsed']}"
                f"\t{r['frac_collapsed']:.4f}"
                f"\t{r['n_anchors_with_variants']}\t{r['n_anchors_fully_fixed']}"
                f"\t{r['n_anchors_partial']}\t{r['frac_anchors_fully_fixed']:.4f}\n"
            )
    logger.info(f"[OUT] {args.out_tsv}")

    # ---- Console summary ----
    print()
    print("=" * 80)
    print(f"Verification: {args.n_anchors} anchors, bank size={len(bank.terms)}")
    print("-" * 80)
    print(
        f"{'mode':<14}  {'pairs':>7}  {'collapsed':>10}  {'frac':>6}  "
        f"{'anchors_fixed':>14}  {'frac_anchors':>12}"
    )
    for mode in TERM_ID_NORMALIZE_MODES:
        r = results[mode]
        print(
            f"{mode:<14}  {r['n_variant_pairs']:>7}  "
            f"{r['n_collapsed']:>10}  {r['frac_collapsed']:>6.2%}  "
            f"{r['n_anchors_fully_fixed']:>14}  "
            f"{r['frac_anchors_fully_fixed']:>12.2%}"
        )
    print("=" * 80)
    print(
        "Interpretation: higher 'frac_collapsed' = more near-variants that\n"
        "would have been (erroneously) surfaced as hard negatives are now\n"
        "correctly treated as positives / false-negatives by the existing\n"
        "gt_match + fn_mask paths.  'aggressive' should reach ~100% by\n"
        "construction; 'none' is the pre-fix baseline (0%)."
    )


if __name__ == "__main__":
    main()
