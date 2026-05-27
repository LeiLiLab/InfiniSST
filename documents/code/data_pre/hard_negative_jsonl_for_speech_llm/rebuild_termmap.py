#!/usr/bin/env python3
"""
Rebuild training JSONL for Speech LLM with density-parameterized term_map.

Input:  JSONL with `retriever_results_by_chunk` + `chunk_metadata`
        (from generate_termmap_maxsim.py)
Output: JSONL ready for megatron sft (term_map injected into user messages)

Strategy per chunk (unified baseline):
  Case A (has GT terms): GT terms all injected + neg terms fill to cap, shuffled
  Case B (no GT terms):  neg terms fill to cap (all-noise term_map)
  Case C (random empty): EMPTY_PROB chance of clearing term_map entirely

cap = density_coeff * multiplier   (multiplier = ceil(duration / 0.96))

GT terms use chunk-specific zh from gt_terms_by_chunk (NOT glossary zh).
Neg terms use retriever-returned zh.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple  # noqa: F401

# ======Configuration=====
SEED = 42
UNIT_DURATION_SEC = 0.96
DEFAULT_DENSITY_COEFF = 5.0

EMPTY_PROB_NO_GT = 0.50
EMPTY_PROB_HAS_GT = 0.15

SYSTEM_PROMPTS = {
    "zh": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate the audio into Chinese text. "
        "Use the 'term_map' as a reference for terminology if provided."
    ),
    "de": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate the audio into German text. "
        "Use the 'term_map' as a reference for terminology if provided."
    ),
    "ja": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate the audio into Japanese text. "
        "Use the 'term_map' as a reference for terminology if provided."
    ),
}
# ======Configuration=====


def _get_cap(multiplier: int, density_coeff: float, max_terms: int = 0,
             no_gt_max_terms: int = 0, has_gt: bool = True) -> int:
    cap = max(1, round(density_coeff * multiplier))
    if max_terms > 0:
        cap = min(cap, max_terms)
    if (not has_gt) and no_gt_max_terms > 0:
        cap = min(cap, no_gt_max_terms)
    return cap


def _format_term_map(terms: List[Dict]) -> str:
    if not terms:
        return ""
    lines = ["term_map:"]
    seen = set()
    for t in terms:
        key = t["term"].lower()
        if key in seen:
            continue
        seen.add(key)
        term_str = t.get("term", "")
        zh_str = t.get("zh", "")
        if term_str and zh_str:
            lines.append(f"{term_str}={zh_str}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def build_chunk_termmap(
    gt_terms: List[Dict],
    retriever_results: List[Dict],
    multiplier: int,
    density_coeff: float,
    rng: random.Random,
    max_terms: int = 0,
    no_gt_max_terms: int = 0,
    empty_prob_no_gt: Optional[float] = None,
    empty_prob_has_gt: Optional[float] = None,
) -> str:
    """Build term_map string for one chunk using unified baseline strategy.

    Case A: has GT → inject all GT + fill neg to cap
    Case B: no GT but has retriever results → fill neg to cap (all-noise)
    Case C: random empty → return "" with EMPTY_PROB

    Phase 6 knobs (default behavior unchanged when left at defaults):
      - no_gt_max_terms: when > 0, caps the term_map size on no-GT chunks to
        this value (independent of density_coeff * multiplier). Used to
        reduce training-time noise on no-GT chunks without affecting has-GT.
      - empty_prob_no_gt: overrides EMPTY_PROB_NO_GT when not None. Used to
        make no-GT chunks more often carry empty term_maps (teach the LLM
        that term_map can be safely ignored when it's noisy).
      - empty_prob_has_gt: overrides EMPTY_PROB_HAS_GT when not None.
    """
    gt_keys = {t["term"].lower() for t in gt_terms}
    has_gt = len(gt_keys) > 0

    if has_gt:
        empty_prob = empty_prob_has_gt if empty_prob_has_gt is not None else EMPTY_PROB_HAS_GT
    else:
        empty_prob = empty_prob_no_gt if empty_prob_no_gt is not None else EMPTY_PROB_NO_GT
    if rng.random() < empty_prob:
        return ""

    cap = _get_cap(multiplier, density_coeff, max_terms)
    if not has_gt and no_gt_max_terms > 0:
        cap = min(cap, no_gt_max_terms)

    gt_formatted = [{"term": t["term"], "zh": t["zh"]} for t in gt_terms]

    neg_formatted = []
    for r in retriever_results:
        if r["term"].lower() not in gt_keys:
            neg_formatted.append({"term": r["term"], "zh": r["zh"]})

    neg_count = max(0, cap - len(gt_formatted))
    selected_neg = neg_formatted[:neg_count]
    combined = gt_formatted + selected_neg

    rng.shuffle(combined)
    return _format_term_map(combined)


def _term_key(term: str) -> str:
    return str(term or "").strip().casefold()


def _format_candidate(item: Dict) -> Optional[Dict]:
    term = str(item.get("term") or "").strip()
    zh = str(item.get("zh") or item.get("translation") or "").strip()
    if not term or not zh:
        return None
    return {"term": term, "zh": zh}


def build_tcm_filtered_with_gt_backfill(
    gt_terms: List[Dict],
    retriever_results: List[Dict],
    rng: random.Random,
    max_terms: int = 0,
) -> str:
    """Use tau-filtered retriever order, while forcing GT translations.

    If the retriever already returned a GT term, keep its retrieved position but
    replace the translation with the chunk-specific ``gt_terms_by_chunk`` value.
    This avoids training the SLM on glossary/wiki translations that disagree
    with the reference-aligned GT translation.

    When ``max_terms`` is set, the final map is capped after GT backfill.  GT
    terms keep priority and retriever-only negatives fill the remaining slots.
    """
    combined: List[Dict] = []
    seen = set()
    gt_by_key = {}
    for gt in gt_terms:
        formatted_gt = _format_candidate(gt)
        if formatted_gt is None:
            continue
        key = _term_key(formatted_gt["term"])
        if key:
            gt_by_key[key] = formatted_gt

    for item in retriever_results:
        formatted = _format_candidate(item)
        if formatted is None:
            continue
        key = _term_key(formatted["term"])
        if not key or key in seen:
            continue
        seen.add(key)
        if key in gt_by_key:
            formatted = gt_by_key[key]
        combined.append(formatted)

    for gt in gt_terms:
        formatted_gt = _format_candidate(gt)
        if formatted_gt is None:
            continue
        key = _term_key(formatted_gt["term"])
        if not key or key in seen:
            continue
        insert_at = rng.randint(0, len(combined))
        combined.insert(insert_at, formatted_gt)
        seen.add(key)

    if max_terms > 0 and len(combined) > max_terms:
        capped: List[Dict] = []
        capped_keys = set()

        # First keep GT terms in their current order/positions as much as the
        # hard cap allows; this reduces negatives on GT-rich chunks.
        for item in combined:
            key = _term_key(item.get("term", ""))
            if key and key in gt_by_key and key not in capped_keys:
                capped.append(item)
                capped_keys.add(key)
                if len(capped) >= max_terms:
                    break

        # Fill any remaining capacity with earliest retriever-only negatives.
        if len(capped) < max_terms:
            for item in combined:
                key = _term_key(item.get("term", ""))
                if not key or key in capped_keys:
                    continue
                capped.append(item)
                capped_keys.add(key)
                if len(capped) >= max_terms:
                    break

        combined = capped

    return _format_term_map(combined)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True,
                        help="JSONL with retriever_results_by_chunk")
    parser.add_argument("--output_jsonl", required=True,
                        help="Output JSONL for megatron sft")
    parser.add_argument("--density_coeff", type=float, default=DEFAULT_DENSITY_COEFF,
                        help="terms per 0.96s unit (default: 5.0)")
    parser.add_argument("--termmap_mode", choices=["legacy_density", "tcm_filtered_with_gt_backfill"],
                        default="legacy_density",
                        help="legacy_density keeps the historical density/noise behavior; "
                             "tcm_filtered_with_gt_backfill uses retriever order plus missing-GT random insertion.")
    parser.add_argument("--max_terms", type=int, default=0,
                        help="hard cap on total terms per chunk (0 = no cap)")
    parser.add_argument("--no_gt_max_terms", type=int, default=0,
                        help="hard cap on term_map size for no-GT chunks only "
                             "(0 = use same cap as has-GT). Phase 6 Sub-problem B fix.")
    parser.add_argument("--empty_prob_no_gt", type=float, default=-1.0,
                        help=f"override EMPTY_PROB_NO_GT (default {EMPTY_PROB_NO_GT}). "
                             "Negative value = keep default. Phase 6 Sub-problem B fix.")
    parser.add_argument("--empty_prob_has_gt", type=float, default=-1.0,
                        help=f"override EMPTY_PROB_HAS_GT (default {EMPTY_PROB_HAS_GT}). "
                             "Negative value = keep default.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--target_lang",
        choices=sorted(SYSTEM_PROMPTS),
        default="zh",
        help="Target language for the system prompt. Term-map values still use the backward-compatible 'zh' field internally.",
    )
    parser.add_argument("--max_conversations", type=int, default=0,
                        help="0 = all; >0 = limit for smoke test")
    args = parser.parse_args()

    density_coeff = args.density_coeff
    max_terms = args.max_terms
    no_gt_max_terms = args.no_gt_max_terms
    empty_prob_no_gt_override = args.empty_prob_no_gt if args.empty_prob_no_gt >= 0 else None
    empty_prob_has_gt_override = args.empty_prob_has_gt if args.empty_prob_has_gt >= 0 else None
    assert density_coeff > 0, f"density_coeff must be positive, got {density_coeff}"
    assert max_terms >= 0, f"max_terms must be non-negative, got {max_terms}"
    assert no_gt_max_terms >= 0, f"no_gt_max_terms must be non-negative, got {no_gt_max_terms}"
    if empty_prob_no_gt_override is not None:
        assert 0.0 <= empty_prob_no_gt_override <= 1.0, (
            f"empty_prob_no_gt must be in [0,1], got {empty_prob_no_gt_override}"
        )
    if empty_prob_has_gt_override is not None:
        assert 0.0 <= empty_prob_has_gt_override <= 1.0, (
            f"empty_prob_has_gt must be in [0,1], got {empty_prob_has_gt_override}"
        )

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    stats = {
        "total_convs": 0,
        "total_chunks": 0,
        "chunks_with_termmap": 0,
        "chunks_empty_termmap": 0,
        "chunks_with_gt": 0,
        "chunks_no_gt_with_neg": 0,
        "chunks_no_gt_no_neg": 0,
        "gt_in_topk": 0,
        "gt_total": 0,
        "chunks_gt_over_max_terms": 0,
    }
    termmap_sizes = []
    cap_hist: Dict[int, int] = {}

    effective_empty_no_gt = (
        empty_prob_no_gt_override if empty_prob_no_gt_override is not None else EMPTY_PROB_NO_GT
    )
    effective_empty_has_gt = (
        empty_prob_has_gt_override if empty_prob_has_gt_override is not None else EMPTY_PROB_HAS_GT
    )
    print(f"=== Rebuild Config ===", flush=True)
    print(f"  termmap_mode = {args.termmap_mode}", flush=True)
    print(f"  density_coeff = {density_coeff}", flush=True)
    print(f"  max_terms = {max_terms} ({'no cap' if max_terms == 0 else f'hard cap at {max_terms}'})", flush=True)
    print(f"  no_gt_max_terms = {no_gt_max_terms} ({'no override' if no_gt_max_terms == 0 else f'no-GT cap at {no_gt_max_terms}'})", flush=True)
    print(f"  EMPTY_PROB_NO_GT = {effective_empty_no_gt}" +
          (f"  (override of default {EMPTY_PROB_NO_GT})" if empty_prob_no_gt_override is not None else ""),
          flush=True)
    print(f"  EMPTY_PROB_HAS_GT = {effective_empty_has_gt}" +
          (f"  (override of default {EMPTY_PROB_HAS_GT})" if empty_prob_has_gt_override is not None else ""),
          flush=True)
    print(f"  seed = {args.seed}", flush=True)
    print(f"  target_lang = {args.target_lang}", flush=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as f_in, \
         open(args.output_jsonl, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError:
                continue

            if 0 < args.max_conversations <= stats["total_convs"]:
                break

            gt_by_chunk = conv.get("gt_terms_by_chunk", [])
            ret_by_chunk = conv.get("retriever_results_by_chunk", [])
            chunk_metadata = conv.get("chunk_metadata", [])
            fallback_multiplier = conv.get("merge_multiplier", 1)
            messages = conv.get("messages", [])

            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = SYSTEM_PROMPTS[args.target_lang]

            user_idx = 0
            for msg_idx, msg in enumerate(messages):
                if msg.get("role") != "user" or "<audio>" not in (msg.get("content") or ""):
                    continue

                chunk_idx = user_idx
                user_idx += 1

                gt_terms = gt_by_chunk[chunk_idx] if chunk_idx < len(gt_by_chunk) else []
                ret_results = ret_by_chunk[chunk_idx] if chunk_idx < len(ret_by_chunk) else []

                if chunk_idx < len(chunk_metadata):
                    multiplier = chunk_metadata[chunk_idx].get("multiplier", fallback_multiplier)
                else:
                    multiplier = fallback_multiplier

                cap = _get_cap(multiplier, density_coeff, max_terms)
                cap_hist[cap] = cap_hist.get(cap, 0) + 1

                stats["total_chunks"] += 1
                if gt_terms:
                    stats["chunks_with_gt"] += 1
                    stats["gt_total"] += len(gt_terms)
                    gt_keys_lower = {t["term"].lower() for t in gt_terms}
                    for r in ret_results:
                        if r["term"].lower() in gt_keys_lower:
                            stats["gt_in_topk"] += 1
                elif ret_results:
                    stats["chunks_no_gt_with_neg"] += 1
                else:
                    stats["chunks_no_gt_no_neg"] += 1

                if args.termmap_mode == "tcm_filtered_with_gt_backfill":
                    if max_terms > 0 and len(gt_terms) > max_terms:
                        stats["chunks_gt_over_max_terms"] += 1
                    termmap_str = build_tcm_filtered_with_gt_backfill(
                        gt_terms, ret_results, rng, max_terms=max_terms
                    )
                else:
                    termmap_str = build_chunk_termmap(
                        gt_terms, ret_results, multiplier, density_coeff, rng,
                        max_terms=max_terms,
                        no_gt_max_terms=no_gt_max_terms,
                        empty_prob_no_gt=empty_prob_no_gt_override,
                        empty_prob_has_gt=empty_prob_has_gt_override,
                    )

                if termmap_str:
                    msg["content"] = f"<audio>\n\n{termmap_str}"
                    stats["chunks_with_termmap"] += 1
                    n_terms = termmap_str.count("\n")
                    termmap_sizes.append(n_terms)
                else:
                    msg["content"] = "<audio>"
                    stats["chunks_empty_termmap"] += 1

            for field in ("retriever_results_by_chunk", "chunk_metadata"):
                conv.pop(field, None)

            f_out.write(json.dumps(conv, ensure_ascii=False) + "\n")
            stats["total_convs"] += 1

    if termmap_sizes:
        stats["avg_termmap_size"] = sum(termmap_sizes) / len(termmap_sizes)
    else:
        stats["avg_termmap_size"] = 0.0

    tc = max(1, stats["total_chunks"])
    print("=== Rebuild Stats ===", flush=True)
    print(f"density_coeff: {density_coeff}", flush=True)
    print(f"Total conversations: {stats['total_convs']}", flush=True)
    print(f"Total chunks: {stats['total_chunks']}", flush=True)
    print(f"  with GT terms:         {stats['chunks_with_gt']:>6d} ({stats['chunks_with_gt']/tc*100:.1f}%)", flush=True)
    print(f"  no GT, has neg:        {stats['chunks_no_gt_with_neg']:>6d} ({stats['chunks_no_gt_with_neg']/tc*100:.1f}%)", flush=True)
    print(f"  no GT, no neg:         {stats['chunks_no_gt_no_neg']:>6d} ({stats['chunks_no_gt_no_neg']/tc*100:.1f}%)", flush=True)
    print(f"  with term_map:         {stats['chunks_with_termmap']:>6d} ({stats['chunks_with_termmap']/tc*100:.1f}%)", flush=True)
    print(f"  empty term_map:        {stats['chunks_empty_termmap']:>6d} ({stats['chunks_empty_termmap']/tc*100:.1f}%)", flush=True)
    print(f"  avg term_map size:     {stats['avg_termmap_size']:.1f}", flush=True)
    gt_recall = stats["gt_in_topk"] / max(1, stats["gt_total"])
    print(f"GT recall in retriever: {stats['gt_in_topk']}/{stats['gt_total']} ({gt_recall*100:.1f}%)", flush=True)
    print(f"Cap distribution:", flush=True)
    for cap_val in sorted(cap_hist.keys()):
        print(f"  cap={cap_val:>3d}: {cap_hist[cap_val]:>6d} chunks", flush=True)
    print(f"Output: {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
