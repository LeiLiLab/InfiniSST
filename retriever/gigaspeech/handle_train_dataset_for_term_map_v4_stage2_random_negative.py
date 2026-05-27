#!/usr/bin/env python3
"""
Stage 2 (v4 Random Negatives): build term_map by mixing GT terms with random negatives.

Input JSONL schema (per line) is expected to contain:
  - "messages": list[{"role": "...", "content": "..."}]
  - "audios": list[str]
  - "gt_terms_by_chunk": list[list[{"term": str, "zh": str}]]

For each audio turn (a user message with content "<audio>"), we:
  1) Take GT terms for this chunk (dedup by lowercased term).
  2) Sample n ~ Uniform{min_neg, ..., max_neg} negatives from the GLOBAL glossary:
       glossary = all deduplicated GT terms in the training set
     excluding the current chunk's GT set.
  3) Shuffle (GT + negatives) and inject as:
       "<audio>\n\nterm_map:\nSRC=ZH\n..."

This is used to test the effect of different negative-term strategies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


def _term_key(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _stable_int(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _build_global_glossary(input_jsonl: str) -> Dict[str, Tuple[str, str]]:
    """
    Return: term_key -> (term_surface, zh_translation)
    """
    out: Dict[str, Tuple[str, str]] = {}
    total = 0
    kept = 0
    for rec in _iter_jsonl(input_jsonl):
        gt_chunks = rec.get("gt_terms_by_chunk") or []
        for chunk in gt_chunks:
            for it in chunk or []:
                total += 1
                term = (it.get("term") or "").strip()
                zh = (it.get("zh") or "").strip()
                if not term or not zh:
                    continue
                k = _term_key(term)
                if not k:
                    continue
                if k not in out:
                    out[k] = (term, zh)
                    kept += 1
    logger.info("Global glossary built: %d unique terms (from %d GT items)", kept, total)
    return out


def _format_term_map(pairs: List[Tuple[str, str]]) -> str:
    if not pairs:
        return ""
    lines = ["term_map:"]
    for s, t in pairs:
        ss = (s or "").replace("\n", " ").strip()
        tt = (t or "").replace("\n", " ").strip()
        if not ss or not tt:
            continue
        lines.append(f"{ss}={tt}")
    return "\n".join(lines)


def _sample_negatives(
    rng: random.Random,
    glossary_keys: List[str],
    glossary_map: Dict[str, Tuple[str, str]],
    exclude: Set[str],
    n: int,
    max_attempts: int = 2000,
) -> List[Tuple[str, str]]:
    if n <= 0 or not glossary_keys:
        return []
    out: List[Tuple[str, str]] = []
    seen = set(exclude)
    attempts = 0
    # Rejection sampling (exclude set is small; glossary can be huge).
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        k = glossary_keys[rng.randrange(0, len(glossary_keys))]
        if k in seen:
            continue
        v = glossary_map.get(k)
        if not v:
            continue
        out.append(v)
        seen.add(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Inject random negatives into term_map for each chunk.")
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_neg", type=int, default=0)
    ap.add_argument("--max_neg", type=int, default=9)
    ap.add_argument("--max_messages", type=int, default=0, help="If >0, only process first N records")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--total_shards", type=int, default=1)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.total_shards < 1 or args.shard_id < 0 or args.shard_id >= args.total_shards:
        raise SystemExit(f"Invalid sharding: shard_id={args.shard_id} total_shards={args.total_shards}")
    if args.min_neg < 0 or args.max_neg < args.min_neg:
        raise SystemExit(f"Invalid neg range: [{args.min_neg}, {args.max_neg}]")

    logger.info("Building global glossary from: %s", args.input_jsonl)
    glossary_map = _build_global_glossary(args.input_jsonl)
    glossary_keys = sorted(glossary_map.keys())
    logger.info("Glossary keys ready: %d", len(glossary_keys))

    processed = 0
    wrote = 0
    skipped = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, rec in enumerate(_iter_jsonl(args.input_jsonl)):
            if args.total_shards > 1 and (idx % args.total_shards) != args.shard_id:
                continue
            if args.max_messages and args.max_messages > 0 and processed >= args.max_messages:
                break

            processed += 1
            messages = rec.get("messages") or []
            audios = rec.get("audios") or []
            gt_chunks = rec.get("gt_terms_by_chunk") or []

            # Build a new messages list with term_map injected for each audio turn.
            new_messages = []
            audio_turn_idx = 0

            # Use utter_id if available for deterministic sampling; otherwise fall back to index.
            uid = str(rec.get("utter_id") or rec.get("id") or idx)

            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if role == "user" and isinstance(content, str) and content.strip() == "<audio>":
                    if audio_turn_idx >= len(audios):
                        new_messages.append(m)
                        audio_turn_idx += 1
                        continue

                    gt_list = gt_chunks[audio_turn_idx] if audio_turn_idx < len(gt_chunks) else []

                    gt_pairs: List[Tuple[str, str]] = []
                    gt_keys: Set[str] = set()
                    for it in gt_list or []:
                        s = (it.get("term") or "").strip()
                        t = (it.get("zh") or "").strip()
                        if not s or not t:
                            continue
                        k = _term_key(s)
                        if not k or k in gt_keys:
                            continue
                        gt_keys.add(k)
                        gt_pairs.append((s, t))

                    if not gt_pairs:
                        # No GT -> keep as is (no term_map).
                        new_messages.append(m)
                        audio_turn_idx += 1
                        continue

                    # Deterministic RNG per (uid, turn) for reproducibility across sharding.
                    rng = random.Random(args.seed + _stable_int(f"{uid}::{audio_turn_idx}"))
                    n_neg = rng.randint(int(args.min_neg), int(args.max_neg))
                    neg_pairs = _sample_negatives(
                        rng=rng,
                        glossary_keys=glossary_keys,
                        glossary_map=glossary_map,
                        exclude=gt_keys,
                        n=n_neg,
                    )

                    final_pairs = list(gt_pairs) + list(neg_pairs)
                    rng.shuffle(final_pairs)

                    # Dedup once more by term key
                    seen_final: Set[str] = set()
                    deduped: List[Tuple[str, str]] = []
                    for s, t in final_pairs:
                        k = _term_key(s)
                        if not k or k in seen_final:
                            continue
                        seen_final.add(k)
                        deduped.append((s, t))

                    term_map_str = _format_term_map(deduped)
                    new_messages.append({"role": "user", "content": f"<audio>\n\n{term_map_str}"})
                    audio_turn_idx += 1
                else:
                    new_messages.append(m)

            out_rec = dict(rec)
            out_rec["messages"] = new_messages
            # Keep audios as-is; keep other metadata for debugging.
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            wrote += 1

    logger.info(
        "Done. processed=%d wrote=%d skipped=%d output=%s",
        processed,
        wrote,
        skipped,
        args.output_jsonl,
    )


if __name__ == "__main__":
    main()









