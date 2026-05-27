#!/usr/bin/env python3
"""
Build dev-eval wiki glossaries from P31-ranked terms that were not used for
training.

The training recipe uses `wiki_rank < 1_000_000`.  This script samples from
`rank >= min_untrained_rank`, excludes dev GT terms and known inference
glossaries, then writes JSON lists compatible with
qwen3_glossary_neg_train.py --eval_wiki_glossary.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Iterable, List, Set


# ======Configuration=====
DEFAULT_RANKED_TERMS = (
    "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/p31_full/"
    "wiki_synth_terms_p31_ranked.jsonl"
)
DEFAULT_DEV_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev"
DEFAULT_INFERENCE_GLOSSARIES = [
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/"
    "documents/code/data_pre/glossary_scale/wiki_glossary_medicine.json",
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/"
    "documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json",
]
DEFAULT_SIZES = [10_000, 1_000_000]
DEFAULT_MIN_UNTRAINED_RANK = 1_000_000
DEFAULT_SEED = 42
# ======Configuration=====


def _term_key(text: str) -> str:
    return (text or "").strip().lower()


def load_json_glossary_terms(paths: Iterable[str]) -> Set[str]:
    terms: Set[str] = set()
    for path in paths:
        if not path:
            continue
        assert os.path.isfile(path), f"Glossary not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), f"Expected list in {path}, got {type(data)}"
        for item in data:
            if isinstance(item, dict):
                key = _term_key(item.get("term", ""))
            else:
                key = _term_key(str(item))
            if key:
                terms.add(key)
    return terms


def load_dev_terms(path: str) -> Set[str]:
    terms: Set[str] = set()
    if not path:
        return terms
    assert os.path.isfile(path), f"Dev JSONL not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("term_key", "term_text", "term"):
                term = _term_key(obj.get(key, ""))
                if term:
                    terms.add(term)
    return terms


def reservoir_sample_ranked_terms(
    *,
    ranked_terms_path: str,
    blocked_terms: Set[str],
    min_untrained_rank: int,
    sample_size: int,
    seed: int,
) -> tuple[List[Dict], Dict[str, int]]:
    rng = random.Random(seed)
    reservoir: List[Dict] = []
    seen_terms: Set[str] = set()
    stats = {
        "total_rows": 0,
        "eligible_rank_rows": 0,
        "blocked": 0,
        "duplicate": 0,
        "sampled_from": 0,
    }

    assert os.path.isfile(ranked_terms_path), f"Ranked terms not found: {ranked_terms_path}"
    with open(ranked_terms_path, "r", encoding="utf-8") as f:
        for line in f:
            stats["total_rows"] += 1
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rank = int(obj.get("rank", stats["total_rows"] - 1))
            if rank < min_untrained_rank:
                continue
            stats["eligible_rank_rows"] += 1

            term = (obj.get("term") or "").strip()
            key = _term_key(obj.get("term_key", term))
            if not term or not key:
                continue
            if key in blocked_terms:
                stats["blocked"] += 1
                continue
            if key in seen_terms:
                stats["duplicate"] += 1
                continue
            seen_terms.add(key)

            entry = {
                "term": term,
                "term_key": key,
                "rank": rank,
                "source": "p31_untrained",
            }
            stats["sampled_from"] += 1
            n_seen = stats["sampled_from"]
            if len(reservoir) < sample_size:
                reservoir.append(entry)
            else:
                j = rng.randrange(n_seen)
                if j < sample_size:
                    reservoir[j] = entry

    rng.shuffle(reservoir)
    return reservoir, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build untrained P31 wiki glossaries for dev evaluation."
    )
    parser.add_argument("--ranked_terms", type=str, default=DEFAULT_RANKED_TERMS)
    parser.add_argument("--dev_jsonl", type=str, default=DEFAULT_DEV_JSONL)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--min_untrained_rank", type=int, default=DEFAULT_MIN_UNTRAINED_RANK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--exclude_glossary",
        type=str,
        nargs="*",
        default=DEFAULT_INFERENCE_GLOSSARIES,
        help="JSON glossary files whose terms must not appear in the output.",
    )
    args = parser.parse_args()

    sizes = sorted(set(int(s) for s in args.sizes))
    assert sizes and sizes[0] > 0, f"Bad sizes: {args.sizes}"
    max_size = sizes[-1]

    dev_terms = load_dev_terms(args.dev_jsonl)
    glossary_terms = load_json_glossary_terms(args.exclude_glossary)
    blocked = dev_terms | glossary_terms
    print(f"[BLOCK] dev_terms={len(dev_terms):,} glossary_terms={len(glossary_terms):,}")

    sampled, stats = reservoir_sample_ranked_terms(
        ranked_terms_path=args.ranked_terms,
        blocked_terms=blocked,
        min_untrained_rank=args.min_untrained_rank,
        sample_size=max_size,
        seed=args.seed,
    )
    assert len(sampled) >= max_size, (
        f"Only sampled {len(sampled):,} terms, need {max_size:,}. Stats: {stats}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    outputs: Dict[str, str] = {}
    for size in sizes:
        out_path = os.path.join(
            args.output_dir,
            f"wiki_p31_untrained_rank{args.min_untrained_rank}_sample{size}.json",
        )
        payload = sampled[:size]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        outputs[str(size)] = out_path
        print(f"[WRITE] size={size:,} path={out_path}")

    manifest = {
        "ranked_terms": args.ranked_terms,
        "dev_jsonl": args.dev_jsonl,
        "output_dir": args.output_dir,
        "sizes": sizes,
        "min_untrained_rank": args.min_untrained_rank,
        "seed": args.seed,
        "blocked_count": len(blocked),
        "stats": stats,
        "outputs": outputs,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[WRITE] manifest={manifest_path}")


if __name__ == "__main__":
    main()
