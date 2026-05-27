#!/usr/bin/env python3
"""
Build untrained P31 wiki glossaries that require target-language translations.

The older dev glossary builder only preserved term/rank fields. For speech-LLM
term-map construction we need actual target translations, so this script joins
the P31-ranked list with `glossary_filtered_from_wiki.json` and samples only
entries with `target_translations[target_lang]`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Counter as CounterType, Dict, Iterable, List, Set, Tuple
from collections import Counter, defaultdict

import ijson


DEFAULT_RANKED_TERMS = (
    "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/p31_full/"
    "wiki_synth_terms_p31_ranked.jsonl"
)
DEFAULT_TRANSLATION_GLOSSARY = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/"
    "retriever/gigaspeech/data/terms/glossary_filtered_from_wiki.json"
)
DEFAULT_DEV_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
DEFAULT_TRAIN_JSONL = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_cleaned.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev"
DEFAULT_INFERENCE_GLOSSARIES = [
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/"
    "documents/code/data_pre/glossary_scale/wiki_glossary_medicine.json",
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/"
    "documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json",
]
DEFAULT_MIN_UNTRAINED_RANK = 1_000_000
DEFAULT_SIZES = [10_000]
DEFAULT_SEED = 42


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
            key = _term_key(item.get("term", "") if isinstance(item, dict) else str(item))
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


def load_ranked_candidates(
    ranked_terms_path: str,
    min_untrained_rank: int,
    blocked_terms: Set[str],
) -> Dict[str, Dict]:
    candidates: Dict[str, Dict] = {}
    with open(ranked_terms_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rank = int(obj.get("rank", 0))
            if rank < min_untrained_rank:
                continue
            term = (obj.get("term") or "").strip()
            key = _term_key(obj.get("term_key", term))
            if not term or not key or key in blocked_terms or key in candidates:
                continue
            candidates[key] = {
                "term": term,
                "term_key": key,
                "rank": rank,
                "source": "p31_untrained",
            }
    return candidates


def collect_translated_candidates(
    translation_glossary: str,
    ranked_candidates: Dict[str, Dict],
    target_lang: str,
) -> List[Dict]:
    translated: List[Dict] = []
    with open(translation_glossary, "rb") as f:
        for key, entry in ijson.kvitems(f, ""):
            candidate = ranked_candidates.get(_term_key(key))
            if candidate is None or not isinstance(entry, dict):
                continue
            translations = entry.get("target_translations")
            translation = ""
            if isinstance(translations, dict):
                translation = str(translations.get(target_lang) or "").strip()
            if not translation:
                continue
            term = str(entry.get("term") or candidate["term"]).strip()
            out = dict(candidate)
            out["term"] = term
            out["translation"] = translation
            out["target_translations"] = {target_lang: translation}
            short_description = str(entry.get("short_description") or "").strip()
            if short_description:
                out["short_description"] = short_description
            translated.append(out)
    return translated


def reservoir_sample(items: List[Dict], sample_size: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    reservoir: List[Dict] = []
    for idx, item in enumerate(items, 1):
        if len(reservoir) < sample_size:
            reservoir.append(item)
            continue
        j = rng.randrange(idx)
        if j < sample_size:
            reservoir[j] = item
    rng.shuffle(reservoir)
    return reservoir


def load_train_gt_terms(path: str) -> Tuple[List[Dict], Dict[str, int]]:
    term_to_translations: Dict[str, CounterType[str]] = defaultdict(Counter)
    key_to_term: Dict[str, str] = {}
    stats = {
        "rows": 0,
        "bad_json_rows": 0,
        "recovered_bad_json_rows": 0,
        "chunks": 0,
        "gt_mentions": 0,
        "empty_gt": 0,
    }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stats["rows"] += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json_rows"] += 1
                chunk_terms_payload = _extract_gt_terms_by_chunk_fragment(line)
                if chunk_terms_payload is None:
                    continue
                stats["recovered_bad_json_rows"] += 1
            else:
                chunk_terms_payload = obj.get("gt_terms_by_chunk", []) or []
            for chunk_terms in chunk_terms_payload:
                stats["chunks"] += 1
                for item in chunk_terms or []:
                    term = str(item.get("term") or "").strip()
                    translation = str(
                        item.get("zh")
                        or item.get("translation")
                        or item.get("target_translation")
                        or ""
                    ).strip()
                    if not term or not translation:
                        stats["empty_gt"] += 1
                        continue
                    key = _term_key(term)
                    key_to_term.setdefault(key, term)
                    term_to_translations[key][translation] += 1
                    stats["gt_mentions"] += 1

    out: List[Dict] = []
    conflicts = 0
    for key, translations in term_to_translations.items():
        if len(translations) > 1:
            conflicts += 1
        translation, count = translations.most_common(1)[0]
        out.append(
            {
                "term": key_to_term[key],
                "term_key": key,
                "translation": translation,
                "target_translations": {"zh": translation},
                "source": "train_gt",
                "train_gt_count": count,
            }
        )
    stats["unique_gt_terms"] = len(out)
    stats["translation_conflict_terms"] = conflicts
    out.sort(key=lambda x: x["term_key"])
    return out, stats


def _extract_gt_terms_by_chunk_fragment(line: str) -> List[List[Dict]] | None:
    key_pos = line.find('"gt_terms_by_chunk"')
    if key_pos < 0:
        return None
    start = line.find("[", key_pos)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(line[start:], start):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(line[start : idx + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, list) else None
    return None


def merge_train_gt_terms(sampled: List[Dict], train_gt_terms: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    merged: Dict[str, Dict] = {}
    stats = {
        "wiki_terms": len(sampled),
        "train_gt_terms": len(train_gt_terms),
        "train_gt_added": 0,
        "train_gt_overrode_existing": 0,
    }
    for item in sampled:
        merged[_term_key(item["term_key"])] = dict(item)
    for item in train_gt_terms:
        key = _term_key(item["term_key"])
        if key in merged:
            old = merged[key]
            old["translation"] = item["translation"]
            old["target_translations"] = dict(item["target_translations"])
            old["source"] = f"{old.get('source', 'unknown')}+train_gt"
            old["train_gt_count"] = item.get("train_gt_count", 0)
            stats["train_gt_overrode_existing"] += 1
        else:
            merged[key] = dict(item)
            stats["train_gt_added"] += 1
    stats["merged_terms"] = len(merged)
    return list(merged.values()), stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build translated untrained P31 wiki glossaries.")
    parser.add_argument("--ranked_terms", default=DEFAULT_RANKED_TERMS)
    parser.add_argument("--translation_glossary", default=DEFAULT_TRANSLATION_GLOSSARY)
    parser.add_argument("--dev_jsonl", default=DEFAULT_DEV_JSONL)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--min_untrained_rank", type=int, default=DEFAULT_MIN_UNTRAINED_RANK)
    parser.add_argument("--target_lang", default="zh")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--suffix", default="zh", help="Output filename suffix before .json.")
    parser.add_argument("--exclude_glossary", nargs="*", default=DEFAULT_INFERENCE_GLOSSARIES)
    parser.add_argument("--train_jsonl", default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--include_train_gt_terms", action="store_true")
    args = parser.parse_args()

    sizes = sorted(set(int(s) for s in args.sizes))
    assert sizes and sizes[0] > 0, f"Bad sizes: {args.sizes}"
    max_size = sizes[-1]

    dev_terms = load_dev_terms(args.dev_jsonl)
    glossary_terms = load_json_glossary_terms(args.exclude_glossary)
    blocked = dev_terms | glossary_terms
    print(f"[BLOCK] dev_terms={len(dev_terms):,} glossary_terms={len(glossary_terms):,}")

    ranked_candidates = load_ranked_candidates(
        args.ranked_terms,
        min_untrained_rank=args.min_untrained_rank,
        blocked_terms=blocked,
    )
    print(f"[RANKED] candidates={len(ranked_candidates):,}")

    translated = collect_translated_candidates(
        args.translation_glossary,
        ranked_candidates=ranked_candidates,
        target_lang=args.target_lang,
    )
    print(f"[TRANSLATED] target_lang={args.target_lang} translated={len(translated):,}")
    assert len(translated) >= max_size, (
        f"Only {len(translated):,} translated candidates, need {max_size:,}. "
        "Lower requested size or expand source pool."
    )

    sampled = reservoir_sample(translated, max_size, seed=args.seed)
    train_gt_stats: Dict[str, int] = {}
    merge_stats_by_size: Dict[str, Dict[str, int]] = {}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    for size in sizes:
        payload = sampled[:size]
        if args.include_train_gt_terms:
            train_gt_terms, train_gt_stats = load_train_gt_terms(args.train_jsonl)
            payload, merge_stats = merge_train_gt_terms(payload, train_gt_terms)
            merge_stats_by_size[str(size)] = merge_stats
        out_path = Path(args.output_dir) / (
            f"wiki_p31_untrained_rank{args.min_untrained_rank}_sample{size}_{args.suffix}.json"
        )
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        outputs[str(size)] = str(out_path)
        print(f"[WRITE] requested_wiki_size={size:,} final_size={len(payload):,} path={out_path}")

    manifest_path = Path(args.output_dir) / f"manifest_{args.suffix}.json"
    manifest = {
        "ranked_terms": args.ranked_terms,
        "translation_glossary": args.translation_glossary,
        "dev_jsonl": args.dev_jsonl,
        "output_dir": args.output_dir,
        "sizes": sizes,
        "min_untrained_rank": args.min_untrained_rank,
        "target_lang": args.target_lang,
        "seed": args.seed,
        "train_jsonl": args.train_jsonl if args.include_train_gt_terms else "",
        "include_train_gt_terms": args.include_train_gt_terms,
        "blocked_count": len(blocked),
        "ranked_candidates": len(ranked_candidates),
        "translated_candidates": len(translated),
        "train_gt_stats": train_gt_stats,
        "merge_stats_by_size": merge_stats_by_size,
        "outputs": outputs,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[WRITE] manifest={manifest_path}")


if __name__ == "__main__":
    main()
