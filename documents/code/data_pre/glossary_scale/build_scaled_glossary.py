#!/usr/bin/env python3

"""
Build scaled glossaries by merging GT terms (union of per-paper extracted
glossaries) with padding terms from wiki_glossary_nlp_ai_cs_enriched.json.

Output glossaries are in list format compatible with build_maxsim_index.py.

Usage:
    python build_scaled_glossary.py --target-sizes 1000 10000
"""

from __future__ import annotations

# ======Configuration=====
import argparse
import glob
import json
import random
from pathlib import Path
from typing import Dict, List, Set

DEFAULT_EXTRACTED_GLOSSARY_DIR = (
    "/home/jiaxuanluo/InfiniSST/documents/data/data_pre/extracted_glossaries_by_paper"
)
DEFAULT_WIKI_GLOSSARY = (
    "/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/"
    "wiki_glossary_nlp_ai_cs_enriched.json"
)
DEFAULT_OUTPUT_DIR = (
    "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre"
)
DEFAULT_TARGET_LANG = "zh"
RANDOM_SEED = 42
OUTPUT_PREFIX = "glossary_acl6060_gt_union"
# ======Configuration=====


def load_gt_from_extracted_glossaries(
    glossary_dir: Path, target_lang: str,
) -> List[Dict]:
    """
    Load all per-paper extracted glossaries and merge into a de-duplicated
    union. Each paper glossary is a dict: {key: {term, target_translations}}.
    """
    pattern = str(glossary_dir / "extracted_glossary__2022.acl-long.*.json")
    files = sorted(glob.glob(pattern))
    assert files, f"No extracted glossary files found matching: {pattern}"

    seen: Dict[str, Dict] = {}
    for fpath in files:
        if "ablation" in fpath:
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            items = data.values()
        elif isinstance(data, list):
            items = data
        else:
            assert False, f"Unexpected format in {fpath}: {type(data)}"

        for entry in items:
            assert isinstance(entry, dict), f"Bad entry in {fpath}: {entry}"
            term = entry.get("term", "")
            translations = entry.get("target_translations", {})
            assert translations.get(target_lang), (
                f"GT term '{term}' in {fpath} missing '{target_lang}' translation"
            )
            key = term.lower()
            if key not in seen:
                seen[key] = {
                    "term": term,
                    "target_translations": translations,
                }

    entries = list(seen.values())
    print(f"Loaded {len(files)} per-paper glossary files -> {len(entries)} unique GT terms")
    return entries


def load_wiki_glossary(path: Path) -> List[Dict]:
    """Load wiki list-format glossary."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    return data


def build_scaled(
    gt_entries: List[Dict],
    wiki_entries: List[Dict],
    target_size: int,
    gt_term_keys: Set[str],
    target_lang: str,
) -> List[Dict]:
    """Build a glossary of exactly target_size entries."""
    assert target_size >= len(gt_entries), (
        f"target_size={target_size} < GT count={len(gt_entries)}"
    )

    eligible_wiki = [
        e for e in wiki_entries
        if (
            e.get("term", "").lower() not in gt_term_keys
            and e.get("target_translations", {}).get(target_lang, "")
        )
    ]
    padding_needed = target_size - len(gt_entries)
    assert len(eligible_wiki) >= padding_needed, (
        f"Not enough wiki terms: need {padding_needed}, have {len(eligible_wiki)}"
    )

    rng = random.Random(RANDOM_SEED)
    padding = rng.sample(eligible_wiki, padding_needed)

    merged = []
    for e in gt_entries:
        merged.append({
            "term": e["term"],
            "target_translations": e["target_translations"],
        })
    for e in padding:
        merged.append({
            "term": e["term"],
            "target_translations": e.get("target_translations", {}),
        })

    assert len(merged) == target_size
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scaled glossaries.")
    parser.add_argument(
        "--extracted-glossary-dir", type=str,
        default=DEFAULT_EXTRACTED_GLOSSARY_DIR,
        help="Directory containing per-paper extracted glossary JSON files.",
    )
    parser.add_argument(
        "--wiki-glossary", type=str, default=DEFAULT_WIKI_GLOSSARY,
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--target-sizes", type=int, nargs="+", default=[1000, 10000],
    )
    parser.add_argument(
        "--target-lang", type=str, default=DEFAULT_TARGET_LANG,
    )
    args = parser.parse_args()

    gt_entries = load_gt_from_extracted_glossaries(
        Path(args.extracted_glossary_dir), args.target_lang,
    )
    wiki_entries = load_wiki_glossary(Path(args.wiki_glossary))

    gt_term_keys = {e["term"].lower() for e in gt_entries}
    print(f"GT terms (union): {len(gt_entries)}")
    print(f"Wiki terms: {len(wiki_entries)}")
    overlap = sum(
        1 for e in wiki_entries if e.get("term", "").lower() in gt_term_keys
    )
    print(f"Overlap (excluded from padding): {overlap}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for size in args.target_sizes:
        merged = build_scaled(
            gt_entries, wiki_entries, size, gt_term_keys, args.target_lang,
        )
        out_path = out_dir / f"{OUTPUT_PREFIX}_gs{size}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path} ({len(merged)} entries)")


if __name__ == "__main__":
    main()
