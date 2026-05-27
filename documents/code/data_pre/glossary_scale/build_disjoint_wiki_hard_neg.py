#!/usr/bin/env python3
"""
Build a disjoint wiki hard-negative glossary for training.

Crawls Wikipedia categories from domains OUTSIDE the inference glossary
(NLP, AI, CS, Medicine, Law are excluded). Applies quality filters and
explicitly removes any term that appears in the inference glossary.

Usage:
    python build_disjoint_wiki_hard_neg.py
    python build_disjoint_wiki_hard_neg.py --output /path/to/output.json
    python build_disjoint_wiki_hard_neg.py --max_total 10000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import OrderedDict
from typing import Dict, List, Set, Tuple

import requests

# ======Configuration=====
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "InfiniSST-DisjointHardNeg/1.0 (research; jiaxuanluo@example.com)",
}

CATEGORY_TIERS: List[Tuple[str, str, int]] = [
    ("math", "Mathematics", 2),
    ("statistics", "Statistics", 2),
    ("physics", "Physics", 2),
    ("chemistry", "Chemistry", 2),
    ("biology", "Biology", 2),
    ("economics", "Economics", 2),
    ("ee", "Electrical_engineering", 2),
    ("mech_eng", "Mechanical_engineering", 2),
    ("philosophy", "Philosophy", 2),
    ("geography", "Geography", 2),
]

INFERENCE_GLOSSARY_PATH = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/"
    "documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
)

MAX_WORDS = 4
MAX_TOTAL = 10000
REQUEST_DELAY_SEC = 0.05
MAX_TERMS_PER_TIER = 20000
REQUEST_TIMEOUT_SEC = 30

HAS_DIGIT = re.compile(r"\d")
PARENS = re.compile(r"\s*\([^)]*\)")

SKIP_PREFIXES = (
    "List of", "Outline of", "History of", "Index of",
    "Category:", "Wikipedia:", "Template:", "Portal:",
    "Draft:", "Talk:", "User:", "File:",
)
# ======Configuration=====


def fetch_category_members(
    category: str,
    cmtype: str = "page",
    limit: int = 500,
) -> List[str]:
    members: List[str] = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": cmtype,
        "cmlimit": min(limit, 500),
        "format": "json",
    }
    while True:
        resp = requests.get(
            WIKI_API, params=params, headers=WIKI_HEADERS,
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
        for m in data.get("query", {}).get("categorymembers", []):
            members.append(m["title"])
        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
            time.sleep(REQUEST_DELAY_SEC)
        else:
            break
    return members


def is_valid_raw_title(title: str) -> bool:
    if any(title.startswith(prefix) for prefix in SKIP_PREFIXES):
        return False
    if "disambiguation" in title.lower():
        return False
    if len(title) < 2 or len(title) > 80:
        return False
    if title.startswith("."):
        return False
    if title.startswith("(") and title.endswith(")"):
        return False
    return True


def clean_term(raw_title: str) -> str:
    """Strip parenthetical disambiguation and whitespace."""
    return PARENS.sub("", raw_title).strip()


def is_high_quality(cleaned: str) -> bool:
    if not cleaned or len(cleaned) < 2:
        return False
    if not cleaned[0].isalpha():
        return False
    if HAS_DIGIT.search(cleaned):
        return False
    if len(cleaned.split()) > MAX_WORDS:
        return False
    return True


def crawl_category_tree(
    root_category: str,
    max_depth: int,
    verbose: bool = True,
) -> Set[str]:
    visited_cats: Set[str] = set()
    terms: Set[str] = set()
    queue: List[Tuple[str, int]] = [(root_category, 0)]
    last_report = 0

    while queue:
        cat, depth = queue.pop(0)
        if cat in visited_cats:
            continue
        visited_cats.add(cat)

        try:
            pages = fetch_category_members(cat, cmtype="page")
        except Exception as exc:
            print(f"  [WARN] Failed to fetch pages for '{cat}': {exc}")
            continue

        for page in pages:
            if is_valid_raw_title(page):
                terms.add(page)

        if depth < max_depth:
            try:
                subcats = fetch_category_members(cat, cmtype="subcat")
            except Exception as exc:
                print(f"  [WARN] Failed to fetch subcats for '{cat}': {exc}")
                subcats = []
            for subcat in subcats:
                subcat_name = subcat.replace("Category:", "")
                queue.append((subcat_name, depth + 1))

        time.sleep(REQUEST_DELAY_SEC)

        if verbose and len(terms) - last_report >= 500:
            last_report = len(terms)
            print(
                f"  [{root_category}] cats_visited={len(visited_cats)}, "
                f"queue={len(queue)}, terms={len(terms)}"
            )

    return terms


def load_inference_glossary(path: str) -> Set[str]:
    """Load inference glossary and return lowercase term set for exclusion."""
    if not os.path.isfile(path):
        print(f"[WARN] Inference glossary not found at {path}, skipping exclusion")
        return set()
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    assert isinstance(entries, list)
    terms = set()
    for e in entries:
        raw = e["term"].strip()
        cleaned = PARENS.sub("", raw).strip().lower()
        terms.add(cleaned)
        terms.add(raw.lower())
    print(f"[INFO] Loaded {len(terms)} inference glossary keys for exclusion")
    return terms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build disjoint wiki hard-negative glossary for training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "wiki_hard_neg_disjoint.json"),
    )
    parser.add_argument("--max_total", type=int, default=MAX_TOTAL)
    parser.add_argument("--max_per_tier", type=int, default=MAX_TERMS_PER_TIER)
    parser.add_argument(
        "--inference_glossary",
        type=str,
        default=INFERENCE_GLOSSARY_PATH,
        help="Path to inference glossary JSON (terms here are excluded)",
    )
    args = parser.parse_args()

    inference_exclude = load_inference_glossary(args.inference_glossary)

    all_terms: OrderedDict[str, Dict] = OrderedDict()
    stats: Dict[str, Dict] = {}

    for tier_name, category, depth in CATEGORY_TIERS:
        print(
            f"\n=== Crawling tier '{tier_name}': "
            f"Category:{category} (max_depth={depth}) ==="
        )
        raw_terms = crawl_category_tree(category, depth)

        tier_stats = {
            "raw": len(raw_terms),
            "cleaned": 0,
            "filtered_quality": 0,
            "filtered_overlap": 0,
            "filtered_dup": 0,
            "added": 0,
        }

        for raw_title in sorted(raw_terms):
            if not is_valid_raw_title(raw_title):
                continue

            cleaned = clean_term(raw_title)
            tier_stats["cleaned"] += 1

            if not is_high_quality(cleaned):
                tier_stats["filtered_quality"] += 1
                continue

            key = cleaned.lower()
            if key in inference_exclude:
                tier_stats["filtered_overlap"] += 1
                continue

            if key in all_terms:
                tier_stats["filtered_dup"] += 1
                continue

            all_terms[key] = {"term": cleaned, "tier": tier_name}
            tier_stats["added"] += 1

        stats[tier_name] = tier_stats
        print(
            f"  Tier '{tier_name}': {tier_stats['raw']} raw → "
            f"{tier_stats['added']} added "
            f"(quality_drop={tier_stats['filtered_quality']}, "
            f"overlap_drop={tier_stats['filtered_overlap']}, "
            f"dup_drop={tier_stats['filtered_dup']})"
        )
        print(f"  Total so far: {len(all_terms)}")

    output = list(all_terms.values())
    if len(output) > args.max_total:
        tier_groups: Dict[str, List[Dict]] = {}
        for entry in output:
            tier_groups.setdefault(entry["tier"], []).append(entry)

        total_available = sum(len(v) for v in tier_groups.values())
        sampled: List[Dict] = []
        remainder = args.max_total
        tier_list = list(tier_groups.keys())

        for i, tier in enumerate(tier_list):
            entries = tier_groups[tier]
            if i == len(tier_list) - 1:
                n = remainder
            else:
                n = max(1, round(len(entries) / total_available * args.max_total))
            n = min(n, len(entries), remainder)
            sampled.extend(entries[:n])
            remainder -= n
            if remainder <= 0:
                break

        print(
            f"\n[INFO] Proportional cap: {len(output)} -> {len(sampled)} "
            f"(target={args.max_total})"
        )
        output = sampled

    # Final sanity checks
    seen: Set[str] = set()
    for entry in output:
        key = entry["term"].lower()
        assert key not in inference_exclude, f"Leakage: {entry['term']!r}"
        assert "(" not in entry["term"], f"Parenthesis: {entry['term']!r}"
        assert not HAS_DIGIT.search(entry["term"]), f"Digit: {entry['term']!r}"
        assert len(entry["term"].split()) <= MAX_WORDS, f"Too long: {entry['term']!r}"
        assert key not in seen, f"Duplicate: {entry['term']!r}"
        seen.add(key)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(output)} disjoint hard-negative terms to {args.output}")
    print(f"{'='*60}")
    for tier_name in stats:
        count = sum(1 for e in output if e["tier"] == tier_name)
        print(f"  {tier_name}: {count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
