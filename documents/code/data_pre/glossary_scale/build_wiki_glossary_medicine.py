#!/usr/bin/env python3
"""
Build a ~10k medicine glossary from Wikipedia categories for inference eval.

Same approach as build_wiki_glossary.py but targeting the medical domain.
Tier hierarchy: pharmacology → medicine → health, ensuring narrower domains
are prioritized.

Output: a JSON file with a list of {"term": ..., "tier": ...} objects.
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
    "User-Agent": "InfiniSST-GlossaryBuilder/1.0 (research; jiaxuanluo@example.com)",
}

CATEGORY_TIERS: List[Tuple[str, str, int]] = [
    ("pharmacology", "Pharmacology", 3),
    ("medicine", "Medicine", 2),
    ("health", "Health", 2),
]

REQUEST_DELAY_SEC = 0.05
MAX_TERMS_PER_TIER = 20000
REQUEST_TIMEOUT_SEC = 30
TARGET_TOTAL = 10000

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
    """Fetch all members (pages or subcats) of a Wikipedia category."""
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


def is_valid_term(title: str) -> bool:
    """Filter out non-term-like article titles."""
    if any(title.startswith(prefix) for prefix in SKIP_PREFIXES):
        return False
    if "disambiguation" in title.lower():
        return False
    if len(title) < 2 or len(title) > 80:
        return False
    if re.match(r"^\d", title):
        return False
    if title.startswith("."):
        return False
    if title.startswith("(") and title.endswith(")"):
        return False
    return True


def crawl_category_tree(
    root_category: str,
    max_depth: int,
    verbose: bool = True,
) -> Set[str]:
    """BFS crawl of a Wikipedia category tree, collecting page titles."""
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
            if is_valid_term(page):
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

        if verbose and len(terms) - last_report >= 200:
            last_report = len(terms)
            print(
                f"  [{root_category}] cats_visited={len(visited_cats)}, "
                f"queue={len(queue)}, terms={len(terms)}"
            )

    return terms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build wiki-based medicine glossary for inference eval"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "wiki_glossary_medicine.json"
        ),
    )
    parser.add_argument(
        "--max_per_tier",
        type=int,
        default=MAX_TERMS_PER_TIER,
    )
    parser.add_argument(
        "--target_total",
        type=int,
        default=TARGET_TOTAL,
    )
    args = parser.parse_args()

    all_terms: OrderedDict[str, Dict] = OrderedDict()

    for tier_name, category, depth in CATEGORY_TIERS:
        print(
            f"\n=== Crawling tier '{tier_name}': "
            f"Category:{category} (max_depth={depth}) ==="
        )
        tier_terms = crawl_category_tree(category, depth)

        new_count = 0
        for term in sorted(tier_terms):
            if not is_valid_term(term):
                continue
            term_lower = term.strip().lower()
            if term_lower not in all_terms:
                all_terms[term_lower] = {"term": term.strip(), "tier": tier_name}
                new_count += 1
                if new_count >= args.max_per_tier:
                    print(f"  Reached max_per_tier={args.max_per_tier}, stopping.")
                    break

        print(
            f"  Tier '{tier_name}': "
            f"{len(tier_terms)} raw -> {new_count} new unique terms"
        )
        print(f"  Total so far: {len(all_terms)}")

        if len(all_terms) >= args.target_total:
            print(f"  Reached target_total={args.target_total}, stopping tiers.")
            break

    output = list(all_terms.values())
    if len(output) > args.target_total:
        output = output[:args.target_total]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    tier_counts = {}
    for v in output:
        t = v["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    print(f"\n{'='*60}")
    print(f"Saved {len(output)} terms to {args.output}")
    for t, c in tier_counts.items():
        print(f"  {t}: {c}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
