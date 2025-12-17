#!/usr/bin/env python3
"""
Build a task-specific glossary for the ACL6060 dev set.

Steps:
1. Extract terminology candidates from the tagged source file where terms are
   wrapped with square brackets, e.g. `[example]`.
2. Look up each term inside an existing glossary JSON file to reuse metadata
   such as translations and descriptions.
3. Mark entries that cannot be found as `confused = True` so that downstream
   indexing code can optionally filter them out.

The output glossary keeps the JSON-dict structure used by
`glossary_cleaned.json`, keyed by the surface term.
"""

import argparse
import json
import re
from collections import OrderedDict
from typing import Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ACL6060 glossary JSON.")
    parser.add_argument(
        "--tagged-file",
        type=str,
        required=True,
        help="Path to ACL6060 tagged terminology file (terms are wrapped by []).",
    )
    parser.add_argument(
        "--glossary-file",
        type=str,
        required=True,
        help="Existing glossary JSON used as the lookup source.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Destination path for the task-specific glossary JSON.",
    )
    return parser.parse_args()


def extract_terms(tagged_file: str) -> Iterable[str]:
    term_pattern = re.compile(r"\[([^\[\]]+)\]")
    seen = OrderedDict()
    with open(tagged_file, "r", encoding="utf-8") as f:
        for line in f:
            for raw_term in term_pattern.findall(line):
                term = raw_term.strip()
                if not term:
                    continue
                if term not in seen:
                    seen[term] = None
    return seen.keys()


def build_lookup(glossary_data: Dict[str, Dict]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    lower_to_key = {}
    for key in glossary_data:
        normalized = key.strip().lower()
        lower_to_key.setdefault(normalized, key)
    return glossary_data, lower_to_key


def build_entry(
    term: str,
    glossary_lookup: Dict[str, Dict],
    lower_to_key: Dict[str, str],
) -> Dict:
    # Exact match first
    if term in glossary_lookup:
        entry = dict(glossary_lookup[term])
        entry["term"] = term
        return entry

    normalized = term.strip().lower()
    matched_key = lower_to_key.get(normalized)
    if matched_key:
        entry = dict(glossary_lookup[matched_key])
        entry["term"] = term
        return entry

    return {
        "term": term,
        "classification_reason": "acl6060_auto",
        "confused": True,
        "short_description": "",
        "target_translations": {},
        "url": "",
    }


def main() -> None:
    args = parse_args()

    with open(args.glossary_file, "r", encoding="utf-8") as f:
        base_glossary = json.load(f)
        if not isinstance(base_glossary, dict):
            raise ValueError("Expected glossary JSON to be a dict keyed by term.")

    glossary_lookup, lower_to_key = build_lookup(base_glossary)

    specialized_glossary = OrderedDict()
    for term in extract_terms(args.tagged_file):
        entry = build_entry(term, glossary_lookup, lower_to_key)
        if not entry.get("confused", False):
            # Keep original flag if present; ensure explicit False
            entry["confused"] = False
        entry.setdefault("classification_reason", "acl6060_auto")
        entry.setdefault("short_description", "")
        entry.setdefault("target_translations", {})
        entry.setdefault("url", "")
        specialized_glossary[term] = entry

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(specialized_glossary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved ACL6060 glossary with {len(specialized_glossary)} terms to {args.output_file}")


if __name__ == "__main__":
    main()

