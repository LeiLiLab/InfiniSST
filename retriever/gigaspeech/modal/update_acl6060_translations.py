#!/usr/bin/env python3
"""
Update Chinese translations for the ACL6060 glossary.

For each line in the paired English and Chinese tagged transcripts, we extract
terms wrapped in square brackets. When the number of bracketed terms matches on
both sides, we treat them as aligned pairs and update the glossary with the
Chinese translation. Entries that receive a translation are marked as
confused = False.

Lines with mismatched bracket counts are ignored to avoid introducing noisy
translations.
"""

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update ACL6060 glossary translations.")
    parser.add_argument(
        "--english-file",
        type=str,
        required=True,
        help="English tagged transcript with terms wrapped in [].",
    )
    parser.add_argument(
        "--chinese-file",
        type=str,
        required=True,
        help="Chinese tagged transcript with terms wrapped in [].",
    )
    parser.add_argument(
        "--glossary-file",
        type=str,
        required=True,
        help="Glossary JSON file to update in place.",
    )
    return parser.parse_args()


TERM_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def extract_terms(line: str) -> List[str]:
    return [term.strip() for term in TERM_PATTERN.findall(line)]


def collect_term_pairs(english_lines: Iterable[str], chinese_lines: Iterable[str]) -> Dict[str, List[str]]:
    term_to_translations: Dict[str, List[str]] = defaultdict(list)
    skipped = 0
    total = 0

    for en_line, zh_line in zip(english_lines, chinese_lines):
        en_terms = extract_terms(en_line)
        zh_terms = extract_terms(zh_line)

        if not en_terms and not zh_terms:
            continue

        total += 1
        if len(en_terms) != len(zh_terms):
            skipped += 1
            continue

        for en_term, zh_term in zip(en_terms, zh_terms):
            if not en_term:
                continue
            term_to_translations[en_term].append(zh_term)

    print(f"[INFO] Collected aligned terms from {total} lines (skipped {skipped} mismatched lines).")
    return term_to_translations


def choose_translation(translations: List[str]) -> Tuple[str, bool]:
    cleaned = [t.strip() for t in translations if t.strip()]
    if not cleaned:
        return "", False
    first = cleaned[0]
    all_same = all(t == first for t in cleaned)
    if not all_same:
        unique = sorted(set(cleaned))
        print(f"[WARN] Multiple translations found for term: {unique}. Using '{first}'.")
    return first, True


def update_glossary(glossary_path: str, term_map: Dict[str, List[str]]) -> None:
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)

    if not isinstance(glossary, dict):
        raise ValueError("Expected glossary JSON to be a dict keyed by term.")

    missing_terms = []
    updated = 0

    for term, translations in term_map.items():
        translation, ok = choose_translation(translations)
        if not ok:
            continue

        entry = glossary.get(term)
        if entry is None:
            missing_terms.append(term)
            continue

        target_translations = entry.get("target_translations")
        if not isinstance(target_translations, dict):
            target_translations = {}
        target_translations["zh"] = translation
        entry["target_translations"] = target_translations
        entry["confused"] = False
        glossary[term] = entry
        updated += 1

    with open(glossary_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Updated {updated} glossary entries.")
    if missing_terms:
        print(f"[WARN] {len(missing_terms)} terms not found in glossary (first 10): {missing_terms[:10]}")


def main() -> None:
    args = parse_args()

    with open(args.english_file, "r", encoding="utf-8") as f_en, open(args.chinese_file, "r", encoding="utf-8") as f_zh:
        english_lines = list(f_en)
        chinese_lines = list(f_zh)

    if len(english_lines) != len(chinese_lines):
        raise ValueError(f"Mismatch in line counts: English={len(english_lines)}, Chinese={len(chinese_lines)}")

    term_pairs = collect_term_pairs(english_lines, chinese_lines)
    update_glossary(args.glossary_file, term_pairs)


if __name__ == "__main__":
    main()

