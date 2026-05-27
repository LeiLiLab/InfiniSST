#!/usr/bin/env python3
"""Build the medicine eval gs10k glossary from medicine GT + medicine wiki terms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


TARGET_LANGS = ("zh", "ja", "de")


DEFAULT_MEDICINE_JSONL = (
    "/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/"
    "medicine_dev_dataset.jsonl"
)
DEFAULT_FILLER_GLOSSARY = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/"
    "glossary_scale/wiki_glossary_medicine_enriched.json"
)
DEFAULT_GT_GLOSSARY = (
    "/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/"
    "medicine_glossary_gt_union_gs10000.json"
)
DEFAULT_OUTPUT = (
    "/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/"
    "medicine_glossary_gt_plus_medicine_wiki_gs10000.json"
)


def _term_norm(term: str) -> str:
    return " ".join(term.strip().lower().split())


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def iter_gt_terms(jsonl_path: Path) -> Iterable[str]:
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("term", "term_text", "term_key"):
                term = _term_norm(str(row.get(key) or ""))
                if term:
                    yield term
            for key in ("_chunk_positive_terms", "positive_terms"):
                value = row.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            term = _term_norm(str(item.get("term") or item.get("text") or ""))
                        else:
                            term = _term_norm(str(item or ""))
                        if term:
                            yield term


def _translation_priority(entry: Dict[str, Any]) -> Tuple[int, int]:
    translations = entry.get("target_translations") or {}
    all_langs = all(translations.get(lang) for lang in TARGET_LANGS)
    any_lang = any(translations.get(lang) for lang in TARGET_LANGS)
    return (0 if all_langs else 1 if any_lang else 2, -len(translations))


def load_filler_terms(path: Path) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    entries = _load_json(path)
    if not isinstance(entries, list):
        raise ValueError(f"Expected list JSON glossary: {path}")

    candidates: List[Tuple[Tuple[int, int], int, str]] = []
    stats = {
        "filler_source_terms": 0,
        "filler_source_all_target_langs": 0,
        "filler_source_any_target_lang": 0,
    }
    for idx, entry in enumerate(entries):
        if isinstance(entry, dict):
            term = _term_norm(str(entry.get("term") or ""))
            translations = entry.get("target_translations") or {}
            if all(translations.get(lang) for lang in TARGET_LANGS):
                stats["filler_source_all_target_langs"] += 1
            if any(translations.get(lang) for lang in TARGET_LANGS):
                stats["filler_source_any_target_lang"] += 1
            priority = _translation_priority(entry)
        else:
            term = _term_norm(str(entry or ""))
            priority = (2, 0)
        if term:
            stats["filler_source_terms"] += 1
            candidates.append((priority, idx, term))

    terms: List[str] = []
    seen: Set[str] = set()
    priority_by_term: Dict[str, int] = {}
    for priority, _, term in sorted(candidates):
        if term not in seen:
            seen.add(term)
            terms.append(term)
            priority_by_term[term] = priority[0]
    return terms, stats, priority_by_term


def iter_gt_terms_from_glossary(path: Path) -> Iterable[str]:
    if not path.is_file():
        return []
    entries = _load_json(path)
    if not isinstance(entries, list):
        raise ValueError(f"Expected list JSON glossary: {path}")
    terms: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("source") != "medicine_gt":
            continue
        term = _term_norm(str(entry.get("term") or ""))
        if term:
            terms.append(term)
    return terms


def write_union(
    *,
    output: Path,
    gt_terms: Iterable[str],
    filler_terms: Iterable[str],
    target_size: int,
    filler_source: str,
) -> Dict[str, int]:
    seen: Set[str] = set()
    rows: List[Dict[str, str]] = []
    for term in sorted({_term_norm(t) for t in gt_terms if _term_norm(t)}):
        if term in seen:
            continue
        seen.add(term)
        rows.append({"term": term, "source": "medicine_gt"})
    gt_count = len(rows)

    filler_count = 0
    for term in filler_terms:
        term = _term_norm(term)
        if not term or term in seen:
            continue
        seen.add(term)
        rows.append({"term": term, "source": filler_source})
        filler_count += 1
        if target_size > 0 and len(rows) >= target_size:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fout:
        json.dump(rows, fout, indent=2, ensure_ascii=False)
    os.replace(tmp, output)
    return {
        "total": len(rows),
        "medicine_gt": gt_count,
        filler_source: filler_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build medicine eval union gs glossary from medicine GT + medicine wiki terms"
    )
    parser.add_argument("--medicine-jsonl", default=DEFAULT_MEDICINE_JSONL)
    parser.add_argument(
        "--gt-glossary",
        default=DEFAULT_GT_GLOSSARY,
        help="Optional previous union glossary; only source=medicine_gt terms are reused.",
    )
    parser.add_argument("--filler-glossary", default=DEFAULT_FILLER_GLOSSARY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-output", default="")
    parser.add_argument("--target-size", type=int, default=10000)
    parser.add_argument("--filler-source", default="medicine_wiki_filler")
    args = parser.parse_args()

    gt_terms = list(iter_gt_terms(Path(args.medicine_jsonl)))
    gt_terms.extend(iter_gt_terms_from_glossary(Path(args.gt_glossary)))
    output = Path(args.output)
    filler_terms, filler_stats, filler_priority = load_filler_terms(
        Path(args.filler_glossary)
    )
    stats = write_union(
        output=output,
        gt_terms=gt_terms,
        filler_terms=filler_terms,
        target_size=args.target_size,
        filler_source=args.filler_source,
    )
    payload = {
        "output": args.output,
        "medicine_jsonl": args.medicine_jsonl,
        "gt_glossary": args.gt_glossary,
        "filler_glossary": args.filler_glossary,
        "target_size": args.target_size,
        "filler_source": args.filler_source,
        **filler_stats,
        **stats,
    }
    with open(output, "r", encoding="utf-8") as fin:
        union_rows = json.load(fin)
    used_filler_terms = [
        _term_norm(str(row.get("term") or ""))
        for row in union_rows
        if row.get("source") == args.filler_source
    ]
    payload["filler_used_all_target_langs"] = sum(
        1 for term in used_filler_terms if filler_priority.get(term) == 0
    )
    payload["filler_used_any_target_lang"] = sum(
        1 for term in used_filler_terms if filler_priority.get(term, 2) <= 1
    )
    stats_output = (
        Path(args.stats_output)
        if args.stats_output
        else output.with_name(output.stem + "_stats.json")
    )
    with open(stats_output, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps({**payload, "stats_output": str(stats_output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
