#!/usr/bin/env python3
"""Build union ACL paper-extracted glossaries for streaming eval.

The raw output is the de-duplicated union of the five ACL paper-extracted
glossaries.  The scaled outputs preserve every raw term first, then append
deterministic wiki filler terms until the requested bank sizes are reached.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_ROOT = Path("/home/jiaxuanluo/InfiniSST")
DEFAULT_EXTRACTED_DIR = DEFAULT_ROOT / "documents/data/data_pre/extracted_glossaries_by_paper"
DEFAULT_FILLER = (
    DEFAULT_ROOT
    / "documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs_enriched.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "retriever/gigaspeech/data_pre"
DEFAULT_PAPERS = (
    "2022.acl-long.268",
    "2022.acl-long.367",
    "2022.acl-long.590",
    "2022.acl-long.110",
    "2022.acl-long.117",
)
DEFAULT_TARGET_SIZES = (1000, 10000)
OUTPUT_PREFIX = "acl6060_paper_extracted_union"
WS_RE = re.compile(r"\s+")


def _norm_term(text: Any) -> str:
    return WS_RE.sub(" ", str(text or "").strip()).casefold()


def _clean_text(text: Any) -> str:
    return WS_RE.sub(" ", str(text or "").strip())


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _entries(data: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                yield str(key), dict(value)
        return
    if isinstance(data, list):
        for idx, value in enumerate(data):
            if isinstance(value, dict):
                yield str(idx), dict(value)
        return
    raise ValueError(f"unsupported glossary JSON type: {type(data).__name__}")


def _paper_path(glossary_dir: Path, paper_id: str) -> Path:
    return glossary_dir / f"extracted_glossary__{paper_id}.json"


def _merge_target_translations(
    old: Dict[str, str], new: Dict[str, str], stats: Dict[str, int]
) -> Dict[str, str]:
    merged = dict(old)
    for lang, value in sorted(new.items()):
        cleaned = _clean_text(value)
        if not cleaned:
            continue
        previous = merged.get(lang, "")
        if previous and previous != cleaned:
            stats["translation_conflicts"] += 1
            continue
        merged[lang] = cleaned
    return merged


def _load_raw_union(
    glossary_dir: Path,
    papers: Sequence[str],
    target_lang: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    merged: Dict[str, Dict[str, Any]] = {}
    stats = {
        "paper_count": len(papers),
        "raw_mentions": 0,
        "kept_mentions": 0,
        "skipped_missing_term": 0,
        "skipped_missing_translation": 0,
        "duplicate_terms": 0,
        "translation_conflicts": 0,
    }

    for paper_id in papers:
        path = _paper_path(glossary_dir, paper_id)
        data = _load_json(path)
        for key, entry in _entries(data):
            stats["raw_mentions"] += 1
            term = _clean_text(entry.get("term") or key)
            norm = _norm_term(term)
            if not norm:
                stats["skipped_missing_term"] += 1
                continue
            translations_raw = entry.get("target_translations") or {}
            if not isinstance(translations_raw, dict):
                translations_raw = {}
            target_value = _clean_text(translations_raw.get(target_lang, ""))
            if not target_value:
                stats["skipped_missing_translation"] += 1
                continue
            stats["kept_mentions"] += 1

            target_translations = {
                str(lang): _clean_text(value)
                for lang, value in translations_raw.items()
                if _clean_text(value)
            }
            if target_lang not in target_translations:
                target_translations[target_lang] = target_value

            if norm in merged:
                stats["duplicate_terms"] += 1
                row = merged[norm]
                row["source_papers"] = sorted(
                    set([*row.get("source_papers", []), paper_id])
                )
                row["target_translations"] = _merge_target_translations(
                    row.get("target_translations", {}),
                    target_translations,
                    stats,
                )
                continue

            merged[norm] = {
                "term": term,
                "source": "paper_extracted_union_raw",
                "target_translations": target_translations,
                "source_papers": [paper_id],
            }

    raw_terms = [merged[key] for key in sorted(merged)]
    stats["raw_unique_terms"] = len(raw_terms)
    return raw_terms, stats


def _load_filler_terms(path: Path, target_lang: str, excluded_keys: set[str]) -> List[Dict[str, Any]]:
    data = _load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    for key, entry in _entries(data):
        term = _clean_text(entry.get("term") or key)
        norm = _norm_term(term)
        if not norm or norm in excluded_keys or norm in out:
            continue
        translations_raw = entry.get("target_translations") or {}
        if not isinstance(translations_raw, dict):
            translations_raw = {}
        target_value = _clean_text(translations_raw.get(target_lang, ""))
        if not target_value:
            continue
        translations = {
            str(lang): _clean_text(value)
            for lang, value in translations_raw.items()
            if _clean_text(value)
        }
        translations[target_lang] = target_value
        out[norm] = {
            "term": term,
            "source": "wiki_filler",
            "target_translations": translations,
        }
    return [out[key] for key in sorted(out)]


def _validate_terms(rows: Sequence[Dict[str, Any]], target_lang: str, expected: int) -> None:
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} rows, got {len(rows)}")
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        term = _clean_text(row.get("term"))
        norm = _norm_term(term)
        if not norm:
            raise AssertionError(f"row {idx} missing term")
        if norm in seen:
            raise AssertionError(f"duplicate normalized term in output: {term}")
        seen.add(norm)
        translations = row.get("target_translations") or {}
        if not isinstance(translations, dict) or not _clean_text(translations.get(target_lang, "")):
            raise AssertionError(f"row {idx} missing {target_lang} translation: {term}")


def _write_json(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-glossary-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument("--filler-glossary", type=Path, default=DEFAULT_FILLER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--papers", nargs="+", default=list(DEFAULT_PAPERS))
    parser.add_argument("--target-sizes", type=int, nargs="+", default=list(DEFAULT_TARGET_SIZES))
    parser.add_argument("--expected-raw-count", type=int, default=253)
    parser.add_argument("--stats-json", type=Path, default=None)
    args = parser.parse_args()

    raw_terms, stats = _load_raw_union(
        args.extracted_glossary_dir,
        args.papers,
        args.target_lang,
    )
    _validate_terms(raw_terms, args.target_lang, args.expected_raw_count)

    raw_keys = {_norm_term(row["term"]) for row in raw_terms}
    filler_terms = _load_filler_terms(args.filler_glossary, args.target_lang, raw_keys)
    stats["filler_candidates"] = len(filler_terms)

    outputs: Dict[str, str] = {}
    raw_path = args.output_dir / f"{OUTPUT_PREFIX}_raw_{args.target_lang}.json"
    _write_json(raw_path, raw_terms)
    outputs["raw"] = str(raw_path)

    for size in sorted(set(args.target_sizes)):
        if size < len(raw_terms):
            raise AssertionError(f"target size {size} is smaller than raw count {len(raw_terms)}")
        needed = size - len(raw_terms)
        if len(filler_terms) < needed:
            raise AssertionError(f"need {needed} filler terms, only have {len(filler_terms)}")
        rows = [*raw_terms, *filler_terms[:needed]]
        _validate_terms(rows, args.target_lang, size)
        out_path = args.output_dir / f"{OUTPUT_PREFIX}_gs{size}_{args.target_lang}.json"
        _write_json(out_path, rows)
        outputs[f"gs{size}"] = str(out_path)
        stats[f"gs{size}_filler_added"] = needed

    stats.update(
        {
            "target_lang": args.target_lang,
            "papers": list(args.papers),
            "output_paths": outputs,
            "filler_glossary": str(args.filler_glossary),
        }
    )
    stats_path = args.stats_json or (
        args.output_dir / f"{OUTPUT_PREFIX}_stats_{args.target_lang}.json"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
