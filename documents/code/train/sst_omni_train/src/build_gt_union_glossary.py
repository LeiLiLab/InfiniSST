#!/usr/bin/env python3
"""Build a translated glossary that explicitly covers SFT gt_terms_by_chunk.

The retriever-SFT data builder evaluates whether retrieved term_map entries hit
the source JSONL's gt_terms_by_chunk.  If the retrieval bank does not contain
those source terms, that statistic is meaningless.  This helper creates a
deterministic union glossary:

1. every translated term observed in the supplied SFT JSONLs, then
2. filler terms from an existing glossary, skipping duplicate normalized terms.

It does not invent translations.  Entries without a target translation for the
requested language are counted and skipped.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def _extract_translation(entry: Mapping[str, Any], lang_code: str) -> str:
    value = entry.get("translation") or entry.get("target_translation") or entry.get(lang_code)
    if value is None and isinstance(entry.get("target_translations"), Mapping):
        value = entry["target_translations"].get(lang_code)
    return str(value or "").strip()


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on {path}:{lineno}")
            yield lineno, obj


def _load_filler(path: Path, lang_code: str) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, Mapping):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = enumerate(data)
    else:
        raise ValueError(f"Unsupported filler glossary format: {path}")

    out: list[Dict[str, Any]] = []
    for key, entry in iterable:
        if isinstance(entry, str):
            term = str(key).strip()
            translation = entry.strip()
            raw: Dict[str, Any] = {
                "term": term,
                "translation": translation,
                "target_translations": {lang_code: translation},
            }
        elif isinstance(entry, Mapping):
            term = str(entry.get("term") or entry.get("source") or key).strip()
            translation = _extract_translation(entry, lang_code)
            raw = dict(entry)
            raw["term"] = term
            raw["translation"] = translation
            target_translations = raw.get("target_translations")
            if isinstance(target_translations, Mapping):
                target_translations = dict(target_translations)
            else:
                target_translations = {}
            if translation:
                target_translations[lang_code] = translation
            raw["target_translations"] = target_translations
        else:
            continue
        if term and translation:
            raw["term_key"] = raw.get("term_key") or _term_key(term)
            out.append(raw)
    return out


def build_union(args: argparse.Namespace) -> None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)

    gt_entries: dict[str, Dict[str, Any]] = {}
    gt_occ = Counter()
    gt_translations = defaultdict(Counter)
    split_counts = Counter()
    skipped_missing_translation = 0
    skipped_empty_term = 0
    rows_seen = 0
    chunks_seen = 0

    for split_name, path in args.input_jsonl:
        for _, obj in _iter_jsonl(path):
            rows_seen += 1
            gt_by_chunk = obj.get("gt_terms_by_chunk")
            if gt_by_chunk is None:
                continue
            if not isinstance(gt_by_chunk, list):
                raise ValueError(f"gt_terms_by_chunk must be a list in {path}")
            for chunk_terms in gt_by_chunk:
                chunks_seen += 1
                if not chunk_terms:
                    continue
                if not isinstance(chunk_terms, list):
                    raise ValueError(f"gt_terms_by_chunk item must be a list in {path}")
                for item in chunk_terms:
                    if not isinstance(item, Mapping):
                        raise ValueError(f"gt term item must be an object in {path}")
                    term = str(item.get("term") or item.get("source") or "").strip()
                    key = _term_key(term)
                    if not key:
                        skipped_empty_term += 1
                        continue
                    translation = _extract_translation(item, args.lang_code)
                    if not translation:
                        skipped_missing_translation += 1
                        continue
                    gt_occ[key] += 1
                    split_counts[f"{split_name}:{key}"] += 1
                    gt_translations[key][translation] += 1
                    if key not in gt_entries:
                        gt_entries[key] = {
                            "term": term,
                            "term_key": key,
                            "translation": translation,
                            "target_translations": {args.lang_code: translation},
                            "source": "speech_llm_gt_terms",
                        }

    filler = _load_filler(args.filler_glossary, args.lang_code)
    seen = set()
    union: list[Dict[str, Any]] = []
    for key, entry in gt_entries.items():
        entry = dict(entry)
        entry["gt_occurrences"] = int(gt_occ[key])
        entry["gt_translation_options"] = dict(gt_translations[key].most_common(args.max_translation_options))
        union.append(entry)
        seen.add(key)

    filler_kept = 0
    filler_skipped_duplicate = 0
    for entry in filler:
        key = _term_key(str(entry.get("term") or ""))
        if not key:
            continue
        if key in seen:
            filler_skipped_duplicate += 1
            continue
        raw = dict(entry)
        raw.setdefault("source", "filler_glossary")
        raw["term_key"] = raw.get("term_key") or key
        union.append(raw)
        seen.add(key)
        filler_kept += 1

    args.output_json.write_text(json.dumps(union, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit = {
        "lang_code": args.lang_code,
        "rows_seen": rows_seen,
        "chunks_seen": chunks_seen,
        "gt_unique_terms": len(gt_entries),
        "gt_occurrences": int(sum(gt_occ.values())),
        "gt_skipped_empty_term": skipped_empty_term,
        "gt_skipped_missing_translation": skipped_missing_translation,
        "filler_glossary": str(args.filler_glossary),
        "filler_valid_entries": len(filler),
        "filler_kept": filler_kept,
        "filler_skipped_duplicate_with_gt": filler_skipped_duplicate,
        "union_entries": len(union),
        "output_json": str(args.output_json),
        "top_gt_terms": [{"term_key": k, "occurrences": int(v)} for k, v in gt_occ.most_common(50)],
    }
    if args.audit_json:
        args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", action="append", nargs=2, metavar=("SPLIT", "PATH"), required=True)
    parser.add_argument("--filler-glossary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--lang-code", default="zh")
    parser.add_argument("--max-translation-options", type=int, default=5)
    args = parser.parse_args()
    args.input_jsonl = [(split, Path(path)) for split, path in args.input_jsonl]
    for _, path in args.input_jsonl:
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.filler_glossary.exists():
        raise FileNotFoundError(args.filler_glossary)
    return args


if __name__ == "__main__":
    build_union(parse_args())
