#!/usr/bin/env python3
"""Backfill target translations into the strict medicine glossary.

The strict medicine retriever glossary is source-term-only because the retriever
only needs English terms.  SimulEval term_map injection needs target-language
translations, so this script copies translations from
``outputs_v2/test/sample_*_v2/sentences_v2.json`` for strict GT terms, then
fills remaining wiki-filler terms from ``wiki_glossary_medicine_enriched.json``.
Entries without any translation source are preserved and reported.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple


DEFAULT_ESO_TEST_ROOT = Path("/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test")
DEFAULT_INPUT_GLOSSARY = Path(
    "/mnt/gemini/home/jiaxuanluo/"
    "medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/"
    "medicine_glossary_gt_plus_medicine_wiki_gs10000.json"
)
DEFAULT_OUTPUT_GLOSSARY = Path(
    "/mnt/gemini/home/jiaxuanluo/"
    "medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/"
    "medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json"
)
DEFAULT_STATS = Path(
    "/mnt/gemini/home/jiaxuanluo/"
    "medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/"
    "medicine_glossary_gt_plus_medicine_wiki_gs10000_translated_stats.json"
)
DEFAULT_WIKI_ENRICHED_GLOSSARY = Path(
    "/home/jiaxuanluo/InfiniSST/"
    "documents/code/data_pre/glossary_scale/wiki_glossary_medicine_enriched.json"
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _iter_sentence_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("sample_*_v2/sentences_v2.json"))


def _collect_eso_translations(root: Path) -> Tuple[Dict[str, Dict[str, Counter]], Dict[str, Any]]:
    by_term: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    surface_by_term: Dict[str, Counter] = defaultdict(Counter)
    sample_term_counts: Dict[str, int] = {}
    sentence_files = list(_iter_sentence_files(root))
    if not sentence_files:
        raise FileNotFoundError(f"No ESO sentence files found under {root}")

    for path in sentence_files:
        sample_id = path.parent.name.split("_")[1]
        sentences = _read_json(path)
        if not isinstance(sentences, list):
            raise ValueError(f"Expected list in {path}")
        n_terms = 0
        for sent in sentences:
            for term_entry in sent.get("terms") or []:
                if not isinstance(term_entry, Mapping):
                    continue
                term = str(term_entry.get("term") or "").strip()
                term_key = _norm(term)
                if not term_key:
                    continue
                surface_by_term[term_key][term] += 1
                translations = term_entry.get("target_translations") or {}
                if not isinstance(translations, Mapping):
                    continue
                kept = False
                for lang in ("zh", "de", "ja"):
                    value = str(translations.get(lang) or "").strip()
                    if value:
                        by_term[term_key][lang][value] += 1
                        kept = True
                if kept:
                    n_terms += 1
        sample_term_counts[sample_id] = n_terms

    stats = {
        "eso_test_root": str(root),
        "sentence_files": [str(p) for p in sentence_files],
        "sample_term_counts": sample_term_counts,
        "unique_eso_terms_with_translation": len(by_term),
    }
    return by_term, {"stats": stats, "surface_by_term": surface_by_term}


def _collect_wiki_translations(path: Path) -> Tuple[Dict[str, Dict[str, Counter]], Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in wiki enriched glossary: {path}")
    by_term: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    entries_with_any_translation = 0
    entries_with_all_translations = 0
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        term = str(entry.get("term") or "").strip()
        term_key = _norm(term)
        translations = entry.get("target_translations") or {}
        if not term_key or not isinstance(translations, Mapping):
            continue
        kept_langs = 0
        for lang in ("zh", "de", "ja"):
            value = str(translations.get(lang) or "").strip()
            if value:
                by_term[term_key][lang][value] += 1
                kept_langs += 1
        if kept_langs:
            entries_with_any_translation += 1
        if kept_langs == 3:
            entries_with_all_translations += 1
    stats = {
        "wiki_enriched_glossary": str(path),
        "wiki_entries": len(data),
        "wiki_unique_terms_with_translation": len(by_term),
        "wiki_entries_with_any_translation": entries_with_any_translation,
        "wiki_entries_with_all_translations": entries_with_all_translations,
    }
    return by_term, {"stats": stats}


def _choose_translation(counter: Counter) -> Tuple[str, List[Dict[str, Any]]]:
    if not counter:
        return "", []
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = ranked[0][0]
    alternatives = [
        {"translation": text, "count": count}
        for text, count in ranked
    ]
    return chosen, alternatives


def _entry_term(entry: Mapping[str, Any], fallback: str = "") -> str:
    return str(entry.get("term") or entry.get("source") or fallback or "").strip()


def _enrich_entry(
    entry: MutableMapping[str, Any],
    term_key: str,
    translations_by_lang: Mapping[str, Counter],
    *,
    overwrite_existing: bool,
    source_name: str,
) -> Tuple[bool, Dict[str, Any]]:
    existing = dict(entry.get("target_translations") or {})
    existing_sources = dict(entry.get("translation_sources") or {})
    chosen: Dict[str, str] = {}
    alternatives: Dict[str, List[Dict[str, Any]]] = {}
    changed = False

    for lang in ("zh", "de", "ja"):
        candidate, ranked = _choose_translation(translations_by_lang.get(lang, Counter()))
        if ranked:
            alternatives[lang] = ranked
        if not candidate:
            continue
        if overwrite_existing or not str(existing.get(lang) or "").strip():
            existing[lang] = candidate
            existing_sources[lang] = source_name
            chosen[lang] = candidate
            changed = True

    if changed:
        entry["target_translations"] = existing
        entry["translation_sources"] = existing_sources
        entry["translation_source"] = "+".join(sorted(set(existing_sources.values())))
        entry["translation_source_term_key"] = term_key
        if alternatives:
            current_alternatives = dict(entry.get("translation_alternatives") or {})
            current_alternatives[source_name] = alternatives
            entry["translation_alternatives"] = current_alternatives
    return changed, {"chosen": chosen, "alternatives": alternatives}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eso-test-root", type=Path, default=DEFAULT_ESO_TEST_ROOT)
    parser.add_argument("--input-glossary", type=Path, default=DEFAULT_INPUT_GLOSSARY)
    parser.add_argument("--wiki-enriched-glossary", type=Path, default=DEFAULT_WIKI_ENRICHED_GLOSSARY)
    parser.add_argument("--output-glossary", type=Path, default=DEFAULT_OUTPUT_GLOSSARY)
    parser.add_argument("--stats-json", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--conflict-examples", type=int, default=50)
    args = parser.parse_args()

    data = _read_json(args.input_glossary)
    eso_by_term, eso_meta = _collect_eso_translations(args.eso_test_root)
    wiki_by_term, wiki_meta = _collect_wiki_translations(args.wiki_enriched_glossary)
    surface_by_term: Dict[str, Counter] = eso_meta["surface_by_term"]

    entries_with_translation_after = 0
    entries_with_existing_translation = 0
    untranslated_entries = 0
    entries_filled_by_eso = 0
    entries_filled_by_wiki = 0
    conflict_examples: List[Dict[str, Any]] = []
    matched_terms: List[str] = []

    def enrich_one(entry: MutableMapping[str, Any], fallback: str = "") -> None:
        nonlocal entries_with_translation_after, entries_with_existing_translation
        nonlocal untranslated_entries, entries_filled_by_eso, entries_filled_by_wiki
        term = _entry_term(entry, fallback)
        term_key = _norm(term)
        if not term_key:
            untranslated_entries += 1
            return
        if entry.get("target_translations"):
            entries_with_existing_translation += 1
        changed_eso = False
        changed_wiki = False
        detail: Dict[str, Any] = {"alternatives": {}}

        translations = eso_by_term.get(term_key)
        if translations:
            changed_eso, detail = _enrich_entry(
                entry,
                term_key,
                translations,
                overwrite_existing=args.overwrite_existing,
                source_name="eso_v2_sentence_terms",
            )
            if changed_eso:
                entries_filled_by_eso += 1

        translations = wiki_by_term.get(term_key)
        if translations:
            changed_wiki, wiki_detail = _enrich_entry(
                entry,
                term_key,
                translations,
                overwrite_existing=False,
                source_name="wiki_medicine_enriched",
            )
            if changed_wiki:
                entries_filled_by_wiki += 1
            merged_alternatives = dict(detail.get("alternatives") or {})
            merged_alternatives.update(wiki_detail.get("alternatives") or {})
            detail = {"alternatives": merged_alternatives}

        if entry.get("target_translations"):
            entries_with_translation_after += 1
            matched_terms.append(term_key)
            has_conflict = any(len(v) > 1 for v in detail.get("alternatives", {}).values())
            if has_conflict and len(conflict_examples) < args.conflict_examples:
                conflict_examples.append(
                    {
                        "term_key": term_key,
                        "surface_forms": surface_by_term.get(term_key, Counter()).most_common(),
                        "alternatives": detail.get("alternatives", {}),
                    }
                )
        else:
            untranslated_entries += 1

    if isinstance(data, dict):
        output = {}
        for key, raw_entry in data.items():
            if not isinstance(raw_entry, MutableMapping):
                output[key] = raw_entry
                untranslated_entries += 1
                continue
            entry = dict(raw_entry)
            enrich_one(entry, str(key))
            output[key] = entry
    elif isinstance(data, list):
        output = []
        for raw_entry in data:
            if not isinstance(raw_entry, MutableMapping):
                output.append(raw_entry)
                untranslated_entries += 1
                continue
            entry = dict(raw_entry)
            enrich_one(entry)
            output.append(entry)
    else:
        raise ValueError(
            f"Unsupported glossary format in {args.input_glossary}: {type(data).__name__}"
        )

    total_entries = len(data)
    stats = {
        "input_glossary": str(args.input_glossary),
        "output_glossary": str(args.output_glossary),
        "eso_test_root": str(args.eso_test_root),
        "wiki_enriched_glossary": str(args.wiki_enriched_glossary),
        "total_entries": total_entries,
        "translated_entries": entries_with_translation_after,
        "untranslated_entries": untranslated_entries,
        "entries_with_existing_translation_before_enrich": entries_with_existing_translation,
        "entries_filled_by_eso": entries_filled_by_eso,
        "entries_filled_by_wiki": entries_filled_by_wiki,
        "unique_matched_terms": len(set(matched_terms)),
        "unique_eso_terms_with_translation": eso_meta["stats"]["unique_eso_terms_with_translation"],
        "unique_wiki_terms_with_translation": wiki_meta["stats"]["wiki_unique_terms_with_translation"],
        "wiki_entries_with_all_translations": wiki_meta["stats"]["wiki_entries_with_all_translations"],
        "sample_term_counts": eso_meta["stats"]["sample_term_counts"],
        "sentence_files": eso_meta["stats"]["sentence_files"],
        "conflict_examples": conflict_examples,
        "policy": {
            "match": "casefolded whitespace-normalized source term exact match",
            "translation_choice": "ESO v2 sentence term translation first; wiki enriched glossary fills missing languages; lexical tie-break within each source",
            "overwrite_existing": bool(args.overwrite_existing),
            "preserve_untranslated_entries": True,
        },
    }
    _write_json(args.output_glossary, output)
    _write_json(args.stats_json, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
