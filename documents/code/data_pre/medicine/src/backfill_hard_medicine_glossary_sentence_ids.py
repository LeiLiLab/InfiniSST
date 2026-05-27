#!/usr/bin/env python3
"""Backfill scorer-aligned ESO sentence ids for the hard-medicine glossary.

The manual hard-term glossary stores the evidence sentences that selected each
term. Offline oracle translation, however, needs every sentence where that term
can be supplied as a term-map entry under the same target-translation dedup rule
used by TERM_ACC. This script preserves the manual evidence fields and writes
both full source occurrences and scorer-aligned per-language sentence ids.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_GLOSSARY = Path(
    "/mnt/gemini/home/jiaxuanluo/medicine_eval_hard_terms_llm_judge_manual_20260524/"
    "hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json"
)
DEFAULT_STATS = Path(
    "/mnt/gemini/home/jiaxuanluo/medicine_eval_hard_terms_llm_judge_manual_20260524/"
    "hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212_stats.json"
)
DEFAULT_ESO_ROOT = Path("/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2_final/test")
DEFAULT_LANGS = ("zh", "de", "ja")


def normalise_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _source_pattern(term: str) -> Optional[re.Pattern[str]]:
    term_norm = normalise_space(term).casefold()
    if not term_norm:
        return None
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        return re.compile(r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])")
    return re.compile(re.escape(term_norm))


def find_source_match(source_text: str, variants: Sequence[str]) -> Optional[Tuple[str, int, int]]:
    source_norm = normalise_space(source_text).casefold()
    if not source_norm:
        return None
    for variant in sorted({normalise_space(v) for v in variants if normalise_space(v)}, key=len, reverse=True):
        pattern = _source_pattern(variant)
        if pattern is None:
            continue
        match = pattern.search(source_norm)
        if match:
            return variant, match.start(), match.end()
    return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def backup_existing(path: Path, suffix: str) -> Optional[Path]:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
    shutil.copy2(path, backup)
    return backup


def iter_entries(data: Any) -> Iterable[Tuple[Optional[str], Mapping[str, Any]]]:
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, Mapping):
                raise ValueError(f"non-object glossary row: {item!r}")
            yield None, item
    elif isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"non-object glossary entry for key={key!r}")
            yield str(key), value
    else:
        raise ValueError(f"unsupported glossary type: {type(data).__name__}")


def sample_dir(root: Path, sample_id: str) -> Path:
    for candidate in (root / f"sample_{sample_id}_v2", root / f"sample_{sample_id}"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No ESO sample dir for sample_id={sample_id} under {root}")


def infer_sample_ids(data: Any) -> List[str]:
    seen: List[str] = []
    used = set()
    for _, entry in iter_entries(data):
        for sample_id in entry.get("sample_ids") or []:
            sample = str(sample_id)
            if sample and sample not in used:
                used.add(sample)
                seen.append(sample)
    if not seen:
        raise ValueError("Could not infer sample ids from glossary sample_ids")
    return sorted(seen)


def load_sentences(root: Path, sample_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        path = sample_dir(root, sample_id) / "sentences_v2.json"
        sentences = read_json(path)
        if not isinstance(sentences, list):
            raise ValueError(f"Expected list in {path}")
        for sentence_index, sent in enumerate(sentences):
            if not isinstance(sent, Mapping):
                raise ValueError(f"non-object sentence row in {path}")
            sentence_id = str(sent.get("sentence_id"))
            source_text = str(sent.get("text") or "")
            translations = sent.get("translations") or {}
            if not sentence_id or sentence_id == "None":
                raise ValueError(f"missing sentence_id in {path} row={sentence_index}")
            if not source_text:
                raise ValueError(f"missing source text in {path} row={sentence_index}")
            if not isinstance(translations, Mapping):
                raise ValueError(f"missing translations dict in {path} row={sentence_index}")
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "sentence_id": sentence_id,
                    "sentence_index": sentence_index,
                    "source_text": source_text,
                    "translations": {str(k): str(v) for k, v in translations.items()},
                    "source_path": str(path),
                }
            )
    return rows


def unique_preserve(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def entry_variants(entry: Mapping[str, Any]) -> List[str]:
    variants = [str(entry.get("term") or "")]
    raw_sources = entry.get("raw_sources") or []
    if isinstance(raw_sources, Sequence) and not isinstance(raw_sources, (str, bytes)):
        variants.extend(str(x) for x in raw_sources)
    return [normalise_space(v) for v in variants if normalise_space(v)]


def entry_occurrences(
    entry: Mapping[str, Any],
    sentences: Sequence[Mapping[str, Any]],
    langs: Sequence[str],
) -> List[Dict[str, Any]]:
    variants = entry_variants(entry)
    translations = entry.get("target_translations") or {}
    if not variants:
        raise ValueError(f"entry has no source variants: {entry}")
    if not isinstance(translations, Mapping):
        raise ValueError(f"entry has malformed target_translations: {entry}")

    occurrences: List[Dict[str, Any]] = []
    for sent in sentences:
        match = find_source_match(str(sent["source_text"]), variants)
        if not match:
            continue
        matched_variant, start, end = match
        target_has_term: Dict[str, bool] = {}
        eligible_langs: List[str] = []
        for lang in langs:
            target = normalise_space(translations.get(lang, ""))
            has_term = bool(target) and target in str(sent["translations"].get(lang, ""))
            target_has_term[lang] = has_term
            if has_term:
                eligible_langs.append(lang)
        occurrences.append(
            {
                "sample_id": str(sent["sample_id"]),
                "sentence_id": str(sent["sentence_id"]),
                "sentence_index": int(sent["sentence_index"]),
                "sample_sentence_id": f"{sent['sample_id']}:{sent['sentence_id']}",
                "source_path": str(sent["source_path"]),
                "matched_source": matched_variant,
                "char_start_in_normalized_source": start,
                "char_end_in_normalized_source": end,
                "eligible_langs": eligible_langs,
                "target_has_term": target_has_term,
                "occurrence_source": "eso_v2_final_source_exact_match",
            }
        )
    return occurrences


def scorer_totals_by_target_dedup(data: Any, sentences: Sequence[Mapping[str, Any]], langs: Sequence[str]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for lang in langs:
        target_to_source: Dict[str, str] = {}
        for _, entry in iter_entries(data):
            translations = entry.get("target_translations") or {}
            target = normalise_space(translations.get(lang, ""))
            if not target:
                continue
            if target not in target_to_source:
                target_to_source[target] = str(entry.get("term") or "")
        total = 0
        for target, source in target_to_source.items():
            for sent in sentences:
                if find_source_match(str(sent["source_text"]), [source]) and target in str(sent["translations"].get(lang, "")):
                    total += 1
        totals[lang] = total
    return totals


def enrich_data(data: Any, sentences: Sequence[Mapping[str, Any]], langs: Sequence[str]) -> Tuple[Any, Counter, Dict[str, int]]:
    stats: Counter = Counter()

    def enrich_one(entry: Mapping[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(dict(entry))
        occurrences = entry_occurrences(entry, sentences, langs)
        if not occurrences:
            raise ValueError(f"No source occurrences found for term={entry.get('term')!r}")

        if "manual_evidence_sample_ids" not in out:
            out["manual_evidence_sample_ids"] = list(out.get("sample_ids") or [])
        if "manual_evidence_sentence_ids" not in out:
            out["manual_evidence_sentence_ids"] = list(out.get("sentence_ids") or [])

        out["sample_ids"] = unique_preserve(x["sample_id"] for x in occurrences)
        out["sentence_ids"] = unique_preserve(x["sentence_id"] for x in occurrences)
        out["sample_sentence_ids"] = unique_preserve(x["sample_sentence_id"] for x in occurrences)
        out["doc_sentence_ids"] = list(out["sample_sentence_ids"])
        out["occurrence_count"] = len(occurrences)
        out["occurrences"] = occurrences
        full_lang_sample_sentence_ids = {
            lang: unique_preserve(
                x["sample_sentence_id"] for x in occurrences if lang in x["eligible_langs"]
            )
            for lang in langs
        }
        full_lang_sentence_ids = {
            lang: unique_preserve(
                x["sentence_id"] for x in occurrences if lang in x["eligible_langs"]
            )
            for lang in langs
        }
        out["lang_sample_sentence_ids_before_target_dedup"] = full_lang_sample_sentence_ids
        out["lang_sentence_ids_before_target_dedup"] = full_lang_sentence_ids
        out["lang_occurrence_count_before_target_dedup"] = {
            lang: len(full_lang_sample_sentence_ids[lang]) for lang in langs
        }
        out["lang_sample_sentence_ids"] = copy.deepcopy(full_lang_sample_sentence_ids)
        out["lang_sentence_ids"] = copy.deepcopy(full_lang_sentence_ids)
        out["lang_occurrence_count"] = copy.deepcopy(out["lang_occurrence_count_before_target_dedup"])
        source_paths = unique_preserve(x["source_path"] for x in occurrences)
        out["sentence_id_source"] = source_paths[0] if len(source_paths) == 1 else source_paths
        out["sentence_id_scope"] = "full_source_occurrences_in_eso_v2_final_test"
        return out

    if isinstance(data, list):
        output: Any = []
        for entry in data:
            enriched = enrich_one(entry)
            output.append(enriched)
            stats["entries"] += 1
            stats["occurrence_count_sum"] += int(enriched["occurrence_count"])
    elif isinstance(data, dict):
        output = {}
        for key, entry in data.items():
            enriched = enrich_one(entry)
            output[key] = enriched
            stats["entries"] += 1
            stats["occurrence_count_sum"] += int(enriched["occurrence_count"])
    else:
        raise ValueError(f"unsupported glossary type: {type(data).__name__}")

    # Match stream_laal_term.py: for each target translation, keep only the
    # first source term in glossary order. Later entries with the same target
    # translation are not counted by TERM_ACC and should not receive oracle
    # sentence ids for that language.
    iterable = output if isinstance(output, list) else list(output.values())
    first_target_owner: Dict[str, Dict[str, int]] = {lang: {} for lang in langs}
    for idx, entry in enumerate(iterable):
        translations = entry.get("target_translations") or {}
        for lang in langs:
            target = normalise_space(translations.get(lang, ""))
            if target and target not in first_target_owner[lang]:
                first_target_owner[lang][target] = idx

    target_dedup_removed = Counter()
    for idx, entry in enumerate(iterable):
        translations = entry.get("target_translations") or {}
        status: Dict[str, str] = {}
        target_owner_index: Dict[str, Optional[int]] = {}
        for lang in langs:
            target = normalise_space(translations.get(lang, ""))
            owner = first_target_owner[lang].get(target) if target else None
            target_owner_index[lang] = owner
            if target and owner is not None and owner != idx:
                target_dedup_removed[lang] += len(entry["lang_sample_sentence_ids"].get(lang, []))
                entry["lang_sample_sentence_ids"][lang] = []
                entry["lang_sentence_ids"][lang] = []
                entry["lang_occurrence_count"][lang] = 0
                status[lang] = "dropped_duplicate_target_translation"
            elif target:
                status[lang] = "kept_first_target_translation"
            else:
                status[lang] = "missing_target_translation"
        entry["lang_target_dedup_status"] = status
        entry["lang_target_owner_index"] = target_owner_index

    stats["target_dedup_removed_zh"] = int(target_dedup_removed["zh"])
    stats["target_dedup_removed_de"] = int(target_dedup_removed["de"])
    stats["target_dedup_removed_ja"] = int(target_dedup_removed["ja"])
    dedup_totals = scorer_totals_by_target_dedup(output, sentences, langs)
    return output, stats, dedup_totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_GLOSSARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_GLOSSARY)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--eso-test-root", type=Path, default=DEFAULT_ESO_ROOT)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--langs", nargs="+", default=list(DEFAULT_LANGS))
    parser.add_argument("--backup-suffix", default=".before_sentence_backfill_20260525T055610")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    data = read_json(args.input)
    sample_ids = args.sample_ids or infer_sample_ids(data)
    sentences = load_sentences(args.eso_test_root, sample_ids)

    before_sentence_sum = 0
    for _, entry in iter_entries(data):
        before_sentence_sum += len(entry.get("sentence_ids") or [])

    output, stats, dedup_totals = enrich_data(data, sentences, args.langs)

    lang_occurrence_count_sum = Counter()
    for _, entry in iter_entries(output):
        for lang, count in (entry.get("lang_occurrence_count") or {}).items():
            lang_occurrence_count_sum[str(lang)] += int(count)

    output_backup = None
    stats_backup = None
    if not args.no_backup:
        if args.output.resolve() == args.input.resolve():
            output_backup = backup_existing(args.output, args.backup_suffix)
        if args.stats_output.exists():
            stats_backup = backup_existing(args.stats_output, args.backup_suffix)

    write_json_atomic(args.output, output)

    stats_json = {
        "input": str(args.input),
        "output": str(args.output),
        "eso_test_root": str(args.eso_test_root),
        "sample_ids": sample_ids,
        "sentence_count": len(sentences),
        "entries": int(stats["entries"]),
        "before_sentence_ids_sum": before_sentence_sum,
        "after_source_occurrence_count_sum": int(stats["occurrence_count_sum"]),
        "after_lang_occurrence_count_sum": dict(sorted(lang_occurrence_count_sum.items())),
        "scorer_term_total_by_target_dedup": dedup_totals,
        "target_dedup_removed_occurrences": {
            "zh": int(stats["target_dedup_removed_zh"]),
            "de": int(stats["target_dedup_removed_de"]),
            "ja": int(stats["target_dedup_removed_ja"]),
        },
        "output_backup": str(output_backup) if output_backup else None,
        "stats_backup": str(stats_backup) if stats_backup else None,
        "sentence_id_scope": "full_source_occurrences_in_eso_v2_final_test",
        "lang_sentence_id_scope": (
            "scorer_aligned_first_target_translation_source_occurrences_whose_target_translation_is_in_final_reference"
        ),
    }
    write_json_atomic(args.stats_output, stats_json)
    print(json.dumps(stats_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
