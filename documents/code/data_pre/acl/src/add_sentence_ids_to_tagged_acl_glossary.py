#!/usr/bin/env python3
"""Add ACL6060 sentence ids to tagged ACL glossaries.

The tagged ACL glossary is term-indexed, while offline SLM readouts often need
sentence-aligned term maps. This script keeps the original glossary order and
schema, adds sentence occurrence metadata, and writes explicit per-language
sentence term-map files compatible with offline_sst_eval.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[5]

DEFAULT_XML = (
    "/mnt/data/siqiouyang/datasets/acl6060/dev/text/xml/"
    "ACL.6060.dev.en-xx.en.xml"
)
DEFAULT_TAGGED_TEXT = (
    "/mnt/data/siqiouyang/datasets/acl6060/dev/text/tagged_terminology/"
    "ACL.6060.dev.tagged.en-xx.en.txt"
)
DEFAULT_SOURCE_TEXT = (
    "/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/"
    "ACL.6060.dev.en-xx.en.txt"
)
DEFAULT_BASE_GLOSSARY = REPO_ROOT / "documents/data/data_pre/glossary_acl6060.json"
DEFAULT_OUT_DIR = Path("/mnt/gemini/home/jiaxuanluo/eval_glossaries")

TERM_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
GLOSSARY_MATCH_PUNCT_RE = re.compile(r"[^a-z0-9']+")


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _normalize_glossary_match_word(word: str) -> str:
    word = word.strip().lower().replace("\u2019", "'")
    if word.endswith("'s"):
        word = word[:-2]
    word = GLOSSARY_MATCH_PUNCT_RE.sub("", word)
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("es") and not word.endswith(("ses", "xes")):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def _glossary_match_norm_char_count(text: str) -> int:
    return sum(
        len(tok)
        for tok in (_normalize_glossary_match_word(w) for w in str(text or "").split())
        if tok
    )


def _term_key(text: str) -> str:
    return str(text or "").strip().lower()


def _clean_tagged_line(line: str) -> str:
    text = TERM_BRACKET_RE.sub(lambda m: m.group(1), line)
    return re.sub(r"\s+", " ", text.strip())


def load_xml_sentence_meta(xml_path: Path) -> Dict[int, Dict[str, Any]]:
    tree = ET.parse(xml_path)
    meta: Dict[int, Dict[str, Any]] = {}
    for doc in tree.getroot().iter("doc"):
        docid = str(doc.get("docid") or "").strip()
        if not docid:
            raise ValueError(f"Missing docid in XML: {xml_path}")
        segs = doc.findall("seg")
        if not segs:
            raise ValueError(f"No segments for docid={docid}: {xml_path}")
        first_seg_id = int(segs[0].get("id", "0"))
        last_seg_id = int(segs[-1].get("id", "0"))
        for local_idx, seg in enumerate(segs):
            sentence_id = int(seg.get("id", "0"))
            if sentence_id <= 0:
                raise ValueError(f"Invalid sentence id in docid={docid}: {xml_path}")
            if sentence_id in meta:
                raise ValueError(f"Duplicate sentence id={sentence_id}: {xml_path}")
            meta[sentence_id] = {
                "sentence_id": sentence_id,
                "sentence_index": sentence_id - 1,
                "docid": docid,
                "doc_sentence_index": local_idx,
                "doc_first_sentence_id": first_seg_id,
                "doc_last_sentence_id": last_seg_id,
            }
    expected = set(range(1, len(meta) + 1))
    if set(meta) != expected:
        missing = sorted(expected - set(meta))[:10]
        extra = sorted(set(meta) - expected)[:10]
        raise ValueError(f"XML sentence ids are not contiguous: missing={missing} extra={extra}")
    return meta


def validate_tagged_against_source(tagged_lines: Sequence[str], source_lines: Sequence[str]) -> None:
    if len(tagged_lines) != len(source_lines):
        raise ValueError(f"tagged lines {len(tagged_lines)} != source lines {len(source_lines)}")
    mismatches: List[Tuple[int, str, str]] = []
    for idx, (tagged, source) in enumerate(zip(tagged_lines, source_lines), start=1):
        clean = _clean_tagged_line(tagged)
        source_norm = re.sub(r"\s+", " ", source.strip())
        if clean != source_norm:
            mismatches.append((idx, clean, source_norm))
            if len(mismatches) >= 5:
                break
    if mismatches:
        details = "; ".join(
            f"id={idx} tagged={clean[:80]!r} source={source[:80]!r}"
            for idx, clean, source in mismatches
        )
        raise ValueError(f"Tagged/source sentence mismatch: {details}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_glossary_entries(data: Any) -> Iterable[Tuple[Optional[str], Mapping[str, Any]]]:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, Mapping):
                yield str(key), value
            else:
                yield str(key), {"term": str(key)}
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping):
                yield None, item
    else:
        raise ValueError(f"Unsupported glossary format: {type(data).__name__}")


def entry_term_key(key: Optional[str], entry: Mapping[str, Any]) -> str:
    if key is not None and str(key).strip():
        return _term_key(key)
    return _term_key(str(entry.get("term") or entry.get("source") or ""))


def load_base_terms(base_glossary_path: Path, min_norm_chars: int) -> Tuple[Dict[str, Dict[str, Any]], Counter]:
    data = load_json(base_glossary_path)
    base: Dict[str, Dict[str, Any]] = {}
    stats: Counter = Counter()
    for key, entry in iter_glossary_entries(data):
        term_key = entry_term_key(key, entry)
        if not term_key:
            stats["base_skipped_empty"] += 1
            continue
        if _glossary_match_norm_char_count(term_key) < min_norm_chars:
            stats["base_skipped_short"] += 1
            continue
        item = dict(entry)
        item.setdefault("term", str(entry.get("term") or key or term_key))
        base[term_key] = item
    stats["base_terms_kept"] = len(base)
    return base, stats


def collect_occurrences(
    tagged_lines: Sequence[str],
    sentence_meta: Mapping[int, Mapping[str, Any]],
    base_terms: Mapping[str, Mapping[str, Any]],
    min_norm_chars: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Counter]:
    by_term: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    for sentence_index, tagged in enumerate(tagged_lines):
        sentence_id = sentence_index + 1
        meta = sentence_meta.get(sentence_id)
        if meta is None:
            raise KeyError(f"Missing XML metadata for sentence_id={sentence_id}")
        stats["sentences_seen"] += 1
        for match in TERM_BRACKET_RE.finditer(tagged):
            surface = match.group(1).strip()
            term_key = _term_key(surface)
            stats["raw_bracket_terms"] += 1
            if not term_key:
                stats["skipped_empty_term"] += 1
                continue
            if _glossary_match_norm_char_count(term_key) < min_norm_chars:
                stats["skipped_short_term"] += 1
                continue
            if term_key not in base_terms:
                stats["skipped_not_in_base_glossary"] += 1
                continue
            row = {
                "term": str(base_terms[term_key].get("term") or surface),
                "term_key": term_key,
                "surface": surface,
                "occurrence_source": "tagged_bracket_exact",
                "sentence_id": sentence_id,
                "sentence_index": sentence_index,
                "docid": meta["docid"],
                "doc_sentence_index": meta["doc_sentence_index"],
                "char_start_in_tagged_line": match.start(1),
                "char_end_in_tagged_line": match.end(1),
            }
            by_term[term_key].append(row)
            rows.append(row)
            stats["kept_occurrences"] += 1
    stats["observed_unique_terms"] = len(by_term)
    return dict(by_term), rows, stats


def _source_term_pattern(term_key: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z0-9])" + re.escape(term_key) + r"(?![a-z0-9])")


def backfill_source_text_exact_occurrences(
    *,
    source_lines: Sequence[str],
    sentence_meta: Mapping[int, Mapping[str, Any]],
    base_terms: Mapping[str, Mapping[str, Any]],
    occurrences_by_term: MutableMapping[str, List[Dict[str, Any]]],
    occurrence_rows: List[Dict[str, Any]],
) -> Counter:
    """Backfill glossary terms that are exact source phrases but not bracketed.

    The tagged ACL source glossary contains a few phrase-level terms that the
    tagged transcript splits into adjacent bracketed components. These are not
    used by the historical 238-term raw eval glossary, but the source glossary
    itself can still receive sentence ids via deterministic source-text exact
    matching.
    """
    stats: Counter = Counter()
    missing_terms = [
        term_key
        for term_key in sorted(base_terms)
        if not occurrences_by_term.get(term_key)
    ]
    for term_key in missing_terms:
        pattern = _source_term_pattern(term_key)
        term_rows: List[Dict[str, Any]] = []
        for sentence_index, source_text in enumerate(source_lines):
            source_lower = source_text.lower()
            for match in pattern.finditer(source_lower):
                sentence_id = sentence_index + 1
                meta = sentence_meta[sentence_id]
                row = {
                    "term": str(base_terms[term_key].get("term") or term_key),
                    "term_key": term_key,
                    "surface": source_text[match.start():match.end()],
                    "occurrence_source": "source_text_exact_phrase_backfill",
                    "sentence_id": sentence_id,
                    "sentence_index": sentence_index,
                    "docid": meta["docid"],
                    "doc_sentence_index": meta["doc_sentence_index"],
                    "char_start_in_source_line": match.start(),
                    "char_end_in_source_line": match.end(),
                    "char_start_in_tagged_line": None,
                    "char_end_in_tagged_line": None,
                }
                term_rows.append(row)
        if term_rows:
            occurrences_by_term.setdefault(term_key, []).extend(term_rows)
            occurrence_rows.extend(term_rows)
            stats["backfilled_terms"] += 1
            stats["backfilled_occurrences"] += len(term_rows)
        else:
            stats["missing_base_terms_after_backfill"] += 1
    return stats


def _unique_preserve(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def occurrence_metadata(occurrences: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "sentence_ids": _unique_preserve(int(x["sentence_id"]) for x in occurrences),
        "sentence_indices": _unique_preserve(int(x["sentence_index"]) for x in occurrences),
        "utter_ids": _unique_preserve(str(x["docid"]) for x in occurrences),
        "doc_sentence_ids": _unique_preserve(
            f"{x['docid']}:{int(x['sentence_id'])}" for x in occurrences
        ),
        "occurrence_count": len(occurrences),
        "occurrences": list(occurrences),
    }


def enrich_entry(
    key: Optional[str],
    entry: Mapping[str, Any],
    occurrences_by_term: Mapping[str, Sequence[Mapping[str, Any]]],
    min_norm_chars: int,
) -> Dict[str, Any]:
    out = copy.deepcopy(dict(entry))
    term_key = entry_term_key(key, entry)
    source = str(entry.get("source") or "").strip()
    eligible_source = source in {"", "acl_tagged_gt", "tagged_gold"}
    eligible = (
        bool(term_key)
        and eligible_source
        and _glossary_match_norm_char_count(term_key) >= min_norm_chars
    )
    occurrences = list(occurrences_by_term.get(term_key, [])) if eligible else []
    out.setdefault("term", str(entry.get("term") or key or term_key))
    out["term_key"] = term_key
    out.update(occurrence_metadata(occurrences))
    out["sentence_id_source"] = (
        "ACL.6060.dev.en-xx.en.xml + ACL.6060.dev.tagged.en-xx.en.txt"
        if occurrences else ""
    )
    return out


def enrich_glossary(
    input_path: Path,
    output_path: Path,
    occurrences_by_term: Mapping[str, Sequence[Mapping[str, Any]]],
    min_norm_chars: int,
    allow_missing_tagged_gt: bool,
) -> Dict[str, Any]:
    data = load_json(input_path)
    stats: Counter = Counter()
    missing_gt: List[str] = []
    if isinstance(data, dict):
        out: Any = {}
        for key, entry in iter_glossary_entries(data):
            enriched = enrich_entry(key, entry, occurrences_by_term, min_norm_chars)
            out[str(key)] = enriched
            stats["entries"] += 1
            if enriched["occurrence_count"]:
                stats["entries_with_sentence_ids"] += 1
    elif isinstance(data, list):
        out = []
        for key, entry in iter_glossary_entries(data):
            enriched = enrich_entry(key, entry, occurrences_by_term, min_norm_chars)
            out.append(enriched)
            stats["entries"] += 1
            source = str(entry.get("source") or "")
            if enriched["occurrence_count"]:
                stats["entries_with_sentence_ids"] += 1
            elif (
                source == "acl_tagged_gt"
                and _glossary_match_norm_char_count(enriched["term_key"]) >= min_norm_chars
            ):
                missing_gt.append(enriched["term_key"])
    else:
        raise ValueError(f"Unsupported glossary format: {input_path}")

    if missing_gt and not allow_missing_tagged_gt:
        preview = ", ".join(sorted(missing_gt)[:20])
        raise ValueError(f"Missing sentence ids for {len(missing_gt)} tagged GT entries: {preview}")

    _write_json_atomic(output_path, out)
    stats["output_path"] = str(output_path)
    stats["input_path"] = str(input_path)
    stats["missing_tagged_gt_entries"] = len(missing_gt)
    return dict(stats)


def build_sentence_maps(
    tagged_lines: Sequence[str],
    source_lines: Sequence[str],
    sentence_meta: Mapping[int, Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
    base_terms: Mapping[str, Mapping[str, Any]],
    langs: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    refs_by_sentence: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for occ in occurrences:
        term_key = str(occ["term_key"])
        base_entry = base_terms[term_key]
        translations = base_entry.get("target_translations") or {}
        refs_by_sentence[int(occ["sentence_id"])][term_key] = {
            "term": str(base_entry.get("term") or occ.get("term") or term_key),
            "key": term_key,
            "target_translations": translations,
            "source": "acl_tagged_gt",
        }

    maps: Dict[str, List[Dict[str, Any]]] = {}
    for lang in langs:
        rows: List[Dict[str, Any]] = []
        for sentence_index, tagged in enumerate(tagged_lines):
            sentence_id = sentence_index + 1
            meta = sentence_meta[sentence_id]
            references: List[Dict[str, Any]] = []
            for term_key, ref in sorted(refs_by_sentence.get(sentence_id, {}).items()):
                translations = ref.get("target_translations") or {}
                translation = str(translations.get(lang) or "").strip()
                if not translation:
                    continue
                references.append(
                    {
                        "term": ref["term"],
                        "key": term_key,
                        "translation": translation,
                        "target_lang": lang,
                        "source": "acl_tagged_gt",
                        "sentence_id": sentence_id,
                        "sentence_index": sentence_index,
                        "docid": meta["docid"],
                        "doc_sentence_index": meta["doc_sentence_index"],
                    }
                )
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "sentence_index": sentence_index,
                    "docid": meta["docid"],
                    "doc_sentence_index": meta["doc_sentence_index"],
                    "source_text": source_lines[sentence_index],
                    "tagged_text": tagged,
                    "references": references,
                }
            )
        maps[lang] = rows

    multilang_rows: List[Dict[str, Any]] = []
    for sentence_index, tagged in enumerate(tagged_lines):
        sentence_id = sentence_index + 1
        meta = sentence_meta[sentence_id]
        references = []
        for term_key, ref in sorted(refs_by_sentence.get(sentence_id, {}).items()):
            references.append(
                {
                    "term": ref["term"],
                    "key": term_key,
                    "target_translations": ref.get("target_translations") or {},
                    "source": "acl_tagged_gt",
                    "sentence_id": sentence_id,
                    "sentence_index": sentence_index,
                    "docid": meta["docid"],
                    "doc_sentence_index": meta["doc_sentence_index"],
                }
            )
        multilang_rows.append(
            {
                "sentence_id": sentence_id,
                "sentence_index": sentence_index,
                "docid": meta["docid"],
                "doc_sentence_index": meta["doc_sentence_index"],
                "source_text": source_lines[sentence_index],
                "tagged_text": tagged,
                "references": references,
            }
        )
    maps["multilang"] = multilang_rows
    return maps


def parse_glossary_arg(raw: str) -> Tuple[Path, Path]:
    if "=" not in raw:
        raise ValueError(f"--glossary must be INPUT=OUTPUT, got: {raw}")
    left, right = raw.split("=", 1)
    if not left.strip() or not right.strip():
        raise ValueError(f"--glossary must be INPUT=OUTPUT, got: {raw}")
    return Path(left), Path(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=Path(DEFAULT_XML))
    parser.add_argument("--tagged-text", type=Path, default=Path(DEFAULT_TAGGED_TEXT))
    parser.add_argument("--source-text", type=Path, default=Path(DEFAULT_SOURCE_TEXT))
    parser.add_argument("--base-glossary", type=Path, default=DEFAULT_BASE_GLOSSARY)
    parser.add_argument("--min-norm-chars", type=int, default=2)
    parser.add_argument(
        "--glossary",
        action="append",
        default=[],
        help="Input and output glossary pair, formatted as INPUT=OUTPUT.",
    )
    parser.add_argument(
        "--sentence-term-map-prefix",
        type=Path,
        default=DEFAULT_OUT_DIR / "acl6060_tagged_sentence_term_map_min_norm2",
    )
    parser.add_argument(
        "--term-occurrences-jsonl",
        type=Path,
        default=DEFAULT_OUT_DIR / "acl6060_tagged_term_occurrences_min_norm2.jsonl",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=DEFAULT_OUT_DIR / "acl6060_tagged_sentence_ids_min_norm2_stats.json",
    )
    parser.add_argument("--langs", nargs="+", default=["zh", "ja", "de"])
    parser.add_argument("--allow-missing-tagged-gt", action="store_true")
    parser.add_argument(
        "--no-backfill-source-text-exact",
        action="store_true",
        help="Disable exact source-text sentence-id backfill for source glossary terms.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for label, path in [
        ("xml", args.xml),
        ("tagged_text", args.tagged_text),
        ("source_text", args.source_text),
        ("base_glossary", args.base_glossary),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    glossary_pairs = [parse_glossary_arg(x) for x in args.glossary]
    if not glossary_pairs:
        raise ValueError("At least one --glossary INPUT=OUTPUT pair is required")
    for input_path, _ in glossary_pairs:
        if not input_path.is_file():
            raise FileNotFoundError(f"glossary input not found: {input_path}")

    sentence_meta = load_xml_sentence_meta(args.xml)
    tagged_lines = _read_lines(args.tagged_text)
    source_lines = _read_lines(args.source_text)
    if len(sentence_meta) != len(tagged_lines):
        raise ValueError(f"XML sentences {len(sentence_meta)} != tagged lines {len(tagged_lines)}")
    validate_tagged_against_source(tagged_lines, source_lines)

    base_terms, base_stats = load_base_terms(args.base_glossary, args.min_norm_chars)
    occurrences_by_term, occurrence_rows, occurrence_stats = collect_occurrences(
        tagged_lines=tagged_lines,
        sentence_meta=sentence_meta,
        base_terms=base_terms,
        min_norm_chars=args.min_norm_chars,
    )
    backfill_stats: Counter = Counter()
    if not args.no_backfill_source_text_exact:
        backfill_stats = backfill_source_text_exact_occurrences(
            source_lines=source_lines,
            sentence_meta=sentence_meta,
            base_terms=base_terms,
            occurrences_by_term=occurrences_by_term,
            occurrence_rows=occurrence_rows,
        )
        occurrence_stats["observed_unique_terms_after_backfill"] = len(occurrences_by_term)
        occurrence_stats["kept_occurrences_after_backfill"] = len(occurrence_rows)
    if not occurrence_rows:
        raise ValueError("No tagged ACL occurrences were kept; refusing to write empty metadata")

    glossary_stats = {}
    for input_path, output_path in glossary_pairs:
        glossary_stats[str(output_path)] = enrich_glossary(
            input_path=input_path,
            output_path=output_path,
            occurrences_by_term=occurrences_by_term,
            min_norm_chars=args.min_norm_chars,
            allow_missing_tagged_gt=args.allow_missing_tagged_gt,
        )

    sentence_maps = build_sentence_maps(
        tagged_lines=tagged_lines,
        source_lines=source_lines,
        sentence_meta=sentence_meta,
        occurrences=occurrence_rows,
        base_terms=base_terms,
        langs=args.langs,
    )
    sentence_map_paths: Dict[str, str] = {}
    sentence_map_stats: Dict[str, Any] = {}
    for lang, rows in sentence_maps.items():
        if lang == "multilang":
            out_path = args.sentence_term_map_prefix.with_name(
                args.sentence_term_map_prefix.name + "_multilang.json"
            )
        else:
            out_path = args.sentence_term_map_prefix.with_name(
                args.sentence_term_map_prefix.name + f"_{lang}.json"
            )
        _write_json_atomic(out_path, rows)
        sentence_map_paths[lang] = str(out_path)
        sentence_map_stats[lang] = {
            "sentences": len(rows),
            "sentences_with_references": sum(1 for row in rows if row.get("references")),
            "references": sum(len(row.get("references") or []) for row in rows),
        }

    _write_jsonl_atomic(args.term_occurrences_jsonl, occurrence_rows)

    stats = {
        "xml": str(args.xml),
        "tagged_text": str(args.tagged_text),
        "source_text": str(args.source_text),
        "base_glossary": str(args.base_glossary),
        "min_norm_chars": args.min_norm_chars,
        "sentence_count": len(tagged_lines),
        "base": dict(base_stats),
        "occurrences": dict(occurrence_stats),
        "source_text_exact_backfill": dict(backfill_stats),
        "glossaries": glossary_stats,
        "sentence_term_maps": sentence_map_stats,
        "sentence_term_map_paths": sentence_map_paths,
        "term_occurrences_jsonl": str(args.term_occurrences_jsonl),
    }
    _write_json_atomic(args.stats_json, stats)

    print(f"[ACL-TAGGED-SENTIDS] stats={args.stats_json}")
    print(f"[ACL-TAGGED-SENTIDS] term_occurrences={args.term_occurrences_jsonl}")
    for lang, path in sentence_map_paths.items():
        print(f"[ACL-TAGGED-SENTIDS] sentence_term_map_{lang}={path}")
    for output_path in glossary_stats:
        print(f"[ACL-TAGGED-SENTIDS] enriched_glossary={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
