#!/usr/bin/env python3
"""Prepare one ESO medicine talk for SimulEval-style RAG evaluation.

The legacy oracle term_map used ESO v2 ``sentence.terms`` annotations, filtered
through a strict MFA-only term set.  That is useful for reproducing earlier
runs, but it is not a stable source of truth when the evaluation glossary
changes.  New oracle experiments should use ``--term-source glossary_match`` so
both the oracle prompt and TERM metrics are derived from the same glossary by
matching source and reference sentences.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


DEFAULT_ESO_TEST_ROOT = Path("/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test")
DEFAULT_STRICT_JSONL = Path(
    "/mnt/gemini/home/jiaxuanluo/"
    "medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/"
    "medicine_dev_dataset.jsonl"
)
DEFAULT_STRICT_GLOSSARY = Path(
    "/mnt/gemini/home/jiaxuanluo/"
    "medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/"
    "medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _norm(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _source_contains(source_text: str, term: str) -> bool:
    source_norm = _normalise_space(source_text).casefold()
    term_norm = _normalise_space(term).casefold()
    if not source_norm or not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        return re.search(pattern, source_norm) is not None
    return term_norm in source_norm


def _text_contains(text: str, needle: str) -> bool:
    text_norm = _normalise_space(text)
    needle_norm = _normalise_space(needle)
    return bool(needle_norm) and needle_norm in text_norm


def _resolve_cluster_path(path: str) -> str:
    s = str(path)
    if s.startswith("/home/"):
        candidate = Path("/mnt/taurus") / s.lstrip("/")
        if candidate.exists():
            return str(candidate)
    raw = Path(path)
    if raw.exists():
        return str(raw)
    return s


def _resolve_audio_path(sample_dir: Path, sample_id: str, metadata: Dict[str, Any]) -> str:
    """Prefer the audio staged next to the selected ESO sample directory.

    ESO metadata can carry absolute paths from the machine where the JSON was
    produced.  For cross-cluster evals, the portable contract is that
    --eso-test-root points at a staged sample tree containing the wav files.
    """

    local_audio = sample_dir / f"{sample_id}_v2.wav"
    if local_audio.is_file():
        return str(local_audio)
    return _resolve_cluster_path(
        str(metadata.get("converted_audio_path") or local_audio)
    )


def _sample_dir(root: Path, sample_id: str) -> Path:
    candidates = [
        root / f"sample_{sample_id}_v2",
        root / f"sample_{sample_id}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No ESO v2 sample dir found for sample_id={sample_id} under {root}")


def _strict_terms_for_sample(path: Path, sample_id: str) -> Tuple[set[str], Dict[str, int]]:
    terms: set[str] = set()
    stats = {
        "rows": 0,
        "term_rows": 0,
        "non_mfa_exact_term_rows": 0,
    }
    sample_ids = {sample_id, f"medicine_{sample_id}"}
    for rec in _iter_jsonl(path):
        if str(rec.get("sample_id") or "") not in sample_ids and str(rec.get("utter_id") or "") not in sample_ids:
            continue
        stats["rows"] += 1
        term = _norm(rec.get("term") or rec.get("term_key") or "")
        if not term:
            continue
        stats["term_rows"] += 1
        if rec.get("mfa_locate_method") != "mfa_exact":
            stats["non_mfa_exact_term_rows"] += 1
            continue
        terms.add(term)
    if not terms:
        raise ValueError(f"No strict MFA-exact terms found for sample_id={sample_id} in {path}")
    return terms, stats


def _strict_glossary_terms(path: Path) -> set[str]:
    data = _read_json(path)
    if isinstance(data, dict):
        entries = data.values()
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unsupported glossary format: {path}")
    out = set()
    for entry in entries:
        if isinstance(entry, dict):
            term = _norm(entry.get("term") or entry.get("source") or "")
            if term:
                out.add(term)
    return out


def _load_translated_glossary(path: Path, lang_code: str) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if isinstance(data, dict):
        entries = data.values()
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unsupported glossary format: {path}")

    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        term = _normalise_space(entry.get("term") or entry.get("source") or "")
        target_translations = entry.get("target_translations") or {}
        translation = ""
        if isinstance(target_translations, dict):
            translation = _normalise_space(target_translations.get(lang_code) or "")
        if not translation:
            translation = _normalise_space(
                entry.get("translation")
                or entry.get("target_translation")
                or entry.get(lang_code)
                or ""
            )
        if not term or not translation:
            continue
        key = (_norm(term), translation)
        if key in seen:
            continue
        seen.add(key)
        row = dict(entry)
        row["term"] = term
        row["target_translations"] = {
            **(
                target_translations
                if isinstance(target_translations, dict)
                else {}
            ),
            lang_code: translation,
        }
        out.append(row)
    return out


def _filter_glossary_by_source(
    entries: Iterable[Dict[str, Any]],
    source_filter: Optional[str],
) -> List[Dict[str, Any]]:
    if not source_filter:
        return list(entries)
    allowed = {x.strip() for x in source_filter.split(",") if x.strip()}
    if not allowed:
        return list(entries)
    return [entry for entry in entries if str(entry.get("source") or "") in allowed]


def _match_glossary_terms(
    *,
    source_text: str,
    reference_text: str,
    entries: Iterable[Dict[str, Any]],
    lang_code: str,
) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for entry in entries:
        term = _normalise_space(entry.get("term") or entry.get("source") or "")
        translations = entry.get("target_translations") or {}
        translation = ""
        if isinstance(translations, dict):
            translation = _normalise_space(translations.get(lang_code) or "")
        if not term or not translation:
            continue
        if not _source_contains(source_text, term):
            continue
        if not _text_contains(reference_text, translation):
            continue
        key = (_norm(term), translation)
        if key in seen:
            continue
        seen.add(key)
        refs.append({"term": term, "translation": translation})
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-id", default="404")
    ap.add_argument("--lang-code", default="zh", choices=["zh", "de", "ja"])
    ap.add_argument("--eso-test-root", type=Path, default=DEFAULT_ESO_TEST_ROOT)
    ap.add_argument("--strict-jsonl", type=Path, default=DEFAULT_STRICT_JSONL)
    ap.add_argument("--strict-glossary", type=Path, default=DEFAULT_STRICT_GLOSSARY)
    ap.add_argument(
        "--term-source",
        choices=["sentence_terms", "glossary_match"],
        default="sentence_terms",
        help=(
            "sentence_terms reproduces legacy ESO sentence.terms oracle rows; "
            "glossary_match derives oracle rows by matching source/reference "
            "against a translated glossary."
        ),
    )
    ap.add_argument(
        "--oracle-glossary",
        type=Path,
        default=None,
        help=(
            "Translated glossary used to build oracle rows in glossary_match mode. "
            "Defaults to --strict-glossary."
        ),
    )
    ap.add_argument(
        "--eval-glossary",
        type=Path,
        default=None,
        help=(
            "Translated glossary used to write the TERM metric glossary in "
            "glossary_match mode. Defaults to --oracle-glossary."
        ),
    )
    ap.add_argument(
        "--glossary-source-filter",
        default="",
        help=(
            "Optional comma-separated entry.source allowlist applied to "
            "oracle/eval glossaries in glossary_match mode, e.g. medicine_gt."
        ),
    )
    ap.add_argument(
        "--glossary-tag",
        default=None,
        help=(
            "Output glossary filename stem. Defaults to "
            "medicine_gt_strict_translated__medicine_<sample_id>."
        ),
    )
    ap.add_argument(
        "--oracle-term-map-tag",
        default=None,
        help=(
            "Output oracle term-map filename stem. Defaults to "
            "medicine.oracle_term_map__medicine_<sample_id>."
        ),
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-sentences", type=int, default=0)
    args = ap.parse_args()

    sample_dir = _sample_dir(args.eso_test_root, args.sample_id)
    sentences_path = sample_dir / "sentences_v2.json"
    metadata_path = sample_dir / "metadata_v2.json"
    if not sentences_path.is_file():
        raise FileNotFoundError(sentences_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    sentences = _read_json(sentences_path)
    metadata = _read_json(metadata_path)
    if not isinstance(sentences, list):
        raise ValueError(f"Expected list in {sentences_path}")

    strict_stats: Dict[str, int] = {}
    strict_terms: set[str] = set()
    if args.term_source == "sentence_terms":
        strict_terms, strict_stats = _strict_terms_for_sample(args.strict_jsonl, args.sample_id)
        strict_glossary_terms = _strict_glossary_terms(args.strict_glossary)
        strict_terms &= strict_glossary_terms
        if not strict_terms:
            raise ValueError(
                f"Strict JSONL terms for sample_id={args.sample_id} do not intersect strict glossary"
            )

    oracle_glossary_path = args.oracle_glossary or args.strict_glossary
    eval_glossary_path = args.eval_glossary or oracle_glossary_path
    oracle_glossary_entries: List[Dict[str, Any]] = []
    eval_glossary_entries: List[Dict[str, Any]] = []
    if args.term_source == "glossary_match":
        oracle_glossary_entries = _filter_glossary_by_source(
            _load_translated_glossary(oracle_glossary_path, args.lang_code),
            args.glossary_source_filter,
        )
        eval_glossary_entries = _filter_glossary_by_source(
            _load_translated_glossary(eval_glossary_path, args.lang_code),
            args.glossary_source_filter,
        )
        if not oracle_glossary_entries:
            raise ValueError(f"No translated oracle glossary entries loaded from {oracle_glossary_path}")
        if not eval_glossary_entries:
            raise ValueError(f"No translated eval glossary entries loaded from {eval_glossary_path}")

    if args.max_sentences > 0:
        sentences = sentences[: args.max_sentences]

    audio_path = _resolve_audio_path(sample_dir, args.sample_id, metadata)

    source_lines: List[str] = []
    ref_lines: List[str] = []
    audio_yaml: List[Dict[str, Any]] = []
    glossary_by_key: Dict[str, Dict[str, Any]] = {}
    oracle_rows: List[Dict[str, Any]] = []
    skipped_terms_no_translation = 0
    skipped_terms_not_strict = 0

    for sent in sentences:
        text = str(sent.get("text") or "").strip()
        translations = sent.get("translations") or {}
        ref = str(translations.get(args.lang_code) or "").strip()
        if not text or not ref:
            raise ValueError(
                f"Sentence {sent.get('sentence_id')} missing text or {args.lang_code} translation"
            )
        start = float(sent.get("start"))
        end = float(sent.get("end"))
        if end <= start:
            raise ValueError(f"Invalid time span for sentence {sent.get('sentence_id')}: {start}-{end}")
        source_lines.append(text)
        ref_lines.append(ref)
        audio_yaml.append({"wav": audio_path, "offset": start, "duration": end - start})

        sent_refs: List[Dict[str, str]] = []
        if args.term_source == "sentence_terms":
            sent_ref_seen: set[Tuple[str, str]] = set()
            for term_entry in sent.get("terms") or []:
                if not isinstance(term_entry, dict):
                    continue
                term = str(term_entry.get("term") or "").strip()
                term_key = _norm(term)
                if not term_key:
                    continue
                if term_key not in strict_terms:
                    skipped_terms_not_strict += 1
                    continue
                target_translations = term_entry.get("target_translations") or {}
                translation = str(target_translations.get(args.lang_code) or "").strip()
                if not translation:
                    skipped_terms_no_translation += 1
                    continue
                sent_key = (term_key, translation)
                if sent_key not in sent_ref_seen:
                    sent_ref_seen.add(sent_key)
                    sent_refs.append({"term": term, "translation": translation})
                glossary_by_key.setdefault(
                    term_key,
                    {
                        "term": term,
                        "target_translations": {
                            k: v for k, v in target_translations.items() if isinstance(v, str) and v.strip()
                        },
                        "source": "medicine_eso_v2_strict_mfa_exact",
                        "sample_id": args.sample_id,
                    },
                )
        else:
            sent_refs = _match_glossary_terms(
                source_text=text,
                reference_text=ref,
                entries=oracle_glossary_entries,
                lang_code=args.lang_code,
            )
            eval_refs = _match_glossary_terms(
                source_text=text,
                reference_text=ref,
                entries=eval_glossary_entries,
                lang_code=args.lang_code,
            )
            for matched in eval_refs:
                term_key = _norm(matched["term"])
                glossary_by_key.setdefault(
                    term_key,
                    {
                        "term": matched["term"],
                        "target_translations": {args.lang_code: matched["translation"]},
                        "source": "medicine_glossary_match_source_ref",
                        "sample_id": args.sample_id,
                    },
                )
        oracle_rows.append(
            {
                "sentence_id": sent.get("sentence_id"),
                "start_sec": start,
                "end_sec": end,
                "references": sent_refs,
            }
        )

    if not glossary_by_key:
        raise ValueError(
            "No translated strict terms remain; cannot build a SimulEval term_map glossary"
        )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"medicine_{args.sample_id}"
    glossary_tag = args.glossary_tag or f"medicine_gt_strict_translated__{prefix}"
    oracle_term_map_tag = args.oracle_term_map_tag or f"medicine.oracle_term_map__{prefix}"

    source_list = out_dir / f"medicine.source__{prefix}.txt"
    target_list = out_dir / f"medicine.target.{args.lang_code}__{prefix}.txt"
    source_text = out_dir / f"medicine.source_text.en__{prefix}.txt"
    ref_text = out_dir / f"medicine.ref.{args.lang_code}__{prefix}.txt"
    audio_out = out_dir / f"medicine.audio__{prefix}.yaml"
    glossary_out = out_dir / f"{glossary_tag}.json"
    oracle_out = out_dir / f"{oracle_term_map_tag}.json"
    manifest_out = out_dir / f"medicine_inputs_manifest__{prefix}.json"

    source_list.write_text(audio_path + "\n", encoding="utf-8")
    target_list.write_text(" ".join(ref_lines) + "\n", encoding="utf-8")
    source_text.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    ref_text.write_text("\n".join(ref_lines) + "\n", encoding="utf-8")
    audio_out.write_text(yaml.safe_dump(audio_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    glossary_out.write_text(
        json.dumps(glossary_by_key, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    oracle_out.write_text(
        json.dumps(oracle_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "sample_id": args.sample_id,
        "lang_code": args.lang_code,
        "term_source": args.term_source,
        "sentence_count": len(source_lines),
        "audio_path": audio_path,
        "strict_jsonl": str(args.strict_jsonl),
        "strict_glossary": str(args.strict_glossary),
        "oracle_glossary": str(oracle_glossary_path),
        "eval_glossary": str(eval_glossary_path),
        "glossary_source_filter": args.glossary_source_filter,
        "glossary_tag": glossary_tag,
        "oracle_term_map_tag": oracle_term_map_tag,
        "strict_jsonl_stats_for_sample": strict_stats,
        "strict_term_count_for_sample": len(strict_terms) if strict_terms else None,
        "oracle_glossary_loaded_terms": len(oracle_glossary_entries) if oracle_glossary_entries else None,
        "eval_glossary_loaded_terms": len(eval_glossary_entries) if eval_glossary_entries else None,
        "translated_term_count": len(glossary_by_key),
        "skipped_terms_not_strict": skipped_terms_not_strict,
        "skipped_terms_no_translation": skipped_terms_no_translation,
        "note": (
            "The source strict glossary has no target_translations for filler terms; "
            "this SimulEval glossary contains translated strict GT terms only."
        ),
        "files": {
            "source_list": str(source_list),
            "target_list": str(target_list),
            "source_text": str(source_text),
            "reference_text": str(ref_text),
            "audio_yaml": str(audio_out),
            "glossary": str(glossary_out),
            "oracle_term_map": str(oracle_out),
        },
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
