#!/usr/bin/env python3
"""
Build InfiniSST term-level training data with balanced no-term samples.

This script scans cleaned term-level chunk files, fetches glossary translations,
and emits JSONL samples that follow the specified conversational format.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

try:
    import ijson  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ijson = None


SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter. You will be given chunks "
    "of English audio and you need to translate the audio into Chinese text."
)


@dataclass
class ChunkSample:
    segment_id: str
    audio: str
    text: str
    terms: List[str]


@dataclass
class DialogueEntry:
    user_content: str
    assistant_content: str
    audio_path: str


def normalize_term(term: str) -> str:
    return term.strip()


def iter_chunk_entries(file_path: Path) -> Iterable[dict]:
    if ijson is None:
        logging.warning(
            "ijson is not installed; loading %s fully into memory. "
            "Install ijson for streaming parsing.", file_path
        )
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            yield item
        return

    with open(file_path, "r", encoding="utf-8") as f:
        for item in ijson.items(f, "item"):
            yield item


def detect_no_term(entry: dict) -> bool:
    if entry.get("is_no_term_chunk"):
        return True
    segment_id = entry.get("segment_id", "")
    if isinstance(segment_id, str) and "no_term" in segment_id.lower():
        return True
    return False


def collect_samples(
    chunk_files: Sequence[Path],
    target_no_term: int,
    term_candidate_goal: int,
) -> Tuple[List[ChunkSample], List[ChunkSample], Dict[str, int]]:
    term_samples: List[ChunkSample] = []
    no_term_samples: List[ChunkSample] = []
    stats = {
        "total_entries": 0,
        "missing_fields": 0,
        "term_without_terms": 0,
    }

    for file_path in chunk_files:
        logging.info("Scanning %s", file_path)
        for entry in iter_chunk_entries(file_path):
            stats["total_entries"] += 1
            audio = entry.get("term_chunk_audio", "")
            text = entry.get("term_chunk_text", "")
            if not isinstance(audio, str) or not audio.strip() or not isinstance(text, str) or not text.strip():
                stats["missing_fields"] += 1
                continue

            terms_raw = entry.get("term_chunk_audio_ground_truth_terms") or []
            terms = [
                normalize_term(t)
                for t in terms_raw
                if isinstance(t, str) and normalize_term(t)
            ]

            is_no_term = detect_no_term(entry)
            if not is_no_term and not terms:
                stats["term_without_terms"] += 1
                is_no_term = True

            sample = ChunkSample(
                segment_id=str(entry.get("segment_id", "")),
                audio=audio.strip(),
                text=text.strip(),
                terms=terms,
            )

            if is_no_term:
                if len(no_term_samples) < target_no_term:
                    no_term_samples.append(sample)
            else:
                if len(term_samples) < term_candidate_goal:
                    term_samples.append(sample)

            if (
                len(no_term_samples) >= target_no_term
                and len(term_samples) >= term_candidate_goal
            ):
                logging.info(
                    "Collected %d no-term and %d candidate term samples; stopping.",
                    len(no_term_samples),
                    len(term_samples),
                )
                return term_samples, no_term_samples, stats

    if len(no_term_samples) < target_no_term or len(term_samples) < term_candidate_goal:
        logging.warning(
            "Insufficient samples collected (terms=%d/%d, no_term=%d/%d). "
            "Consider adding more chunk files or lowering the target.",
            len(term_samples),
            term_candidate_goal,
            len(no_term_samples),
            target_no_term,
        )

    return term_samples, no_term_samples, stats


def extract_translation(
    translations: Dict[str, str], target_lang: str
) -> str:
    if not translations:
        return ""
    lowered = target_lang.lower()
    for key in (
        target_lang,
        lowered,
        target_lang.upper(),
        target_lang.replace("-", "_"),
        lowered.split("_")[0],
        lowered.split("-")[0],
    ):
        value = translations.get(key)
        if value:
            return value.strip()
    for key, value in translations.items():
        if key.lower() == lowered and value:
            return value.strip()
    return ""


def load_translations_for_terms(
    glossary_path: Path,
    needed_terms: Set[str],
    target_lang: str,
) -> Tuple[Dict[str, str], Set[str]]:
    normalized_needed = {normalize_term(term) for term in needed_terms if term}
    if not normalized_needed:
        return {}, set()

    translations: Dict[str, str] = {}
    missing: Set[str] = set()

    def maybe_add(term_label: str, entry: dict, canonical_term: str | None = None) -> None:
        normalized = normalize_term(term_label)
        if normalized not in normalized_needed or normalized in translations:
            return
        target_trans = entry.get("target_translations") or {}
        zh_value = extract_translation(target_trans if isinstance(target_trans, dict) else {}, target_lang)
        if zh_value:
            translations[normalized] = zh_value
        else:
            translations[normalized] = canonical_term or term_label
            missing.add(normalized)

    if ijson is None:
        logging.warning(
            "ijson not found; loading full glossary from %s. This may require significant memory.",
            glossary_path,
        )
        with open(glossary_path, "r", encoding="utf-8") as f:
            glossary_data = json.load(f)
        items = (
            glossary_data.items()
            if isinstance(glossary_data, dict)
            else enumerate(glossary_data)
        )
        for key, entry in items:
            if not isinstance(entry, dict):
                continue
            term_value = entry.get("term")
            canonical = term_value if isinstance(term_value, str) else None
            maybe_add(str(key), entry, canonical_term=canonical)
            if canonical:
                maybe_add(canonical, entry, canonical_term=canonical)
    else:
        with open(glossary_path, "r", encoding="utf-8") as f:
            for key, entry in ijson.kvitems(f, ""):
                if not isinstance(entry, dict):
                    continue
                term_value = entry.get("term")
                canonical = term_value if isinstance(term_value, str) else None
                maybe_add(str(key), entry, canonical_term=canonical)
                if canonical:
                    maybe_add(canonical, entry, canonical_term=canonical)
                if len(translations) == len(normalized_needed):
                    break

    if missing:
        logging.warning(
            "Missing zh translations for %d/%d terms (%.2f%%). Falling back to original terms.",
            len(missing),
            len(normalized_needed),
            len(missing) * 100.0 / len(normalized_needed),
        )

    return translations, missing


def format_reference_block(refs: List[Dict[str, str]]) -> str:
    if not refs:
        return "[]"
    return ", ".join(json.dumps(ref, ensure_ascii=False) for ref in refs)


def build_term_entries(
    term_samples: List[ChunkSample],
    translations: Dict[str, str],
    target_term_count: int,
) -> Tuple[List[DialogueEntry], int]:
    selected: List[DialogueEntry] = []
    skipped = 0

    for sample in term_samples:
        refs: List[Dict[str, str]] = []
        missing_translation = False
        for term in sample.terms:
            normalized = normalize_term(term)
            translation = translations.get(normalized)
            if translation is None:
                missing_translation = True
                break
            refs.append({"term": term, "translation": translation})
        if missing_translation or not refs:
            skipped += 1
            continue
        user_content = f"<audio>, references: {format_reference_block(refs)}"
        selected.append(
            DialogueEntry(
                user_content=user_content,
                assistant_content=sample.text,
                audio_path=sample.audio,
            )
        )
        if len(selected) >= target_term_count:
            break

    return selected, skipped


def build_no_term_entries(no_term_samples: List[ChunkSample], target_no_term: int) -> List[DialogueEntry]:
    if len(no_term_samples) >= target_no_term:
        samples = no_term_samples[:target_no_term]
    else:
        logging.warning(
            "Requested %d no-term samples but only %d available.",
            target_no_term,
            len(no_term_samples),
        )
        samples = no_term_samples

    entries: List[DialogueEntry] = []
    for sample in samples:
        entries.append(
            DialogueEntry(
                user_content="<audio>, references: []",
                assistant_content=sample.text,
                audio_path=sample.audio,
            )
        )
    return entries


def interleave_entries(
    term_entries: Sequence[DialogueEntry],
    no_term_entries: Sequence[DialogueEntry],
) -> List[DialogueEntry]:
    final_entries: List[DialogueEntry] = []
    term_idx = 0
    no_term_idx = 0
    total_needed = len(term_entries) + len(no_term_entries)
    for idx in range(total_needed):
        use_term = idx % 2 == 0
        if use_term and term_idx < len(term_entries):
            final_entries.append(term_entries[term_idx])
            term_idx += 1
        elif not use_term and no_term_idx < len(no_term_entries):
            final_entries.append(no_term_entries[no_term_idx])
            no_term_idx += 1
        elif term_idx < len(term_entries):
            final_entries.append(term_entries[term_idx])
            term_idx += 1
        elif no_term_idx < len(no_term_entries):
            final_entries.append(no_term_entries[no_term_idx])
            no_term_idx += 1
    return final_entries


def build_conversation(entries: Sequence[DialogueEntry]) -> Dict[str, object]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    audios: List[str] = []
    for entry in entries:
        messages.append({"role": "user", "content": entry.user_content})
        messages.append({"role": "assistant", "content": entry.assistant_content})
        audios.append(entry.audio_path)
    return {"messages": messages, "audios": audios}


def write_records(
    entries: Sequence[DialogueEntry],
    output_path: Path,
    output_format: str = "json",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        conversation = build_conversation(entries)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False)
    elif output_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": entry.user_content},
                        {"role": "assistant", "content": entry.assistant_content},
                    ],
                    "audios": [entry.audio_path],
                }
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def natural_sort_key(path: Path) -> Tuple:
    numbers = [int(chunk) for chunk in re.findall(r"(\d+)", path.name)]
    return (*numbers, path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct InfiniSST term-level training data.")
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path("/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/samples/xl_cleaned"),
        help="Directory containing term_level_chunks_*.json files.",
    )
    parser.add_argument(
        "--glossary-path",
        type=Path,
        default=Path("/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json"),
        help="Path to glossary_cleaned.json.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/samples/term_training_180k.jsonl"),
        help="Destination file for the generated dataset.",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=180_000,
        help="Total number of samples to output.",
    )
    parser.add_argument(
        "--no-term-ratio",
        type=float,
        default=0.5,
        help="Fraction of samples that should be no-term.",
    )
    parser.add_argument(
        "--term-oversample-factor",
        type=float,
        default=3.0,
        help="Multiplier for the number of term samples to collect before filtering by translations.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on the number of chunk files to scan.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="zh",
        help="Glossary target language key to use for translations.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "jsonl"],
        help="Output format for the generated dataset.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not (0 < args.no_term_ratio < 1):
        raise ValueError("--no-term-ratio must be between 0 and 1.")

    chunk_dir = args.chunk_dir
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    chunk_files = sorted(
        [
            path
            for path in chunk_dir.glob("term_level_chunks_*.json")
            if path.name != "term_level_chunks_merged.json"
        ],
        key=natural_sort_key,
    )
    if args.max_files:
        chunk_files = chunk_files[: args.max_files]

    if not chunk_files:
        raise FileNotFoundError(f"No term_level_chunks_*.json files found under {chunk_dir}")

    target_no_term = math.floor(args.target_samples * args.no_term_ratio)
    target_term = args.target_samples - target_no_term
    term_candidate_goal = math.ceil(target_term * args.term_oversample_factor)

    logging.info(
        "Target samples: %d (term=%d, no_term=%d, term_candidate_goal=%d)",
        args.target_samples,
        target_term,
        target_no_term,
        term_candidate_goal,
    )

    term_samples, no_term_samples, stats = collect_samples(
        chunk_files,
        target_no_term=target_no_term,
        term_candidate_goal=term_candidate_goal,
    )

    unique_terms: Set[str] = {term for sample in term_samples for term in sample.terms}
    logging.info(
        "Collected %d candidate term samples with %d unique terms.",
        len(term_samples),
        len(unique_terms),
    )

    translations, missing_terms = load_translations_for_terms(
        args.glossary_path,
        unique_terms,
        args.target_lang,
    )
    logging.info(
        "Resolved %d/%d translations.",
        len(translations),
        len(unique_terms),
    )

    term_entries, skipped_terms = build_term_entries(
        term_samples,
        translations,
        target_term_count=target_term,
    )
    if len(term_entries) < target_term:
        raise RuntimeError(
            f"Unable to collect enough term samples with translations "
            f"({len(term_entries)}/{target_term}). "
            f"Increase --term-oversample-factor or provide more chunk files."
        )

    no_term_entries = build_no_term_entries(no_term_samples, target_no_term)
    if len(no_term_entries) < target_no_term:
        raise RuntimeError(
            f"Unable to collect enough no-term samples "
            f"({len(no_term_entries)}/{target_no_term})."
        )

    final_entries = interleave_entries(term_entries, no_term_entries)
    if len(final_entries) != args.target_samples:
        raise AssertionError(
            f"Final record count mismatch: expected {args.target_samples}, got {len(final_entries)}"
        )

    write_records(final_entries, args.output_path, args.output_format)

    logging.info("Wrote %d samples to %s", len(final_entries), args.output_path)
    logging.info(
        "Stats: processed=%d missing_fields=%d term_without_terms=%d skipped_terms=%d missing_translations=%d",
        stats["total_entries"],
        stats["missing_fields"],
        stats["term_without_terms"],
        skipped_terms,
        len(missing_terms),
    )


if __name__ == "__main__":
    main()

