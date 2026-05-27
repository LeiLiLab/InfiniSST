#!/usr/bin/env python3
"""Prepare medicine eval JSONL with balanced variable speech contexts.

The ESO medicine source provides full WAVs, sentence-level timestamps, term
annotations, and MFA short TextGrids.  This script mirrors the ACL6060 eval
JSONL shape used by the retriever: for each 1.92s/0.96s base chunk it selects
one requested context duration, writes one row per term fully contained in that
selected context, and writes one no-term row when no term is contained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from documents.code.data_pre.tts.prepare_acl6060_dev_dataset import (  # noqa: E402
    WordInterval,
    parse_short_textgrid,
)


SAMPLE_RATE = 16000
OLD_CHUNK_SEC = 1.92
STRIDE_SEC = 0.96
DEFAULT_DURATION_SECS = (2.88, 3.84, 4.80, 5.76)
DEFAULT_INPUT_DIR = "/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test"
DEFAULT_MFA_TEXTGRID_DIR = "/home/jiaxingxu/rag-sst/eso-dataset/mfa_v1/textgrids"
DEFAULT_OUTPUT_DIR = "/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76"
DEFAULT_CHUNK_AUDIO_DIR = (
    "/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/audio_chunks"
)
DEFAULT_FILLER_GLOSSARY = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/"
    "glossary_scale/wiki_glossary_medicine_enriched.json"
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
WORD_NORMALIZE_RE = re.compile(r"[^a-z0-9']+")


@dataclass(frozen=True)
class TermSpan:
    term: str
    start: float
    end: float
    sample_id: str
    sentence_id: int
    locate_method: str


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    return value


def duration_tag(sec: float) -> str:
    return f"{sec:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_duration_secs(value: str) -> List[float]:
    durations = [float(v) for v in value.replace(",", " ").split() if v.strip()]
    if not durations:
        raise ValueError("--duration-secs must contain at least one duration")
    out: List[float] = []
    for dur in durations:
        if dur <= 0:
            raise ValueError(f"Duration must be positive: {dur}")
        rounded = round(dur, 4)
        if rounded not in out:
            out.append(rounded)
    return out


def stable_u64(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def choose_duration(
    *,
    row_counts: Counter,
    duration_order: List[float],
    assignment: str,
    stable_key: str,
    n_rows_by_duration: Dict[float, int],
) -> float:
    preferred_idx = stable_u64(stable_key) % len(duration_order)
    rotated = duration_order[preferred_idx:] + duration_order[:preferred_idx]
    rank = {dur: idx for idx, dur in enumerate(rotated)}
    if assignment == "hash_group":
        return rotated[0]
    return min(
        duration_order,
        key=lambda d: (
            row_counts[duration_tag(d)] + n_rows_by_duration.get(d, 1),
            row_counts[duration_tag(d)],
            rank[d],
        ),
    )


def clamp_centered_start(
    old_start_sec: float,
    wav_duration_sec: float,
    *,
    old_chunk_sec: float,
    new_chunk_sec: float,
) -> float:
    desired = old_start_sec + (old_chunk_sec - new_chunk_sec) * 0.5
    if wav_duration_sec >= new_chunk_sec:
        return min(max(desired, 0.0), wav_duration_sec - new_chunk_sec)
    return 0.0


def normalize_word(word: str) -> str:
    word = str(word or "").strip().lower().replace("\u2019", "'")
    if word.endswith("'s"):
        word = word[:-2]
    word = WORD_NORMALIZE_RE.sub("", word)
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("es") and not word.endswith(("ses", "xes")):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def tokenize_with_char_spans(text: str) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for match in TOKEN_RE.finditer(str(text or "")):
        tok = normalize_word(match.group(0))
        if tok:
            out.append((tok, match.start(), match.end()))
    return out


def term_token_variants(term: str) -> List[List[str]]:
    variants: List[List[str]] = []

    def add(tokens: Iterable[str]) -> None:
        cleaned = [t for t in tokens if t]
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

    add(tok for tok, _, _ in tokenize_with_char_spans(term))
    add(normalize_word(piece) for piece in re.split(r"[^A-Za-z0-9']+", str(term or "")))
    return variants


def find_all_subseq(haystack: Sequence[str], needle: Sequence[str]) -> List[int]:
    if not needle or len(needle) > len(haystack):
        return []
    return [
        i
        for i in range(0, len(haystack) - len(needle) + 1)
        if list(haystack[i : i + len(needle)]) == list(needle)
    ]


def find_term_char_span(text: str, term: str) -> Optional[Tuple[int, int]]:
    text_tokens = tokenize_with_char_spans(text)
    norm_words = [tok for tok, _, _ in text_tokens]
    for variant in term_token_variants(term):
        for hit in find_all_subseq(norm_words, variant):
            return text_tokens[hit][1], text_tokens[hit + len(variant) - 1][2]
    return None


def locate_term_span(
    words: Sequence[WordInterval],
    *,
    sentence_start: float,
    sentence_end: float,
    sentence_text: str,
    term: str,
    sample_id: str,
    sentence_id: int,
    unmatched_term_policy: str,
) -> Optional[TermSpan]:
    term_lc = str(term or "").strip().lower()
    if not term_lc:
        raise ValueError("empty term")

    window_words = [
        wi
        for wi in words
        if wi.start < sentence_end + 0.4 and wi.end > sentence_start - 0.4
    ]
    indexed_norm_words = [
        (idx, normalize_word(wi.word))
        for idx, wi in enumerate(window_words)
        if normalize_word(wi.word)
    ]
    norm_words = [w for _, w in indexed_norm_words]
    for variant in term_token_variants(term_lc):
        for hit in find_all_subseq(norm_words, variant):
            first = indexed_norm_words[hit][0]
            last = indexed_norm_words[hit + len(variant) - 1][0]
            return TermSpan(
                term=term_lc,
                start=float(window_words[first].start),
                end=float(window_words[last].end),
                sample_id=sample_id,
                sentence_id=sentence_id,
                locate_method="mfa_exact",
            )

    char_span = find_term_char_span(sentence_text, term_lc)
    if char_span is not None:
        char_start, char_end = char_span
        text_len = max(1, len(sentence_text))
        sent_dur = max(0.05, sentence_end - sentence_start)
        return TermSpan(
            term=term_lc,
            start=sentence_start + (char_start / text_len) * sent_dur,
            end=sentence_start + (char_end / text_len) * sent_dur,
            sample_id=sample_id,
            sentence_id=sentence_id,
            locate_method="char_proportional",
        )

    if unmatched_term_policy == "drop":
        return None
    if unmatched_term_policy != "center_fallback":
        raise ValueError(f"Unknown unmatched_term_policy: {unmatched_term_policy}")

    # Legacy behavior for reproducing older medicine eval data.  This can assign
    # a GT term even when the source term string is not present in the speech.
    center = (sentence_start + sentence_end) * 0.5
    half_width = min(0.35, max(0.05, (sentence_end - sentence_start) * 0.1))
    return TermSpan(
        term=term_lc,
        start=max(sentence_start, center - half_width),
        end=min(sentence_end, center + half_width),
        sample_id=sample_id,
        sentence_id=sentence_id,
        locate_method="sentence_center_fallback",
    )


def contained_terms(
    term_spans: Sequence[TermSpan],
    start_sec: float,
    end_sec: float,
) -> Set[str]:
    terms: Set[str] = set()
    for ts in term_spans:
        if ts.start >= start_sec and ts.end <= end_sec:
            terms.add(ts.term.strip().lower())
    return terms


def build_window_text(words: Sequence[WordInterval], start_sec: float, end_sec: float) -> str:
    return " ".join(
        wi.word for wi in words if wi.start < end_sec and wi.end > start_sec and wi.word.strip()
    )


def write_chunk_audio(
    wav_path: str,
    output_path: str,
    start_sec: float,
    chunk_sec: float,
    *,
    overwrite: bool,
) -> Tuple[int, int]:
    start_sample = int(round(start_sec * SAMPLE_RATE))
    chunk_samples = int(round(chunk_sec * SAMPLE_RATE))
    if os.path.isfile(output_path) and not overwrite:
        return start_sample, chunk_samples
    audio, sr = sf.read(
        wav_path,
        start=start_sample,
        frames=chunk_samples,
        dtype="float32",
    )
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected SR={SAMPLE_RATE}, got {sr}: {wav_path}")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - len(audio)), mode="constant")
    elif len(audio) > chunk_samples:
        audio = audio[:chunk_samples]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")
    return start_sample, chunk_samples


def sample_id_from_dir(path: Path) -> str:
    match = re.match(r"sample_(.+)_v2$", path.name)
    if not match:
        raise ValueError(f"Unexpected sample directory name: {path}")
    return match.group(1)


def iter_sample_dirs(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.glob("sample_*_v2") if p.is_dir())


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def load_filler_terms(path: str) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    entries = load_json(p)
    candidates: List[Tuple[int, int, str]] = []
    if isinstance(entries, list):
        for idx, entry in enumerate(entries):
            if isinstance(entry, dict):
                term = str(entry.get("term") or "").strip().lower()
                translations = entry.get("target_translations") or {}
                all_langs = all(translations.get(lang) for lang in ("zh", "ja", "de"))
                any_lang = any(translations.get(lang) for lang in ("zh", "ja", "de"))
                priority = 0 if all_langs else 1 if any_lang else 2
            else:
                term = str(entry or "").strip().lower()
                priority = 2
            if term:
                candidates.append((priority, idx, term))

    terms: List[str] = []
    seen: Set[str] = set()
    for _, _, term in sorted(candidates):
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def write_glossary(
    *,
    output_path: Path,
    medicine_terms: Iterable[str],
    filler_terms: Iterable[str],
    target_size: int,
    filler_source: str = "medicine_wiki_filler",
) -> int:
    seen: Set[str] = set()
    entries: List[Dict[str, str]] = []
    for term in sorted({t.strip().lower() for t in medicine_terms if t.strip()}):
        if term not in seen:
            seen.add(term)
            entries.append({"term": term, "source": "medicine_gt"})
    for term in filler_terms:
        term_lc = term.strip().lower()
        if not term_lc or term_lc in seen:
            continue
        seen.add(term_lc)
        entries.append({"term": term_lc, "source": filler_source})
        if target_size > 0 and len(entries) >= target_size:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fout:
        json.dump(entries, fout, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)
    return len(entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--mfa-textgrid-dir", default=DEFAULT_MFA_TEXTGRID_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl-name", default="medicine_dev_dataset.jsonl")
    parser.add_argument("--chunk-audio-dir", default=DEFAULT_CHUNK_AUDIO_DIR)
    parser.add_argument(
        "--duration-secs",
        default=" ".join(str(x) for x in DEFAULT_DURATION_SECS),
    )
    parser.add_argument("--old-chunk-sec", type=float, default=OLD_CHUNK_SEC)
    parser.add_argument("--stride-sec", type=float, default=STRIDE_SEC)
    parser.add_argument(
        "--duration-assignment",
        choices=["balance_rows", "hash_group"],
        default="balance_rows",
    )
    parser.add_argument("--stats-json", default="")
    parser.add_argument("--glossary-json", default="")
    parser.add_argument("--filler-glossary", default=DEFAULT_FILLER_GLOSSARY)
    parser.add_argument("--filler-source", default="medicine_wiki_filler")
    parser.add_argument("--glossary-target-size", type=int, default=10000)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument(
        "--unmatched-term-policy",
        choices=["drop", "center_fallback"],
        default="drop",
        help=(
            "How to handle source terms that cannot be located in MFA words or "
            "sentence text. 'drop' is the cleaned default; 'center_fallback' "
            "reproduces the legacy behavior."
        ),
    )
    parser.add_argument(
        "--dropped-terms-json",
        default="",
        help="Audit JSON for terms dropped by --unmatched-term-policy=drop.",
    )
    parser.add_argument(
        "--allowed-locate-methods",
        default="mfa_exact char_proportional",
        help=(
            "Whitespace/comma separated locate methods to keep as positives. "
            "Use 'mfa_exact' for strict MFA-only cleaning."
        ),
    )
    parser.add_argument(
        "--max-base-chunks-per-sample",
        type=int,
        default=0,
        help="Debug/smoke option. 0 means process every base chunk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    mfa_dir = Path(args.mfa_textgrid_dir)
    output_dir = Path(args.output_dir)
    chunk_audio_dir = Path(args.chunk_audio_dir)
    durations = parse_duration_secs(args.duration_secs)
    duration_tags = [duration_tag(d) for d in durations]
    allowed_locate_methods = {
        method.strip()
        for method in args.allowed_locate_methods.replace(",", " ").split()
        if method.strip()
    }
    unknown_locate_methods = allowed_locate_methods - {
        "mfa_exact",
        "char_proportional",
        "sentence_center_fallback",
    }
    if unknown_locate_methods:
        raise ValueError(f"Unknown allowed locate methods: {sorted(unknown_locate_methods)}")
    if not allowed_locate_methods:
        raise ValueError("--allowed-locate-methods must keep at least one locate method")
    build_tag = "medicine_mfa_varctx_" + "_".join(duration_tags)
    if args.unmatched_term_policy == "drop":
        build_tag += "_drop_unmatched"
    if allowed_locate_methods != {"mfa_exact", "char_proportional"}:
        build_tag += "_loc_" + "_".join(sorted(allowed_locate_methods))

    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if not mfa_dir.is_dir():
        raise FileNotFoundError(mfa_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_audio_dir.mkdir(parents=True, exist_ok=True)

    row_counts: Counter = Counter()
    stats: Counter = Counter()
    locate_counts: Counter = Counter()
    dropped_by_term: Counter = Counter()
    dropped_by_sample: Counter = Counter()
    dropped_by_reason: Counter = Counter()
    dropped_by_locate_method: Counter = Counter()
    dropped_examples: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    all_medicine_terms: Set[str] = set()

    for sample_dir in iter_sample_dirs(input_dir):
        sample_id = sample_id_from_dir(sample_dir)
        wav_path = sample_dir / f"{sample_id}_v2.wav"
        sentences_path = sample_dir / "sentences_v2.json"
        tg_path = mfa_dir / f"test_{sample_id}_full.TextGrid"
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        if not sentences_path.is_file():
            raise FileNotFoundError(sentences_path)
        if not tg_path.is_file():
            raise FileNotFoundError(tg_path)

        info = sf.info(str(wav_path))
        if info.samplerate != SAMPLE_RATE:
            raise ValueError(f"Expected SR={SAMPLE_RATE}, got {info.samplerate}: {wav_path}")
        wav_duration = float(info.duration)
        words = parse_short_textgrid(str(tg_path))
        sentences = load_json(sentences_path)
        if not isinstance(sentences, list):
            raise ValueError(f"Expected list in {sentences_path}, got {type(sentences)}")

        term_spans: List[TermSpan] = []
        for sentence in sentences:
            sentence_id = int(sentence.get("sentence_id", len(term_spans)))
            sentence_start = max(0.0, float(sentence.get("start", 0.0)))
            sentence_end = min(wav_duration, float(sentence.get("end", sentence_start)))
            if sentence_end <= sentence_start:
                continue
            sentence_text = str(sentence.get("text") or "")
            for entry in sentence.get("terms") or []:
                term = str(entry.get("term") or "").strip()
                if not term:
                    continue
                span = locate_term_span(
                    words,
                    sentence_start=sentence_start,
                    sentence_end=sentence_end,
                    sentence_text=sentence_text,
                    term=term,
                    sample_id=sample_id,
                    sentence_id=sentence_id,
                    unmatched_term_policy=args.unmatched_term_policy,
                )
                if span is None:
                    term_lc = term.strip().lower()
                    stats["dropped_unmatched_terms"] += 1
                    dropped_by_reason["unmatched"] += 1
                    dropped_by_term[term_lc] += 1
                    dropped_by_sample[sample_id] += 1
                    if len(dropped_examples) < 250:
                        dropped_examples.append(
                            {
                                "sample_id": sample_id,
                                "sentence_id": sentence_id,
                                "sentence_start": round(sentence_start, 4),
                                "sentence_end": round(sentence_end, 4),
                                "sentence_text": sentence_text,
                                "term": term,
                                "drop_reason": "unmatched",
                                "target_translations": entry.get("target_translations") or {},
                            }
                        )
                    continue
                span = TermSpan(
                    term=span.term,
                    start=max(0.0, min(wav_duration, span.start)),
                    end=max(0.0, min(wav_duration, span.end)),
                    sample_id=span.sample_id,
                    sentence_id=span.sentence_id,
                    locate_method=span.locate_method,
                )
                if span.end <= span.start:
                    continue
                if span.locate_method not in allowed_locate_methods:
                    stats["dropped_disallowed_locate_method_terms"] += 1
                    dropped_by_reason["disallowed_locate_method"] += 1
                    dropped_by_locate_method[span.locate_method] += 1
                    dropped_by_term[span.term] += 1
                    dropped_by_sample[sample_id] += 1
                    if len(dropped_examples) < 250:
                        dropped_examples.append(
                            {
                                "sample_id": sample_id,
                                "sentence_id": sentence_id,
                                "sentence_start": round(sentence_start, 4),
                                "sentence_end": round(sentence_end, 4),
                                "sentence_text": sentence_text,
                                "term": term,
                                "drop_reason": "disallowed_locate_method",
                                "locate_method": span.locate_method,
                                "target_translations": entry.get("target_translations") or {},
                            }
                        )
                    continue
                term_spans.append(span)
                all_medicine_terms.add(span.term)
                locate_counts[span.locate_method] += 1

        base_chunks = max(
            1,
            int(math.ceil((wav_duration - args.old_chunk_sec) / args.stride_sec)) + 1,
        )
        if args.max_base_chunks_per_sample > 0:
            base_chunks = min(base_chunks, args.max_base_chunks_per_sample)

        for chunk_idx in range(base_chunks):
            old_start_sec = chunk_idx * args.stride_sec
            if old_start_sec >= wav_duration:
                break

            terms_by_duration: Dict[float, Set[str]] = {}
            start_by_duration: Dict[float, float] = {}
            for dur in durations:
                start_sec = clamp_centered_start(
                    old_start_sec,
                    wav_duration,
                    old_chunk_sec=args.old_chunk_sec,
                    new_chunk_sec=dur,
                )
                end_sec = start_sec + dur
                terms_by_duration[dur] = contained_terms(term_spans, start_sec, end_sec)
                start_by_duration[dur] = start_sec

            n_rows_by_duration = {
                dur: max(1, len(terms))
                for dur, terms in terms_by_duration.items()
            }
            chosen = choose_duration(
                row_counts=row_counts,
                duration_order=durations,
                assignment=args.duration_assignment,
                stable_key=f"{sample_id}\t{chunk_idx}",
                n_rows_by_duration=n_rows_by_duration,
            )
            chosen_tag = duration_tag(chosen)
            start_sec = start_by_duration[chosen]
            end_sec = start_sec + chosen
            terms = terms_by_duration[chosen]
            audio_path = chunk_audio_dir / chosen_tag / (
                f"medicine_{sample_id}_ctx{chosen_tag}_chunk_{chunk_idx}.wav"
            )
            context_start_sample, read_frames = write_chunk_audio(
                str(wav_path),
                str(audio_path),
                start_sec,
                chosen,
                overwrite=args.overwrite_audio,
            )
            src_text = build_window_text(words, start_sec, end_sec)
            base_row = {
                "chunk_src_text": src_text,
                "utter_id": f"medicine_{sample_id}",
                "sample_id": sample_id,
                "domain": "medicine",
                "chunk_idx": chunk_idx,
                "chunk_audio_path": str(audio_path),
                "chunk_duration_sec": round(chosen, 4),
                "context_duration_sec": round(chosen, 4),
                "context_duration_tag": chosen_tag,
                "source_chunk_idx_1p92": chunk_idx,
                "context_start_sample": context_start_sample,
                "context_read_frames": read_frames,
                "context_reused_source_audio": False,
                "context_build": build_tag,
            }
            stats["chunks_total"] += 1
            stats[f"chunks_dur_{chosen_tag}"] += 1
            if terms:
                stats["chunks_with_terms"] += 1
                matching_spans = [
                    ts for ts in term_spans if ts.term in terms and ts.start >= start_sec and ts.end <= end_sec
                ]
                method_by_term: Dict[str, str] = {}
                for ts in matching_spans:
                    method_by_term.setdefault(ts.term, ts.locate_method)
                for term in sorted(terms):
                    matching = [ts for ts in matching_spans if ts.term == term]
                    rel_start = min((ts.start for ts in matching), default=start_sec) - start_sec
                    rel_end = max((ts.end for ts in matching), default=start_sec) - start_sec
                    row = dict(base_row)
                    row.update(
                        {
                            "term": term,
                            "term_key": term,
                            "mfa_term_start_in_chunk": round(max(0.0, rel_start), 4),
                            "mfa_term_end_in_chunk": round(min(chosen, rel_end), 4),
                            "mfa_term_duration": round(max(0.0, rel_end - rel_start), 4),
                            "mfa_locate_method": method_by_term.get(term, ""),
                        }
                    )
                    all_rows.append(row)
                    row_counts[chosen_tag] += 1
                    stats["written_term_rows"] += 1
                    stats[f"written_rows_dur_{chosen_tag}"] += 1
            else:
                stats["chunks_without_terms"] += 1
                row = dict(base_row)
                row.update(
                    {
                        "term": "",
                        "term_key": "",
                        "mfa_term_start_in_chunk": None,
                        "mfa_term_end_in_chunk": None,
                        "mfa_term_duration": None,
                        "mfa_locate_method": "",
                    }
                )
                all_rows.append(row)
                row_counts[chosen_tag] += 1
                stats["written_empty_rows"] += 1
                stats[f"written_rows_dur_{chosen_tag}"] += 1

        stats["samples_processed"] += 1
        stats[f"sample_{sample_id}_term_spans"] = len(term_spans)
        print(
            f"[MED-VARCTX] sample={sample_id} "
            f"duration={wav_duration:.2f}s term_spans={len(term_spans)} "
            f"rows_so_far={len(all_rows)}",
            flush=True,
        )

    output_jsonl = output_dir / args.output_jsonl_name
    tmp_jsonl = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    with open(tmp_jsonl, "w", encoding="utf-8") as fout:
        for row in all_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, output_jsonl)

    glossary_json = Path(args.glossary_json) if args.glossary_json else (
        output_dir / "medicine_glossary_gt_plus_medicine_wiki_gs10000.json"
    )
    glossary_size = write_glossary(
        output_path=glossary_json,
        medicine_terms=all_medicine_terms,
        filler_terms=load_filler_terms(args.filler_glossary),
        target_size=args.glossary_target_size,
        filler_source=args.filler_source,
    )

    stats["written_total_rows"] = len(all_rows)
    stats["unique_medicine_terms"] = len(all_medicine_terms)
    payload: Dict[str, Any] = dict(sorted(stats.items()))
    payload.update(
        {
            "input_dir": str(input_dir),
            "mfa_textgrid_dir": str(mfa_dir),
            "output": str(output_jsonl),
            "audio_output_dir": str(chunk_audio_dir),
            "glossary_json": str(glossary_json),
            "glossary_size": glossary_size,
            "filler_glossary": args.filler_glossary,
            "duration_secs": durations,
            "duration_tags": duration_tags,
            "old_chunk_sec": args.old_chunk_sec,
            "stride_sec": args.stride_sec,
            "duration_assignment": args.duration_assignment,
            "context_build": build_tag,
            "unmatched_term_policy": args.unmatched_term_policy,
            "allowed_locate_methods": sorted(allowed_locate_methods),
            "locate_method_counts": dict(sorted(locate_counts.items())),
            "dropped_terms_by_reason": dict(sorted(dropped_by_reason.items())),
            "dropped_terms_by_locate_method": dict(sorted(dropped_by_locate_method.items())),
            "dropped_unmatched_terms_by_term": dict(sorted(dropped_by_term.items())),
            "dropped_unmatched_terms_by_sample": dict(sorted(dropped_by_sample.items())),
        }
    )
    for tag in duration_tags:
        payload[f"duration_row_count_{tag}"] = row_counts[tag]

    stats_json = Path(args.stats_json) if args.stats_json else (
        output_jsonl.with_name(output_jsonl.stem + "_stats.json")
    )
    with open(stats_json, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False, sort_keys=True)

    dropped_terms_json = Path(args.dropped_terms_json) if args.dropped_terms_json else (
        output_jsonl.with_name(output_jsonl.stem + "_dropped_terms.json")
    )
    dropped_payload = {
        "input_dir": str(input_dir),
        "mfa_textgrid_dir": str(mfa_dir),
        "output": str(output_jsonl),
        "unmatched_term_policy": args.unmatched_term_policy,
        "allowed_locate_methods": sorted(allowed_locate_methods),
        "dropped_unmatched_terms": int(stats["dropped_unmatched_terms"]),
        "dropped_disallowed_locate_method_terms": int(
            stats["dropped_disallowed_locate_method_terms"]
        ),
        "dropped_terms_by_reason": dict(sorted(dropped_by_reason.items())),
        "dropped_terms_by_locate_method": dict(sorted(dropped_by_locate_method.items())),
        "dropped_unmatched_terms_by_term": dict(sorted(dropped_by_term.items())),
        "dropped_unmatched_terms_by_sample": dict(sorted(dropped_by_sample.items())),
        "examples": dropped_examples,
    }
    with open(dropped_terms_json, "w", encoding="utf-8") as fout:
        json.dump(make_jsonable(dropped_payload), fout, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[MED-VARCTX] output={output_jsonl}")
    print(f"[MED-VARCTX] stats={stats_json}")
    print(f"[MED-VARCTX] dropped_terms={dropped_terms_json}")
    print(f"[MED-VARCTX] glossary={glossary_json} size={glossary_size}")
    print(f"[MED-VARCTX] written_total_rows={payload['written_total_rows']}")
    for tag in duration_tags:
        print(f"[MED-VARCTX] duration_row_count_{tag}={row_counts[tag]}")
    print(f"[MED-VARCTX] locate_method_counts={dict(sorted(locate_counts.items()))}")
    print(f"[MED-VARCTX] dropped_unmatched_terms={stats['dropped_unmatched_terms']}")
    print(
        "[MED-VARCTX] dropped_disallowed_locate_method_terms="
        f"{stats['dropped_disallowed_locate_method_terms']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
