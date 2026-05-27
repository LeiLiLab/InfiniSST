#!/usr/bin/env python3
"""
Prepare ACL6060 dev dataset using the *extracted paper glossary*
(extracted_glossary_list.json) with raw transcript text (NOT tagged text).

Term occurrences are found by case-insensitive word-boundary matching
against the raw transcript, avoiding the inaccurate tagged annotations.

Reuses existing MFA TextGrids and chunk audio from prepare_acl6060_dev_dataset.py.
"""

from __future__ import annotations

import json
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from documents.code.data_pre.tts.prepare_acl6060_dev_dataset import (
    TalkInfo,
    SegmentInfo,
    ChunkRecord,
    TermTimeSpan,
    WordInterval,
    parse_xml,
    find_textgrid,
    parse_short_textgrid,
    align_expected_to_mfa,
    build_global_word_list,
    _log,
    _warn,
    ACL_WAV_DIR,
    ACL_XML_PATH,
    WORK_DIR as BASE_WORK_DIR,
    MFA_TEXTGRID_SUBDIR,
    CHUNK_AUDIO_SUBDIR,
    SAMPLE_RATE,
    CHUNK_SEC,
    STRIDE_SEC,
)

# ======Configuration=====
ACL_RAW_TEXT_PATH = (
    "/mnt/data/siqiouyang/datasets/acl6060/dev/text/txt/"
    "ACL.6060.dev.en-xx.en.txt"
)
GLOSSARY_BY_PAPER_DIR = (
    "/home/jiaxuanluo/InfiniSST/documents/data/data_pre/"
    "extracted_glossary_lists_by_paper"
)
GLOSSARY_PAPER_IDS = [
    "2022.acl-long.110",
    "2022.acl-long.117",
    "2022.acl-long.268",
    "2022.acl-long.367",
    "2022.acl-long.590",
]
TTS_TERM_TO_PATH = "/mnt/gemini/data/siqiouyang/acl_terms/term_to_path.json"

OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary"
OUTPUT_JSONL_NAME = "acl6060_dev_dataset.jsonl"
OUTPUT_TTS_JSONL_NAME = "acl6060_dev_dataset_with_tts.jsonl"
# ======Configuration=====


def load_glossary_by_paper(glossary_dir: str, paper_ids: List[str]) -> Set[str]:
    """Load and merge per-paper glossary JSONs, dedup by lowercase."""
    glossary: Set[str] = set()
    total_raw = 0
    for pid in paper_ids:
        path = os.path.join(glossary_dir, f"extracted_glossary_list__{pid}.json")
        assert os.path.isfile(path), f"Glossary not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            terms = json.load(f)
        assert isinstance(terms, list), f"Expected list in {path}, got {type(terms)}"
        total_raw += len(terms)
        for t in terms:
            glossary.add(t.strip().lower())
        _log(f"  {pid}: {len(terms)} terms")
    _log(f"Merged glossary: {total_raw} raw -> {len(glossary)} unique lowercased")
    return glossary


def load_raw_text(txt_path: str) -> Dict[int, str]:
    """Return {seg_id: raw_text} where seg_id is 1-based."""
    result: Dict[int, str] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            result[i] = line
    _log(f"Raw text: {len(result)} segments loaded")
    return result


def _build_term_patterns(glossary: Set[str]) -> List[Tuple[str, re.Pattern]]:
    """
    Build regex patterns for each glossary term, sorted longest-first
    so longer terms match before shorter substrings.

    Handles:
      - Optional plural suffix (s/es/ies)
      - Hyphen / space / no-separator variants (e.g. code-switching ↔ code switching ↔ codeswitching)
    """
    sorted_terms = sorted(glossary, key=len, reverse=True)
    patterns: List[Tuple[str, re.Pattern]] = []
    for term in sorted_terms:
        token_parts = re.split(r"([\s\-])", term)
        regex_parts: List[str] = []
        for part in token_parts:
            if part in (" ", "-"):
                regex_parts.append(r"[\s\-]?")
            else:
                regex_parts.append(re.escape(part))
        base = "".join(regex_parts)
        base_with_plural = base + r"(?:e?s|ies)?"
        pattern = re.compile(r"\b" + base_with_plural + r"\b", re.IGNORECASE)
        patterns.append((term, pattern))
    return patterns


def find_terms_in_text(
    text: str,
    term_patterns: List[Tuple[str, re.Pattern]],
) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    """
    Find all glossary term occurrences in text via word-boundary regex.
    Returns (term_list, [(term_lc, char_start, char_end), ...]).
    Longer terms take priority; matched spans are excluded from shorter matches.
    """
    occupied = set()  # char positions already claimed
    terms: List[str] = []
    spans: List[Tuple[str, int, int]] = []

    for term_lc, pattern in term_patterns:
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if any(pos in occupied for pos in range(start, end)):
                continue
            occupied.update(range(start, end))
            terms.append(term_lc)
            spans.append((term_lc, start, end))

    spans.sort(key=lambda x: x[1])
    terms = [s[0] for s in spans]
    return terms, spans


def build_talks_from_raw_text(
    xml_talks: List[Tuple[str, int, int]],
    raw_text: Dict[int, str],
    glossary: Set[str],
    wav_dir: str,
) -> List[TalkInfo]:
    """Build TalkInfo from raw text with term matching."""
    term_patterns = _build_term_patterns(glossary)
    talks: List[TalkInfo] = []
    total_found = 0

    for docid, first_seg, last_seg in xml_talks:
        wav_path = os.path.join(wav_dir, f"{docid}.wav")
        assert os.path.isfile(wav_path), f"WAV not found: {wav_path}"

        segments: List[SegmentInfo] = []
        for sid in range(first_seg, last_seg + 1):
            assert sid in raw_text, f"Segment {sid} not in raw text"
            text = raw_text[sid]
            terms, spans = find_terms_in_text(text, term_patterns)
            total_found += len(terms)

            segments.append(SegmentInfo(
                seg_id=sid,
                clean_text=text,
                terms=terms,
                term_spans=spans,
            ))

        talks.append(TalkInfo(docid=docid, wav_path=wav_path, segments=segments))
        talk_terms = sum(len(s.terms) for s in segments)
        _log(f"  Talk {docid}: segs={len(segments)}, terms_found={talk_terms}")

    _log(f"Total term occurrences found: {total_found}")
    return talks


def map_terms_to_timestamps(
    talk: TalkInfo,
    mfa_intervals: List[WordInterval],
) -> List[TermTimeSpan]:
    """
    Map term occurrences (found by text matching) to MFA time spans.
    Reuses word alignment logic from the base module.
    """
    global_words = build_global_word_list(talk)
    expected_words = [w for w, _, _ in global_words]
    alignment = align_expected_to_mfa(expected_words, mfa_intervals)

    seg_word_offsets: List[int] = []
    offset = 0
    for seg in talk.segments:
        seg_word_offsets.append(offset)
        offset += len(seg.clean_text.split())

    term_spans: List[TermTimeSpan] = []

    for seg_i, seg in enumerate(talk.segments):
        if not seg.terms:
            continue

        seg_words = seg.clean_text.split()
        seg_offset = seg_word_offsets[seg_i]

        for term, char_start, char_end in seg.term_spans:
            prefix = seg.clean_text[:char_start]
            prefix_words = prefix.split() if prefix.strip() else []
            term_words = seg.clean_text[char_start:char_end].split()

            first_word_idx = len(prefix_words)
            last_word_idx = first_word_idx + len(term_words) - 1

            if last_word_idx >= len(seg_words):
                _warn(f"Term '{term}' word index out of range in seg {seg.seg_id}")
                continue

            global_first = seg_offset + first_word_idx
            global_last = seg_offset + last_word_idx

            if global_first >= len(alignment) or global_last >= len(alignment):
                _warn(f"Term '{term}' global index out of range")
                continue

            first_interval = alignment[global_first]
            last_interval = alignment[global_last]

            if first_interval is None and last_interval is None:
                if seg_offset < len(alignment) and seg_offset + len(seg_words) - 1 < len(alignment):
                    seg_start_int = None
                    seg_end_int = None
                    for k in range(seg_offset, seg_offset + len(seg_words)):
                        if alignment[k] is not None:
                            if seg_start_int is None:
                                seg_start_int = alignment[k]
                            seg_end_int = alignment[k]
                    if seg_start_int is not None and seg_end_int is not None:
                        seg_dur = seg_end_int.end - seg_start_int.start
                        seg_chars = len(seg.clean_text)
                        if seg_chars > 0 and seg_dur > 0:
                            t_start = seg_start_int.start + (char_start / seg_chars) * seg_dur
                            t_end = seg_start_int.start + (char_end / seg_chars) * seg_dur
                            term_spans.append(TermTimeSpan(term=term, start=t_start, end=t_end))
                            continue
                _warn(f"Term '{term}' (seg {seg.seg_id}): no MFA match, skipping")
                continue

            t_start = first_interval.start if first_interval else last_interval.start
            t_end = last_interval.end if last_interval else first_interval.end
            term_spans.append(TermTimeSpan(term=term, start=t_start, end=t_end))

    _log(f"  Term spans: {len(term_spans)} mapped to timestamps")
    return term_spans


def generate_chunks_reuse_audio(
    talk: TalkInfo,
    mfa_intervals: List[WordInterval],
    term_spans: List[TermTimeSpan],
    glossary: Set[str],
    chunk_audio_dir: str,
    *,
    overwrite_audio: bool = False,
) -> List[ChunkRecord]:
    """Generate chunk records and ensure each chunk WAV exists.

    Full-coverage only for GT terms: a term is assigned to a chunk iff its
    MFA span is fully contained in the chunk.
    """
    info = sf.info(talk.wav_path)
    wav_duration = info.duration
    assert info.samplerate == SAMPLE_RATE, (
        f"Expected SR={SAMPLE_RATE}, got {info.samplerate}"
    )

    num_chunks = max(1, int(np.ceil((wav_duration - CHUNK_SEC) / STRIDE_SEC)) + 1)
    chunks: List[ChunkRecord] = []

    for ci in range(num_chunks):
        c_start = ci * STRIDE_SEC
        c_end = c_start + CHUNK_SEC
        if c_start >= wav_duration:
            break

        words_in_chunk = [
            wi for wi in mfa_intervals
            if wi.start < c_end and wi.end > c_start
        ]
        src_text = " ".join(wi.word for wi in words_in_chunk)

        terms_in_chunk: Set[str] = set()
        for ts in term_spans:
            if ts.start >= c_start and ts.end <= c_end:
                term_lower = ts.term.strip().lower()
                if term_lower in glossary:
                    terms_in_chunk.add(term_lower)

        audio_path = os.path.join(chunk_audio_dir, f"{talk.docid}_chunk_{ci}.wav")
        if overwrite_audio or not os.path.isfile(audio_path):
            write_chunk_audio(talk.wav_path, audio_path, c_start, CHUNK_SEC)

        chunks.append(ChunkRecord(
            talk_id=talk.docid,
            chunk_idx=ci,
            start_sec=c_start,
            end_sec=min(c_end, wav_duration),
            src_text=src_text,
            terms=terms_in_chunk,
            audio_path=audio_path,
        ))

    with_term = sum(1 for c in chunks if c.terms)
    _log(f"  Talk {talk.docid}: {len(chunks)} chunks "
         f"(with_term={with_term}, no_term={len(chunks) - with_term})")
    return chunks


def write_chunk_audio(
    wav_path: str,
    output_path: str,
    start_sec: float,
    chunk_sec: float,
) -> None:
    start_sample = int(round(start_sec * SAMPLE_RATE))
    chunk_samples = int(round(chunk_sec * SAMPLE_RATE))
    audio, sr = sf.read(
        wav_path,
        start=start_sample,
        frames=chunk_samples,
        dtype="float32",
    )
    assert sr == SAMPLE_RATE, f"Expected SR={SAMPLE_RATE}, got {sr}: {wav_path}"
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - len(audio)), mode="constant")
    elif len(audio) > chunk_samples:
        audio = audio[:chunk_samples]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")


def save_jsonl_with_tts(
    all_chunks: List[ChunkRecord],
    jsonl_path: str,
    tts_jsonl_path: str,
    tts_map: Dict[str, str],
) -> None:
    rows = 0
    with open(jsonl_path, "w", encoding="utf-8") as f, \
         open(tts_jsonl_path, "w", encoding="utf-8") as f_tts:

        for chunk in all_chunks:
            if chunk.terms:
                for term in sorted(chunk.terms):
                    row = {
                        "term": term,
                        "term_key": term,
                        "chunk_src_text": chunk.src_text,
                        "utter_id": chunk.talk_id,
                        "chunk_idx": chunk.chunk_idx,
                        "chunk_audio_path": chunk.audio_path,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    row_tts = dict(row)
                    row_tts["tts_audio_path"] = tts_map.get(term, "")
                    f_tts.write(json.dumps(row_tts, ensure_ascii=False) + "\n")
                    rows += 1
            else:
                row = {
                    "term": "",
                    "term_key": "",
                    "chunk_src_text": chunk.src_text,
                    "utter_id": chunk.talk_id,
                    "chunk_idx": chunk.chunk_idx,
                    "chunk_audio_path": chunk.audio_path,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_tts = dict(row)
                row_tts["tts_audio_path"] = ""
                f_tts.write(json.dumps(row_tts, ensure_ascii=False) + "\n")
                rows += 1

    unique_chunks = len(all_chunks)
    with_term = sum(1 for c in all_chunks if c.terms)
    unique_terms: Set[str] = set()
    for c in all_chunks:
        unique_terms.update(c.terms)
    total_gt = sum(len(c.terms) for c in all_chunks)

    _log(f"JSONL: {rows} rows, {unique_chunks} chunks "
         f"(with_term={with_term}, no_term={unique_chunks - with_term})")
    _log(f"  GT occurrences={total_gt}, unique_terms={len(unique_terms)}")
    _log(f"  Saved: {jsonl_path}")
    _log(f"  Saved: {tts_jsonl_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--output-jsonl-name", default=OUTPUT_JSONL_NAME)
    parser.add_argument("--output-tts-jsonl-name", default=OUTPUT_TTS_JSONL_NAME)
    parser.add_argument(
        "--chunk-audio-dir",
        default="",
        help="Directory for generated chunk WAVs. Defaults to BASE_WORK_DIR/audio_chunks.",
    )
    parser.add_argument("--chunk-sec", type=float, default=CHUNK_SEC)
    parser.add_argument("--stride-sec", type=float, default=STRIDE_SEC)
    parser.add_argument("--overwrite-audio", action="store_true")
    return parser.parse_args()


def main() -> int:
    global CHUNK_SEC, STRIDE_SEC
    args = parse_args()
    CHUNK_SEC = float(args.chunk_sec)
    STRIDE_SEC = float(args.stride_sec)

    _log("=" * 70)
    _log("ACL6060 Dev — Extracted Paper Glossary (raw text matching)")
    _log("=" * 70)
    _log(f"chunk_sec={CHUNK_SEC:.2f} stride_sec={STRIDE_SEC:.2f}")

    for name, path in [
        ("ACL_WAV_DIR", ACL_WAV_DIR),
        ("ACL_XML_PATH", ACL_XML_PATH),
        ("ACL_RAW_TEXT_PATH", ACL_RAW_TEXT_PATH),
        ("GLOSSARY_BY_PAPER_DIR", GLOSSARY_BY_PAPER_DIR),
        ("TTS_TERM_TO_PATH", TTS_TERM_TO_PATH),
    ]:
        assert os.path.exists(path), f"{name} not found: {path}"

    os.makedirs(args.output_dir, exist_ok=True)
    mfa_output_dir = os.path.join(BASE_WORK_DIR, MFA_TEXTGRID_SUBDIR)
    chunk_audio_dir = args.chunk_audio_dir or os.path.join(BASE_WORK_DIR, CHUNK_AUDIO_SUBDIR)
    assert os.path.isdir(mfa_output_dir), f"MFA TextGrids not found: {mfa_output_dir}"
    os.makedirs(chunk_audio_dir, exist_ok=True)

    glossary = load_glossary_by_paper(GLOSSARY_BY_PAPER_DIR, GLOSSARY_PAPER_IDS)

    with open(TTS_TERM_TO_PATH, "r", encoding="utf-8") as f:
        raw_tts = json.load(f)
    tts_map: Dict[str, str] = {}
    for t, p in raw_tts.items():
        lc = t.strip().lower()
        if lc not in tts_map:
            tts_map[lc] = p

    xml_talks = parse_xml(ACL_XML_PATH)
    raw_text = load_raw_text(ACL_RAW_TEXT_PATH)
    talks = build_talks_from_raw_text(xml_talks, raw_text, glossary, ACL_WAV_DIR)

    all_chunks: List[ChunkRecord] = []
    for talk in talks:
        _log(f"Processing talk: {talk.docid}")
        tg_path = find_textgrid(mfa_output_dir, talk.docid)
        mfa_words = parse_short_textgrid(tg_path)
        term_spans = map_terms_to_timestamps(talk, mfa_words)
        talk_chunks = generate_chunks_reuse_audio(
            talk,
            mfa_words,
            term_spans,
            glossary,
            chunk_audio_dir,
            overwrite_audio=args.overwrite_audio,
        )
        all_chunks.extend(talk_chunks)

    output_jsonl = os.path.join(args.output_dir, args.output_jsonl_name)
    output_tts_jsonl = os.path.join(args.output_dir, args.output_tts_jsonl_name)
    save_jsonl_with_tts(all_chunks, output_jsonl, output_tts_jsonl, tts_map)

    # TTS coverage
    unique_terms: Set[str] = set()
    for c in all_chunks:
        unique_terms.update(c.terms)
    with_tts = sum(1 for t in unique_terms if tts_map.get(t))
    _log(f"\nTTS coverage: {with_tts}/{len(unique_terms)} terms have TTS audio")
    for t in sorted(unique_terms):
        has = "YES" if tts_map.get(t) else "NO"
        _log(f"  {t:<30s}  tts={has}")

    # Term-in-text check
    total_gt = 0
    in_text = 0
    for c in all_chunks:
        for t in c.terms:
            total_gt += 1
            if t in c.src_text.lower():
                in_text += 1
    if total_gt > 0:
        _log(f"\nTerm-in-text: {in_text}/{total_gt} = {in_text / total_gt * 100:.1f}%")
    else:
        _log("\nNo GT terms found.")

    _log("=" * 70)
    _log("DONE")
    _log("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
