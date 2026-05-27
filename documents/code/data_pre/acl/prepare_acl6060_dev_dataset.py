#!/usr/bin/env python3
"""
Prepare ACL6060 dev dataset for offline terminology retrieval evaluation.

Pipeline:
  1. Parse XML for per-talk segments
  2. Parse tagged terminology for terms per segment
  3. Prepare MFA input directories
  4. Run MFA forced alignment
  5. Parse TextGrid output for word timestamps
  6. Generate 1.92s overlapping chunks (0.96s stride)
  7. Map terms to chunks via word-level timestamps
  8. Save chunk WAV files + output JSONL
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf

# ======Configuration=====
ACL_WAV_DIR = "/mnt/data/siqiouyang/datasets/acl6060/dev/full_wavs"
ACL_XML_PATH = (
    "/mnt/data/siqiouyang/datasets/acl6060/dev/text/xml/"
    "ACL.6060.dev.en-xx.en.xml"
)
ACL_TAGGED_PATH = (
    "/mnt/data/siqiouyang/datasets/acl6060/dev/text/tagged_terminology/"
    "ACL.6060.dev.tagged.en-xx.en.txt"
)
GLOSSARY_NPY = "/mnt/gemini/data/siqiouyang/acl_terms.npy"

WORK_DIR = "/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval"
MFA_INPUT_SUBDIR = "mfa_input"
MFA_TEXTGRID_SUBDIR = "mfa_textgrids"
MFA_CACHE_SUBDIR = "mfa_cache"
CHUNK_AUDIO_SUBDIR = "audio_chunks"
OUTPUT_JSONL_NAME = "acl6060_dev_dataset.jsonl"

SAMPLE_RATE = 16000
BASE_UNIT_SEC = 0.96
CHUNK_SEC = 1.92
STRIDE_SEC = 0.96
CHUNK_SAMPLES = int(CHUNK_SEC * SAMPLE_RATE)

MFA_CONDA_ENV = "mfa"
MFA_NUM_JOBS = 64
MFA_ACOUSTIC_MODEL = "english_mfa"
MFA_DICTIONARY = "english_mfa"

TERM_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
# ======Configuration=====


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SegmentInfo:
    seg_id: int
    clean_text: str
    terms: List[str]
    term_spans: List[Tuple[str, int, int]]  # (term, char_start, char_end) in clean_text


@dataclass
class TalkInfo:
    docid: str
    wav_path: str
    segments: List[SegmentInfo]

    @property
    def full_clean_text(self) -> str:
        return " ".join(seg.clean_text for seg in self.segments)


@dataclass
class WordInterval:
    word: str
    start: float
    end: float


@dataclass
class TermTimeSpan:
    term: str
    start: float
    end: float


@dataclass
class ChunkRecord:
    talk_id: str
    chunk_idx: int
    start_sec: float
    end_sec: float
    src_text: str
    terms: Set[str]
    audio_path: str


# ---------------------------------------------------------------------------
# Step 1: Parse XML for per-talk segment ranges
# ---------------------------------------------------------------------------


def parse_xml(xml_path: str) -> List[Tuple[str, int, int]]:
    """Return [(docid, first_seg_id, last_seg_id), ...]."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    talks: List[Tuple[str, int, int]] = []
    for doc in root.iter("doc"):
        docid = doc.get("docid", "")
        assert docid, "Missing docid in XML"
        segs = doc.findall("seg")
        assert segs, f"No segments in doc {docid}"
        first_id = int(segs[0].get("id", "0"))
        last_id = int(segs[-1].get("id", "0"))
        talks.append((docid, first_id, last_id))
    _log(f"XML: {len(talks)} talks parsed")
    return talks


# ---------------------------------------------------------------------------
# Step 2: Parse tagged terminology
# ---------------------------------------------------------------------------


def _extract_terms_and_clean(tagged_line: str) -> Tuple[str, List[str], List[Tuple[str, int, int]]]:
    """
    Extract terms from bracketed text.
    Returns (clean_text, term_list, term_char_spans).
    """
    terms: List[str] = []
    spans: List[Tuple[str, int, int]] = []
    clean_parts: List[str] = []
    char_offset = 0
    last_end = 0

    for m in TERM_BRACKET_RE.finditer(tagged_line):
        before = tagged_line[last_end:m.start()]
        clean_parts.append(before)
        char_offset += len(before)

        term_text = m.group(1)
        term_start = char_offset
        term_end = char_offset + len(term_text)
        clean_parts.append(term_text)
        char_offset = term_end

        terms.append(term_text)
        spans.append((term_text, term_start, term_end))
        last_end = m.end()

    remaining = tagged_line[last_end:]
    clean_parts.append(remaining)
    clean_text = "".join(clean_parts)
    return clean_text, terms, spans


def load_tagged_text(tagged_path: str) -> Dict[int, Tuple[str, List[str], List[Tuple[str, int, int]]]]:
    """Return {seg_id: (clean_text, terms, char_spans)} where seg_id is 1-based."""
    result: Dict[int, Tuple[str, List[str], List[Tuple[str, int, int]]]] = {}
    with open(tagged_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            clean, terms, spans = _extract_terms_and_clean(line)
            result[i] = (clean, terms, spans)
    _log(f"Tagged text: {len(result)} segments parsed")
    return result


def load_glossary(npy_path: str) -> Set[str]:
    arr = np.load(npy_path, allow_pickle=True)
    glossary = set(str(t).strip().lower() for t in arr if str(t).strip())
    _log(f"Glossary: {len(glossary)} unique terms (lowercased)")
    return glossary


def build_talks(
    xml_talks: List[Tuple[str, int, int]],
    tagged_data: Dict[int, Tuple[str, List[str], List[Tuple[str, int, int]]]],
    glossary: Set[str],
    wav_dir: str,
) -> List[TalkInfo]:
    """Build TalkInfo objects with filtered terms (only glossary terms kept)."""
    talks: List[TalkInfo] = []
    total_terms_raw = 0
    total_terms_kept = 0

    for docid, first_seg, last_seg in xml_talks:
        wav_path = os.path.join(wav_dir, f"{docid}.wav")
        assert os.path.isfile(wav_path), f"WAV not found: {wav_path}"

        segments: List[SegmentInfo] = []
        for sid in range(first_seg, last_seg + 1):
            assert sid in tagged_data, f"Segment {sid} not found in tagged text"
            clean_text, raw_terms, char_spans = tagged_data[sid]

            filtered_terms: List[str] = []
            filtered_spans: List[Tuple[str, int, int]] = []
            for term, cs, ce in char_spans:
                total_terms_raw += 1
                if term.strip().lower() in glossary:
                    filtered_terms.append(term)
                    filtered_spans.append((term, cs, ce))
                    total_terms_kept += 1

            segments.append(SegmentInfo(
                seg_id=sid,
                clean_text=clean_text,
                terms=filtered_terms,
                term_spans=filtered_spans,
            ))

        talks.append(TalkInfo(docid=docid, wav_path=wav_path, segments=segments))
        _log(f"  Talk {docid}: segs={len(segments)}, "
             f"terms={sum(len(s.terms) for s in segments)}")

    _log(f"Terms: raw={total_terms_raw}, kept (in glossary)={total_terms_kept}")
    return talks


# ---------------------------------------------------------------------------
# Step 3: Prepare MFA input
# ---------------------------------------------------------------------------


def prepare_mfa_input(talks: List[TalkInfo], mfa_input_dir: str) -> None:
    os.makedirs(mfa_input_dir, exist_ok=True)
    for talk in talks:
        talk_dir = os.path.join(mfa_input_dir, talk.docid)
        os.makedirs(talk_dir, exist_ok=True)

        wav_link = os.path.join(talk_dir, f"{talk.docid}.wav")
        if not os.path.exists(wav_link):
            os.symlink(talk.wav_path, wav_link)

        lab_path = os.path.join(talk_dir, f"{talk.docid}.lab")
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(talk.full_clean_text)

        _log(f"  MFA input: {talk_dir}")


# ---------------------------------------------------------------------------
# Step 4: Run MFA
# ---------------------------------------------------------------------------


def run_mfa(mfa_input_dir: str, mfa_output_dir: str, mfa_cache_dir: str) -> None:
    os.makedirs(mfa_output_dir, exist_ok=True)
    os.makedirs(mfa_cache_dir, exist_ok=True)

    cmd = [
        "conda", "run", "-n", MFA_CONDA_ENV,
        "mfa", "align",
        "--clean", "--final_clean",
        "--single_speaker",
        "--num_jobs", str(MFA_NUM_JOBS),
        "--temporary_directory", mfa_cache_dir,
        "--overwrite",
        "--output_format", "short_textgrid",
        mfa_input_dir,
        MFA_DICTIONARY,
        MFA_ACOUSTIC_MODEL,
        mfa_output_dir,
    ]

    _log(f"Running MFA: {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        _warn(f"MFA stderr:\n{proc.stderr}")
        raise RuntimeError(
            f"MFA failed with exit code {proc.returncode}.\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    _log("MFA alignment completed successfully.")


# ---------------------------------------------------------------------------
# Step 5: Parse TextGrid (short format)
# ---------------------------------------------------------------------------


def parse_short_textgrid(tg_path: str) -> List[WordInterval]:
    """Parse MFA short-format TextGrid; return word intervals."""
    with open(tg_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]

    assert len(lines) >= 7, f"TextGrid too short: {tg_path}"

    idx = 0
    while idx < len(lines) and lines[idx] != "<exists>":
        idx += 1
    assert idx < len(lines), f"No <exists> tag in {tg_path}"
    idx += 1

    num_tiers = int(lines[idx])
    idx += 1

    words: List[WordInterval] = []

    for _ in range(num_tiers):
        tier_class = lines[idx].strip('"')
        idx += 1
        tier_name = lines[idx].strip('"')
        idx += 1
        _tier_xmin = float(lines[idx])
        idx += 1
        _tier_xmax = float(lines[idx])
        idx += 1
        num_intervals = int(lines[idx])
        idx += 1

        for _ in range(num_intervals):
            xmin = float(lines[idx])
            idx += 1
            xmax = float(lines[idx])
            idx += 1
            text = lines[idx].strip('"')
            idx += 1

            if tier_name == "words" and text.strip():
                words.append(WordInterval(word=text, start=xmin, end=xmax))

    assert words, f"No word intervals found in {tg_path}"
    _log(f"  TextGrid {Path(tg_path).name}: {len(words)} word intervals")
    return words


def find_textgrid(mfa_output_dir: str, docid: str) -> str:
    """Locate the TextGrid file for a talk in MFA output."""
    candidates = [
        os.path.join(mfa_output_dir, docid, f"{docid}.TextGrid"),
        os.path.join(mfa_output_dir, f"{docid}.TextGrid"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Recursive search
    for root, _dirs, files in os.walk(mfa_output_dir):
        for fname in files:
            if fname.endswith(".TextGrid") and docid in fname:
                return os.path.join(root, fname)

    raise FileNotFoundError(
        f"TextGrid not found for {docid} in {mfa_output_dir}. "
        f"Checked: {candidates}"
    )


# ---------------------------------------------------------------------------
# Step 6: Align words + Map terms to timestamps
# ---------------------------------------------------------------------------


def _normalize_word(w: str) -> str:
    """Normalize a word for matching: lowercase, strip punctuation."""
    w = w.lower().strip()
    w = re.sub(r"[^\w'-]", "", w)
    return w


def build_global_word_list(talk: TalkInfo) -> List[Tuple[str, int, int]]:
    """
    Build a global ordered word list for the talk.
    Returns [(word, seg_idx, word_in_seg_idx), ...].
    """
    result: List[Tuple[str, int, int]] = []
    for seg_i, seg in enumerate(talk.segments):
        words = seg.clean_text.split()
        for w_i, w in enumerate(words):
            result.append((w, seg_i, w_i))
    return result


def align_expected_to_mfa(
    expected_words: List[str],
    mfa_intervals: List[WordInterval],
) -> List[Optional[WordInterval]]:
    """
    Align expected word list to MFA word intervals using greedy forward matching.
    Returns a list parallel to expected_words, with the matched MFA interval or None.
    """
    result: List[Optional[WordInterval]] = [None] * len(expected_words)
    mfa_ptr = 0

    for i, exp_word in enumerate(expected_words):
        exp_norm = _normalize_word(exp_word)
        if not exp_norm:
            continue

        best_j = None
        search_window = 5
        for j in range(mfa_ptr, min(mfa_ptr + search_window, len(mfa_intervals))):
            mfa_norm = _normalize_word(mfa_intervals[j].word)
            if mfa_norm == exp_norm:
                best_j = j
                break

        if best_j is not None:
            result[i] = mfa_intervals[best_j]
            mfa_ptr = best_j + 1
        else:
            pass

    matched = sum(1 for r in result if r is not None)
    _log(f"  Word alignment: {matched}/{len(expected_words)} matched "
         f"({matched/len(expected_words)*100:.1f}%)")
    return result


def map_terms_to_timestamps(
    talk: TalkInfo,
    mfa_intervals: List[WordInterval],
) -> List[TermTimeSpan]:
    """
    Map each term occurrence to a time span using MFA word timestamps.
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
            term_words = term.split()

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


# ---------------------------------------------------------------------------
# Step 7: Generate overlapping chunks
# ---------------------------------------------------------------------------


def generate_chunks(
    talk: TalkInfo,
    mfa_intervals: List[WordInterval],
    term_spans: List[TermTimeSpan],
    glossary: Set[str],
) -> List[ChunkRecord]:
    """Generate 1.92s overlapping chunks with 0.96s stride."""
    info = sf.info(talk.wav_path)
    wav_duration = info.duration
    assert info.samplerate == SAMPLE_RATE, (
        f"Expected SR={SAMPLE_RATE}, got {info.samplerate} for {talk.wav_path}"
    )

    num_chunks = max(1, int(np.ceil((wav_duration - CHUNK_SEC) / STRIDE_SEC)) + 1)
    chunks: List[ChunkRecord] = []

    chunk_audio_dir = os.path.join(WORK_DIR, CHUNK_AUDIO_SUBDIR)
    os.makedirs(chunk_audio_dir, exist_ok=True)

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

        audio_path = os.path.join(
            chunk_audio_dir, f"{talk.docid}_chunk_{ci}.wav"
        )
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
         f"(with_term={with_term}, no_term={len(chunks)-with_term})")
    return chunks


# ---------------------------------------------------------------------------
# Step 8: Save chunk audio + JSONL
# ---------------------------------------------------------------------------


def save_chunk_audio(wav_path: str, chunks: List[ChunkRecord]) -> None:
    """Read full WAV and write each chunk as a separate file."""
    audio, sr = sf.read(wav_path)
    assert sr == SAMPLE_RATE
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    for chunk in chunks:
        start_sample = int(chunk.start_sec * SAMPLE_RATE)
        end_sample = start_sample + CHUNK_SAMPLES

        if start_sample >= len(audio):
            chunk_audio = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        else:
            chunk_audio = audio[start_sample:end_sample]
            if len(chunk_audio) < CHUNK_SAMPLES:
                chunk_audio = np.pad(
                    chunk_audio,
                    (0, CHUNK_SAMPLES - len(chunk_audio)),
                    mode="constant",
                )

        sf.write(chunk.audio_path, chunk_audio, SAMPLE_RATE)


def save_jsonl(all_chunks: List[ChunkRecord], output_path: str) -> None:
    """Save dataset in GigaSpeech dev JSONL format."""
    rows_written = 0
    with open(output_path, "w", encoding="utf-8") as f:
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
                    rows_written += 1
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
                rows_written += 1

    unique_chunks = len(all_chunks)
    with_term = sum(1 for c in all_chunks if c.terms)
    unique_terms = set()
    for c in all_chunks:
        unique_terms.update(c.terms)

    _log(f"JSONL: {rows_written} rows, {unique_chunks} unique chunks "
         f"(with_term={with_term}, no_term={unique_chunks - with_term}), "
         f"unique_terms={len(unique_terms)}")
    _log(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _log("=" * 70)
    _log("ACL6060 Dev Dataset Preparation")
    _log("=" * 70)

    for path_name, path_val in [
        ("ACL_WAV_DIR", ACL_WAV_DIR),
        ("ACL_XML_PATH", ACL_XML_PATH),
        ("ACL_TAGGED_PATH", ACL_TAGGED_PATH),
        ("GLOSSARY_NPY", GLOSSARY_NPY),
    ]:
        assert os.path.exists(path_val), f"{path_name} not found: {path_val}"

    os.makedirs(WORK_DIR, exist_ok=True)
    mfa_input_dir = os.path.join(WORK_DIR, MFA_INPUT_SUBDIR)
    mfa_output_dir = os.path.join(WORK_DIR, MFA_TEXTGRID_SUBDIR)
    mfa_cache_dir = os.path.join(WORK_DIR, MFA_CACHE_SUBDIR)
    output_jsonl = os.path.join(WORK_DIR, OUTPUT_JSONL_NAME)

    # --- Step 1: Parse XML ---
    _log("--- Step 1: Parse XML ---")
    xml_talks = parse_xml(ACL_XML_PATH)

    # --- Step 2: Parse tagged text + glossary ---
    _log("--- Step 2: Parse tagged terminology ---")
    tagged_data = load_tagged_text(ACL_TAGGED_PATH)
    glossary = load_glossary(GLOSSARY_NPY)
    talks = build_talks(xml_talks, tagged_data, glossary, ACL_WAV_DIR)

    # --- Step 3: Prepare MFA input ---
    _log("--- Step 3: Prepare MFA input ---")
    prepare_mfa_input(talks, mfa_input_dir)

    # --- Step 4: Run MFA ---
    _log("--- Step 4: Run MFA forced alignment ---")
    run_mfa(mfa_input_dir, mfa_output_dir, mfa_cache_dir)

    # --- Step 5-7: Parse TextGrid + generate chunks + map terms ---
    _log("--- Steps 5-7: Parse alignment, generate chunks, map terms ---")
    all_chunks: List[ChunkRecord] = []

    for talk in talks:
        _log(f"Processing talk: {talk.docid}")
        tg_path = find_textgrid(mfa_output_dir, talk.docid)
        mfa_words = parse_short_textgrid(tg_path)

        term_spans = map_terms_to_timestamps(talk, mfa_words)
        talk_chunks = generate_chunks(talk, mfa_words, term_spans, glossary)

        _log(f"  Saving chunk audio for {talk.docid}...")
        save_chunk_audio(talk.wav_path, talk_chunks)
        all_chunks.extend(talk_chunks)

    # --- Step 8: Save JSONL ---
    _log("--- Step 8: Save JSONL ---")
    save_jsonl(all_chunks, output_jsonl)

    # --- Summary ---
    _log("=" * 70)
    _log("SUMMARY")
    _log("=" * 70)
    total_chunks = len(all_chunks)
    with_term = sum(1 for c in all_chunks if c.terms)
    all_terms: Set[str] = set()
    for c in all_chunks:
        all_terms.update(c.terms)
    _log(f"Total chunks:     {total_chunks}")
    _log(f"With-term chunks: {with_term}")
    _log(f"No-term chunks:   {total_chunks - with_term}")
    _log(f"Unique terms:     {len(all_terms)}")
    _log(f"Output JSONL:     {output_jsonl}")
    _log(f"Chunk audio dir:  {os.path.join(WORK_DIR, CHUNK_AUDIO_SUBDIR)}")
    _log("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
