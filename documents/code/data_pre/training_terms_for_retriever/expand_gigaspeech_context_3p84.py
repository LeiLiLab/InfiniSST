#!/usr/bin/env python3
"""Expand retriever rows from 1.92s chunks to 3.84s chunks.

The input JSONL is usually one row per positive term event, with optional
empty-term rows for eval.  For each existing GigaSpeech speech group
`(utter_id, chunk_idx)`, this script cuts a longer 3.84s chunk from the
original GigaSpeech opus file and writes one output row for every known term
event whose MFA span overlaps the longer window.  That keeps newly audible
terms in the same speech group, so the trainer can mask them as positives
instead of treating them as false negatives.

Rows whose `utter_id` starts with `wiki_synth_` are recut from the original TTS
WAV and wiki-synth MFA TextGrid inferred from the 1.92s chunk path.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf


WIKI_SYNTH_PREFIX = "wiki_synth_"
SAMPLE_RATE = 16000
OLD_CHUNK_SEC = 1.92
NEW_CHUNK_SEC = 3.84
STRIDE_SEC = 0.96
PROGRESS_EVERY = 100_000

GS_TEXTGRID_DIR = "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"
GS_SQLITE_INDEX = "/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite"

DEFAULT_INPUT = (
    "/mnt/gemini/home/jiaxuanluo/"
    "term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl"
)
DEFAULT_OUTPUT = (
    "/mnt/gemini/home/jiaxuanluo/"
    "term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_gsctx3p84.jsonl"
)
DEFAULT_AUDIO_DIR = (
    "/mnt/gemini/data1/jiaxuanluo/"
    "term_train_audio_chunks_gsv2full_gsdedup_gsctx3p84"
)
WORD_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9']+")


@dataclass(frozen=True)
class AlignInfo:
    opus: str
    start_sample: int
    end_sample: int


@dataclass
class TermEvent:
    row: Dict[str, Any]
    utter_id: str
    source_chunk_idx: int
    term_key: str
    abs_start_sec: float
    abs_end_sec: float
    line_no: int

    @property
    def duration_sec(self) -> float:
        return self.abs_end_sec - self.abs_start_sec


@dataclass(frozen=True)
class WikiSourceInfo:
    source_wav: str
    lab: str
    textgrid: str
    source_tag: str
    shard_tag: str
    utt_id: str


def normalize_word(w: str) -> str:
    w = w.strip().lower().replace("\u2019", "'")
    if w.endswith("'s"):
        w = w[:-2]
    return WORD_NORMALIZE_PATTERN.sub("", w)


def tokenize_text(s: str) -> List[str]:
    raw = s.strip().lower().replace("\u2019", "'")
    return [t for t in (normalize_word(w) for w in raw.split()) if t]


def tokenize_text_variants(s: str) -> List[List[str]]:
    raw = s.strip().lower().replace("\u2019", "'")
    variants: List[List[str]] = []

    def add(tokens: List[str]) -> None:
        if tokens and tokens not in variants:
            variants.append(tokens)

    add(tokenize_text(raw))
    split_tokens = [
        t for t in (normalize_word(w) for w in WORD_NORMALIZE_PATTERN.split(raw)) if t
    ]
    add(split_tokens)
    return variants


def find_all_subseq(haystack: List[str], needle: List[str]) -> List[int]:
    if not needle or len(needle) > len(haystack):
        return []
    hits = []
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            hits.append(i)
    return hits


def term_key(row: Dict[str, Any]) -> str:
    return str(row.get("term_key") or row.get("term") or "").strip().casefold()


def stable_shard_id(key: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % num_shards


def row_selected_for_shard(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    num_shards = int(getattr(args, "num_shards", 1) or 1)
    if num_shards <= 1:
        return True
    utter_id = str(row.get("utter_id") or "").strip()
    if not utter_id:
        return int(getattr(args, "shard_id", 0) or 0) == 0
    return stable_shard_id(utter_id, num_shards) == int(getattr(args, "shard_id", 0) or 0)


def parse_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_textgrid_words(tg_path: str) -> List[Tuple[float, float, str]]:
    with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f.readlines()]
    tier_name_idx = None
    for i, line in enumerate(lines):
        if line == '"words"':
            tier_name_idx = i
            break
    if tier_name_idx is None:
        raise ValueError(f'No "words" tier in {tg_path}')

    n_intervals = int(lines[tier_name_idx + 3])
    cursor = tier_name_idx + 4
    intervals: List[Tuple[float, float, str]] = []
    for _ in range(n_intervals):
        start = float(lines[cursor])
        end = float(lines[cursor + 1])
        word = lines[cursor + 2]
        if word.startswith('"') and word.endswith('"') and len(word) >= 2:
            word = word[1:-1]
        intervals.append((start, end, word))
        cursor += 3
    return intervals


class GigaSpeechLookup:
    def __init__(self, sqlite_path: str, textgrid_dir: str, sample_rate: int):
        if not os.path.isfile(sqlite_path):
            raise FileNotFoundError(sqlite_path)
        self._con = sqlite3.connect(sqlite_path)
        self._con.execute("PRAGMA read_uncommitted=1;")
        self._cur_align = self._con.cursor()
        self._cur_manifest = self._con.cursor()
        self._textgrid_dir = textgrid_dir
        self._sample_rate = sample_rate
        self._align_cache: Dict[str, Optional[AlignInfo]] = {}

    def get_align(self, utter_id: str) -> Optional[AlignInfo]:
        if utter_id in self._align_cache:
            return self._align_cache[utter_id]
        self._cur_align.execute(
            "SELECT opus, start, end FROM align_segments WHERE align_id = ?",
            (utter_id,),
        )
        row = self._cur_align.fetchone()
        if row is None:
            self._align_cache[utter_id] = None
            return None
        info = AlignInfo(str(row[0]), int(row[1]), int(row[2]))
        self._align_cache[utter_id] = info
        return info

    def manifest_segments_for_align(
        self, align: AlignInfo,
    ) -> List[Tuple[str, int, int]]:
        self._cur_manifest.execute(
            """SELECT seg_id, start, end FROM manifest_segments
               WHERE opus = ? AND start < ? AND end > ?
               ORDER BY start""",
            (align.opus, align.end_sample, align.start_sample),
        )
        return [(str(r[0]), int(r[1]), int(r[2])) for r in self._cur_manifest.fetchall()]

    def words_for_align(
        self, align: AlignInfo,
    ) -> List[Tuple[float, float, str]]:
        words: List[Tuple[float, float, str]] = []
        for seg_id, seg_start, seg_end in self.manifest_segments_for_align(align):
            tg_path = os.path.join(self._textgrid_dir, f"{seg_id}.TextGrid")
            if not os.path.isfile(tg_path):
                continue
            try:
                intervals = parse_textgrid_words(tg_path)
            except Exception:
                continue
            seg_start_sec = seg_start / self._sample_rate
            align_start_sec = align.start_sample / self._sample_rate
            align_end_sec = align.end_sample / self._sample_rate
            for start, end, word in intervals:
                abs_start = seg_start_sec + start
                abs_end = seg_start_sec + end
                if abs_start < align_end_sec and abs_end > align_start_sec:
                    words.append((abs_start, abs_end, word))
        words.sort(key=lambda x: (x[0], x[1]))
        return words


def clamp_context_start(
    old_start_sample: int,
    align: AlignInfo,
    *,
    old_chunk_samples: int,
    new_chunk_samples: int,
) -> Tuple[int, int]:
    """Return `(context_start_sample, read_frames)` inside the align segment."""

    left_extra = max(0, (new_chunk_samples - old_chunk_samples) // 2)
    desired_start = old_start_sample - left_extra
    align_len = max(0, align.end_sample - align.start_sample)
    if align_len >= new_chunk_samples:
        max_start = align.end_sample - new_chunk_samples
        context_start = min(max(desired_start, align.start_sample), max_start)
        return context_start, new_chunk_samples
    return align.start_sample, align_len


def build_window_text(
    words: Iterable[Tuple[float, float, str]],
    start_sec: float,
    end_sec: float,
) -> str:
    toks: List[str] = []
    for start, end, word in words:
        if not normalize_word(word):
            continue
        midpoint = (start + end) * 0.5
        if start_sec <= midpoint < end_sec:
            toks.append(word.strip())
    return " ".join(toks)


def locate_term_abs_in_words(
    words: List[Tuple[float, float, str]],
    term: str,
    old_chunk_start_sec: float,
    old_chunk_end_sec: float,
) -> Optional[Tuple[float, float]]:
    """Locate a term span in absolute-time word intervals.

    Used for dev JSONLs that have term labels but no pre-enriched MFA
    `mfa_term_*_in_chunk` fields.  If the term appears multiple times in the
    utterance, prefer the occurrence that overlaps the original 1.92s chunk.
    """

    term_variants = tokenize_text_variants(term)
    if not term_variants:
        return None

    words_with_idx: List[Tuple[int, str]] = []
    for idx, (_, _, word) in enumerate(words):
        norm = normalize_word(word)
        if norm:
            words_with_idx.append((idx, norm))
    norm_words = [w for _, w in words_with_idx]

    best: Optional[Tuple[float, float]] = None
    best_overlap = -1.0
    for tokens in term_variants:
        for hit in find_all_subseq(norm_words, tokens):
            first_orig = words_with_idx[hit][0]
            last_orig = words_with_idx[hit + len(tokens) - 1][0]
            start = words[first_orig][0]
            end = words[last_orig][1]
            overlap = max(0.0, min(end, old_chunk_end_sec) - max(start, old_chunk_start_sec))
            if overlap > best_overlap:
                best = (start, end)
                best_overlap = overlap
    if best is None or best_overlap <= 0.0:
        return None
    return best


def locate_term_in_full_words(
    words: List[Tuple[float, float, str]],
    term: str,
) -> Optional[Tuple[float, float]]:
    if not words:
        return None
    window_start = min(start for start, _, _ in words)
    window_end = max(end for _, end, _ in words)
    return locate_term_abs_in_words(words, term, window_start, window_end)


def nonempty_word_intervals(
    words: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    return [(start, end, word) for start, end, word in words if str(word).strip()]


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def locate_wiki_term_span(
    words: List[Tuple[float, float, str]],
    term: str,
    utterance: str,
) -> Optional[Tuple[float, float]]:
    span = locate_term_in_full_words(words, term)
    if span is not None:
        return span

    utt_words = tokenize_text(utterance)
    term_variants = tokenize_text_variants(term)
    word_intervals = nonempty_word_intervals(words)
    if not utt_words or len(word_intervals) != len(utt_words):
        return None

    for tokens in term_variants:
        for hit in find_all_subseq(utt_words, tokens):
            first = hit
            last = hit + len(tokens) - 1
            return word_intervals[first][0], word_intervals[last][1]
    return None


def build_wiki_window_text(
    words: List[Tuple[float, float, str]],
    start_sec: float,
    end_sec: float,
    utterance: str,
) -> str:
    word_intervals = nonempty_word_intervals(words)
    utt_tokens = [
        re.sub(r"[^\w'-]", "", token)
        for token in utterance.split()
    ]
    utt_tokens = [token for token in utt_tokens if token]
    use_original = len(word_intervals) == len(utt_tokens) and bool(utt_tokens)

    toks: List[str] = []
    for idx, (start, end, word) in enumerate(word_intervals):
        midpoint = (start + end) * 0.5
        if start_sec <= midpoint < end_sec:
            toks.append(utt_tokens[idx] if use_original else word.strip())
    return " ".join(toks)


def safe_wav_name(utter_id: str, chunk_idx: int) -> str:
    safe_utter = re.sub(r"[^A-Za-z0-9_.-]+", "_", utter_id)
    return f"{safe_utter}_ctx3p84_chunk_{chunk_idx}.wav"


def safe_wiki_wav_name(utter_id: str, chunk_idx: int) -> str:
    safe_utter = re.sub(r"[^A-Za-z0-9_.-]+", "_", utter_id)
    return f"{safe_utter}_ctx3p84_chunk_{chunk_idx}_clean.wav"


def ensure_audio_chunk(
    *,
    opus_path: str,
    output_path: str,
    start_sample: int,
    read_frames: int,
    new_chunk_samples: int,
    sample_rate: int,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    if dry_run:
        return True
    if os.path.isfile(output_path) and not overwrite:
        return True
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if read_frames <= 0:
        audio = np.zeros(new_chunk_samples, dtype=np.float32)
    else:
        audio, sr = sf.read(
            opus_path,
            start=start_sample,
            frames=read_frames,
            dtype="float32",
        )
        if sr != sample_rate:
            raise ValueError(f"Expected {sample_rate}Hz, got {sr}Hz: {opus_path}")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < new_chunk_samples:
            audio = np.pad(audio, (0, new_chunk_samples - len(audio)), mode="constant")
        elif len(audio) > new_chunk_samples:
            audio = audio[:new_chunk_samples]
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")
    return True


def infer_wiki_source(row: Dict[str, Any]) -> Optional[WikiSourceInfo]:
    chunk_path = str(row.get("chunk_audio_path") or "")
    match = re.match(
        r"^(?P<base>.+?/MFA/(?P<source_tag>3variant_gsv2_[^/]+))"
        r"/chunks/(?P<shard>shard_\d+)/(?P<utt>utt_\d+)_clean\.wav$",
        chunk_path,
    )
    if not match:
        return None
    base = match.group("base")
    shard = match.group("shard")
    utt_id = match.group("utt")
    source_wav = os.path.join(base, "work", shard, "mfa_input", f"{utt_id}.wav")
    lab = os.path.join(base, "work", shard, "mfa_input", f"{utt_id}.lab")
    textgrid = os.path.join(base, "work", shard, "mfa_output", f"{utt_id}.TextGrid")
    return WikiSourceInfo(
        source_wav=source_wav,
        lab=lab,
        textgrid=textgrid,
        source_tag=match.group("source_tag"),
        shard_tag=shard,
        utt_id=utt_id,
    )


def clamp_wiki_context_start(
    old_start_sample: int,
    total_samples: int,
    *,
    old_chunk_samples: int,
    new_chunk_samples: int,
) -> Tuple[int, int]:
    left_extra = max(0, (new_chunk_samples - old_chunk_samples) // 2)
    desired_start = old_start_sample - left_extra
    if total_samples >= new_chunk_samples:
        max_start = total_samples - new_chunk_samples
        return min(max(desired_start, 0), max_start), new_chunk_samples
    return 0, total_samples


def recover_old_wiki_chunk_start_sec(
    row: Dict[str, Any],
    term_start_sec: float,
    term_end_sec: float,
    total_duration_sec: float,
    old_chunk_sec: float,
) -> float:
    old_rel_start = parse_float(row.get("mfa_term_start_in_chunk"))
    old_rel_end = parse_float(row.get("mfa_term_end_in_chunk"))
    term_dur = max(0.0, term_end_sec - term_start_sec)
    eps = 1e-4

    if old_rel_start is not None and old_rel_end is not None and old_rel_end > old_rel_start:
        if term_dur >= old_chunk_sec - eps and old_rel_start <= eps and old_rel_end >= old_chunk_sec - eps:
            old_start = (term_start_sec + term_end_sec) * 0.5 - old_chunk_sec * 0.5
        elif old_rel_start > eps:
            old_start = term_start_sec - old_rel_start
        elif old_rel_end < old_chunk_sec - eps:
            old_start = term_end_sec - old_rel_end
        else:
            old_start = term_start_sec - old_rel_start
    else:
        old_start = (term_start_sec + term_end_sec) * 0.5 - old_chunk_sec * 0.5

    max_start = max(0.0, total_duration_sec - old_chunk_sec)
    return min(max(old_start, 0.0), max_start)


def collect_gigaspeech_events(
    args: argparse.Namespace,
    lookup: GigaSpeechLookup,
) -> Tuple[
    Dict[str, List[TermEvent]],
    Dict[str, float],
    Dict[Tuple[str, int], AlignInfo],
    Dict[Tuple[str, int], Dict[str, Any]],
    List[Dict[str, Any]],
    Counter,
]:
    events_by_utter: Dict[str, List[TermEvent]] = defaultdict(list)
    max_duration_by_utter: Dict[str, float] = defaultdict(float)
    groups: Dict[Tuple[str, int], AlignInfo] = {}
    group_representative: Dict[Tuple[str, int], Dict[str, Any]] = {}
    fallback_gs_rows: List[Dict[str, Any]] = []
    stats: Counter = Counter()

    old_chunk_start_offset_sec_by_key: Dict[Tuple[str, int], float] = {}
    words_cache: Dict[str, List[Tuple[float, float, str]]] = {}

    with open(args.input, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if args.max_lines and line_no > args.max_lines:
                break
            line = line.strip()
            if not line:
                stats["blank_lines"] += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["json_parse_errors"] += 1
                continue

            stats["input_rows"] += 1
            if not row_selected_for_shard(row, args):
                stats["input_rows_skipped_by_shard"] += 1
                continue
            stats["input_rows_selected_by_shard"] += 1
            utter_id = str(row.get("utter_id") or "").strip()
            if utter_id.startswith(WIKI_SYNTH_PREFIX):
                stats["wiki_rows_seen"] += 1
                continue

            stats["gigaspeech_rows_seen"] += 1
            align = lookup.get_align(utter_id)
            chunk_idx = parse_int(row.get("chunk_idx"))
            mfa_start = parse_float(row.get("mfa_term_start_in_chunk"))
            mfa_end = parse_float(row.get("mfa_term_end_in_chunk"))
            tk = term_key(row)

            if align is None or chunk_idx is None:
                stats["gigaspeech_unexpandable_rows"] += 1
                if args.copy_unexpandable_gs:
                    fallback_gs_rows.append(row)
                continue

            group_key = (utter_id, chunk_idx)
            groups[group_key] = align
            group_representative.setdefault(group_key, row)
            chunk_start_offset = old_chunk_start_offset_sec_by_key.get(group_key)
            if chunk_start_offset is None:
                chunk_start_offset = chunk_idx * args.stride_sec
                old_chunk_start_offset_sec_by_key[group_key] = chunk_start_offset

            align_start_sec = align.start_sample / args.sample_rate
            old_chunk_abs_start = align_start_sec + chunk_start_offset
            old_chunk_abs_end = old_chunk_abs_start + args.old_chunk_sec

            if not tk:
                stats["gigaspeech_empty_term_rows"] += 1
                continue

            if mfa_start is not None and mfa_end is not None and mfa_end > mfa_start:
                abs_start = old_chunk_abs_start + mfa_start
                abs_end = old_chunk_abs_start + mfa_end
            else:
                words = words_cache.get(utter_id)
                if words is None:
                    words = lookup.words_for_align(align)
                    words_cache[utter_id] = words
                located = locate_term_abs_in_words(
                    words,
                    tk,
                    old_chunk_abs_start,
                    old_chunk_abs_end,
                )
                if located is None:
                    stats["gigaspeech_term_mfa_lookup_failed"] += 1
                    continue
                abs_start, abs_end = located

            event = TermEvent(
                row=row,
                utter_id=utter_id,
                source_chunk_idx=chunk_idx,
                term_key=tk,
                abs_start_sec=abs_start,
                abs_end_sec=abs_end,
                line_no=line_no,
            )
            events_by_utter[utter_id].append(event)
            max_duration_by_utter[utter_id] = max(
                max_duration_by_utter[utter_id],
                max(0.0, event.duration_sec),
            )
            stats["gigaspeech_expandable_rows"] += 1

            if stats["input_rows"] % PROGRESS_EVERY == 0:
                print(
                    "[PASS1] "
                    f"rows={stats['input_rows']:,} "
                    f"gs_events={stats['gigaspeech_expandable_rows']:,} "
                    f"gs_groups={len(groups):,}",
                    flush=True,
                )

    for utter_events in events_by_utter.values():
        utter_events.sort(key=lambda ev: (ev.abs_start_sec, ev.abs_end_sec, ev.term_key))

    stats["gigaspeech_utterances"] = len(events_by_utter)
    stats["gigaspeech_context_groups"] = len(groups)
    return (
        events_by_utter,
        max_duration_by_utter,
        groups,
        group_representative,
        fallback_gs_rows,
        stats,
    )


def select_events_for_window(
    events: List[TermEvent],
    starts: List[float],
    max_duration_sec: float,
    start_sec: float,
    end_sec: float,
    include_mode: str,
) -> List[TermEvent]:
    if include_mode == "contained":
        lo = bisect.bisect_left(starts, start_sec)
        hi = bisect.bisect_left(starts, end_sec)
        return [ev for ev in events[lo:hi] if ev.abs_end_sec <= end_sec]

    lo = bisect.bisect_left(starts, start_sec - max_duration_sec - 0.01)
    hi = bisect.bisect_left(starts, end_sec)
    return [
        ev for ev in events[lo:hi]
        if ev.abs_start_sec < end_sec and ev.abs_end_sec > start_sec
    ]


def write_expanded_gigaspeech(
    args: argparse.Namespace,
    lookup: GigaSpeechLookup,
    events_by_utter: Dict[str, List[TermEvent]],
    max_duration_by_utter: Dict[str, float],
    groups: Dict[Tuple[str, int], AlignInfo],
    group_representative: Dict[Tuple[str, int], Dict[str, Any]],
    fallback_gs_rows: List[Dict[str, Any]],
    fout,
    stats: Counter,
) -> None:
    audio_dir = Path(args.audio_output_dir)
    if not args.dry_run:
        audio_dir.mkdir(parents=True, exist_ok=True)

    old_chunk_samples = int(round(args.old_chunk_sec * args.sample_rate))
    new_chunk_samples = int(round(args.new_chunk_sec * args.sample_rate))
    stride_samples = int(round(args.stride_sec * args.sample_rate))

    current_utter: Optional[str] = None
    current_words: List[Tuple[float, float, str]] = []
    current_starts: List[float] = []

    for group_no, ((utter_id, chunk_idx), align) in enumerate(sorted(groups.items()), start=1):
        if args.max_gs_groups and group_no > args.max_gs_groups:
            stats["gigaspeech_groups_skipped_by_limit"] += len(groups) - args.max_gs_groups
            break

        if utter_id != current_utter:
            current_utter = utter_id
            current_words = lookup.words_for_align(align)
            current_starts = [ev.abs_start_sec for ev in events_by_utter.get(utter_id, [])]

        old_start_sample = align.start_sample + chunk_idx * stride_samples
        context_start_sample, read_frames = clamp_context_start(
            old_start_sample,
            align,
            old_chunk_samples=old_chunk_samples,
            new_chunk_samples=new_chunk_samples,
        )
        context_start_sec = context_start_sample / args.sample_rate
        context_end_sec = context_start_sec + args.new_chunk_sec
        wav_path = str(audio_dir / safe_wav_name(utter_id, chunk_idx))

        events = events_by_utter.get(utter_id, [])
        selected = select_events_for_window(
            events,
            current_starts,
            max_duration_by_utter.get(utter_id, 0.0),
            context_start_sec,
            context_end_sec,
            args.include_mode,
        )

        seen_event_keys = set()
        unique_events: List[TermEvent] = []
        for ev in selected:
            key = (ev.term_key, round(ev.abs_start_sec, 4), round(ev.abs_end_sec, 4))
            if key in seen_event_keys:
                continue
            seen_event_keys.add(key)
            unique_events.append(ev)
        selected = unique_events
        if not selected:
            stats["gigaspeech_groups_without_selected_terms"] += 1
            if not args.write_empty_groups:
                continue

        try:
            ensure_audio_chunk(
                opus_path=align.opus,
                output_path=wav_path,
                start_sample=context_start_sample,
                read_frames=read_frames,
                new_chunk_samples=new_chunk_samples,
                sample_rate=args.sample_rate,
                overwrite=args.overwrite_audio,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            stats["audio_write_failures"] += 1
            if stats["audio_write_failures"] <= 20:
                print(
                    f"[WARN] audio write failed utter_id={utter_id} chunk_idx={chunk_idx}: {exc}",
                    flush=True,
            )
            continue

        expanded_text = build_window_text(current_words, context_start_sec, context_end_sec)
        if not expanded_text and selected:
            expanded_text = str(selected[0].row.get("chunk_src_text") or "")
        if not expanded_text:
            expanded_text = str(
                group_representative.get((utter_id, chunk_idx), {}).get("chunk_src_text") or ""
            )

        stats["written_gigaspeech_groups"] += 1
        stats["max_terms_per_expanded_group"] = max(
            stats["max_terms_per_expanded_group"],
            len(selected),
        )
        if len(selected) > 1:
            stats["multi_term_expanded_groups"] += 1

        if not selected:
            out_row = dict(group_representative.get((utter_id, chunk_idx), {}))
            out_row["term"] = ""
            out_row["term_key"] = ""
            out_row["chunk_idx"] = chunk_idx
            out_row["chunk_audio_path"] = wav_path
            out_row["chunk_src_text"] = expanded_text
            out_row["mfa_term_start_in_chunk"] = None
            out_row["mfa_term_end_in_chunk"] = None
            out_row["mfa_term_duration"] = None
            out_row["chunk_duration_sec"] = round(args.new_chunk_sec, 4)
            out_row["source_chunk_idx_1p92"] = chunk_idx
            out_row["context_start_sample"] = context_start_sample
            out_row["context_read_frames"] = read_frames
            out_row["context_build"] = "gigaspeech_mfa_expand_3p84"
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            stats["written_empty_gigaspeech_rows"] += 1
            continue

        for ev in selected:
            out_row = dict(ev.row)
            out_row["chunk_idx"] = chunk_idx
            out_row["chunk_audio_path"] = wav_path
            out_row["chunk_src_text"] = expanded_text
            out_row["mfa_term_start_in_chunk"] = round(ev.abs_start_sec - context_start_sec, 4)
            out_row["mfa_term_end_in_chunk"] = round(ev.abs_end_sec - context_start_sec, 4)
            out_row["mfa_term_duration"] = round(ev.duration_sec, 4)
            out_row["chunk_duration_sec"] = round(args.new_chunk_sec, 4)
            out_row["source_chunk_idx_1p92"] = ev.source_chunk_idx
            out_row["context_start_sample"] = context_start_sample
            out_row["context_read_frames"] = read_frames
            out_row["context_build"] = "gigaspeech_mfa_expand_3p84"
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            stats["written_gigaspeech_rows"] += 1
            if ev.source_chunk_idx != chunk_idx:
                stats["written_new_context_positive_rows"] += 1

        if stats["written_gigaspeech_groups"] % PROGRESS_EVERY == 0:
            print(
                "[PASS2-GS] "
                f"groups={stats['written_gigaspeech_groups']:,} "
                f"rows={stats['written_gigaspeech_rows']:,} "
                f"new_context_pos={stats['written_new_context_positive_rows']:,}",
                flush=True,
            )

    for row in fallback_gs_rows:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["written_unexpandable_gigaspeech_rows"] += 1


def write_expanded_wiki_rows(args: argparse.Namespace, fout, stats: Counter) -> None:
    if args.no_copy_wiki:
        return
    wiki_root = Path(args.wiki_audio_output_dir or os.path.join(args.audio_output_dir, "wiki_synth"))
    old_chunk_samples = int(round(args.old_chunk_sec * args.sample_rate))
    new_chunk_samples = int(round(args.new_chunk_sec * args.sample_rate))

    with open(args.input, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if args.max_lines and line_no > args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row_selected_for_shard(row, args):
                continue
            utter_id = str(row.get("utter_id") or "").strip()
            if not utter_id.startswith(WIKI_SYNTH_PREFIX):
                continue

            stats["wiki_rows_for_expand"] += 1
            chunk_idx = parse_int(row.get("chunk_idx")) or 0
            source = infer_wiki_source(row)
            if source is None:
                stats["wiki_source_infer_failures"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_unexpanded_wiki_rows"] += 1
                continue
            if not os.path.isfile(source.source_wav):
                stats["wiki_source_wav_missing"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_unexpanded_wiki_rows"] += 1
                continue
            if not os.path.isfile(source.lab):
                stats["wiki_lab_missing"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_unexpanded_wiki_rows"] += 1
                continue
            if not os.path.isfile(source.textgrid):
                stats["wiki_textgrid_missing"] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_unexpanded_wiki_rows"] += 1
                continue

            try:
                info = sf.info(source.source_wav)
                if info.samplerate != args.sample_rate:
                    raise ValueError(
                        f"Expected {args.sample_rate}Hz, got {info.samplerate}Hz: "
                        f"{source.source_wav}"
                    )
                words = parse_textgrid_words(source.textgrid)
                utterance = read_text(source.lab)
                term_span = locate_wiki_term_span(
                    words,
                    str(row.get("term") or ""),
                    utterance,
                )
                if term_span is None:
                    raise ValueError(f"term not located in TextGrid: {row.get('term')}")
                term_start_sec, term_end_sec = term_span
                old_start_sec = recover_old_wiki_chunk_start_sec(
                    row,
                    term_start_sec,
                    term_end_sec,
                    info.duration,
                    args.old_chunk_sec,
                )
                old_start_sample = int(round(old_start_sec * args.sample_rate))
                context_start_sample, read_frames = clamp_wiki_context_start(
                    old_start_sample,
                    int(info.frames),
                    old_chunk_samples=old_chunk_samples,
                    new_chunk_samples=new_chunk_samples,
                )
                context_start_sec = context_start_sample / args.sample_rate
                context_end_sec = context_start_sec + args.new_chunk_sec
                wav_path = str(
                    wiki_root
                    / source.source_tag
                    / source.shard_tag
                    / safe_wiki_wav_name(utter_id, chunk_idx)
                )
                ensure_audio_chunk(
                    opus_path=source.source_wav,
                    output_path=wav_path,
                    start_sample=context_start_sample,
                    read_frames=read_frames,
                    new_chunk_samples=new_chunk_samples,
                    sample_rate=args.sample_rate,
                    overwrite=args.overwrite_audio,
                    dry_run=args.dry_run,
                )
            except Exception as exc:
                stats["wiki_expand_failures"] += 1
                if stats["wiki_expand_failures"] <= 20:
                    print(
                        f"[WARN] wiki expand failed utter_id={utter_id} line={line_no}: {exc}",
                        flush=True,
                    )
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_unexpanded_wiki_rows"] += 1
                continue

            expanded_text = build_wiki_window_text(
                words,
                context_start_sec,
                context_end_sec,
                utterance,
            )
            if not expanded_text:
                expanded_text = str(row.get("chunk_src_text") or "")

            out_row = dict(row)
            out_row["chunk_idx"] = chunk_idx
            out_row["chunk_audio_path"] = wav_path
            out_row["chunk_src_text"] = expanded_text
            out_row["mfa_term_start_in_chunk"] = round(term_start_sec - context_start_sec, 4)
            out_row["mfa_term_end_in_chunk"] = round(term_end_sec - context_start_sec, 4)
            out_row["mfa_term_duration"] = round(max(0.0, term_end_sec - term_start_sec), 4)
            out_row["chunk_duration_sec"] = round(args.new_chunk_sec, 4)
            out_row["source_chunk_audio_path_1p92"] = row.get("chunk_audio_path", "")
            out_row["source_chunk_idx_1p92"] = chunk_idx
            out_row["source_tts_wav_path"] = os.path.realpath(source.source_wav)
            out_row["source_textgrid_path"] = source.textgrid
            out_row["context_start_sample"] = context_start_sample
            out_row["context_read_frames"] = read_frames
            out_row["context_build"] = "wiki_synth_mfa_expand_3p84"
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            stats["written_wiki_rows"] += 1
            if stats["written_wiki_rows"] % PROGRESS_EVERY == 0:
                print(
                    f"[PASS2-WIKI] rows={stats['written_wiki_rows']:,} "
                    f"unexpanded={stats['written_unexpanded_wiki_rows']:,}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--audio-output-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--wiki-audio-output-dir",
        default="",
        help=(
            "Directory for recut wiki_synth 3.84s WAVs. "
            "Default: <audio-output-dir>/wiki_synth."
        ),
    )
    parser.add_argument("--stats-json", default="")
    parser.add_argument("--sqlite-index", default=GS_SQLITE_INDEX)
    parser.add_argument("--gs-textgrid-dir", default=GS_TEXTGRID_DIR)
    parser.add_argument("--old-chunk-sec", type=float, default=OLD_CHUNK_SEC)
    parser.add_argument("--new-chunk-sec", type=float, default=NEW_CHUNK_SEC)
    parser.add_argument("--stride-sec", type=float, default=STRIDE_SEC)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument(
        "--include-mode",
        choices=["overlap", "contained"],
        default="overlap",
        help="overlap is conservative for false-negative masking.",
    )
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--max-gs-groups", type=int, default=0)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Stable utter_id-hash shard count. 1 disables sharding.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard id in [0, num_shards). All rows for one utter_id stay together.",
    )
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--write-empty-groups",
        action="store_true",
        help="Write a no-term row for expanded GigaSpeech groups with no selected terms.",
    )
    parser.add_argument(
        "--copy-unexpandable-gs",
        action="store_true",
        default=True,
        help="Copy GigaSpeech rows that lack enough MFA metadata unchanged.",
    )
    parser.add_argument(
        "--no-copy-unexpandable-gs",
        dest="copy_unexpandable_gs",
        action="store_false",
    )
    parser.add_argument(
        "--no-copy-wiki",
        action="store_true",
        help="Do not expand or append wiki_synth rows from the input JSONL.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)
    if args.new_chunk_sec <= args.old_chunk_sec:
        raise ValueError("--new-chunk-sec must be larger than --old-chunk-sec")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")

    lookup = GigaSpeechLookup(args.sqlite_index, args.gs_textgrid_dir, args.sample_rate)
    (
        events_by_utter,
        max_duration_by_utter,
        groups,
        group_representative,
        fallback_gs_rows,
        stats,
    ) = (
        collect_gigaspeech_events(args, lookup)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fout:
        write_expanded_gigaspeech(
            args,
            lookup,
            events_by_utter,
            max_duration_by_utter,
            groups,
            group_representative,
            fallback_gs_rows,
            fout,
            stats,
        )
        write_expanded_wiki_rows(args, fout, stats)

    os.replace(tmp_path, output_path)
    stats["written_total_rows"] = (
        stats["written_gigaspeech_rows"]
        + stats["written_empty_gigaspeech_rows"]
        + stats["written_unexpandable_gigaspeech_rows"]
        + stats["written_wiki_rows"]
        + stats["written_unexpanded_wiki_rows"]
    )
    stats_payload = dict(sorted(stats.items()))
    stats_payload.update(
        {
            "input": args.input,
            "output": args.output,
            "audio_output_dir": args.audio_output_dir,
            "wiki_audio_output_dir": args.wiki_audio_output_dir
            or os.path.join(args.audio_output_dir, "wiki_synth"),
            "old_chunk_sec": args.old_chunk_sec,
            "new_chunk_sec": args.new_chunk_sec,
            "stride_sec": args.stride_sec,
            "include_mode": args.include_mode,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "dry_run": args.dry_run,
            "write_empty_groups": args.write_empty_groups,
        }
    )

    stats_json = args.stats_json or args.output.replace(".jsonl", "_stats.json")
    with open(stats_json, "w", encoding="utf-8") as fout:
        json.dump(stats_payload, fout, indent=2, ensure_ascii=False, sort_keys=True)

    print("[DONE]", flush=True)
    for key, value in stats_payload.items():
        print(f"  {key}: {value}", flush=True)
    print(f"  stats_json: {stats_json}", flush=True)


if __name__ == "__main__":
    main()
