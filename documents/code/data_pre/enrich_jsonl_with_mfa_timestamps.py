#!/usr/bin/env python3
"""
Enrich the term training JSONL with MFA-derived term positions.

For each sample, locates the TextGrid, finds the term, and adds:
  mfa_term_start_in_chunk: float (seconds, relative to chunk start)
  mfa_term_end_in_chunk:   float (seconds, relative to chunk start)

These chunk-relative positions allow the MFA-supervised MaxSim loss to know
exactly which encoder windows fully cover the term's time range.

Handles two data sources:
  - GigaSpeech: SQLite index -> manifest TextGrid -> absolute position ->
    subtract chunk_abs_start (align_start + chunk_idx * STRIDE)
  - Wiki-synth: TextGrid -> find term abs position + anchor chunk_start
    via chunk_src_text matching -> chunk-relative position
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

# ======Configuration=====
CHUNK_SEC = 1.92
STRIDE_SEC = 0.96
SAMPLE_RATE = 16000
WORD_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9']+")
WIKI_SYNTH_PREFIX = "wiki_synth_"

GS_TEXTGRID_DIR = "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"
GS_SQLITE_INDEX = "/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite"

WIKI_MFA_BASES = [
    "/mnt/aries/data4/jiaxuanluo/MFA/3variant",
    "/mnt/aries/data6/jiaxuanluo/MFA/1third_aries",
    "/mnt/aries/data6/jiaxuanluo/MFA/aries",
]

WIKI_SHARD_SIZE = 149936
WIKI_NUM_SHARDS = 20
PROGRESS_EVERY = 500_000
SEARCH_EXPAND_TOKENS = 8
OVERLAP_QUERY_LIMIT = 64

TEXTGRID_LRU_MAX = 20_000
ALIGN_LRU_MAX = 50_000
CANDIDATES_LRU_MAX = 50_000
# ======Configuration=====


def normalize_word(w: str) -> str:
    w = w.strip().lower()
    w = w.replace("\u2019", "'")
    if w.endswith("'s"):
        w = w[:-2]
    return WORD_NORMALIZE_PATTERN.sub("", w)


def tokenize_text(s: str) -> List[str]:
    raw = s.strip().lower().replace("\u2019", "'")
    return [t for t in (normalize_word(w) for w in raw.split()) if t]


def tokenize_text_variants(s: str) -> List[List[str]]:
    """Tokenize text with conservative variants for MFA/TextGrid mismatch.

    GigaSpeech TextGrids often split hyphenated terms (trade-offs -> trade offs)
    while JSONL term keys keep them as one surface token. Keep the original
    whitespace tokenization first, then add a punctuation-split variant.
    """
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


def parse_textgrid_words(tg_path: str) -> List[Tuple[float, float, str]]:
    with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f.readlines()]
    tier_name_idx = None
    for i, line in enumerate(lines):
        if line == '"words"':
            tier_name_idx = i
            break
    assert tier_name_idx is not None, f'No "words" tier in {tg_path}'
    n_intervals = int(lines[tier_name_idx + 3])
    intervals = []
    cursor = tier_name_idx + 4
    for _ in range(n_intervals):
        start = float(lines[cursor])
        end = float(lines[cursor + 1])
        word = lines[cursor + 2]
        if word.startswith('"') and word.endswith('"') and len(word) >= 2:
            word = word[1:-1]
        intervals.append((start, end, word))
        cursor += 3
    return intervals


def find_all_subseq(haystack: List[str], needle: List[str]) -> List[int]:
    if not needle or len(needle) > len(haystack):
        return []
    hits = []
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            hits.append(i)
    return hits


class LruDict:
    def __init__(self, max_items: int):
        self._max = max_items
        self._d: Dict[str, object] = {}
        self._order: deque = deque()

    def get(self, key: str):
        return self._d.get(key)

    def put(self, key: str, value):
        if key in self._d:
            self._d[key] = value
            return
        self._d[key] = value
        self._order.append(key)
        while len(self._order) > self._max:
            old = self._order.popleft()
            self._d.pop(old, None)


# ---- Wiki-synth ----

def wiki_utter_id_to_paths(utter_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (textgrid_path, lab_path) for a wiki-synth utter_id."""
    numeric_str = utter_id[len(WIKI_SYNTH_PREFIX):]
    assert numeric_str.isdigit(), f"Invalid wiki utter_id: {utter_id}"
    numeric_id = int(numeric_str)
    shard_idx = min(numeric_id // WIKI_SHARD_SIZE, WIKI_NUM_SHARDS - 1)
    utt_tag = f"utt_{numeric_str}"
    tg_name = f"{utt_tag}.TextGrid"
    lab_name = f"{utt_tag}.lab"

    def _check(base: str, s: int) -> Tuple[Optional[str], Optional[str]]:
        shard_dir = os.path.join(base, "work", f"shard_{s:02d}")
        tg = os.path.join(shard_dir, "mfa_output", tg_name)
        lab = os.path.join(shard_dir, "mfa_input", lab_name)
        if os.path.exists(tg):
            return tg, (lab if os.path.exists(lab) else None)
        return None, None

    for base in WIKI_MFA_BASES:
        tg, lab = _check(base, shard_idx)
        if tg:
            return tg, lab
    for base in WIKI_MFA_BASES:
        for s in range(WIKI_NUM_SHARDS):
            tg, lab = _check(base, s)
            if tg:
                return tg, lab
    return None, None


def _read_lab_utterance(lab_path: Optional[str]) -> str:
    if lab_path is None or not os.path.exists(lab_path):
        return ""
    try:
        with open(lab_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _nonempty_words(
    intervals: List[Tuple[float, float, str]],
) -> List[Tuple[int, str]]:
    """Return (original_index, normalized_word) pairs, skipping silence/empty."""
    result = []
    for i, (_, _, w) in enumerate(intervals):
        nw = normalize_word(w)
        if nw:
            result.append((i, nw))
    return result


def find_term_abs_in_textgrid(
    intervals: List[Tuple[float, float, str]], term_key: str,
    utterance: str = "",
) -> Optional[Tuple[float, float]]:
    """Find term's absolute (start, end) in TextGrid.

    Strategy 1: normalized word matching (skipping silence intervals).
    Strategy 2 (fallback): positional matching via utterance word indices,
    handles MFA <unk> for OOV words.
    """
    term_token_variants = tokenize_text_variants(term_key)
    if not term_token_variants:
        return None

    words_with_idx = _nonempty_words(intervals)
    norm_words = [nw for _, nw in words_with_idx]

    for term_tokens in term_token_variants:
        hits = find_all_subseq(norm_words, term_tokens)
        if hits:
            first_orig = words_with_idx[hits[0]][0]
            last_orig = words_with_idx[hits[0] + len(term_tokens) - 1][0]
            return (intervals[first_orig][0], intervals[last_orig][1])

    if not utterance:
        return None

    utt_tokens = tokenize_text(utterance)
    if not utt_tokens or len(utt_tokens) != len(words_with_idx):
        return None

    for term_tokens in term_token_variants:
        utt_hits = find_all_subseq(utt_tokens, term_tokens)
        if not utt_hits:
            continue
        pos = utt_hits[0]
        first_orig = words_with_idx[pos][0]
        last_orig = words_with_idx[pos + len(term_tokens) - 1][0]
        return (intervals[first_orig][0], intervals[last_orig][1])
    return None


def estimate_chunk_start_from_src_text(
    intervals: List[Tuple[float, float, str]], chunk_src_text: str,
) -> Optional[float]:
    """Estimate chunk_start by matching chunk_src_text in TextGrid.

    align_and_cut_wiki_synth.py's get_chunk_text() selects words whose
    midpoints fall in [chunk_start, chunk_end]. We reverse this:
    find the matched words and estimate chunk_start as slightly before
    the first matched word's midpoint.
    """
    chunk_tokens = tokenize_text(chunk_src_text)
    if not chunk_tokens:
        return None

    words_with_idx = []
    for i, (s, e, w) in enumerate(intervals):
        nw = normalize_word(w)
        if nw:
            words_with_idx.append((i, nw))

    norm_words = [nw for _, nw in words_with_idx]
    hits = find_all_subseq(norm_words, chunk_tokens)
    if not hits:
        return None

    first_match = words_with_idx[hits[0]][0]
    last_match = words_with_idx[hits[0] + len(chunk_tokens) - 1][0]

    first_mid = (intervals[first_match][0] + intervals[first_match][1]) / 2
    last_mid = (intervals[last_match][0] + intervals[last_match][1]) / 2

    # chunk_start must be <= first_mid and chunk_end >= last_mid
    # chunk_end = chunk_start + CHUNK_SEC
    # So: chunk_start <= first_mid AND chunk_start + CHUNK_SEC >= last_mid
    # => last_mid - CHUNK_SEC <= chunk_start <= first_mid
    lo = max(0.0, last_mid - CHUNK_SEC)
    hi = first_mid
    if lo > hi:
        return lo
    return (lo + hi) / 2


def process_wiki_sample(sample: Dict) -> Dict:
    """Find term chunk-relative position for wiki-synth samples."""
    utter_id = str(sample.get("utter_id", "")).strip()
    term_key = str(sample.get("term_key", "")).strip()
    chunk_src_text = str(sample.get("chunk_src_text", "")).strip()

    tg_path, lab_path = wiki_utter_id_to_paths(utter_id)
    if tg_path is None or not os.path.exists(tg_path):
        return sample

    try:
        intervals = parse_textgrid_words(tg_path)
    except Exception:
        return sample

    utterance = _read_lab_utterance(lab_path)
    term_abs = find_term_abs_in_textgrid(intervals, term_key, utterance=utterance)
    if term_abs is None:
        return sample

    term_abs_start, term_abs_end = term_abs

    chunk_start = estimate_chunk_start_from_src_text(intervals, chunk_src_text)
    if chunk_start is None:
        sample["mfa_term_duration"] = round(term_abs_end - term_abs_start, 4)
        return sample

    t_start_in_chunk = term_abs_start - chunk_start
    t_end_in_chunk = term_abs_end - chunk_start

    if t_start_in_chunk < -0.1 or t_end_in_chunk > CHUNK_SEC + 0.1:
        sample["mfa_term_duration"] = round(term_abs_end - term_abs_start, 4)
        return sample

    sample["mfa_term_start_in_chunk"] = round(max(0.0, t_start_in_chunk), 4)
    sample["mfa_term_end_in_chunk"] = round(min(CHUNK_SEC, t_end_in_chunk), 4)
    sample["mfa_term_duration"] = round(term_abs_end - term_abs_start, 4)
    return sample


# ---- GigaSpeech ----

class GigaSpeechMFALookup:
    def __init__(self, sqlite_path: str, textgrid_dir: str):
        assert os.path.exists(sqlite_path), f"SQLite index not found: {sqlite_path}"
        self._con = sqlite3.connect(sqlite_path)
        self._con.execute("PRAGMA read_uncommitted=1;")
        self._cur_align = self._con.cursor()
        self._cur_manifest = self._con.cursor()
        self._textgrid_dir = textgrid_dir
        self._align_cache = LruDict(ALIGN_LRU_MAX)
        self._candidates_cache = LruDict(CANDIDATES_LRU_MAX)
        self._tg_cache = LruDict(TEXTGRID_LRU_MAX)

    def _get_align_info(self, utter_id: str) -> Optional[Tuple[str, int, int]]:
        cached = self._align_cache.get(utter_id)
        if cached is not None:
            return cached
        self._cur_align.execute(
            "SELECT opus, start, end FROM align_segments WHERE align_id = ?",
            (utter_id,),
        )
        row = self._cur_align.fetchone()
        if row is None:
            return None
        result = (str(row[0]), int(row[1]), int(row[2]))
        self._align_cache.put(utter_id, result)
        return result

    def _get_candidates_with_meta(
        self, utter_id: str, opus: str, a_start: int, a_end: int,
    ) -> Optional[List[Tuple[str, float, float]]]:
        cached = self._candidates_cache.get(utter_id)
        if cached is not None:
            return cached
        self._cur_manifest.execute(
            """SELECT seg_id, start, end FROM manifest_segments
               WHERE opus = ? AND start < ? AND end > ?
               ORDER BY start LIMIT ?""",
            (opus, a_end, a_start, OVERLAP_QUERY_LIMIT),
        )
        meta = [
            (str(r[0]), int(r[1]) / SAMPLE_RATE, int(r[2]) / SAMPLE_RATE)
            for r in self._cur_manifest.fetchall()
        ]
        if not meta:
            return None
        self._candidates_cache.put(utter_id, meta)
        return meta

    def _get_tg_intervals(self, seg_id: str) -> Optional[List[Tuple[float, float, str]]]:
        cached = self._tg_cache.get(seg_id)
        if cached is not None:
            return cached
        tg_path = os.path.join(self._textgrid_dir, f"{seg_id}.TextGrid")
        if not os.path.exists(tg_path):
            return None
        try:
            intervals = parse_textgrid_words(tg_path)
        except Exception:
            return None
        self._tg_cache.put(seg_id, intervals)
        return intervals

    def find_term_in_chunk(
        self, utter_id: str, term_key: str, chunk_src_text: str, chunk_idx: int,
    ) -> Optional[Tuple[float, float, float]]:
        """Returns (term_start_in_chunk, term_end_in_chunk, duration) or None."""
        align_info = self._get_align_info(utter_id)
        if align_info is None:
            return None
        opus, a_start_samples, a_end_samples = align_info
        a_start_sec = a_start_samples / SAMPLE_RATE

        chunk_abs_start = a_start_sec + chunk_idx * STRIDE_SEC
        chunk_abs_end = chunk_abs_start + CHUNK_SEC

        candidates = self._get_candidates_with_meta(
            utter_id, opus, a_start_samples, a_end_samples,
        )
        if candidates is None:
            return None

        term_token_variants = tokenize_text_variants(term_key)
        chunk_token_variants = tokenize_text_variants(chunk_src_text) if chunk_src_text else []
        if not term_token_variants:
            return None

        for seg_id, m_start_sec, m_end_sec in candidates:
            overlap = max(0, min(m_end_sec, chunk_abs_end) - max(m_start_sec, chunk_abs_start))
            if overlap <= 0:
                continue

            intervals = self._get_tg_intervals(seg_id)
            if intervals is None:
                continue

            words_with_idx = _nonempty_words(intervals)
            norm_words = [nw for _, nw in words_with_idx]

            anchor_start = None
            anchor_end = None
            for chunk_tokens in chunk_token_variants:
                chunk_hits = find_all_subseq(norm_words, chunk_tokens)
                if chunk_hits:
                    pick = min(max(chunk_idx, 0), len(chunk_hits) - 1)
                    anchor_start = chunk_hits[pick]
                    anchor_end = anchor_start + len(chunk_tokens)
                    break

            search_ranges: List[Tuple[int, int]] = []
            if anchor_start is not None and anchor_end is not None:
                lo = max(0, anchor_start - SEARCH_EXPAND_TOKENS)
                hi = min(len(norm_words), anchor_end + SEARCH_EXPAND_TOKENS)
                search_ranges.append((lo, hi))
            search_ranges.append((0, len(norm_words)))

            for lo, hi in search_ranges:
                for term_tokens in term_token_variants:
                    hits = find_all_subseq(norm_words[lo:hi], term_tokens)
                    if hits:
                        first_wi = lo + hits[0]
                        last_wi = first_wi + len(term_tokens) - 1
                        if last_wi >= len(words_with_idx):
                            continue
                        first_orig = words_with_idx[first_wi][0]
                        last_orig = words_with_idx[last_wi][0]
                        tg_term_start = intervals[first_orig][0]
                        tg_term_end = intervals[last_orig][1]
                        term_abs_start = m_start_sec + tg_term_start
                        term_abs_end = m_start_sec + tg_term_end
                        t_in_chunk_start = term_abs_start - chunk_abs_start
                        t_in_chunk_end = term_abs_end - chunk_abs_start
                        duration = term_abs_end - term_abs_start
                        if duration > 0:
                            return (t_in_chunk_start, t_in_chunk_end, duration)
        return None


def process_gigaspeech_sample(sample: Dict, gs_lookup: GigaSpeechMFALookup) -> Dict:
    utter_id = str(sample.get("utter_id", "")).strip()
    term_key = str(sample.get("term_key", "")).strip()
    chunk_src_text = str(sample.get("chunk_src_text", "")).strip()
    chunk_idx = int(sample.get("chunk_idx", 0))

    result = gs_lookup.find_term_in_chunk(utter_id, term_key, chunk_src_text, chunk_idx)
    if result is not None:
        t_start, t_end, duration = result
        sample["mfa_term_start_in_chunk"] = round(t_start, 4)
        sample["mfa_term_end_in_chunk"] = round(t_end, 4)
        sample["mfa_term_duration"] = round(duration, 4)
    return sample


def has_position(sample: Dict) -> bool:
    return sample.get("mfa_term_start_in_chunk") is not None


def has_duration(sample: Dict) -> bool:
    return sample.get("mfa_term_duration") is not None


def clear_mfa_fields(sample: Dict) -> None:
    sample["mfa_term_start_in_chunk"] = None
    sample["mfa_term_end_in_chunk"] = None
    sample["mfa_term_duration"] = None


def get_term_key(sample: Dict) -> str:
    return str(sample.get("term_key", sample.get("term", ""))).strip().lower()


def main():
    parser = argparse.ArgumentParser(description="Enrich training JSONL with MFA term positions")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_lines", type=int, default=0)
    parser.add_argument("--sqlite_index", default=GS_SQLITE_INDEX)
    parser.add_argument("--gs_textgrid_dir", default=GS_TEXTGRID_DIR)
    parser.add_argument(
        "--only-source",
        choices=["all", "gs", "wiki"],
        default="all",
        help="Restrict enrichment to one source; skipped rows are written unchanged.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip rows that already have mfa_term_start_in_chunk.",
    )
    parser.add_argument(
        "--preserve-existing",
        action="store_true",
        help="Do not clear existing MFA fields before processing a row.",
    )
    parser.add_argument(
        "--drop-empty-term",
        action="store_true",
        help="Drop rows in the selected source whose term_key/term is empty.",
    )
    parser.add_argument(
        "--drop-unmatched",
        action="store_true",
        help="Drop selected-source rows that still lack MFA position/duration after processing.",
    )
    args = parser.parse_args()

    assert os.path.exists(args.input), f"Input not found: {args.input}"

    gs_lookup = GigaSpeechMFALookup(args.sqlite_index, args.gs_textgrid_dir)
    stats = defaultdict(int)

    with open(args.input, "r") as in_f, open(args.output, "w") as out_f:
        for i, line in enumerate(in_f):
            if args.max_lines and i >= args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_error"] += 1
                continue

            utter_id = str(sample.get("utter_id", "")).strip()
            is_wiki = utter_id.startswith(WIKI_SYNTH_PREFIX)
            source = "wiki" if is_wiki else "gs"

            should_process = args.only_source in ("all", source)
            if should_process and args.drop_empty_term and not get_term_key(sample):
                stats["dropped_empty_term"] += 1
                stats[f"dropped_empty_term_{source}"] += 1
                continue
            if args.only_missing and has_position(sample):
                should_process = False

            if not should_process:
                if has_position(sample):
                    stats["preserved_position"] += 1
                    stats[f"preserved_pos_{source}"] += 1
                elif has_duration(sample):
                    stats["preserved_duration_only"] += 1
                    stats[f"preserved_dur_{source}"] += 1
                else:
                    stats["preserved_unmatched"] += 1
                    stats[f"preserved_unmatched_{source}"] += 1
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["total"] += 1
                continue

            if not args.preserve_existing:
                clear_mfa_fields(sample)

            if is_wiki:
                sample = process_wiki_sample(sample)
            else:
                sample = process_gigaspeech_sample(sample, gs_lookup)

            row_has_position = has_position(sample)
            row_has_duration = has_duration(sample)

            if row_has_position:
                stats["matched_position"] += 1
                stats[f"matched_pos_{source}"] += 1
            elif row_has_duration:
                stats["matched_duration_only"] += 1
                stats[f"matched_dur_{source}"] += 1
            else:
                stats["unmatched"] += 1
                stats[f"unmatched_{source}"] += 1
                if args.drop_unmatched:
                    stats["dropped_unmatched"] += 1
                    stats[f"dropped_unmatched_{source}"] += 1
                    continue

            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stats["total"] += 1

            if stats["total"] % PROGRESS_EVERY == 0:
                total = stats["total"]
                pos = stats.get("matched_position", 0)
                dur = stats.get("matched_duration_only", 0)
                unm = stats.get("unmatched", 0)
                print(
                    f"[PROGRESS] total={total} "
                    f"position={pos} ({pos/max(total,1):.4f}) "
                    f"duration_only={dur} unmatched={unm} "
                    f"gs_pos={stats.get('matched_pos_gs',0)} "
                    f"wiki_pos={stats.get('matched_pos_wiki',0)}",
                    flush=True,
                )

    print(f"\n[DONE] {args.output}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    total = stats["total"]
    pos = stats.get("matched_position", 0)
    dur = stats.get("matched_duration_only", 0)
    print(f"\n  Position rate: {pos}/{total} = {pos/max(total,1):.4f}")
    print(f"  Duration-only rate: {dur}/{total} = {dur/max(total,1):.4f}")
    print(f"  Total with MFA: {pos+dur}/{total} = {(pos+dur)/max(total,1):.4f}")


if __name__ == "__main__":
    main()
