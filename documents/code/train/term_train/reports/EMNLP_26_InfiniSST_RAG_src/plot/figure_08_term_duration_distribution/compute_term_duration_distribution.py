"""
Compute spoken duration distribution for each term using MFA TextGrid word alignments.

This script is designed for very large JSONL inputs:
- It streams line-by-line (no full-file load).
- It caches parsed TextGrids with an LRU to reduce repeated IO.
- It uses reservoir sampling to approximate per-term quantiles without storing all durations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sqlite3
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Tuple


# ======Configuration=====
DEFAULT_TERM_JSONL_PATH = "/mnt/gemini/data1/jiaxuanluo/term_train_dataset_final.jsonl"
DEFAULT_TEXTGRID_DIR = "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"
DEFAULT_SQLITE_INDEX = "outputs/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite"
DEFAULT_OUTPUT_DIR = "outputs/term_duration_distribution"

# Progress logging
PROGRESS_EVERY_N_LINES = 200_000

# Text normalization and matching
WORD_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9']+")
SEARCH_EXPAND_TOKENS = 8

# Sampling and caching
RANDOM_SEED = 0
RESERVOIR_SIZE_OVERALL = 200_000
RESERVOIR_SIZE_PER_TERM = 2048
TEXTGRID_LRU_MAX_UTTERANCES = 8192
ALIGN_LRU_MAX_UTTERANCES = 50_000
OVERLAP_QUERY_LIMIT = 64
UTTERANCE_CANDIDATES_LRU_MAX = 50_000

# Histogram bins (seconds): [0, 0.05, 0.10, ..., 3.00] plus an overflow bin
HIST_BIN_WIDTH_SECONDS = 0.05
HIST_MAX_SECONDS = 3.0


try:
    import orjson  # type: ignore

    def _json_loads(line: str) -> dict:
        return orjson.loads(line)

except Exception:

    def _json_loads(line: str) -> dict:
        return json.loads(line)


@dataclass(frozen=True)
class WordInterval:
    start: float
    end: float
    word: str


@dataclass(frozen=True)
class ParsedTextGrid:
    intervals: List[WordInterval]
    words_norm: List[str]


class ReservoirSampler:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self._capacity = capacity
        self._rng = rng
        self._n_seen = 0
        self._samples: List[float] = []

    def add(self, x: float) -> None:
        self._n_seen += 1
        if len(self._samples) < self._capacity:
            self._samples.append(x)
            return
        j = self._rng.randint(1, self._n_seen)
        if j <= self._capacity:
            self._samples[j - 1] = x

    def samples(self) -> List[float]:
        return list(self._samples)

    def count_seen(self) -> int:
        return self._n_seen


class LruTextGridCache:
    def __init__(self, max_items: int) -> None:
        self._max_items = max_items
        self._cache: Dict[str, ParsedTextGrid] = {}
        self._order: Deque[str] = deque()

    def get(self, key: str) -> Optional[ParsedTextGrid]:
        if key not in self._cache:
            return None
        # Refresh LRU order
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)
        return self._cache[key]

    def put(self, key: str, value: ParsedTextGrid) -> None:
        if key in self._cache:
            self._cache[key] = value
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return

        self._cache[key] = value
        self._order.append(key)
        while len(self._order) > self._max_items:
            old = self._order.popleft()
            self._cache.pop(old, None)


def normalize_word(w: str) -> str:
    w = w.strip().lower()
    w = WORD_NORMALIZE_PATTERN.sub("", w)
    return w


def tokenize_text(s: str) -> List[str]:
    raw = s.strip().lower()
    raw = raw.replace("\u2019", "'")
    tokens = [normalize_word(t) for t in raw.split()]
    return [t for t in tokens if t]


def utter_id_to_textgrid_path(textgrid_dir: Path, utter_id: str) -> Path:
    prefix, idx = utter_id.rsplit("_", 1)
    if not idx.isdigit():
        raise ValueError(f"Unsupported utter_id format: {utter_id}")
    return textgrid_dir / f"{prefix}_S{int(idx):07d}.TextGrid"


def manifest_id_to_textgrid_path(textgrid_dir: Path, manifest_id: str) -> Path:
    return textgrid_dir / f"{manifest_id}.TextGrid"


def parse_audio_field(audio_field: str) -> Tuple[str, int, int]:
    parts = audio_field.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid audio field: {audio_field}")
    opus = parts[0]
    start = int(parts[1])
    length = int(parts[2])
    return opus, start, start + length


class LruDict:
    def __init__(self, max_items: int) -> None:
        self._max_items = max_items
        self._d: Dict[str, Tuple[str, int, int]] = {}
        self._order: Deque[str] = deque()

    def get(self, key: str) -> Optional[Tuple[str, int, int]]:
        if key not in self._d:
            return None
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)
        return self._d[key]

    def put(self, key: str, value: Tuple[str, int, int]) -> None:
        if key in self._d:
            self._d[key] = value
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return
        self._d[key] = value
        self._order.append(key)
        while len(self._order) > self._max_items:
            old = self._order.popleft()
            self._d.pop(old, None)


class LruListCache:
    def __init__(self, max_items: int) -> None:
        self._max_items = max_items
        self._d: Dict[str, List[str]] = {}
        self._order: Deque[str] = deque()

    def get(self, key: str) -> Optional[List[str]]:
        if key not in self._d:
            return None
        try:
            self._order.remove(key)
        except ValueError:
            pass
        self._order.append(key)
        return self._d[key]

    def put(self, key: str, value: List[str]) -> None:
        if key in self._d:
            self._d[key] = value
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return
        self._d[key] = value
        self._order.append(key)
        while len(self._order) > self._max_items:
            old = self._order.popleft()
            self._d.pop(old, None)


def parse_textgrid_words(textgrid_path: Path) -> List[WordInterval]:
    # This parser matches the simplified TextGrid format like:
    # "IntervalTier"
    # "words"
    # xmin
    # xmax
    # n_intervals
    # start
    # end
    # "word"
    lines = textgrid_path.read_text(encoding="utf-8", errors="replace").splitlines()

    tier_name_idx = None
    for i, line in enumerate(lines):
        if line.strip() == '"words"':
            tier_name_idx = i
            break
    if tier_name_idx is None:
        raise ValueError(f'No "words" tier found in {textgrid_path}')

    # After "words": xmin, xmax, n_intervals
    try:
        n_intervals = int(lines[tier_name_idx + 3].strip())
    except Exception as e:
        raise ValueError(f"Failed to parse words tier header in {textgrid_path}") from e

    intervals: List[WordInterval] = []
    cursor = tier_name_idx + 4
    for _ in range(n_intervals):
        start = float(lines[cursor].strip())
        end = float(lines[cursor + 1].strip())
        word = lines[cursor + 2].strip()
        if word.startswith('"') and word.endswith('"') and len(word) >= 2:
            word = word[1:-1]
        intervals.append(WordInterval(start=start, end=end, word=word))
        cursor += 3
    return intervals


def parse_textgrid_words_with_norm(textgrid_path: Path) -> ParsedTextGrid:
    intervals = parse_textgrid_words(textgrid_path)
    words_norm = [normalize_word(w.word) for w in intervals]
    return ParsedTextGrid(intervals=intervals, words_norm=words_norm)


def find_all_subseq(haystack: List[str], needle: List[str]) -> List[int]:
    if not needle or len(needle) > len(haystack):
        return []
    hits: List[int] = []
    last_start = len(haystack) - len(needle)
    for i in range(last_start + 1):
        if haystack[i : i + len(needle)] == needle:
            hits.append(i)
    return hits


def compute_hist_bins() -> List[float]:
    bins: List[float] = [0.0]
    x = 0.0
    while x < HIST_MAX_SECONDS:
        x = round(x + HIST_BIN_WIDTH_SECONDS, 10)
        bins.append(x)
    return bins


def hist_bin_index(duration_s: float, bin_edges: List[float]) -> int:
    if duration_s < 0:
        return 0
    if duration_s >= bin_edges[-1]:
        return len(bin_edges)  # overflow bin
    i = int(duration_s // HIST_BIN_WIDTH_SECONDS) + 1
    return max(1, min(i, len(bin_edges) - 1))


def quantile_from_sorted(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def safe_float_stats(durations: List[float]) -> dict:
    if not durations:
        return {
            "count": 0,
            "mean": float("nan"),
            "stdev": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    durations_sorted = sorted(durations)
    mean = sum(durations) / len(durations)
    stdev = statistics.pstdev(durations) if len(durations) >= 2 else 0.0
    return {
        "count": len(durations),
        "mean": mean,
        "stdev": stdev,
        "min": durations_sorted[0],
        "max": durations_sorted[-1],
        "p50": quantile_from_sorted(durations_sorted, 0.50),
        "p90": quantile_from_sorted(durations_sorted, 0.90),
        "p95": quantile_from_sorted(durations_sorted, 0.95),
        "p99": quantile_from_sorted(durations_sorted, 0.99),
    }


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield _json_loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute term duration distributions from MFA TextGrids.")
    parser.add_argument("--term-jsonl", default=DEFAULT_TERM_JSONL_PATH, help="Path to term_train_dataset_final.jsonl")
    parser.add_argument("--textgrid-dir", default=DEFAULT_TEXTGRID_DIR, help="Directory containing MFA TextGrid files")
    parser.add_argument("--sqlite-index", default=DEFAULT_SQLITE_INDEX, help="SQLite index built by build_gigaspeech_mfa_sqlite_index.py")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory under the repo")
    args = parser.parse_args()

    term_jsonl_path = Path(args.term_jsonl)
    textgrid_dir = Path(args.textgrid_dir)
    sqlite_index = Path(args.sqlite_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sqlite_index.exists():
        raise FileNotFoundError(f"SQLite index not found: {sqlite_index}")

    con = sqlite3.connect(str(sqlite_index))
    con.execute("PRAGMA read_uncommitted=1;")
    cur_align = con.cursor()
    cur_manifest = con.cursor()

    rng = random.Random(RANDOM_SEED)
    overall_sampler = ReservoirSampler(RESERVOIR_SIZE_OVERALL, rng)
    per_term_sampler: Dict[str, ReservoirSampler] = {}

    cache = LruTextGridCache(TEXTGRID_LRU_MAX_UTTERANCES)
    align_cache = LruDict(ALIGN_LRU_MAX_UTTERANCES)
    candidates_cache = LruListCache(UTTERANCE_CANDIDATES_LRU_MAX)
    bin_edges = compute_hist_bins()
    hist_counts = [0 for _ in range(len(bin_edges) + 1)]  # include overflow

    matched = 0
    not_found = 0
    parse_errors = 0
    align_missing = 0
    manifest_overlap_missing = 0
    textgrid_missing = 0
    total = 0

    per_term_count = Counter()
    per_term_sum = defaultdict(float)
    per_term_min = defaultdict(lambda: float("inf"))
    per_term_max = defaultdict(lambda: float("-inf"))

    for ex in iter_jsonl(term_jsonl_path):
        total += 1
        if total % PROGRESS_EVERY_N_LINES == 0:
            rate = 0.0 if total == 0 else matched / total
            print(f"progress lines={total} matched={matched} rate={rate:.4f} not_found={not_found} missing_tg={textgrid_missing} parse_errors={parse_errors}")

        try:
            term = str(ex.get("term", "")).strip()
            term_key = str(ex.get("term_key", term)).strip() or term
            utter_id = str(ex.get("utter_id", "")).strip()
            chunk_text = str(ex.get("chunk_src_text", "")).strip()
            chunk_idx = int(ex.get("chunk_idx", 0))
        except Exception:
            parse_errors += 1
            continue

        if not term_key or not utter_id:
            parse_errors += 1
            continue

        align_info = align_cache.get(utter_id)
        if align_info is None:
            cur_align.execute("SELECT opus, start, end FROM align_segments WHERE align_id = ?", (utter_id,))
            row = cur_align.fetchone()
            if row is None:
                align_missing += 1
                continue
            align_info = (str(row[0]), int(row[1]), int(row[2]))
            align_cache.put(utter_id, align_info)
        opus, a_start, a_end = align_info

        # Find manifest segments overlapping with the align segment time range (cached by utter_id).
        candidates_seg_ids = candidates_cache.get(utter_id)
        candidates_meta: Optional[List[Tuple[str, int, int]]] = None
        if candidates_seg_ids is None:
            cur_manifest.execute(
                """
                SELECT seg_id, start, end
                FROM manifest_segments
                WHERE opus = ?
                  AND start < ?
                  AND end > ?
                ORDER BY start
                LIMIT ?
                """,
                (opus, a_end, a_start, OVERLAP_QUERY_LIMIT),
            )
            candidates_meta = [(str(r[0]), int(r[1]), int(r[2])) for r in cur_manifest.fetchall()]
            if not candidates_meta:
                manifest_overlap_missing += 1
                continue
            # Prefer segments with the largest overlap.
            candidates_meta.sort(key=lambda x: (min(x[2], a_end) - max(x[1], a_start)), reverse=True)
            candidates_seg_ids = [c[0] for c in candidates_meta]
            candidates_cache.put(utter_id, candidates_seg_ids)

        chunk_tokens = tokenize_text(chunk_text)
        term_tokens = tokenize_text(term)

        if not term_tokens:
            parse_errors += 1
            continue

        duration_s = None
        for seg_id in candidates_seg_ids:
            tg_path = manifest_id_to_textgrid_path(textgrid_dir, seg_id)
            if not tg_path.exists():
                continue

            parsed = cache.get(seg_id)
            if parsed is None:
                try:
                    parsed = parse_textgrid_words_with_norm(tg_path)
                except Exception:
                    parse_errors += 1
                    continue
                cache.put(seg_id, parsed)

            intervals = parsed.intervals
            words_norm = parsed.words_norm

            # Prefer finding term within the chunk match (if chunk matches within this manifest segment).
            anchor_start = None
            anchor_end = None
            if chunk_tokens:
                chunk_hits = find_all_subseq(words_norm, chunk_tokens)
                if chunk_hits:
                    pick = min(max(chunk_idx, 0), len(chunk_hits) - 1)
                    anchor_start = chunk_hits[pick]
                    anchor_end = anchor_start + len(chunk_tokens)

            search_ranges: List[Tuple[int, int]] = []
            if anchor_start is not None and anchor_end is not None:
                lo = max(0, anchor_start - SEARCH_EXPAND_TOKENS)
                hi = min(len(words_norm), anchor_end + SEARCH_EXPAND_TOKENS)
                search_ranges.append((lo, hi))
            search_ranges.append((0, len(words_norm)))

            term_hit = None
            for lo, hi in search_ranges:
                hits = find_all_subseq(words_norm[lo:hi], term_tokens)
                if hits:
                    term_hit = lo + hits[0]
                    break
            if term_hit is None:
                continue

            first_idx = term_hit
            last_idx = term_hit + len(term_tokens) - 1
            if last_idx >= len(intervals):
                continue

            d = float(intervals[last_idx].end - intervals[first_idx].start)
            if d >= 0.0 and math.isfinite(d):
                duration_s = d
                break

        if duration_s is None:
            # If no candidate manifest segment contains the term, count as not found.
            not_found += 1
            # If none of the overlapping candidates had a TextGrid file at all, mark missing.
            if all(not manifest_id_to_textgrid_path(textgrid_dir, sid).exists() for sid in candidates_seg_ids):
                textgrid_missing += 1
            continue

        matched += 1
        overall_sampler.add(duration_s)
        hist_counts[hist_bin_index(duration_s, bin_edges)] += 1

        per_term_count[term_key] += 1
        per_term_sum[term_key] += duration_s
        per_term_min[term_key] = min(per_term_min[term_key], duration_s)
        per_term_max[term_key] = max(per_term_max[term_key], duration_s)

        sampler = per_term_sampler.get(term_key)
        if sampler is None:
            sampler = ReservoirSampler(RESERVOIR_SIZE_PER_TERM, rng)
            per_term_sampler[term_key] = sampler
        sampler.add(duration_s)

    overall_stats = safe_float_stats(overall_sampler.samples())
    summary = {
        "inputs": {
            "term_jsonl": str(term_jsonl_path),
            "textgrid_dir": str(textgrid_dir),
            "sqlite_index": str(sqlite_index),
        },
        "counts": {
            "total_lines": total,
            "matched": matched,
            "not_found": not_found,
            "align_missing": align_missing,
            "manifest_overlap_missing": manifest_overlap_missing,
            "textgrid_missing": textgrid_missing,
            "parse_errors": parse_errors,
            "match_rate": 0.0 if total == 0 else matched / total,
        },
        "overall_duration_seconds_sampled_stats": overall_stats,
        "histogram_seconds": {
            "bin_width": HIST_BIN_WIDTH_SECONDS,
            "max_seconds": HIST_MAX_SECONDS,
            "bin_edges": bin_edges,
            "counts": hist_counts,
            "overflow_bin_index": len(bin_edges),
        },
        "notes": {
            "quantiles_are_approximate": True,
            "method": "Reservoir sampling on matched occurrences; exact counts and sums are tracked.",
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Per-term TSV
    per_term_rows: List[Tuple[str, int, float, float, float, float, float, float]] = []
    for term_key, cnt in per_term_count.items():
        mean = per_term_sum[term_key] / cnt if cnt else float("nan")
        s = per_term_sampler[term_key].samples()
        s_sorted = sorted(s)
        p50 = quantile_from_sorted(s_sorted, 0.50) if s_sorted else float("nan")
        p90 = quantile_from_sorted(s_sorted, 0.90) if s_sorted else float("nan")
        p95 = quantile_from_sorted(s_sorted, 0.95) if s_sorted else float("nan")
        per_term_rows.append(
            (
                term_key,
                cnt,
                mean,
                per_term_min[term_key],
                per_term_max[term_key],
                p50,
                p90,
                p95,
            )
        )

    per_term_rows.sort(key=lambda x: (-x[1], x[0]))
    out_tsv = output_dir / "per_term_stats.tsv"
    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("term_key\tcount\tmean_s\tmin_s\tmax_s\tp50_s\tp90_s\tp95_s\n")
        for row in per_term_rows:
            f.write(
                f"{row[0]}\t{row[1]}\t{row[2]:.6f}\t{row[3]:.6f}\t{row[4]:.6f}\t{row[5]:.6f}\t{row[6]:.6f}\t{row[7]:.6f}\n"
            )

    # Also store top terms by count (debug-friendly)
    top_terms_path = output_dir / "top_terms_by_count.txt"
    with top_terms_path.open("w", encoding="utf-8") as f:
        for term_key, cnt in per_term_count.most_common(200):
            f.write(f"{cnt}\t{term_key}\n")

    con.close()
    print(
        f"done output_dir={output_dir} total={total} matched={matched} not_found={not_found} "
        f"align_missing={align_missing} manifest_overlap_missing={manifest_overlap_missing} "
        f"textgrid_missing={textgrid_missing} parse_errors={parse_errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

