#!/usr/bin/env python3
"""Build MFA-grounded ``gt_terms_by_chunk`` for Speech LLM SFT.

This script is intentionally stricter than the older source-trajectory proxy:

1. source terms come from exact n-gram matches over MFA TextGrid word intervals;
2. a term is assigned to an audio chunk by timestamp overlap with that chunk;
3. the target translation must appear as an exact substring in assistant
   messages from the current audio response through the end of the conversation;
4. output can strip existing term maps to ``term_map:NONE`` so this stage only
   establishes clean GT labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import wave
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SAMPLE_RATE = 16000
WORD_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9']+")


@dataclass(frozen=True)
class WordInterval:
    start: float
    end: float
    word: str
    norm: str


class LruDict:
    def __init__(self, max_items: int):
        self._max_items = max_items
        self._data: Dict[str, object] = {}
        self._order: deque[str] = deque()

    def get(self, key: str):
        return self._data.get(key)

    def put(self, key: str, value) -> None:
        if key not in self._data:
            self._order.append(key)
        self._data[key] = value
        while len(self._order) > self._max_items:
            old = self._order.popleft()
            self._data.pop(old, None)


def normalize_word(word: str) -> str:
    word = str(word or "").strip().lower().replace("\u2019", "'")
    if word.endswith("'s"):
        word = word[:-2]
    return WORD_NORMALIZE_PATTERN.sub("", word)


def tokenize_text(text: str) -> List[str]:
    return [x for x in (normalize_word(w) for w in str(text or "").split()) if x]


def tokenize_text_variants(text: str) -> List[Tuple[str, ...]]:
    raw = str(text or "").strip().lower().replace("\u2019", "'")
    variants: List[Tuple[str, ...]] = []

    def add(tokens: Sequence[str]) -> None:
        item = tuple(x for x in tokens if x)
        if item and item not in variants:
            variants.append(item)

    add(tokenize_text(raw))
    add(normalize_word(w) for w in WORD_NORMALIZE_PATTERN.split(raw))
    return variants


def term_display_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def extract_translation(entry: Mapping[str, Any], lang_code: str) -> str:
    value = entry.get("translation") or entry.get("target_translation") or entry.get(lang_code)
    if value is None and isinstance(entry.get("target_translations"), Mapping):
        value = entry["target_translations"].get(lang_code)
    return str(value or "").strip()


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object {path}:{lineno}")
            yield lineno, obj


def parse_textgrid_words(path: Path) -> List[Tuple[float, float, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f]
    tier_name_idx = None
    for idx, line in enumerate(lines):
        if line == '"words"':
            tier_name_idx = idx
            break
    if tier_name_idx is None:
        raise ValueError(f'No "words" tier in {path}')
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


def wav_duration_sec(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            raise ValueError(f"Invalid wav sample rate for {path}: {rate}")
        return frames / float(rate)


def audio_user_indices(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    return [
        idx for idx, msg in enumerate(messages)
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or "")
    ]


def assistant_future_text(messages: Sequence[Mapping[str, Any]], audio_msg_idx: int) -> str:
    return "\n".join(
        str(messages[idx].get("content") or "")
        for idx in range(audio_msg_idx + 1, len(messages))
        if messages[idx].get("role") == "assistant"
    )


def set_audio_term_map_none(messages: Sequence[Mapping[str, Any]]) -> None:
    for msg in messages:
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or ""):
            msg["content"] = "<audio>\n\nterm_map:NONE"


def load_glossary(
    path: Path,
    *,
    lang_code: str,
    max_words: int,
    min_norm_chars: int,
) -> Tuple[Dict[Tuple[str, ...], List[Dict[str, Any]]], Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        raw_items = [(str(k), v) for k, v in data.items()]
    elif isinstance(data, list):
        raw_items = [(str(i), v) for i, v in enumerate(data)]
    else:
        raise ValueError(f"Unsupported glossary format: {path}")

    by_tokens: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    stats = Counter()
    for key, entry in raw_items:
        stats["raw_entries"] += 1
        if isinstance(entry, str):
            term = key.strip()
            translation = entry.strip()
            raw: Dict[str, Any] = {"term": term, "translation": translation}
        elif isinstance(entry, Mapping):
            term = str(entry.get("term") or entry.get("source") or key).strip()
            translation = extract_translation(entry, lang_code)
            raw = dict(entry)
            raw["term"] = term
            raw["translation"] = translation
        else:
            stats["skipped_bad_entry"] += 1
            continue
        if not term or not translation:
            stats["skipped_missing_term_or_translation"] += 1
            continue
        variants = tokenize_text_variants(term)
        if not variants:
            stats["skipped_no_tokens"] += 1
            continue
        kept_variant = False
        for tokens in variants:
            norm_chars = len("".join(tokens))
            if max_words > 0 and len(tokens) > max_words:
                continue
            if norm_chars < min_norm_chars:
                continue
            item = dict(raw)
            item["token_tuple"] = list(tokens)
            item["token_key"] = " ".join(tokens)
            item["term_key"] = term_display_key(term)
            by_tokens.setdefault(tokens, []).append(item)
            kept_variant = True
        if kept_variant:
            stats["kept_entries"] += 1
        else:
            stats["skipped_length_filter"] += 1
    if not by_tokens:
        raise ValueError(f"No usable glossary terms after filtering: {path}")
    return by_tokens, {
        **dict(stats),
        "unique_token_tuples": len(by_tokens),
        "max_term_words": max(len(k) for k in by_tokens),
    }


class GigaSpeechMFA:
    def __init__(self, sqlite_path: Path, textgrid_dir: Path, *, textgrid_cache_size: int = 20000):
        self._con = sqlite3.connect(str(sqlite_path))
        self._con.execute("PRAGMA read_uncommitted=1")
        self._cur_align = self._con.cursor()
        self._cur_manifest = self._con.cursor()
        self._textgrid_dir = textgrid_dir
        self._align_cache = LruDict(50000)
        self._candidates_cache = LruDict(50000)
        self._textgrid_cache = LruDict(textgrid_cache_size)

    def align_info(self, utter_id: str) -> Optional[Tuple[str, int, int]]:
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

    def candidates(self, utter_id: str, opus: str, start_samples: int, end_samples: int, limit: int) -> List[Tuple[str, float, float]]:
        cached = self._candidates_cache.get(utter_id)
        if cached is not None:
            return cached
        self._cur_manifest.execute(
            """SELECT seg_id, start, end FROM manifest_segments
               WHERE opus = ? AND start < ? AND end > ?
               ORDER BY start LIMIT ?""",
            (opus, end_samples, start_samples, limit),
        )
        rows = [
            (str(seg_id), int(start) / SAMPLE_RATE, int(end) / SAMPLE_RATE)
            for seg_id, start, end in self._cur_manifest.fetchall()
        ]
        self._candidates_cache.put(utter_id, rows)
        return rows

    def textgrid_words(self, seg_id: str) -> Optional[List[Tuple[float, float, str]]]:
        cached = self._textgrid_cache.get(seg_id)
        if cached is not None:
            return cached
        path = self._textgrid_dir / f"{seg_id}.TextGrid"
        if not path.exists():
            return None
        try:
            words = parse_textgrid_words(path)
        except Exception:
            return None
        self._textgrid_cache.put(seg_id, words)
        return words

    def row_words(self, utter_id: str, *, candidate_limit: int) -> Tuple[Optional[Tuple[str, int, int]], List[WordInterval]]:
        info = self.align_info(utter_id)
        if info is None:
            return None, []
        opus, align_start, align_end = info
        out: List[WordInterval] = []
        for seg_id, seg_start_sec, _seg_end_sec in self.candidates(utter_id, opus, align_start, align_end, candidate_limit):
            words = self.textgrid_words(seg_id)
            if words is None:
                continue
            for rel_start, rel_end, raw_word in words:
                norm = normalize_word(raw_word)
                if not norm:
                    continue
                abs_start = seg_start_sec + rel_start
                abs_end = seg_start_sec + rel_end
                if abs_end <= align_start / SAMPLE_RATE or abs_start >= align_end / SAMPLE_RATE:
                    continue
                out.append(WordInterval(abs_start, abs_end, raw_word, norm))
        out.sort(key=lambda x: (x.start, x.end, x.word))
        return info, out


def iter_term_occurrences(
    words: Sequence[WordInterval],
    glossary_by_tokens: Mapping[Tuple[str, ...], List[Dict[str, Any]]],
    *,
    max_term_words: int,
) -> Iterable[Dict[str, Any]]:
    norms = [w.norm for w in words]
    seen = set()
    for start_idx in range(len(norms)):
        max_width = min(max_term_words, len(norms) - start_idx)
        for width in range(1, max_width + 1):
            tokens = tuple(norms[start_idx : start_idx + width])
            entries = glossary_by_tokens.get(tokens)
            if not entries:
                continue
            span_start = words[start_idx].start
            span_end = words[start_idx + width - 1].end
            raw_text = " ".join(w.word for w in words[start_idx : start_idx + width])
            for entry in entries:
                translation = str(entry.get("translation") or "").strip()
                dedupe_key = (start_idx, width, str(entry.get("term_key")), translation)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                yield {
                    "term": str(entry.get("term") or "").strip(),
                    "translation": translation,
                    "token_key": str(entry.get("token_key") or " ".join(tokens)),
                    "term_key": str(entry.get("term_key") or entry.get("term") or ""),
                    "mfa_text": raw_text,
                    "mfa_start": span_start,
                    "mfa_end": span_end,
                }


def term_matches_chunk(term: Mapping[str, Any], chunk_start: float, chunk_end: float, policy: str) -> bool:
    start = float(term["mfa_start"])
    end = float(term["mfa_end"])
    eps = 1e-4
    if policy == "contained":
        return start >= chunk_start - eps and end <= chunk_end + eps
    if policy == "overlap":
        return max(start, chunk_start) < min(end, chunk_end)
    if policy == "end_in_chunk":
        return end > chunk_start + eps and end <= chunk_end + eps
    if policy == "midpoint":
        mid = 0.5 * (start + end)
        return mid >= chunk_start - eps and mid < chunk_end + eps
    raise ValueError(f"Unknown chunk assignment policy: {policy}")


def build(args: argparse.Namespace) -> None:
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    glossary_by_tokens, glossary_stats = load_glossary(
        args.glossary_json,
        lang_code=args.lang_code,
        max_words=args.max_words,
        min_norm_chars=args.min_norm_chars,
    )
    mfa = GigaSpeechMFA(args.sqlite_index, args.textgrid_dir)
    max_term_words = int(glossary_stats["max_term_words"])

    stats: Dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "glossary_json": str(args.glossary_json),
        "sqlite_index": str(args.sqlite_index),
        "textgrid_dir": str(args.textgrid_dir),
        "lang_code": args.lang_code,
        "max_words": args.max_words,
        "min_norm_chars": args.min_norm_chars,
        "chunk_assignment_policy": args.chunk_assignment_policy,
        "target_match_policy": "future_ref",
        "term_map_output_policy": args.term_map_output_policy,
        "glossary": glossary_stats,
        "rows_seen": 0,
        "rows_written": 0,
        "rows_missing_mfa_align": 0,
        "rows_no_mfa_words": 0,
        "rows_audio_gt_mismatch": 0,
        "audio_duration_errors": 0,
        "chunks_total": 0,
        "chunks_with_mfa_gt": 0,
        "mfa_term_occurrences_total": 0,
        "mfa_chunk_term_candidates_total": 0,
        "future_ref_kept_terms_total": 0,
        "future_ref_dropped_terms_total": 0,
        "dropped_rows": 0,
        "drop_reasons": Counter(),
    }
    gt_hist = Counter()
    term_freq = Counter()
    chunk_duration_hist = Counter()
    samples: List[Dict[str, Any]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for lineno, obj in iter_jsonl(args.input_jsonl):
            stats["rows_seen"] += 1
            if args.limit_rows and stats["rows_seen"] > args.limit_rows:
                break
            try:
                messages = obj.get("messages")
                audios = obj.get("audios")
                utter_id = str(obj.get("utter_id") or "").strip()
                if not utter_id:
                    raise ValueError("missing utter_id")
                if not isinstance(messages, list) or not isinstance(audios, list) or not audios:
                    raise ValueError("missing messages or audios")
                audio_idxs = audio_user_indices(messages)
                if len(audio_idxs) != len(audios):
                    stats["rows_audio_gt_mismatch"] += 1
                    raise ValueError(f"audio user messages={len(audio_idxs)} audios={len(audios)}")

                align_info, words = mfa.row_words(utter_id, candidate_limit=args.overlap_query_limit)
                if align_info is None:
                    stats["rows_missing_mfa_align"] += 1
                    raise ValueError(f"missing MFA align row for utter_id={utter_id}")
                if not words:
                    stats["rows_no_mfa_words"] += 1
                    raise ValueError(f"no MFA words for utter_id={utter_id}")
                _opus, align_start_samples, _align_end_samples = align_info
                align_start_sec = align_start_samples / SAMPLE_RATE

                durations = []
                for audio_path in audios:
                    try:
                        durations.append(wav_duration_sec(str(audio_path)))
                    except Exception as exc:
                        stats["audio_duration_errors"] += 1
                        raise ValueError(f"could not read audio duration {audio_path}: {exc}") from exc

                chunk_bounds: List[Tuple[float, float]] = []
                cursor = align_start_sec
                for duration in durations:
                    chunk_bounds.append((cursor, cursor + duration))
                    cursor += duration
                    chunk_duration_hist[round(duration, 3)] += 1

                row_occurrences = list(
                    iter_term_occurrences(words, glossary_by_tokens, max_term_words=max_term_words)
                )
                stats["mfa_term_occurrences_total"] += len(row_occurrences)

                gt_by_chunk: List[List[Dict[str, Any]]] = []
                for chunk_idx, ((chunk_start, chunk_end), msg_idx) in enumerate(zip(chunk_bounds, audio_idxs)):
                    future_text = assistant_future_text(messages, msg_idx)
                    chunk_terms: List[Dict[str, Any]] = []
                    seen_chunk = set()
                    for term in row_occurrences:
                        if not term_matches_chunk(term, chunk_start, chunk_end, args.chunk_assignment_policy):
                            continue
                        stats["mfa_chunk_term_candidates_total"] += 1
                        translation = str(term.get("translation") or "").strip()
                        if not translation or translation not in future_text:
                            stats["future_ref_dropped_terms_total"] += 1
                            continue
                        dedupe_key = (str(term.get("term_key")), translation)
                        if dedupe_key in seen_chunk:
                            continue
                        seen_chunk.add(dedupe_key)
                        item = {
                            "term": str(term.get("term") or ""),
                            args.lang_code: translation,
                            "mfa_start": round(float(term["mfa_start"]) - align_start_sec, 4),
                            "mfa_end": round(float(term["mfa_end"]) - align_start_sec, 4),
                            "mfa_chunk_start": round(chunk_start - align_start_sec, 4),
                            "mfa_chunk_end": round(chunk_end - align_start_sec, 4),
                            "mfa_text": str(term.get("mfa_text") or ""),
                        }
                        chunk_terms.append(item)
                        term_freq[str(term.get("token_key") or term.get("term_key") or term.get("term"))] += 1
                    chunk_terms.sort(key=lambda x: (float(x["mfa_start"]), -len(str(x["term"]).split()), str(x["term"]).casefold()))
                    gt_by_chunk.append(chunk_terms)
                    stats["chunks_total"] += 1
                    stats["future_ref_kept_terms_total"] += len(chunk_terms)
                    if chunk_terms:
                        stats["chunks_with_mfa_gt"] += 1
                    gt_hist[len(chunk_terms)] += 1
                    if len(samples) < args.sample_count and chunk_terms:
                        samples.append({
                            "row_line": lineno,
                            "utter_id": utter_id,
                            "chunk_idx": chunk_idx,
                            "chunk_start": round(chunk_start - align_start_sec, 4),
                            "chunk_end": round(chunk_end - align_start_sec, 4),
                            "future_text_prefix": future_text[:240],
                            "gt_terms": chunk_terms[:30],
                        })

                obj["gt_terms_by_chunk"] = gt_by_chunk
                obj["mfa_glossary_future_ref_gt_policy"] = {
                    "version": "mfa_glossary_future_ref_v1",
                    "source": "GigaSpeech MFA TextGrid word timestamps",
                    "glossary_json": str(args.glossary_json),
                    "sqlite_index": str(args.sqlite_index),
                    "textgrid_dir": str(args.textgrid_dir),
                    "source_match": "MFA normalized word n-gram exact match",
                    "chunk_assignment_policy": args.chunk_assignment_policy,
                    "target_match_policy": "future_ref",
                    "target_match_reference": "assistant messages from current audio response through conversation end",
                    "target_match_exact_substring": True,
                    "term_map_output_policy": args.term_map_output_policy,
                }
                if args.term_map_output_policy == "none":
                    set_audio_term_map_none(messages)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
            except Exception as exc:
                if args.drop_bad_rows:
                    stats["dropped_rows"] += 1
                    stats["drop_reasons"][str(exc).splitlines()[0][:240]] += 1
                    continue
                raise RuntimeError(f"Failed {args.input_jsonl}:{lineno}: {exc}") from exc

    stats["drop_reasons"] = dict(stats["drop_reasons"])
    stats["gt_terms_per_chunk_hist"] = dict(gt_hist.most_common(80))
    stats["chunk_duration_hist"] = {str(k): int(v) for k, v in chunk_duration_hist.most_common(80)}
    stats["top_gt_terms"] = [{"token_key": k, "count": int(v)} for k, v in term_freq.most_common(100)]
    stats["chunks_with_mfa_gt_rate"] = (
        stats["chunks_with_mfa_gt"] / stats["chunks_total"] if stats["chunks_total"] else 0.0
    )
    stats["avg_mfa_gt_terms_per_chunk"] = (
        stats["future_ref_kept_terms_total"] / stats["chunks_total"] if stats["chunks_total"] else 0.0
    )
    stats["future_ref_kept_candidate_rate"] = (
        stats["future_ref_kept_terms_total"] / stats["mfa_chunk_term_candidates_total"]
        if stats["mfa_chunk_term_candidates_total"] else 0.0
    )
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sample_json:
        args.sample_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_jsonl": str(args.output_jsonl),
        "rows_written": stats["rows_written"],
        "chunks_total": stats["chunks_total"],
        "mfa_term_occurrences_total": stats["mfa_term_occurrences_total"],
        "mfa_chunk_term_candidates_total": stats["mfa_chunk_term_candidates_total"],
        "future_ref_kept_terms_total": stats["future_ref_kept_terms_total"],
        "chunks_with_mfa_gt_rate": stats["chunks_with_mfa_gt_rate"],
        "avg_mfa_gt_terms_per_chunk": stats["avg_mfa_gt_terms_per_chunk"],
        "future_ref_kept_candidate_rate": stats["future_ref_kept_candidate_rate"],
    }, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path)
    parser.add_argument("--glossary-json", type=Path, required=True)
    parser.add_argument("--sqlite-index", type=Path, default=Path("/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite"))
    parser.add_argument("--textgrid-dir", type=Path, default=Path("/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"))
    parser.add_argument("--lang-code", default="zh")
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument("--min-norm-chars", type=int, default=2)
    parser.add_argument("--chunk-assignment-policy", choices=["overlap", "contained", "end_in_chunk", "midpoint"], default="overlap")
    parser.add_argument("--term-map-output-policy", choices=["none", "preserve"], default="none")
    parser.add_argument("--overlap-query-limit", type=int, default=128)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--drop-bad-rows", action="store_true")
    args = parser.parse_args()
    for path in [args.input_jsonl, args.glossary_json, args.sqlite_index, args.textgrid_dir]:
        if not path.exists():
            raise FileNotFoundError(path)
    return args


if __name__ == "__main__":
    build(parse_args())
