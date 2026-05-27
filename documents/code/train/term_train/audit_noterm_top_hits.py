#!/usr/bin/env python3
"""Inspect high-scoring no-term chunks from a calibration NPZ dump."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SAMPLE_RATE = 16000
CHUNK_SEC = 1.92
WORD_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9']+")


def normalize_word(word: str) -> str:
    word = word.strip().lower().replace("\u2019", "'")
    if word.endswith("'s"):
        word = word[:-2]
    return WORD_NORMALIZE_PATTERN.sub("", word)


def tokenize(text: str) -> List[str]:
    return [tok for tok in (normalize_word(w) for w in text.split()) if tok]


def contains_subseq(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    for i in range(0, len(haystack) - len(needle) + 1):
        if tuple(haystack[i : i + len(needle)]) == tuple(needle):
            return True
    return False


def parse_audio_field(audio_field: str) -> Tuple[str, int, int]:
    path, start, length = audio_field.rsplit(":", 2)
    return path, int(start), int(length)


def load_manifest_starts(paths: Iterable[Path]) -> Dict[str, int]:
    starts: Dict[str, int] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seg_id = row.get("id", "")
                audio = row.get("audio", "")
                if not seg_id or not audio or seg_id in starts:
                    continue
                try:
                    _, start, _ = parse_audio_field(audio)
                except Exception:
                    continue
                starts[seg_id] = start
    return starts


def load_needed_manifest_starts(paths: Iterable[Path], needed_ids: set[str]) -> Dict[str, int]:
    starts: Dict[str, int] = {}
    if not needed_ids:
        return starts
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seg_id = row.get("id", "")
                if not seg_id or seg_id not in needed_ids or seg_id in starts:
                    continue
                audio = row.get("audio", "")
                try:
                    _, start, _ = parse_audio_field(audio)
                except Exception:
                    continue
                starts[seg_id] = start
                if len(starts) == len(needed_ids):
                    return starts
    return starts


def parse_textgrid_words(path: Path) -> List[Tuple[float, float, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    tier_name_idx = None
    for i, line in enumerate(lines):
        if line == '"words"':
            tier_name_idx = i
            break
    if tier_name_idx is None:
        return []
    n_intervals = int(lines[tier_name_idx + 3])
    intervals: List[Tuple[float, float, str]] = []
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


def window_text(
    seg_id: str,
    source_start_sample: str,
    chunk_src_text: str,
    textgrid_dir: Path,
    manifest_starts: Dict[str, int],
) -> str:
    if chunk_src_text:
        return chunk_src_text
    if not seg_id or not source_start_sample:
        return ""
    tg_path = textgrid_dir / f"{seg_id}.TextGrid"
    if not tg_path.exists():
        return ""
    try:
        rel_start = (int(source_start_sample) - manifest_starts[seg_id]) / SAMPLE_RATE
    except Exception:
        return ""
    rel_end = rel_start + CHUNK_SEC
    words = []
    for start, end, word in parse_textgrid_words(tg_path):
        mid = (start + end) * 0.5
        norm = normalize_word(word)
        if norm and rel_start <= mid < rel_end:
            words.append(word)
    return " ".join(words)


def arr_str(data: dict, key: str, idx: int) -> str:
    if key not in data:
        return ""
    return str(data[key][idx])


def load_jsonl_metadata(path: str) -> Dict[str, Dict]:
    if not path:
        return {}
    rows: Dict[str, Dict] = {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio_path = str(row.get("chunk_audio_path", "") or "")
            if audio_path:
                rows[audio_path] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--textgrid_dir", required=True)
    parser.add_argument("--manifest_tsv", action="append", default=[])
    parser.add_argument("--dev_jsonl", default="")
    parser.add_argument("--top_n", type=int, default=40)
    parser.add_argument("--out_jsonl", default="")
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=True) as npz:
        data = {key: npz[key] for key in npz.files}
    scores = np.asarray(data["noterm_topk_sim"], dtype=np.float32)
    top_terms = np.asarray(data.get("noterm_topk_terms", []), dtype=object)
    order = np.argsort(-scores[:, 0])[: args.top_n]
    jsonl_rows = load_jsonl_metadata(args.dev_jsonl)
    needed_seg_ids: set[str] = set()
    for idx in order:
        audio_path = arr_str(data, "noterm_chunk_audio_path", int(idx))
        meta = jsonl_rows.get(audio_path, {})
        seg_id = arr_str(data, "noterm_source_seg_id", int(idx)) or str(meta.get("source_seg_id", "") or "")
        if seg_id:
            needed_seg_ids.add(seg_id)
    manifest_starts = load_needed_manifest_starts(
        [Path(p) for p in args.manifest_tsv],
        needed_seg_ids,
    )
    textgrid_dir = Path(args.textgrid_dir)

    rows = []
    for rank, idx in enumerate(order, 1):
        audio_path = arr_str(data, "noterm_chunk_audio_path", int(idx))
        meta = jsonl_rows.get(audio_path, {})
        chunk_src_text = arr_str(data, "noterm_chunk_src_text", int(idx)) or str(meta.get("chunk_src_text", "") or "")
        source_seg_id = arr_str(data, "noterm_source_seg_id", int(idx)) or str(meta.get("source_seg_id", "") or "")
        source_start_sample = (
            arr_str(data, "noterm_source_start_sample", int(idx))
            or str(meta.get("source_start_sample", "") or "")
        )
        terms = [str(x) for x in top_terms[idx].tolist()] if top_terms.size else []
        text = window_text(
            source_seg_id,
            source_start_sample,
            chunk_src_text,
            textgrid_dir,
            manifest_starts,
        )
        text_tokens = tokenize(text)
        term_hits = [
            term for term in terms
            if contains_subseq(text_tokens, tokenize(term))
        ]
        row = {
            "rank": rank,
            "top1_score": float(scores[idx, 0]),
            "top_terms": terms,
            "term_hits_in_window": term_hits,
            "utter_id": arr_str(data, "noterm_utter_id", idx),
            "chunk_audio_path": audio_path,
            "source_seg_id": source_seg_id,
            "source_start_sample": source_start_sample,
            "window_text": text,
        }
        rows.append(row)

    if args.out_jsonl:
        out = Path(args.out_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
