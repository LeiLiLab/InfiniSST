#!/usr/bin/env python3
"""Build GigaSpeech no-term dev chunks from MFA TextGrids.

The output rows intentionally have empty term fields so the existing eval loader
counts them as audio-ok no-term chunks for detection/noise calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000
CHUNK_SEC = 1.92
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)
WORD_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9']+")


def normalize_word(word: str) -> str:
    word = word.strip().lower().replace("\u2019", "'")
    if word.endswith("'s"):
        word = word[:-2]
    return WORD_NORMALIZE_PATTERN.sub("", word)


def normalize_term_tuple(text: str, max_tokens: int) -> Tuple[str, ...]:
    tokens = tuple(t for t in (normalize_word(w) for w in text.split()) if t)
    if not tokens or len(tokens) > max_tokens:
        return ()
    return tokens


def parse_audio_field(audio_field: str) -> Tuple[str, int, int]:
    parts = audio_field.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid audio field: {audio_field}")
    path = parts[0]
    start = int(parts[1])
    length = int(parts[2])
    return path, start, length


def iter_manifest_rows(paths: Sequence[Path]) -> Iterator[Dict[str, str]]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                yield row


def parse_textgrid_words(path: Path) -> List[Tuple[float, float, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    tier_name_idx = None
    for i, line in enumerate(lines):
        if line == '"words"':
            tier_name_idx = i
            break
    if tier_name_idx is None:
        raise ValueError(f'No "words" tier in {path}')
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


def load_forbidden_terms(
    jsonl_paths: Sequence[Path],
    glossary_paths: Sequence[Path],
    max_tokens: int,
    max_terms: int,
) -> Set[Tuple[str, ...]]:
    forbidden: Set[Tuple[str, ...]] = set()

    def add(text: str) -> None:
        if max_terms > 0 and len(forbidden) >= max_terms:
            return
        tup = normalize_term_tuple(text, max_tokens=max_tokens)
        if tup:
            forbidden.add(tup)

    for path in jsonl_paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if max_terms > 0 and len(forbidden) >= max_terms:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                add(str(obj.get("term_key") or obj.get("term") or obj.get("term_text") or ""))
    for path in glossary_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            values: Iterable[object] = data.keys()
        else:
            values = data
        for item in values:
            if max_terms > 0 and len(forbidden) >= max_terms:
                break
            add(str(item))
    return forbidden


def window_has_forbidden_term(
    intervals: Sequence[Tuple[float, float, str]],
    win_start: float,
    win_end: float,
    forbidden: Set[Tuple[str, ...]],
    max_tokens: int,
) -> bool:
    tokens = [
        normalize_word(word)
        for start, end, word in intervals
        if normalize_word(word) and ((start + end) * 0.5) >= win_start and ((start + end) * 0.5) < win_end
    ]
    if not tokens:
        return True
    for n in range(1, min(max_tokens, len(tokens)) + 1):
        for i in range(0, len(tokens) - n + 1):
            if tuple(tokens[i : i + n]) in forbidden:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_tsv", action="append", required=True)
    parser.add_argument("--textgrid_dir", required=True)
    parser.add_argument("--forbidden_jsonl", action="append", default=[])
    parser.add_argument("--forbidden_glossary", action="append", default=[])
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_audio_dir", required=True)
    parser.add_argument("--base_dev_jsonl", default="")
    parser.add_argument("--combined_output_jsonl", default="")
    parser.add_argument("--target_chunks", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_scan_rows", type=int, default=500000)
    parser.add_argument("--max_forbidden_tokens", type=int, default=5)
    parser.add_argument("--max_forbidden_terms", type=int, default=1500000)
    parser.add_argument("--min_words_in_window", type=int, default=2)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    manifest_paths = [Path(p) for p in args.manifest_tsv]
    textgrid_dir = Path(args.textgrid_dir)
    output_jsonl = Path(args.output_jsonl)
    output_audio_dir = Path(args.output_audio_dir)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    forbidden = load_forbidden_terms(
        [Path(p) for p in args.forbidden_jsonl],
        [Path(p) for p in args.forbidden_glossary],
        max_tokens=args.max_forbidden_tokens,
        max_terms=args.max_forbidden_terms,
    )
    print(f"[SETUP] forbidden_terms={len(forbidden)}")

    rows: List[Dict[str, object]] = []
    scanned = 0
    for row in iter_manifest_rows(manifest_paths):
        if len(rows) >= args.target_chunks:
            break
        scanned += 1
        if args.max_scan_rows > 0 and scanned > args.max_scan_rows:
            break
        seg_id = row.get("id", "")
        audio_field = row.get("audio", "")
        if not seg_id or not audio_field:
            continue
        tg_path = textgrid_dir / f"{seg_id}.TextGrid"
        if not tg_path.exists():
            continue
        try:
            audio_path, abs_start, length = parse_audio_field(audio_field)
            intervals = parse_textgrid_words(tg_path)
        except Exception:
            continue
        if length < CHUNK_SAMPLES:
            continue
        max_offset = length - CHUNK_SAMPLES
        candidates = [
            0,
            max_offset,
            rng.randint(0, max_offset) if max_offset > 0 else 0,
            rng.randint(0, max_offset) if max_offset > 0 else 0,
        ]
        picked = None
        for rel_offset in candidates:
            win_start = rel_offset / SAMPLE_RATE
            win_end = win_start + CHUNK_SEC
            n_words = sum(
                1
                for start, end, word in intervals
                if normalize_word(word)
                and ((start + end) * 0.5) >= win_start
                and ((start + end) * 0.5) < win_end
            )
            if n_words < args.min_words_in_window:
                continue
            if window_has_forbidden_term(
                intervals, win_start, win_end, forbidden, args.max_forbidden_tokens
            ):
                continue
            picked = rel_offset
            break
        if picked is None:
            continue
        try:
            audio, sr = sf.read(
                audio_path,
                start=abs_start + picked,
                frames=CHUNK_SAMPLES,
                dtype="float32",
            )
        except Exception:
            continue
        if sr != SAMPLE_RATE or len(audio) < CHUNK_SAMPLES:
            continue
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        wav_name = f"{seg_id}_noterm_{picked}.wav"
        wav_path = output_audio_dir / wav_name
        sf.write(wav_path, np.asarray(audio, dtype=np.float32), SAMPLE_RATE)
        rows.append(
            {
                "term": "",
                "term_key": "",
                "chunk_src_text": "",
                "utter_id": f"{seg_id}_noterm",
                "chunk_idx": len(rows),
                "chunk_audio_path": str(wav_path),
                "audio_type": "gigaspeech_noterm",
                "source_seg_id": seg_id,
                "source_audio": audio_path,
                "source_start_sample": abs_start + picked,
            }
        )
        if len(rows) % 250 == 0:
            print(f"[PROGRESS] rows={len(rows)} scanned={scanned}")

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.base_dev_jsonl and args.combined_output_jsonl:
        combined_output = Path(args.combined_output_jsonl)
        combined_output.parent.mkdir(parents=True, exist_ok=True)
        with combined_output.open("w", encoding="utf-8") as out_f:
            with Path(args.base_dev_jsonl).open("r", encoding="utf-8", errors="replace") as base_f:
                for line in base_f:
                    if line.strip():
                        out_f.write(line if line.endswith("\n") else line + "\n")
            for row in rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[DONE] combined_jsonl={combined_output}")
    print(f"[DONE] wrote rows={len(rows)} scanned={scanned} jsonl={output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
