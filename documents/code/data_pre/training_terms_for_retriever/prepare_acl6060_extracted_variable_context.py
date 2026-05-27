#!/usr/bin/env python3
"""Prepare ACL6060 extracted-paper eval JSONL with variable context lengths.

The source ACL6060 eval set is regenerated from the full WAVs and cached MFA
TextGrids.  For each 1.92s/0.96s base chunk, this script chooses one of the
requested longer context windows, writes all glossary terms fully contained in
that selected window, and writes a no-term row when no term is contained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from documents.code.data_pre.paper_extracted.prepare_acl6060_extracted_paper_glossary import (  # noqa: E402
    ACL_RAW_TEXT_PATH,
    ACL_WAV_DIR,
    ACL_XML_PATH,
    BASE_WORK_DIR,
    GLOSSARY_BY_PAPER_DIR,
    GLOSSARY_PAPER_IDS,
    MFA_TEXTGRID_SUBDIR,
    OUTPUT_JSONL_NAME,
    OUTPUT_TTS_JSONL_NAME,
    SAMPLE_RATE,
    TTS_TERM_TO_PATH,
    build_talks_from_raw_text,
    find_textgrid,
    load_glossary_by_paper,
    load_raw_text,
    map_terms_to_timestamps,
    parse_short_textgrid,
    parse_xml,
    save_jsonl_with_tts,
)


OLD_CHUNK_SEC = 1.92
STRIDE_SEC = 0.96
DEFAULT_DURATION_SECS = (2.88, 3.84, 4.80, 5.76)


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


def load_tts_map(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as fin:
        raw = json.load(fin)
    out: Dict[str, str] = {}
    for term, audio_path in raw.items():
        key = str(term).strip().lower()
        if key and key not in out:
            out[key] = str(audio_path)
    return out


def build_window_text(words, start_sec: float, end_sec: float) -> str:
    return " ".join(w.word for w in words if w.start < end_sec and w.end > start_sec)


def contained_terms(term_spans, glossary: Set[str], start_sec: float, end_sec: float) -> Set[str]:
    terms: Set[str] = set()
    for ts in term_spans:
        if ts.start >= start_sec and ts.end <= end_sec:
            term = ts.term.strip().lower()
            if term in glossary:
                terms.add(term)
    return terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76",
    )
    parser.add_argument("--output-jsonl-name", default=OUTPUT_JSONL_NAME)
    parser.add_argument("--output-tts-jsonl-name", default=OUTPUT_TTS_JSONL_NAME)
    parser.add_argument(
        "--chunk-audio-dir",
        default="/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_varctx2p88_3p84_4p80_5p76/audio_chunks",
    )
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
    parser.add_argument("--overwrite-audio", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    durations = parse_duration_secs(args.duration_secs)
    duration_tags = [duration_tag(d) for d in durations]

    for name, path in [
        ("ACL_WAV_DIR", ACL_WAV_DIR),
        ("ACL_XML_PATH", ACL_XML_PATH),
        ("ACL_RAW_TEXT_PATH", ACL_RAW_TEXT_PATH),
        ("GLOSSARY_BY_PAPER_DIR", GLOSSARY_BY_PAPER_DIR),
        ("TTS_TERM_TO_PATH", TTS_TERM_TO_PATH),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    mfa_output_dir = os.path.join(BASE_WORK_DIR, MFA_TEXTGRID_SUBDIR)
    if not os.path.isdir(mfa_output_dir):
        raise FileNotFoundError(f"MFA TextGrids not found: {mfa_output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.chunk_audio_dir, exist_ok=True)

    glossary = load_glossary_by_paper(GLOSSARY_BY_PAPER_DIR, GLOSSARY_PAPER_IDS)
    tts_map = load_tts_map(TTS_TERM_TO_PATH)
    xml_talks = parse_xml(ACL_XML_PATH)
    raw_text = load_raw_text(ACL_RAW_TEXT_PATH)
    talks = build_talks_from_raw_text(xml_talks, raw_text, glossary, ACL_WAV_DIR)

    row_counts: Counter = Counter()
    stats: Counter = Counter()
    all_chunks = []
    build_tag = "acl6060_extracted_mfa_varctx_" + "_".join(duration_tags)

    for talk in talks:
        tg_path = find_textgrid(mfa_output_dir, talk.docid)
        mfa_words = parse_short_textgrid(tg_path)
        term_spans = map_terms_to_timestamps(talk, mfa_words)
        info = sf.info(talk.wav_path)
        if info.samplerate != SAMPLE_RATE:
            raise ValueError(f"Expected SR={SAMPLE_RATE}, got {info.samplerate}: {talk.wav_path}")
        wav_duration = float(info.duration)
        base_chunks = max(1, int(math.ceil((wav_duration - args.old_chunk_sec) / args.stride_sec)) + 1)
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
                terms_by_duration[dur] = contained_terms(term_spans, glossary, start_sec, end_sec)
                start_by_duration[dur] = start_sec
            n_rows_by_duration = {dur: max(1, len(terms)) for dur, terms in terms_by_duration.items()}
            chosen = choose_duration(
                row_counts=row_counts,
                duration_order=durations,
                assignment=args.duration_assignment,
                stable_key=f"{talk.docid}\t{chunk_idx}",
                n_rows_by_duration=n_rows_by_duration,
            )
            chosen_tag = duration_tag(chosen)
            start_sec = start_by_duration[chosen]
            end_sec = start_sec + chosen
            terms = terms_by_duration[chosen]
            audio_path = os.path.join(
                args.chunk_audio_dir,
                chosen_tag,
                f"{talk.docid}_ctx{chosen_tag}_chunk_{chunk_idx}.wav",
            )
            context_start_sample, read_frames = write_chunk_audio(
                talk.wav_path,
                audio_path,
                start_sec,
                chosen,
                overwrite=args.overwrite_audio,
            )
            src_text = build_window_text(mfa_words, start_sec, end_sec)
            if terms:
                stats["chunks_with_terms"] += 1
            else:
                stats["chunks_without_terms"] += 1
            stats["chunks_total"] += 1
            stats[f"chunks_dur_{chosen_tag}"] += 1

            if terms:
                for term in sorted(terms):
                    matching = [
                        ts for ts in term_spans
                        if ts.term.strip().lower() == term and ts.start >= start_sec and ts.end <= end_sec
                    ]
                    rel_start = min((ts.start for ts in matching), default=start_sec) - start_sec
                    rel_end = max((ts.end for ts in matching), default=start_sec) - start_sec
                    row_counts[chosen_tag] += 1
                    stats["written_term_rows"] += 1
                    stats[f"written_rows_dur_{chosen_tag}"] += 1
                    all_chunks.append({
                        "term": term,
                        "term_key": term,
                        "chunk_src_text": src_text,
                        "utter_id": talk.docid,
                        "chunk_idx": chunk_idx,
                        "chunk_audio_path": audio_path,
                        "mfa_term_start_in_chunk": round(max(0.0, rel_start), 4),
                        "mfa_term_end_in_chunk": round(min(chosen, rel_end), 4),
                        "mfa_term_duration": round(max(0.0, rel_end - rel_start), 4),
                        "chunk_duration_sec": round(chosen, 4),
                        "context_duration_sec": round(chosen, 4),
                        "context_duration_tag": chosen_tag,
                        "source_chunk_idx_1p92": chunk_idx,
                        "context_start_sample": context_start_sample,
                        "context_read_frames": read_frames,
                        "context_reused_source_audio": False,
                        "context_build": build_tag,
                    })
            else:
                row_counts[chosen_tag] += 1
                stats["written_empty_rows"] += 1
                stats[f"written_rows_dur_{chosen_tag}"] += 1
                all_chunks.append({
                    "term": "",
                    "term_key": "",
                    "chunk_src_text": src_text,
                    "utter_id": talk.docid,
                    "chunk_idx": chunk_idx,
                    "chunk_audio_path": audio_path,
                    "mfa_term_start_in_chunk": None,
                    "mfa_term_end_in_chunk": None,
                    "mfa_term_duration": None,
                    "chunk_duration_sec": round(chosen, 4),
                    "context_duration_sec": round(chosen, 4),
                    "context_duration_tag": chosen_tag,
                    "source_chunk_idx_1p92": chunk_idx,
                    "context_start_sample": context_start_sample,
                    "context_read_frames": read_frames,
                    "context_reused_source_audio": False,
                    "context_build": build_tag,
                })

    output_jsonl = os.path.join(args.output_dir, args.output_jsonl_name)
    output_tts_jsonl = os.path.join(args.output_dir, args.output_tts_jsonl_name)
    tmp_jsonl = output_jsonl + ".tmp"
    tmp_tts = output_tts_jsonl + ".tmp"
    with open(tmp_jsonl, "w", encoding="utf-8") as fout, open(tmp_tts, "w", encoding="utf-8") as fout_tts:
        for row in all_chunks:
            line = json.dumps(row, ensure_ascii=False)
            fout.write(line + "\n")
            row_tts = dict(row)
            row_tts["tts_audio_path"] = tts_map.get(str(row.get("term_key") or "").strip().lower(), "")
            fout_tts.write(json.dumps(row_tts, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, output_jsonl)
    os.replace(tmp_tts, output_tts_jsonl)

    stats["written_total_rows"] = len(all_chunks)
    payload = dict(sorted(stats.items()))
    payload.update({
        "input": "acl6060_full_wavs_cached_mfa",
        "output": output_jsonl,
        "output_tts": output_tts_jsonl,
        "audio_output_dir": args.chunk_audio_dir,
        "duration_secs": durations,
        "duration_tags": duration_tags,
        "old_chunk_sec": args.old_chunk_sec,
        "stride_sec": args.stride_sec,
        "duration_assignment": args.duration_assignment,
        "context_build": build_tag,
    })
    for tag in duration_tags:
        payload[f"duration_row_count_{tag}"] = row_counts[tag]

    stats_json = args.stats_json or output_jsonl.replace(".jsonl", "_stats.json")
    with open(stats_json, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[ACL-VARCTX] output={output_jsonl}")
    print(f"[ACL-VARCTX] tts_output={output_tts_jsonl}")
    print(f"[ACL-VARCTX] stats={stats_json}")
    print(f"[ACL-VARCTX] written_total_rows={payload['written_total_rows']}")
    for tag in duration_tags:
        print(f"[ACL-VARCTX] duration_row_count_{tag}={row_counts[tag]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
