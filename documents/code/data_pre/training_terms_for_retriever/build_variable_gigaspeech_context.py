#!/usr/bin/env python3
"""Build a variable-duration retriever JSONL from MFA-aligned training rows.

The output mixes 0.96s, 1.92s, 2.88s, and 3.84s speech chunks.  For every
generated speech window, all known MFA term events that overlap the window are
written as positive rows, so newly audible terms are not left as hard-negative
false negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import soundfile as sf

from expand_gigaspeech_context_3p84 import (
    DEFAULT_INPUT,
    GigaSpeechLookup,
    TermEvent,
    WIKI_SYNTH_PREFIX,
    build_wiki_window_text,
    build_window_text,
    collect_gigaspeech_events,
    ensure_audio_chunk,
    infer_wiki_source,
    locate_wiki_term_span,
    parse_float,
    parse_int,
    parse_textgrid_words,
    read_text,
    recover_old_wiki_chunk_start_sec,
    row_selected_for_shard,
    select_events_for_window,
)


SAMPLE_RATE = 16000
OLD_CHUNK_SEC = 1.92
STRIDE_SEC = 0.96
DEFAULT_DURATION_SECS = (0.96, 1.92, 2.88, 3.84)
PROGRESS_EVERY = 100_000

DEFAULT_OUTPUT = (
    "/mnt/gemini/home/jiaxuanluo/"
    "term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx"
    "0p96_1p92_2p88_3p84.jsonl"
)
DEFAULT_AUDIO_DIR = (
    "/mnt/gemini/data1/jiaxuanluo/"
    "term_train_audio_chunks_gsv2full_gsdedup_varctx0p96_1p92_2p88_3p84"
)


def duration_tag(sec: float) -> str:
    return f"{sec:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def context_build_tag(prefix: str, durations: Iterable[float]) -> str:
    return f"{prefix}_mfa_varctx_" + "_".join(duration_tag(d) for d in durations)


def parse_duration_secs(value: str) -> List[float]:
    durations = [float(v) for v in value.replace(",", " ").split() if v.strip()]
    if not durations:
        raise ValueError("--duration-secs must contain at least one duration")
    if any(d <= 0 for d in durations):
        raise ValueError(f"All durations must be positive: {durations}")
    rounded = []
    for dur in durations:
        r = round(dur, 4)
        if r not in rounded:
            rounded.append(r)
    return rounded


def stable_u64(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def safe_var_wav_name(utter_id: str, chunk_idx: int, dur_sec: float, *, wiki: bool = False) -> str:
    safe_utter = re.sub(r"[^A-Za-z0-9_.-]+", "_", utter_id)
    suffix = "_clean" if wiki else ""
    return f"{safe_utter}_ctx{duration_tag(dur_sec)}_chunk_{chunk_idx}{suffix}.wav"


def clamp_centered_context_start(
    old_start_sample: int,
    segment_start_sample: int,
    segment_end_sample: int,
    *,
    old_chunk_samples: int,
    new_chunk_samples: int,
) -> Tuple[int, int]:
    """Center a new window on the old 1.92s window, clamped to the source span."""

    desired_start = old_start_sample + (old_chunk_samples - new_chunk_samples) // 2
    segment_len = max(0, segment_end_sample - segment_start_sample)
    if segment_len >= new_chunk_samples:
        max_start = segment_end_sample - new_chunk_samples
        return min(max(desired_start, segment_start_sample), max_start), new_chunk_samples
    return segment_start_sample, segment_len


def clamp_context_start_covering_span(
    preferred_start_sample: int,
    segment_start_sample: int,
    segment_end_sample: int,
    *,
    span_start_sec: float,
    span_end_sec: float,
    sample_rate: int,
    new_chunk_samples: int,
) -> Tuple[int, int]:
    """Keep the preferred window unless it misses the positive term span."""

    segment_len = max(0, segment_end_sample - segment_start_sample)
    if segment_len < new_chunk_samples:
        return segment_start_sample, segment_len

    min_start = segment_start_sample
    max_start = segment_end_sample - new_chunk_samples
    desired_start = min(max(preferred_start_sample, min_start), max_start)

    span_start_sample = int(math.floor(span_start_sec * sample_rate))
    span_end_sample = int(math.ceil(span_end_sec * sample_rate))
    if span_end_sample <= desired_start or span_start_sample >= desired_start + new_chunk_samples:
        span_len = max(0, span_end_sample - span_start_sample)
        if span_len <= new_chunk_samples:
            latest_start_covering_span = span_start_sample
            earliest_start_covering_span = span_end_sample - new_chunk_samples
            desired_start = min(
                max(desired_start, earliest_start_covering_span),
                latest_start_covering_span,
            )
        else:
            desired_start = (span_start_sample + span_end_sample - new_chunk_samples) // 2

    return min(max(desired_start, min_start), max_start), new_chunk_samples


def dedupe_events(events: Iterable[TermEvent]) -> List[TermEvent]:
    seen = set()
    out: List[TermEvent] = []
    for ev in events:
        key = (ev.term_key, round(ev.abs_start_sec, 4), round(ev.abs_end_sec, 4))
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def clipped_rel_span(
    abs_start_sec: float,
    abs_end_sec: float,
    context_start_sec: float,
    duration_sec: float,
) -> Optional[Tuple[float, float]]:
    rel_start = max(0.0, abs_start_sec - context_start_sec)
    rel_end = min(duration_sec, abs_end_sec - context_start_sec)
    if rel_end <= rel_start:
        return None
    return rel_start, rel_end


def choose_duration(
    *,
    candidates: Dict[float, List[TermEvent]],
    row_counts: Counter,
    duration_order: List[float],
    assignment: str,
    stable_key: str,
) -> Optional[float]:
    if not candidates:
        return None

    preferred_idx = stable_u64(stable_key) % len(duration_order)
    rotated = duration_order[preferred_idx:] + duration_order[:preferred_idx]
    rank = {dur: idx for idx, dur in enumerate(rotated)}

    if assignment == "hash_group":
        for dur in rotated:
            if dur in candidates:
                return dur
        return None

    def score(dur: float) -> Tuple[int, int, int]:
        tag = duration_tag(dur)
        n_rows = len(candidates[dur])
        return (row_counts[tag] + n_rows, row_counts[tag], rank[dur])

    return min(candidates, key=score)


def choose_wiki_duration(
    *,
    row_counts: Counter,
    duration_order: List[float],
    assignment: str,
    stable_key: str,
) -> float:
    preferred_idx = stable_u64(stable_key) % len(duration_order)
    rotated = duration_order[preferred_idx:] + duration_order[:preferred_idx]
    rank = {dur: idx for idx, dur in enumerate(rotated)}
    if assignment == "hash_group":
        return rotated[0]
    return min(duration_order, key=lambda d: (row_counts[duration_tag(d)] + 1, row_counts[duration_tag(d)], rank[d]))


def write_variable_gigaspeech(
    args: argparse.Namespace,
    lookup: GigaSpeechLookup,
    events_by_utter: Dict[str, List[TermEvent]],
    max_duration_by_utter: Dict[str, float],
    groups: Dict[Tuple[str, int], Any],
    group_representative: Dict[Tuple[str, int], Dict[str, Any]],
    fallback_gs_rows: List[Dict[str, Any]],
    fout,
    stats: Counter,
    duration_row_counts: Counter,
) -> None:
    audio_dir = Path(args.audio_output_dir)
    if not args.dry_run:
        audio_dir.mkdir(parents=True, exist_ok=True)

    old_chunk_samples = int(round(args.old_chunk_sec * args.sample_rate))
    stride_samples = int(round(args.stride_sec * args.sample_rate))
    durations = args.duration_secs
    gs_build = context_build_tag("gigaspeech", durations)

    words_cache: Dict[str, List[Tuple[float, float, str]]] = {}
    starts_cache: Dict[str, List[float]] = {}

    group_items = sorted(
        groups.items(),
        key=lambda kv: (stable_u64(f"{kv[0][0]}\t{kv[0][1]}"), kv[0][0], kv[0][1]),
    )

    for group_no, ((utter_id, chunk_idx), align) in enumerate(group_items, start=1):
        if args.max_gs_groups and group_no > args.max_gs_groups:
            stats["gigaspeech_groups_skipped_by_limit"] += len(group_items) - args.max_gs_groups
            break

        if utter_id not in words_cache:
            words_cache[utter_id] = lookup.words_for_align(align)
            starts_cache[utter_id] = [
                ev.abs_start_sec for ev in events_by_utter.get(utter_id, [])
            ]
        current_words = words_cache[utter_id]
        current_starts = starts_cache[utter_id]
        events = events_by_utter.get(utter_id, [])
        old_start_sample = align.start_sample + chunk_idx * stride_samples

        candidates: Dict[float, List[TermEvent]] = {}
        context_meta: Dict[float, Tuple[int, int, float, float]] = {}
        for dur_sec in durations:
            new_chunk_samples = int(round(dur_sec * args.sample_rate))
            context_start_sample, read_frames = clamp_centered_context_start(
                old_start_sample,
                align.start_sample,
                align.end_sample,
                old_chunk_samples=old_chunk_samples,
                new_chunk_samples=new_chunk_samples,
            )
            context_start_sec = context_start_sample / args.sample_rate
            context_end_sec = context_start_sec + dur_sec
            selected = dedupe_events(
                select_events_for_window(
                    events,
                    current_starts,
                    max_duration_by_utter.get(utter_id, 0.0),
                    context_start_sec,
                    context_end_sec,
                    args.include_mode,
                )
            )
            selected = [
                ev for ev in selected
                if clipped_rel_span(ev.abs_start_sec, ev.abs_end_sec, context_start_sec, dur_sec)
                is not None
            ]
            if selected or args.write_empty_groups:
                candidates[dur_sec] = selected
                context_meta[dur_sec] = (
                    context_start_sample,
                    read_frames,
                    context_start_sec,
                    context_end_sec,
                )

        chosen_dur = choose_duration(
            candidates=candidates,
            row_counts=duration_row_counts,
            duration_order=durations,
            assignment=args.duration_assignment,
            stable_key=f"{utter_id}\t{chunk_idx}",
        )
        if chosen_dur is None:
            stats["gigaspeech_groups_without_selected_terms"] += 1
            continue

        selected = candidates[chosen_dur]
        chosen_tag = duration_tag(chosen_dur)
        context_start_sample, read_frames, context_start_sec, context_end_sec = context_meta[chosen_dur]
        new_chunk_samples = int(round(chosen_dur * args.sample_rate))
        rep = group_representative.get((utter_id, chunk_idx), {})
        reuse_old_audio = (
            args.reuse_old_audio_for_1p92
            and abs(chosen_dur - args.old_chunk_sec) < 1e-6
            and os.path.isfile(str(rep.get("chunk_audio_path") or ""))
        )
        if reuse_old_audio:
            wav_path = str(rep.get("chunk_audio_path"))
            stats["gigaspeech_reused_old_audio_rows"] += max(1, len(selected))
        else:
            wav_path = str(audio_dir / duration_tag(chosen_dur) / safe_var_wav_name(utter_id, chunk_idx, chosen_dur))
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
                        f"[WARN] audio write failed utter_id={utter_id} "
                        f"chunk_idx={chunk_idx} dur={chosen_dur}: {exc}",
                        flush=True,
                    )
                continue

        expanded_text = build_window_text(current_words, context_start_sec, context_end_sec)
        if not expanded_text and selected:
            expanded_text = str(selected[0].row.get("chunk_src_text") or "")
        if not expanded_text:
            expanded_text = str(rep.get("chunk_src_text") or "")

        stats["written_gigaspeech_groups"] += 1
        stats[f"written_gigaspeech_groups_dur_{chosen_tag}"] += 1
        stats["max_terms_per_expanded_group"] = max(
            stats["max_terms_per_expanded_group"],
            len(selected),
        )
        if len(selected) > 1:
            stats["multi_term_expanded_groups"] += 1

        if not selected:
            out_row = dict(rep)
            out_row["term"] = ""
            out_row["term_key"] = ""
            out_row["chunk_idx"] = chunk_idx
            out_row["chunk_audio_path"] = wav_path
            out_row["chunk_src_text"] = expanded_text
            out_row["mfa_term_start_in_chunk"] = None
            out_row["mfa_term_end_in_chunk"] = None
            out_row["mfa_term_duration"] = None
            out_row["chunk_duration_sec"] = round(chosen_dur, 4)
            out_row["context_duration_sec"] = round(chosen_dur, 4)
            out_row["context_duration_tag"] = chosen_tag
            out_row["source_chunk_idx_1p92"] = chunk_idx
            out_row["context_start_sample"] = context_start_sample
            out_row["context_read_frames"] = read_frames
            out_row["context_reused_source_audio"] = reuse_old_audio
            out_row["context_build"] = gs_build
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            stats["written_empty_gigaspeech_rows"] += 1
            stats[f"written_rows_dur_{chosen_tag}"] += 1
            duration_row_counts[chosen_tag] += 1
            continue

        for ev in selected:
            span = clipped_rel_span(
                ev.abs_start_sec,
                ev.abs_end_sec,
                context_start_sec,
                chosen_dur,
            )
            if span is None:
                stats["gigaspeech_selected_span_empty_after_clip"] += 1
                continue
            rel_start, rel_end = span
            out_row = dict(ev.row)
            out_row["chunk_idx"] = chunk_idx
            out_row["chunk_audio_path"] = wav_path
            out_row["chunk_src_text"] = expanded_text
            out_row["mfa_term_start_in_chunk"] = round(rel_start, 4)
            out_row["mfa_term_end_in_chunk"] = round(rel_end, 4)
            out_row["mfa_term_duration"] = round(rel_end - rel_start, 4)
            out_row["mfa_term_full_duration"] = round(ev.duration_sec, 4)
            out_row["chunk_duration_sec"] = round(chosen_dur, 4)
            out_row["context_duration_sec"] = round(chosen_dur, 4)
            out_row["context_duration_tag"] = chosen_tag
            out_row["source_chunk_idx_1p92"] = ev.source_chunk_idx
            out_row["context_start_sample"] = context_start_sample
            out_row["context_read_frames"] = read_frames
            out_row["context_reused_source_audio"] = reuse_old_audio
            out_row["context_build"] = gs_build
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            stats["written_gigaspeech_rows"] += 1
            stats[f"written_gigaspeech_rows_dur_{chosen_tag}"] += 1
            stats[f"written_rows_dur_{chosen_tag}"] += 1
            duration_row_counts[chosen_tag] += 1
            if ev.source_chunk_idx != chunk_idx:
                stats["written_new_context_positive_rows"] += 1
                stats[f"written_new_context_positive_rows_dur_{chosen_tag}"] += 1

        if stats["written_gigaspeech_groups"] % PROGRESS_EVERY == 0:
            counts = " ".join(
                f"{duration_tag(d)}={duration_row_counts[duration_tag(d)]:,}"
                for d in durations
            )
            print(
                "[PASS2-GS] "
                f"groups={stats['written_gigaspeech_groups']:,} "
                f"rows={stats['written_gigaspeech_rows']:,} "
                f"dur_rows=({counts})",
                flush=True,
            )

    for row in fallback_gs_rows:
        fail_tag = duration_tag(args.old_chunk_sec)
        out_row = dict(row)
        out_row["chunk_duration_sec"] = round(args.old_chunk_sec, 4)
        out_row["context_duration_sec"] = round(args.old_chunk_sec, 4)
        out_row["context_duration_tag"] = fail_tag
        out_row["context_build"] = f"{gs_build}_fallback_1p92"
        out_row["context_reused_source_audio"] = True
        fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        stats["written_unexpandable_gigaspeech_rows"] += 1
        stats[f"written_rows_dur_{fail_tag}"] += 1
        duration_row_counts[fail_tag] += 1


def write_variable_wiki_rows(
    args: argparse.Namespace,
    fout,
    stats: Counter,
    duration_row_counts: Counter,
) -> None:
    if args.no_copy_wiki:
        return

    wiki_root = Path(args.wiki_audio_output_dir or os.path.join(args.audio_output_dir, "wiki_synth"))
    old_chunk_samples = int(round(args.old_chunk_sec * args.sample_rate))
    durations = args.duration_secs
    wiki_build = context_build_tag("wiki_synth", durations)
    source_missing_cache: Dict[Tuple[str, str, str], Optional[str]] = {}
    source_bundle_cache: Dict[
        Tuple[str, str, str],
        Tuple[float, int, List[Tuple[float, float, str]], str],
    ] = {}
    term_span_cache: Dict[Tuple[Tuple[str, str, str], str], Optional[Tuple[float, float]]] = {}

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
            chosen_dur = choose_wiki_duration(
                row_counts=duration_row_counts,
                duration_order=durations,
                assignment=args.duration_assignment,
                stable_key=f"{utter_id}\t{chunk_idx}\t{line_no}",
            )
            chosen_tag = duration_tag(chosen_dur)

            reuse_old_audio = (
                args.reuse_old_audio_for_1p92
                and abs(chosen_dur - args.old_chunk_sec) < 1e-6
                and os.path.isfile(str(row.get("chunk_audio_path") or ""))
            )
            if reuse_old_audio:
                out_row = dict(row)
                out_row["chunk_duration_sec"] = round(chosen_dur, 4)
                out_row["context_duration_sec"] = round(chosen_dur, 4)
                out_row["context_duration_tag"] = chosen_tag
                out_row["source_chunk_audio_path_1p92"] = row.get("chunk_audio_path", "")
                out_row["source_chunk_idx_1p92"] = chunk_idx
                out_row["context_reused_source_audio"] = True
                out_row["context_build"] = wiki_build
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                stats["written_wiki_rows"] += 1
                stats[f"written_wiki_rows_dur_{chosen_tag}"] += 1
                stats[f"written_rows_dur_{chosen_tag}"] += 1
                duration_row_counts[chosen_tag] += 1
                continue

            source = infer_wiki_source(row)
            if source is None:
                if args.wiki_expand_failure_policy == "error":
                    raise RuntimeError(f"wiki source inference failed: utter_id={utter_id} line={line_no}")
                if args.wiki_expand_failure_policy == "drop":
                    stats["dropped_wiki_rows"] += 1
                    stats["dropped_wiki_rows_source_infer_failed"] += 1
                    continue
                fail_tag = duration_tag(args.old_chunk_sec)
                stats["wiki_source_infer_failures"] += 1
                row["context_expand_failure"] = "source_infer_failed"
                row["chunk_duration_sec"] = round(args.old_chunk_sec, 4)
                row["context_duration_sec"] = round(args.old_chunk_sec, 4)
                row["context_duration_tag"] = fail_tag
                row["context_build"] = f"{wiki_build}_fallback_1p92"
                row["context_reused_source_audio"] = True
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_unexpanded_wiki_rows"] += 1
                stats[f"written_rows_dur_{fail_tag}"] += 1
                duration_row_counts[fail_tag] += 1
                continue

            source_key = (source.source_wav, source.lab, source.textgrid)
            if source_key in source_missing_cache:
                missing = source_missing_cache[source_key]
            else:
                missing = None
                for label, path in (
                    ("source_wav", source.source_wav),
                    ("lab", source.lab),
                    ("textgrid", source.textgrid),
                ):
                    if not os.path.isfile(path):
                        missing = label
                        break
                source_missing_cache[source_key] = missing
            if missing is not None:
                if args.wiki_expand_failure_policy == "error":
                    raise FileNotFoundError(f"wiki {missing} missing: utter_id={utter_id} line={line_no}")
                if args.wiki_expand_failure_policy == "drop":
                    stats["dropped_wiki_rows"] += 1
                    stats[f"dropped_wiki_rows_{missing}_missing"] += 1
                    continue
                fail_tag = duration_tag(args.old_chunk_sec)
                stats[f"wiki_{missing}_missing"] += 1
                row["context_expand_failure"] = f"{missing}_missing"
                row["chunk_duration_sec"] = round(args.old_chunk_sec, 4)
                row["context_duration_sec"] = round(args.old_chunk_sec, 4)
                row["context_duration_tag"] = fail_tag
                row["context_build"] = f"{wiki_build}_fallback_1p92"
                row["context_reused_source_audio"] = True
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_unexpanded_wiki_rows"] += 1
                stats[f"written_rows_dur_{fail_tag}"] += 1
                duration_row_counts[fail_tag] += 1
                continue

            try:
                if source_key in source_bundle_cache:
                    source_duration, source_frames, words, utterance = source_bundle_cache[source_key]
                else:
                    info = sf.info(source.source_wav)
                    if info.samplerate != args.sample_rate:
                        raise ValueError(
                            f"Expected {args.sample_rate}Hz, got {info.samplerate}Hz: "
                            f"{source.source_wav}"
                        )
                    words = parse_textgrid_words(source.textgrid)
                    utterance = read_text(source.lab)
                    source_duration = float(info.duration)
                    source_frames = int(info.frames)
                    source_bundle_cache[source_key] = (
                        source_duration,
                        source_frames,
                        words,
                        utterance,
                    )
                term = str(row.get("term") or "")
                term_cache_key = (source_key, term)
                if term_cache_key in term_span_cache:
                    term_span = term_span_cache[term_cache_key]
                else:
                    term_span = locate_wiki_term_span(words, term, utterance)
                    term_span_cache[term_cache_key] = term_span
                if term_span is None:
                    raise ValueError(f"term not located in TextGrid: {row.get('term')}")
                term_start_sec, term_end_sec = term_span
                old_start_sec = recover_old_wiki_chunk_start_sec(
                    row,
                    term_start_sec,
                    term_end_sec,
                    source_duration,
                    args.old_chunk_sec,
                )
                old_start_sample = int(round(old_start_sec * args.sample_rate))
                new_chunk_samples = int(round(chosen_dur * args.sample_rate))
                preferred_start_sample, _ = clamp_centered_context_start(
                    old_start_sample,
                    0,
                    source_frames,
                    old_chunk_samples=old_chunk_samples,
                    new_chunk_samples=new_chunk_samples,
                )
                context_start_sample, read_frames = clamp_context_start_covering_span(
                    preferred_start_sample,
                    0,
                    source_frames,
                    span_start_sec=term_start_sec,
                    span_end_sec=term_end_sec,
                    sample_rate=args.sample_rate,
                    new_chunk_samples=new_chunk_samples,
                )
                if context_start_sample != preferred_start_sample:
                    stats["wiki_context_start_adjusted_to_cover_term"] += 1
                    stats[f"wiki_context_start_adjusted_to_cover_term_dur_{chosen_tag}"] += 1
                context_start_sec = context_start_sample / args.sample_rate
                context_end_sec = context_start_sec + chosen_dur
                wav_path = str(
                    wiki_root
                    / duration_tag(chosen_dur)
                    / source.source_tag
                    / source.shard_tag
                    / safe_var_wav_name(utter_id, chunk_idx, chosen_dur, wiki=True)
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
                span = clipped_rel_span(
                    term_start_sec,
                    term_end_sec,
                    context_start_sec,
                    chosen_dur,
                )
                if span is None:
                    raise ValueError(f"term outside chosen window after clipping: {row.get('term')}")
                rel_start, rel_end = span
            except Exception as exc:
                if args.wiki_expand_failure_policy == "error":
                    raise RuntimeError(
                        f"wiki expand failed utter_id={utter_id} line={line_no} dur={chosen_dur}: {exc}"
                    ) from exc
                if args.wiki_expand_failure_policy == "drop":
                    stats["wiki_expand_failures"] += 1
                    stats["dropped_wiki_rows"] += 1
                    stats["dropped_wiki_rows_expand_failed"] += 1
                    if stats["wiki_expand_failures"] <= 20:
                        print(
                            f"[DROP] wiki expand failed utter_id={utter_id} "
                            f"line={line_no} dur={chosen_dur}: {exc}",
                            flush=True,
                        )
                    continue
                fail_tag = duration_tag(args.old_chunk_sec)
                stats["wiki_expand_failures"] += 1
                if stats["wiki_expand_failures"] <= 20:
                    print(
                        f"[WARN] wiki expand failed utter_id={utter_id} "
                        f"line={line_no} dur={chosen_dur}: {exc}",
                        flush=True,
                    )
                row["context_expand_failure"] = str(exc)[:200]
                row["chunk_duration_sec"] = round(args.old_chunk_sec, 4)
                row["context_duration_sec"] = round(args.old_chunk_sec, 4)
                row["context_duration_tag"] = fail_tag
                row["context_build"] = f"{wiki_build}_fallback_1p92"
                row["context_reused_source_audio"] = True
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written_unexpanded_wiki_rows"] += 1
                stats[f"written_rows_dur_{fail_tag}"] += 1
                duration_row_counts[fail_tag] += 1
                continue

            expanded_text = build_wiki_window_text(words, context_start_sec, context_end_sec, utterance)
            if not expanded_text:
                expanded_text = str(row.get("chunk_src_text") or "")

            out_row = dict(row)
            out_row["chunk_idx"] = chunk_idx
            out_row["chunk_audio_path"] = wav_path
            out_row["chunk_src_text"] = expanded_text
            out_row["mfa_term_start_in_chunk"] = round(rel_start, 4)
            out_row["mfa_term_end_in_chunk"] = round(rel_end, 4)
            out_row["mfa_term_duration"] = round(rel_end - rel_start, 4)
            out_row["mfa_term_full_duration"] = round(max(0.0, term_end_sec - term_start_sec), 4)
            out_row["chunk_duration_sec"] = round(chosen_dur, 4)
            out_row["context_duration_sec"] = round(chosen_dur, 4)
            out_row["context_duration_tag"] = chosen_tag
            out_row["source_chunk_audio_path_1p92"] = row.get("chunk_audio_path", "")
            out_row["source_chunk_idx_1p92"] = chunk_idx
            out_row["source_tts_wav_path"] = os.path.realpath(source.source_wav)
            out_row["source_textgrid_path"] = source.textgrid
            out_row["context_start_sample"] = context_start_sample
            out_row["context_read_frames"] = read_frames
            out_row["context_reused_source_audio"] = False
            out_row["context_build"] = wiki_build
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            stats["written_wiki_rows"] += 1
            stats[f"written_wiki_rows_dur_{chosen_tag}"] += 1
            stats[f"written_rows_dur_{chosen_tag}"] += 1
            duration_row_counts[chosen_tag] += 1

            if stats["written_wiki_rows"] % PROGRESS_EVERY == 0:
                counts = " ".join(
                    f"{duration_tag(d)}={duration_row_counts[duration_tag(d)]:,}"
                    for d in durations
                )
                print(
                    f"[PASS2-WIKI] rows={stats['written_wiki_rows']:,} "
                    f"unexpanded={stats['written_unexpanded_wiki_rows']:,} "
                    f"dur_rows=({counts})",
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
        help="Directory for recut wiki_synth WAVs. Default: <audio-output-dir>/wiki_synth.",
    )
    parser.add_argument("--stats-json", default="")
    parser.add_argument("--sqlite-index", default=None)
    parser.add_argument("--gs-textgrid-dir", default=None)
    parser.add_argument("--old-chunk-sec", type=float, default=OLD_CHUNK_SEC)
    parser.add_argument("--stride-sec", type=float, default=STRIDE_SEC)
    parser.add_argument(
        "--duration-secs",
        default=" ".join(str(x) for x in DEFAULT_DURATION_SECS),
        help="Space/comma-separated duration buckets in seconds.",
    )
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument(
        "--include-mode",
        choices=["overlap", "contained"],
        default="overlap",
        help="overlap is conservative for false-negative masking.",
    )
    parser.add_argument(
        "--duration-assignment",
        choices=["balance_rows", "hash_group"],
        default="balance_rows",
        help="balance_rows targets equal output-row counts per duration.",
    )
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--max-gs-groups", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-empty-groups", action="store_true")
    parser.add_argument("--copy-unexpandable-gs", action="store_true", default=True)
    parser.add_argument("--no-copy-unexpandable-gs", dest="copy_unexpandable_gs", action="store_false")
    parser.add_argument("--no-copy-wiki", action="store_true")
    parser.add_argument(
        "--wiki-expand-failure-policy",
        choices=["fallback", "drop", "error"],
        default="fallback",
        help=(
            "How to handle wiki_synth rows that cannot be recut. "
            "'fallback' preserves legacy 1.92s fallback rows, 'drop' filters "
            "the affected rows with counters, and 'error' fails fast."
        ),
    )
    parser.add_argument(
        "--reuse-old-audio-for-1p92",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing 1.92s chunk_audio_path instead of recutting.",
    )
    args = parser.parse_args()

    if args.sqlite_index is None:
        from expand_gigaspeech_context_3p84 import GS_SQLITE_INDEX
        args.sqlite_index = GS_SQLITE_INDEX
    if args.gs_textgrid_dir is None:
        from expand_gigaspeech_context_3p84 import GS_TEXTGRID_DIR
        args.gs_textgrid_dir = GS_TEXTGRID_DIR

    args.duration_secs = parse_duration_secs(args.duration_secs)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    lookup = GigaSpeechLookup(args.sqlite_index, args.gs_textgrid_dir, args.sample_rate)
    (
        events_by_utter,
        max_duration_by_utter,
        groups,
        group_representative,
        fallback_gs_rows,
        stats,
    ) = collect_gigaspeech_events(args, lookup)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    duration_row_counts: Counter = Counter()
    with open(tmp_path, "w", encoding="utf-8") as fout:
        write_variable_gigaspeech(
            args,
            lookup,
            events_by_utter,
            max_duration_by_utter,
            groups,
            group_representative,
            fallback_gs_rows,
            fout,
            stats,
            duration_row_counts,
        )
        write_variable_wiki_rows(args, fout, stats, duration_row_counts)

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
            "duration_secs": args.duration_secs,
            "duration_tags": [duration_tag(d) for d in args.duration_secs],
            "stride_sec": args.stride_sec,
            "include_mode": args.include_mode,
            "duration_assignment": args.duration_assignment,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "dry_run": args.dry_run,
            "write_empty_groups": args.write_empty_groups,
            "reuse_old_audio_for_1p92": args.reuse_old_audio_for_1p92,
            "wiki_expand_failure_policy": args.wiki_expand_failure_policy,
        }
    )
    for dur in args.duration_secs:
        tag = duration_tag(dur)
        stats_payload[f"duration_row_count_{tag}"] = duration_row_counts[tag]

    stats_json = args.stats_json or args.output.replace(".jsonl", "_stats.json")
    with open(stats_json, "w", encoding="utf-8") as fout:
        json.dump(stats_payload, fout, indent=2, ensure_ascii=False, sort_keys=True)

    print("[DONE]", flush=True)
    for key, value in stats_payload.items():
        print(f"  {key}: {value}", flush=True)
    print(f"  stats_json: {stats_json}", flush=True)


if __name__ == "__main__":
    main()
