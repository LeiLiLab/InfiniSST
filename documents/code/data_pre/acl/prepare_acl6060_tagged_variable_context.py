#!/usr/bin/env python3
"""Prepare ACL6060 tagged-glossary eval JSONL with variable context lengths.

This uses the existing ACL6060 MFA TextGrids, removes brackets from the tagged
English transcript for alignment, keeps only terms present in
``glossary_acl6060.json``, and emits the same offline eval JSONL shape used by
the extracted-paper ACL readout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from documents.code.data_pre.tts.prepare_acl6060_dev_dataset import (  # noqa: E402
    ACL_WAV_DIR,
    ACL_XML_PATH,
    BASE_UNIT_SEC,
    MFA_TEXTGRID_SUBDIR,
    SAMPLE_RATE,
    WORK_DIR as BASE_WORK_DIR,
    SegmentInfo,
    TalkInfo,
    _log,
    find_textgrid,
    map_terms_to_timestamps,
    parse_short_textgrid,
    parse_xml,
)
from documents.code.data_pre.training_terms_for_retriever.prepare_acl6060_extracted_variable_context import (  # noqa: E402
    build_window_text,
    choose_duration,
    clamp_centered_start,
    duration_tag,
    parse_duration_secs,
    write_chunk_audio,
)


ACL_TAGGED_TEXT_PATH = (
    "/mnt/data/siqiouyang/datasets/acl6060/dev/text/tagged_terminology/"
    "ACL.6060.dev.tagged.en-xx.en.txt"
)
TAGGED_GLOSSARY_JSON = (
    "/home/jiaxuanluo/InfiniSST/documents/data/data_pre/glossary_acl6060.json"
)
WIKI_ENRICHED_JSON = (
    "/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/"
    "wiki_glossary_nlp_ai_cs_enriched.json"
)

OUTPUT_DIR = (
    "/mnt/gemini/home/jiaxuanluo/"
    "acl6060_dev_offline_eval_tagged_glossary_varctx2p88_3p84_4p80_5p76"
)
CHUNK_AUDIO_DIR = (
    "/mnt/gemini/home/jiaxuanluo/"
    "acl6060_dev_offline_eval_tagged_varctx2p88_3p84_4p80_5p76/audio_chunks"
)
EVAL_GLOSSARY_OUTPUT = (
    "/mnt/gemini/home/jiaxuanluo/eval_glossaries/"
    "acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json"
)
OUTPUT_JSONL_NAME = "acl6060_tagged_dev_dataset.jsonl"

TERM_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
GLOSSARY_MATCH_PUNCT_RE = re.compile(r"[^a-z0-9']+")


def _normalize_glossary_match_word(word: str) -> str:
    word = word.strip().lower().replace("\u2019", "'")
    if word.endswith("'s"):
        word = word[:-2]
    word = GLOSSARY_MATCH_PUNCT_RE.sub("", word)
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("es") and not word.endswith(("ses", "xes")):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def _glossary_match_norm_char_count(text: str) -> int:
    return sum(
        len(tok)
        for tok in (_normalize_glossary_match_word(w) for w in str(text or "").split())
        if tok
    )


def _extract_terms_and_clean(
    tagged_line: str,
    allowed_terms: Set[str],
    min_norm_chars: int,
) -> Tuple[str, List[str], List[Tuple[str, int, int]]]:
    terms: List[str] = []
    spans: List[Tuple[str, int, int]] = []
    clean_parts: List[str] = []
    char_offset = 0
    last_end = 0

    for match in TERM_BRACKET_RE.finditer(tagged_line):
        before = tagged_line[last_end:match.start()]
        clean_parts.append(before)
        char_offset += len(before)

        surface = match.group(1).strip()
        term_key = surface.lower()
        term_start = char_offset
        term_end = char_offset + len(surface)
        clean_parts.append(surface)
        char_offset = term_end

        if (
            term_key in allowed_terms
            and _glossary_match_norm_char_count(term_key) >= int(min_norm_chars)
        ):
            terms.append(term_key)
            spans.append((term_key, term_start, term_end))
        last_end = match.end()

    remaining = tagged_line[last_end:]
    clean_parts.append(remaining)
    return "".join(clean_parts), terms, spans


def load_tagged_glossary(path: str, min_norm_chars: int) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as fin:
        raw = json.load(fin)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict glossary, got {type(raw).__name__}: {path}")

    out: Dict[str, Dict] = {}
    skipped_short = 0
    for key, value in raw.items():
        term_key = str(key or "").strip().lower()
        if not term_key:
            continue
        if _glossary_match_norm_char_count(term_key) < int(min_norm_chars):
            skipped_short += 1
            continue
        entry = value if isinstance(value, dict) else {"term": term_key}
        out[term_key] = entry
    _log(
        f"Tagged glossary: {len(out)} terms kept from {len(raw)} "
        f"(skipped_short={skipped_short}, min_norm_chars={min_norm_chars})"
    )
    return out


def load_tagged_text(
    tagged_path: str,
    allowed_terms: Set[str],
    min_norm_chars: int,
) -> Dict[int, Tuple[str, List[str], List[Tuple[str, int, int]]]]:
    result: Dict[int, Tuple[str, List[str], List[Tuple[str, int, int]]]] = {}
    raw_terms = 0
    kept_terms = 0
    with open(tagged_path, "r", encoding="utf-8") as fin:
        for seg_id, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            raw_terms += len(TERM_BRACKET_RE.findall(line))
            clean, terms, spans = _extract_terms_and_clean(
                line,
                allowed_terms,
                min_norm_chars,
            )
            kept_terms += len(terms)
            result[seg_id] = (clean, terms, spans)
    _log(
        f"Tagged text: {len(result)} segments, raw_bracket_terms={raw_terms}, "
        f"kept_term_occurrences={kept_terms}"
    )
    return result


def build_talks_from_tagged_text(
    xml_talks: List[Tuple[str, int, int]],
    tagged_data: Dict[int, Tuple[str, List[str], List[Tuple[str, int, int]]]],
    wav_dir: str,
) -> List[TalkInfo]:
    talks: List[TalkInfo] = []
    for docid, first_seg, last_seg in xml_talks:
        wav_path = os.path.join(wav_dir, f"{docid}.wav")
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        segments: List[SegmentInfo] = []
        for seg_id in range(first_seg, last_seg + 1):
            if seg_id not in tagged_data:
                raise KeyError(f"Segment {seg_id} not found in tagged text")
            clean_text, terms, spans = tagged_data[seg_id]
            segments.append(
                SegmentInfo(
                    seg_id=seg_id,
                    clean_text=clean_text,
                    terms=terms,
                    term_spans=spans,
                )
            )

        talks.append(TalkInfo(docid=docid, wav_path=wav_path, segments=segments))
        _log(
            f"  Talk {docid}: segs={len(segments)}, "
            f"tagged_terms={sum(len(s.terms) for s in segments)}"
        )
    return talks


def contained_terms(term_spans, glossary: Set[str], start_sec: float, end_sec: float) -> Set[str]:
    terms: Set[str] = set()
    for term_span in term_spans:
        if term_span.start >= start_sec and term_span.end <= end_sec:
            term = term_span.term.strip().lower()
            if term in glossary:
                terms.add(term)
    return terms


def write_eval_glossary(
    *,
    observed_terms: Set[str],
    tagged_glossary: Dict[str, Dict],
    wiki_json: str,
    output_path: str,
    size: int,
    min_norm_chars: int,
) -> Dict[str, int]:
    gt_entries = []
    seen = set()
    for term_key in sorted(observed_terms):
        if _glossary_match_norm_char_count(term_key) < int(min_norm_chars):
            continue
        entry = tagged_glossary.get(term_key, {})
        surface = str(entry.get("term") or term_key).strip() or term_key
        translations = entry.get("target_translations") or {}
        gt_entries.append(
            {
                "term": surface,
                "target_translations": translations,
                "source": "acl_tagged_gt",
            }
        )
        seen.add(surface.strip().lower())
        seen.add(term_key)

    with open(wiki_json, "r", encoding="utf-8") as fin:
        wiki_entries = json.load(fin)
    if not isinstance(wiki_entries, list):
        raise TypeError(f"Expected list wiki glossary: {wiki_json}")

    filler = []
    for item in wiki_entries:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        key = term.lower()
        if not term or key in seen:
            continue
        if _glossary_match_norm_char_count(key) < int(min_norm_chars):
            continue
        translations = item.get("target_translations") or {}
        filler.append(
            {
                "term": term,
                "target_translations": translations,
                "source": "wiki_fill",
            }
        )
        seen.add(key)
        if len(gt_entries) + len(filler) >= int(size):
            break

    if len(gt_entries) > int(size):
        raise ValueError(f"GT terms ({len(gt_entries)}) exceed requested glossary size {size}")
    if len(gt_entries) + len(filler) < int(size):
        raise ValueError(
            f"Not enough filler terms for size={size}: "
            f"gt={len(gt_entries)} filler={len(filler)}"
        )

    output = gt_entries + filler[: int(size) - len(gt_entries)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(output, fout, ensure_ascii=False, indent=2)
    return {
        "eval_glossary_total": len(output),
        "eval_glossary_gt_terms": len(gt_entries),
        "eval_glossary_filler_terms": len(output) - len(gt_entries),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--output-jsonl-name", default=OUTPUT_JSONL_NAME)
    parser.add_argument("--chunk-audio-dir", default=CHUNK_AUDIO_DIR)
    parser.add_argument("--tagged-text", default=ACL_TAGGED_TEXT_PATH)
    parser.add_argument("--tagged-glossary-json", default=TAGGED_GLOSSARY_JSON)
    parser.add_argument("--wiki-glossary-json", default=WIKI_ENRICHED_JSON)
    parser.add_argument("--eval-glossary-output", default=EVAL_GLOSSARY_OUTPUT)
    parser.add_argument("--eval-glossary-size", type=int, default=10000)
    parser.add_argument(
        "--duration-secs",
        default="2.88 3.84 4.80 5.76",
    )
    parser.add_argument("--old-chunk-sec", type=float, default=2 * BASE_UNIT_SEC)
    parser.add_argument("--stride-sec", type=float, default=BASE_UNIT_SEC)
    parser.add_argument(
        "--duration-assignment",
        choices=["balance_rows", "hash_group"],
        default="balance_rows",
    )
    parser.add_argument("--min-norm-chars", type=int, default=2)
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
        ("ACL_TAGGED_TEXT_PATH", args.tagged_text),
        ("TAGGED_GLOSSARY_JSON", args.tagged_glossary_json),
        ("WIKI_GLOSSARY_JSON", args.wiki_glossary_json),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    mfa_output_dir = os.path.join(BASE_WORK_DIR, MFA_TEXTGRID_SUBDIR)
    if not os.path.isdir(mfa_output_dir):
        raise FileNotFoundError(f"MFA TextGrids not found: {mfa_output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.chunk_audio_dir, exist_ok=True)

    tagged_glossary = load_tagged_glossary(args.tagged_glossary_json, args.min_norm_chars)
    xml_talks = parse_xml(ACL_XML_PATH)
    tagged_data = load_tagged_text(
        args.tagged_text,
        set(tagged_glossary),
        args.min_norm_chars,
    )
    talks = build_talks_from_tagged_text(xml_talks, tagged_data, ACL_WAV_DIR)

    row_counts: Counter = Counter()
    stats: Counter = Counter()
    all_rows: List[Dict] = []
    observed_terms: Set[str] = set()
    build_tag = "acl6060_tagged_mfa_varctx_" + "_".join(duration_tags)

    for talk in talks:
        _log(f"Processing talk: {talk.docid}")
        tg_path = find_textgrid(mfa_output_dir, talk.docid)
        mfa_words = parse_short_textgrid(tg_path)
        term_spans = map_terms_to_timestamps(talk, mfa_words)
        info = sf.info(talk.wav_path)
        if info.samplerate != SAMPLE_RATE:
            raise ValueError(f"Expected SR={SAMPLE_RATE}, got {info.samplerate}: {talk.wav_path}")
        wav_duration = float(info.duration)
        base_chunks = max(
            1,
            int(math.ceil((wav_duration - args.old_chunk_sec) / args.stride_sec)) + 1,
        )

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
                terms_by_duration[dur] = contained_terms(
                    term_spans,
                    set(tagged_glossary),
                    start_sec,
                    end_sec,
                )
                start_by_duration[dur] = start_sec

            n_rows_by_duration = {
                dur: max(1, len(terms))
                for dur, terms in terms_by_duration.items()
            }
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
                        if ts.term.strip().lower() == term
                        and ts.start >= start_sec
                        and ts.end <= end_sec
                    ]
                    rel_start = min((ts.start for ts in matching), default=start_sec) - start_sec
                    rel_end = max((ts.end for ts in matching), default=start_sec) - start_sec
                    observed_terms.add(term)
                    row_counts[chosen_tag] += 1
                    stats["written_term_rows"] += 1
                    stats[f"written_rows_dur_{chosen_tag}"] += 1
                    all_rows.append(
                        {
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
                            "glossary_source": "acl6060_tagged_gold",
                        }
                    )
            else:
                row_counts[chosen_tag] += 1
                stats["written_empty_rows"] += 1
                stats[f"written_rows_dur_{chosen_tag}"] += 1
                all_rows.append(
                    {
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
                        "glossary_source": "acl6060_tagged_gold",
                    }
                )

    output_jsonl = os.path.join(args.output_dir, args.output_jsonl_name)
    tmp_jsonl = output_jsonl + ".tmp"
    with open(tmp_jsonl, "w", encoding="utf-8") as fout:
        for row in all_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, output_jsonl)

    eval_stats = write_eval_glossary(
        observed_terms=observed_terms,
        tagged_glossary=tagged_glossary,
        wiki_json=args.wiki_glossary_json,
        output_path=args.eval_glossary_output,
        size=args.eval_glossary_size,
        min_norm_chars=args.min_norm_chars,
    )

    stats["written_total_rows"] = len(all_rows)
    payload = dict(sorted(stats.items()))
    payload.update(
        {
            "input": "acl6060_tagged_text_cached_mfa",
            "output": output_jsonl,
            "audio_output_dir": args.chunk_audio_dir,
            "tagged_text": args.tagged_text,
            "tagged_glossary_json": args.tagged_glossary_json,
            "eval_glossary_output": args.eval_glossary_output,
            "wiki_glossary_json": args.wiki_glossary_json,
            "duration_secs": durations,
            "duration_tags": duration_tags,
            "old_chunk_sec": args.old_chunk_sec,
            "stride_sec": args.stride_sec,
            "duration_assignment": args.duration_assignment,
            "context_build": build_tag,
            "min_norm_chars": args.min_norm_chars,
            "observed_unique_terms": len(observed_terms),
        }
    )
    payload.update(eval_stats)
    for tag in duration_tags:
        payload[f"duration_row_count_{tag}"] = row_counts[tag]

    stats_json = args.stats_json or output_jsonl.replace(".jsonl", "_stats.json")
    with open(stats_json, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"[ACL-TAGGED-VARCTX] output={output_jsonl}")
    print(f"[ACL-TAGGED-VARCTX] stats={stats_json}")
    print(f"[ACL-TAGGED-VARCTX] eval_glossary={args.eval_glossary_output}")
    print(f"[ACL-TAGGED-VARCTX] written_total_rows={payload['written_total_rows']}")
    print(f"[ACL-TAGGED-VARCTX] written_term_rows={payload.get('written_term_rows', 0)}")
    print(f"[ACL-TAGGED-VARCTX] observed_unique_terms={len(observed_terms)}")
    for tag in duration_tags:
        print(f"[ACL-TAGGED-VARCTX] duration_row_count_{tag}={row_counts[tag]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
