#!/usr/bin/env python3
"""Rebuild gt_terms_by_chunk from source-text exact glossary matches.

This is a stricter alternative to historical gt_terms_by_chunk fields.  It uses
the bilingual TSV as the source-text authority, splits src_trajectory by the
existing streaming chunk structure, and marks every whole-token exact match from
the supplied glossary as a GT term for that chunk.

The script does not invent translations.  If a glossary term lacks a target
translation for the requested language, it is counted and excluded.

Optionally, source matches can be filtered by exact target-side evidence.  The
strict streaming-safe policy is ``future_ref``: for each audio chunk, a term is
kept only when its target translation appears as an exact substring in the
assistant messages from that chunk's assistant turn through the end of the
conversation.  This avoids treating a term as GT for the current chunk only
because the same translation appeared before the current audio.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def _token_key(text: str) -> str:
    return " ".join(tok.casefold() for tok in TOKEN_RE.findall(str(text or "")))


def _parse_token_set(text: str) -> set[str]:
    return {tok.strip().casefold() for tok in str(text or "").split(",") if tok.strip()}


def _extract_translation(entry: Mapping[str, Any], lang_code: str) -> str:
    value = entry.get("translation") or entry.get("target_translation") or entry.get(lang_code)
    if value is None and isinstance(entry.get("target_translations"), Mapping):
        value = entry["target_translations"].get(lang_code)
    return str(value or "").strip()


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on {path}:{lineno}")
            yield lineno, obj


def _audio_user_indices(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    out = []
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or ""):
            out.append(idx)
    return out


def _extract_utter_id_from_audio_path(audio_path: str) -> Optional[str]:
    try:
        parts = Path(str(audio_path)).parts
        if len(parts) >= 3:
            return f"{parts[-3]}_{parts[-2]}"
    except Exception:
        return None
    return None


def _parse_trajectory(value: str) -> List[str]:
    if not value:
        return []
    try:
        obj = ast.literal_eval(value)
    except Exception as exc:
        raise ValueError(f"Could not parse src_trajectory: {value[:120]!r}") from exc
    if not isinstance(obj, list):
        raise ValueError("src_trajectory is not a list")
    return [str(x or "") for x in obj]


def load_tsv_index(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"id", "src_text", "tgt_text", "src_trajectory"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"TSV missing columns {sorted(missing)}: {path}")
        for row in reader:
            uid = str(row.get("id") or "").strip()
            if not uid:
                continue
            out[uid] = {
                "src_text": row.get("src_text") or "",
                "tgt_text": row.get("tgt_text") or "",
                "src_trajectory": _parse_trajectory(row.get("src_trajectory") or ""),
            }
    if not out:
        raise ValueError(f"No TSV rows loaded: {path}")
    return out


def _normalize_glossary_entry(key: str, entry: Any, lang_code: str) -> Optional[Dict[str, Any]]:
    if isinstance(entry, str):
        term = str(key).strip()
        translation = entry.strip()
        raw: Dict[str, Any] = {
            "term": term,
            "translation": translation,
            "target_translations": {lang_code: translation},
        }
    elif isinstance(entry, Mapping):
        term = str(entry.get("term") or entry.get("source") or key).strip()
        translation = _extract_translation(entry, lang_code)
        raw = dict(entry)
        raw["term"] = term
        raw["translation"] = translation
        target_translations = raw.get("target_translations")
        if isinstance(target_translations, Mapping):
            target_translations = dict(target_translations)
        else:
            target_translations = {}
        if translation:
            target_translations[lang_code] = translation
        raw["target_translations"] = target_translations
    else:
        return None
    if not term or not translation:
        return None
    token_key = _token_key(term)
    if not token_key:
        return None
    raw["token_key"] = token_key
    raw["term_key"] = _term_key(term)
    return raw


def load_glossary(
    path: Path,
    lang_code: str,
    max_words: int,
    min_norm_chars: int,
    exclude_source_tokens: set[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, Mapping):
        raw_items = [(str(k), v) for k, v in data.items()]
    elif isinstance(data, list):
        raw_items = [(str(i), v) for i, v in enumerate(data)]
    else:
        raise ValueError(f"Unsupported glossary format: {path}")

    by_token_key: Dict[str, List[Dict[str, Any]]] = {}
    stats = Counter()
    for key, entry in raw_items:
        stats["raw_entries"] += 1
        item = _normalize_glossary_entry(key, entry, lang_code)
        if item is None:
            stats["skipped_missing_term_or_translation"] += 1
            continue
        token_key = str(item["token_key"])
        words = token_key.split()
        norm_chars = len("".join(words))
        if exclude_source_tokens and set(words).intersection(exclude_source_tokens):
            stats["skipped_excluded_source_token"] += 1
            continue
        if max_words > 0 and len(words) > max_words:
            stats["skipped_too_many_words"] += 1
            continue
        if norm_chars < min_norm_chars:
            stats["skipped_too_short"] += 1
            continue
        if token_key not in by_token_key:
            by_token_key[token_key] = []
        by_token_key[token_key].append(item)
        stats["kept_entries"] += 1
        stats[f"kept_words_{len(words)}"] += 1
    if not by_token_key:
        raise ValueError(f"No usable glossary terms after filtering: {path}")
    max_term_words = max(len(k.split()) for k in by_token_key)
    return by_token_key, {
        **dict(stats),
        "unique_token_keys": len(by_token_key),
        "max_term_words": max_term_words,
    }


def split_trajectory_by_chunks(trajectory: List[str], num_chunks: int, merge_multiplier: Optional[int]) -> List[str]:
    if num_chunks <= 0:
        return []
    if not trajectory:
        return []
    if merge_multiplier is not None:
        chunks = []
        for i in range(num_chunks):
            start = i * merge_multiplier
            end = min((i + 1) * merge_multiplier, len(trajectory))
            chunks.append(" ".join(trajectory[start:end]))
        return chunks
    chunk_size = (len(trajectory) + num_chunks - 1) // num_chunks
    return [
        " ".join(trajectory[i * chunk_size : min((i + 1) * chunk_size, len(trajectory))])
        for i in range(num_chunks)
    ]


def _assistant_content_for_chunk(messages: Sequence[Mapping[str, Any]], audio_msg_idx: int) -> str:
    # The data layout is alternating user/audio then assistant.  Use the next
    # assistant message if present; otherwise return empty and count it.
    for msg in messages[audio_msg_idx + 1 : audio_msg_idx + 3]:
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def _assistant_text_from(
    messages: Sequence[Mapping[str, Any]],
    *,
    start_idx: int,
    end_idx: Optional[int] = None,
) -> str:
    stop = len(messages) if end_idx is None else min(end_idx, len(messages))
    return "\n".join(
        str(messages[idx].get("content") or "")
        for idx in range(max(0, start_idx), stop)
        if messages[idx].get("role") == "assistant"
    )


def _target_reference_text_for_policy(
    messages: Sequence[Mapping[str, Any]],
    *,
    audio_msg_idx: int,
    assistant_content: str,
    assistant_full_text: str,
    policy: str,
) -> str:
    if policy == "none":
        return ""
    if policy == "chunk":
        return assistant_content
    if policy == "future_ref":
        return _assistant_text_from(messages, start_idx=audio_msg_idx + 1)
    if policy == "full_ref":
        return assistant_full_text
    raise ValueError(f"Unknown target match policy: {policy}")


def _set_audio_term_map_none(messages: Sequence[Mapping[str, Any]]) -> None:
    for msg in messages:
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or ""):
            msg["content"] = "<audio>\n\nterm_map:NONE"


def match_glossary_terms(
    chunk_text: str,
    glossary_by_token_key: Mapping[str, List[Dict[str, Any]]],
    max_term_words: int,
) -> List[Dict[str, Any]]:
    toks = [tok.casefold() for tok in TOKEN_RE.findall(chunk_text or "")]
    if not toks:
        return []
    out: List[Dict[str, Any]] = []
    seen_terms = set()
    for start in range(len(toks)):
        upper = min(max_term_words, len(toks) - start)
        for width in range(1, upper + 1):
            key = " ".join(toks[start : start + width])
            entries = glossary_by_token_key.get(key)
            if not entries:
                continue
            for entry in entries:
                term_key = str(entry.get("term_key") or entry.get("term") or key).casefold()
                if term_key in seen_terms:
                    continue
                seen_terms.add(term_key)
                out.append(entry)
    out.sort(key=lambda x: (-len(str(x.get("token_key") or "").split()), str(x.get("term") or "").casefold()))
    return out


def build(args: argparse.Namespace) -> None:
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    tsv_index = load_tsv_index(args.input_tsv)
    glossary_by_key, glossary_stats = load_glossary(
        args.glossary_json,
        args.lang_code,
        args.max_words,
        args.min_norm_chars,
        _parse_token_set(args.exclude_source_tokens),
    )
    max_term_words = int(glossary_stats["max_term_words"])

    stats: Dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "input_tsv": str(args.input_tsv),
        "glossary_json": str(args.glossary_json),
        "output_jsonl": str(args.output_jsonl),
        "lang_code": args.lang_code,
        "max_words": args.max_words,
        "min_norm_chars": args.min_norm_chars,
        "target_match_policy": args.target_match_policy,
        "term_map_output_policy": args.term_map_output_policy,
        "glossary": glossary_stats,
        "rows_seen": 0,
        "rows_written": 0,
        "rows_missing_tsv": 0,
        "rows_missing_trajectory": 0,
        "rows_mismatched_audio_message_counts": 0,
        "rows_mismatched_chunk_text_counts": 0,
        "chunks_total": 0,
        "chunks_with_exact_gt": 0,
        "source_exact_terms_total": 0,
        "exact_gt_terms_total": 0,
        "target_match_dropped_terms": 0,
        "source_target_same_terms_total": 0,
        "source_target_same_gt_terms": 0,
        "old_gt_terms_total": 0,
        "old_gt_terms_overlapping_exact": 0,
        "translation_in_assistant_chunk": 0,
        "translation_in_target_match_reference": 0,
        "translation_in_assistant_conversation": 0,
        "assistant_chunks_empty": 0,
        "dropped_rows": 0,
        "drop_reasons": Counter(),
    }
    term_freq = Counter()
    chunks_per_row = Counter()
    gt_count_hist = Counter()
    samples: List[Dict[str, Any]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for lineno, obj in _iter_jsonl(args.input_jsonl):
            stats["rows_seen"] += 1
            try:
                messages = obj.get("messages")
                audios = obj.get("audios")
                if not isinstance(messages, list) or not isinstance(audios, list) or not audios:
                    raise ValueError("missing messages or audios")
                audio_user_idxs = _audio_user_indices(messages)
                if len(audio_user_idxs) != len(audios):
                    stats["rows_mismatched_audio_message_counts"] += 1
                    raise ValueError(f"audio user messages={len(audio_user_idxs)} audios={len(audios)}")
                uid = str(obj.get("utter_id") or "").strip()
                if not uid and audios:
                    uid = _extract_utter_id_from_audio_path(str(audios[0])) or ""
                    obj["utter_id"] = uid
                row = tsv_index.get(uid)
                if row is None:
                    stats["rows_missing_tsv"] += 1
                    raise ValueError(f"missing TSV row for utter_id={uid}")
                trajectory = row.get("src_trajectory") or []
                if not trajectory:
                    stats["rows_missing_trajectory"] += 1
                    raise ValueError(f"missing src_trajectory for utter_id={uid}")
                merge_multiplier = obj.get("merge_multiplier")
                if merge_multiplier is not None:
                    try:
                        merge_multiplier = int(merge_multiplier)
                    except Exception as exc:
                        raise ValueError(f"invalid merge_multiplier={merge_multiplier!r}") from exc
                src_chunks = split_trajectory_by_chunks(trajectory, len(audios), merge_multiplier)
                if len(src_chunks) != len(audios):
                    stats["rows_mismatched_chunk_text_counts"] += 1
                    raise ValueError(f"src_chunks={len(src_chunks)} audios={len(audios)}")
                obj["source_chunk_asr_by_chunk"] = src_chunks
                obj["source_full_asr"] = row.get("src_text") or ""
                assistant_full_text = _assistant_text_from(messages, start_idx=0)

                old_gt_by_chunk = obj.get("gt_terms_by_chunk") or [[] for _ in audios]
                if not isinstance(old_gt_by_chunk, list):
                    old_gt_by_chunk = [[] for _ in audios]

                new_gt_by_chunk: List[List[Dict[str, str]]] = []
                for chunk_idx, (chunk_text, msg_idx) in enumerate(zip(src_chunks, audio_user_idxs)):
                    matches = match_glossary_terms(chunk_text, glossary_by_key, max_term_words)
                    assistant_content = _assistant_content_for_chunk(messages, msg_idx)
                    if not assistant_content:
                        stats["assistant_chunks_empty"] += 1
                    old_gt_terms = old_gt_by_chunk[chunk_idx] if chunk_idx < len(old_gt_by_chunk) else []
                    old_keys = set()
                    if isinstance(old_gt_terms, list):
                        for item in old_gt_terms:
                            if isinstance(item, Mapping):
                                old_key = _token_key(str(item.get("term") or item.get("source") or ""))
                                if old_key:
                                    old_keys.add(old_key)
                    stats["old_gt_terms_total"] += len(old_keys)

                    target_reference_text = _target_reference_text_for_policy(
                        messages,
                        audio_msg_idx=msg_idx,
                        assistant_content=assistant_content,
                        assistant_full_text=assistant_full_text,
                        policy=args.target_match_policy,
                    )
                    out_terms = []
                    for entry in matches:
                        term = str(entry.get("term") or "").strip()
                        translation = _extract_translation(entry, args.lang_code)
                        token_key = str(entry.get("token_key") or _token_key(term))
                        if not term or not translation or not token_key:
                            continue
                        stats["source_exact_terms_total"] += 1
                        if translation and translation in assistant_content:
                            stats["translation_in_assistant_chunk"] += 1
                        if translation and translation in target_reference_text:
                            stats["translation_in_target_match_reference"] += 1
                        if translation and translation in assistant_full_text:
                            stats["translation_in_assistant_conversation"] += 1
                        if args.target_match_policy != "none" and translation not in target_reference_text:
                            stats["target_match_dropped_terms"] += 1
                            continue
                        if _token_key(term) == _token_key(translation):
                            stats["source_target_same_gt_terms"] += 1
                        if str(term).strip().casefold() == str(translation).strip().casefold():
                            stats["source_target_same_terms_total"] += 1
                        out_terms.append({"term": term, args.lang_code: translation})
                        term_freq[token_key] += 1
                        if token_key in old_keys:
                            stats["old_gt_terms_overlapping_exact"] += 1

                    stats["chunks_total"] += 1
                    if out_terms:
                        stats["chunks_with_exact_gt"] += 1
                    stats["exact_gt_terms_total"] += len(out_terms)
                    gt_count_hist[len(out_terms)] += 1
                    new_gt_by_chunk.append(out_terms)

                    if len(samples) < args.sample_count and out_terms:
                        samples.append({
                            "row_line": lineno,
                            "utter_id": uid,
                            "chunk_idx": chunk_idx,
                            "merge_multiplier": merge_multiplier,
                            "source_chunk_text": chunk_text,
                            "assistant_content": assistant_content,
                            "exact_gt_terms": out_terms[:30],
                        })

                obj["gt_terms_by_chunk"] = new_gt_by_chunk
                obj["source_glossary_exact_gt_policy"] = {
                    "version": "source_glossary_exact_v1",
                    "source": "source_chunk_asr_by_chunk",
                    "glossary_json": str(args.glossary_json),
                    "whole_token_exact_match": True,
                    "max_words": args.max_words,
                    "min_norm_chars": args.min_norm_chars,
                    "translation_source": "glossary_entry",
                    "target_match_policy": args.target_match_policy,
                    "target_match_reference": (
                        "assistant messages from current audio response through conversation end"
                        if args.target_match_policy == "future_ref"
                        else args.target_match_policy
                    ),
                    "target_match_exact_substring": args.target_match_policy != "none",
                    "term_map_output_policy": args.term_map_output_policy,
                }
                if args.term_map_output_policy == "none":
                    _set_audio_term_map_none(messages)
                chunks_per_row[len(audios)] += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1

            except Exception as exc:
                if args.drop_bad_rows:
                    stats["dropped_rows"] += 1
                    stats["drop_reasons"][str(exc).splitlines()[0][:200]] += 1
                    continue
                raise RuntimeError(f"Failed processing {args.input_jsonl}:{lineno}: {exc}") from exc

    stats["drop_reasons"] = dict(stats["drop_reasons"])
    stats["chunks_per_row_hist"] = dict(chunks_per_row.most_common(50))
    stats["exact_gt_terms_per_chunk_hist"] = dict(gt_count_hist.most_common(80))
    stats["top_exact_terms"] = [{"token_key": k, "count": int(v)} for k, v in term_freq.most_common(80)]
    stats["chunks_with_exact_gt_rate"] = (
        stats["chunks_with_exact_gt"] / stats["chunks_total"] if stats["chunks_total"] else 0.0
    )
    stats["avg_exact_gt_terms_per_chunk"] = (
        stats["exact_gt_terms_total"] / stats["chunks_total"] if stats["chunks_total"] else 0.0
    )
    stats["old_gt_overlap_rate_by_old_terms"] = (
        stats["old_gt_terms_overlapping_exact"] / stats["old_gt_terms_total"]
        if stats["old_gt_terms_total"] else 0.0
    )
    stats["target_match_kept_term_rate"] = (
        stats["exact_gt_terms_total"] / stats["source_exact_terms_total"]
        if stats["source_exact_terms_total"] else 0.0
    )
    stats["translation_in_assistant_chunk_rate"] = (
        stats["translation_in_assistant_chunk"] / stats["source_exact_terms_total"]
        if stats["source_exact_terms_total"] else 0.0
    )
    stats["translation_in_target_match_reference_rate"] = (
        stats["translation_in_target_match_reference"] / stats["source_exact_terms_total"]
        if stats["source_exact_terms_total"] else 0.0
    )
    stats["translation_in_assistant_conversation_rate"] = (
        stats["translation_in_assistant_conversation"] / stats["source_exact_terms_total"]
        if stats["source_exact_terms_total"] else 0.0
    )
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sample_json:
        args.sample_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_jsonl": str(args.output_jsonl),
        "rows_written": stats["rows_written"],
        "chunks_total": stats["chunks_total"],
        "source_exact_terms_total": stats["source_exact_terms_total"],
        "exact_gt_terms_total": stats["exact_gt_terms_total"],
        "target_match_kept_term_rate": stats["target_match_kept_term_rate"],
        "chunks_with_exact_gt_rate": stats["chunks_with_exact_gt_rate"],
        "avg_exact_gt_terms_per_chunk": stats["avg_exact_gt_terms_per_chunk"],
        "translation_in_assistant_chunk_rate": stats["translation_in_assistant_chunk_rate"],
        "translation_in_target_match_reference_rate": stats["translation_in_target_match_reference_rate"],
        "translation_in_assistant_conversation_rate": stats["translation_in_assistant_conversation_rate"],
    }, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--glossary-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path)
    parser.add_argument("--lang-code", default="zh")
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument("--min-norm-chars", type=int, default=2)
    parser.add_argument(
        "--target-match-policy",
        choices=["none", "chunk", "future_ref", "full_ref"],
        default="none",
        help=(
            "Filter source exact matches by exact target translation evidence. "
            "'future_ref' uses assistant messages from the current audio response "
            "through the end of the conversation."
        ),
    )
    parser.add_argument(
        "--term-map-output-policy",
        choices=["preserve", "none"],
        default="preserve",
        help="Whether to preserve existing user term_map text or rewrite audio user chunks to term_map:NONE.",
    )
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--drop-bad-rows", action="store_true")
    parser.add_argument(
        "--exclude-source-tokens",
        default="",
        help="Comma-separated source tokens; glossary terms containing any token are excluded before matching.",
    )
    args = parser.parse_args()
    for p in [args.input_jsonl, args.input_tsv, args.glossary_json]:
        if not p.exists():
            raise FileNotFoundError(p)
    return args


if __name__ == "__main__":
    build(parse_args())
