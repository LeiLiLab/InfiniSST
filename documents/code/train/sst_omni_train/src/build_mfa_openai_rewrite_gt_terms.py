#!/usr/bin/env python3
"""Build clean de/ja Speech LLM GT terms from GigaSpeech MFA + OpenAI rewrite.

This script intentionally does not read legacy term_map entries as labels.
It uses MFA TextGrid word timestamps to find source-side exact glossary
occurrences, asks OpenAI for an exact assistant reference span plus an uncommon
target translation, rewrites that exact span, and emits gt_terms_by_chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_mfa_glossary_future_ref_gt_terms import (  # noqa: E402
    GigaSpeechMFA,
    WordInterval,
    assistant_future_text,
    audio_user_indices,
    iter_jsonl,
    normalize_word,
    term_display_key,
    term_matches_chunk,
    tokenize_text_variants,
    wav_duration_sec,
)

SAMPLE_RATE = 16000
SOURCE_TOKEN_EXCLUDE_DEFAULT = (
    "a,an,the,this,that,these,those,his,her,hers,him,he,she,it,its,they,them,"
    "their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,"
    "himself,herself,itself,ourselves,yourselves,themselves,what,which,who,"
    "whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,"
    "everybody,everything,there,here,where,when,why,how,all,any,some,one,two"
)


def _parse_legacy_term_map_translations(messages: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    """Read user-side term_map lines as a translation lexicon, not GT labels."""
    out: Dict[str, List[str]] = {}
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "")
        marker = "term_map:"
        idx = content.find(marker)
        if idx < 0 or "term_map:NONE" in content:
            continue
        for raw_line in content[idx + len(marker) :].splitlines():
            line = raw_line.strip()
            if not line or line.upper() == "NONE" or "=" not in line:
                continue
            term, translation = line.split("=", 1)
            term = term.strip()
            translation = _clean_text(translation)
            if not term or not translation:
                continue
            key = term_display_key(term)
            bucket = out.setdefault(key, [])
            if translation not in bucket:
                bucket.append(translation)
    for key in list(out):
        out[key].sort(key=len, reverse=True)
    return out


def _find_exact_translation_span(
    text: str,
    translations: Sequence[str],
    *,
    require_boundaries: bool,
) -> Optional[str]:
    for translation in translations:
        if not translation:
            continue
        start = text.find(translation)
        if start < 0:
            continue
        end = start + len(translation)
        if require_boundaries and not _safe_text_boundaries(text, start, end, translation):
            continue
        return translation
    return None


def _sha_json(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _parse_token_set(text: str) -> set[str]:
    return {x.strip().casefold() for x in str(text or "").split(",") if x.strip()}


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def _is_latin_alnum(ch: str) -> bool:
    return bool(ch) and ch.isalnum() and bool(re.match(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", ch))


def _safe_text_boundaries(text: str, start: int, end: int, replacement: str) -> bool:
    if start < 0 or end > len(text) or start >= end or not replacement:
        return False
    if start > 0 and _is_latin_alnum(text[start - 1]) and _is_latin_alnum(replacement[0]):
        return False
    if end < len(text) and _is_latin_alnum(text[end]) and _is_latin_alnum(replacement[-1]):
        return False
    return True


def _clean_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^[\"'“”‘’`]+|[\"'“”‘’`]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _validate_target_translation(source_term: str, reference_span: str, target_translation: str) -> Tuple[bool, str]:
    target_translation = _clean_text(target_translation)
    reference_span = str(reference_span or "").strip()
    if not reference_span:
        return False, "missing_reference_span"
    if not target_translation:
        return False, "missing_target_translation"
    if target_translation == reference_span:
        return False, "identity_with_reference_span"
    if _term_key(source_term) == _term_key(target_translation):
        return False, "source_equals_target"
    if "<term" in target_translation or "</term>" in target_translation or "=" in target_translation or "\n" in target_translation:
        return False, "bad_target_marker_or_delimiter"
    if len(target_translation) > max(24, len(reference_span) * 4 + 12):
        return False, "target_too_long"
    return True, "ok"


def _extract_utter_id_from_audio(audio_path: str) -> str:
    parts = Path(str(audio_path)).parts
    if len(parts) < 3:
        return ""
    return f"{parts[-3]}_{parts[-2]}"


def _parse_audio_spec(audio_spec: str) -> Tuple[str, int, int]:
    """Parse GigaSpeech TSV audio field: /path/OPUS.opus:start:n_frames."""
    base, start_s, n_frames_s = str(audio_spec).rsplit(":", 2)
    # The MFA SQLite stores the full opus path in manifest_segments.opus.
    opus = base
    start = int(start_s)
    n_frames = int(n_frames_s)
    return opus, start, start + n_frames


def load_audio_map_tsv(path: Optional[Path]) -> Dict[str, Tuple[str, int, int]]:
    if path is None:
        return {}
    out: Dict[str, Tuple[str, int, int]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            id_idx = header.index("id")
            audio_idx = header.index("audio")
        except ValueError as exc:
            raise ValueError(f"TSV must contain id/audio columns: {path}") from exc
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(id_idx, audio_idx):
                continue
            key = parts[id_idx].strip()
            if not key:
                continue
            try:
                out[key] = _parse_audio_spec(parts[audio_idx])
            except Exception:
                continue
    if not out:
        raise ValueError(f"No usable id->audio entries loaded from {path}")
    return out


def mfa_words_for_span(
    mfa: GigaSpeechMFA,
    *,
    cache_key: str,
    opus: str,
    start_samples: int,
    end_samples: int,
    candidate_limit: int,
) -> List[WordInterval]:
    out: List[WordInterval] = []
    for seg_id, seg_start_sec, _seg_end_sec in mfa.candidates(cache_key, opus, start_samples, end_samples, candidate_limit):
        words = mfa.textgrid_words(seg_id)
        if words is None:
            continue
        for rel_start, rel_end, raw_word in words:
            norm = normalize_word(raw_word)
            if not norm:
                continue
            abs_start = seg_start_sec + rel_start
            abs_end = seg_start_sec + rel_end
            if abs_end <= start_samples / SAMPLE_RATE or abs_start >= end_samples / SAMPLE_RATE:
                continue
            out.append(WordInterval(abs_start, abs_end, raw_word, norm))
    out.sort(key=lambda x: (x.start, x.end, x.word))
    return out


def _set_audio_term_map_none(messages: Sequence[MutableMapping[str, Any]]) -> None:
    for msg in messages:
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or ""):
            msg["content"] = "<audio>\n\nterm_map:NONE"


def _assistant_msg_for_chunk(messages: Sequence[Mapping[str, Any]], audio_msg_idx: int) -> str:
    for msg in messages[audio_msg_idx + 1 : audio_msg_idx + 3]:
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def _replace_future_assistant(
    messages: Sequence[MutableMapping[str, Any]],
    *,
    start_idx: int,
    reference_span: str,
    target_translation: str,
    require_boundaries: bool,
) -> Optional[int]:
    for msg_idx in range(start_idx, len(messages)):
        msg = messages[msg_idx]
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        pos = content.find(reference_span)
        if pos < 0:
            continue
        end = pos + len(reference_span)
        if require_boundaries and not _safe_text_boundaries(content, pos, end, target_translation):
            return None
        msg["content"] = content[:pos] + target_translation + content[end:]
        return msg_idx
    return None


def load_source_glossary(
    path: Path,
    *,
    max_words: int,
    min_norm_chars: int,
    exclude_tokens: set[str],
) -> Tuple[Dict[Tuple[str, ...], List[Dict[str, Any]]], Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        raw_items = [(str(k), v) for k, v in data.items()]
    elif isinstance(data, list):
        raw_items = [(str(i), v) for i, v in enumerate(data)]
    else:
        raise ValueError(f"Unsupported glossary JSON format: {path}")

    by_tokens: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    stats = Counter()
    for key, entry in raw_items:
        stats["raw_entries"] += 1
        if isinstance(entry, str):
            term = key.strip()
            raw: Dict[str, Any] = {"term": term}
        elif isinstance(entry, Mapping):
            term = str(entry.get("term") or entry.get("source") or key).strip()
            raw = dict(entry)
            raw["term"] = term
        else:
            stats["skipped_bad_entry"] += 1
            continue
        if not term:
            stats["skipped_missing_term"] += 1
            continue
        variants = tokenize_text_variants(term)
        if not variants:
            stats["skipped_no_tokens"] += 1
            continue
        kept = False
        for tokens in variants:
            norm_chars = len("".join(tokens))
            if exclude_tokens and set(tokens).intersection(exclude_tokens):
                stats["skipped_excluded_source_token_variant"] += 1
                continue
            if max_words > 0 and len(tokens) > max_words:
                continue
            if norm_chars < min_norm_chars:
                continue
            item = dict(raw)
            item["token_tuple"] = list(tokens)
            item["token_key"] = " ".join(tokens)
            item["term_key"] = term_display_key(term)
            by_tokens.setdefault(tokens, []).append(item)
            kept = True
        if kept:
            stats["kept_entries"] += 1
        else:
            stats["skipped_length_or_exclusion"] += 1
    if not by_tokens:
        raise ValueError(f"No usable source glossary terms after filtering: {path}")
    stats["unique_token_tuples"] = len(by_tokens)
    stats["max_term_words"] = max(len(k) for k in by_tokens)
    return by_tokens, dict(stats)


def load_source_candidate_allowlist(path: Optional[Path]) -> Tuple[Dict[str, set[Tuple[str, ...]]], Dict[str, Any]]:
    """Load old-new_v3 style utterance-level NER/noun candidate allowlist.

    The allowlist is only a type/phrase gate.  Actual GT evidence still comes
    from MFA word timestamp exact matching in ``iter_source_occurrences``.
    """
    if path is None:
        return {}, {}
    allowed: Dict[str, set[Tuple[str, ...]]] = {}
    stats = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid source candidate JSONL at {path}:{line_no}: {exc}") from exc
            utter_id = str(obj.get("utter_id") or "").strip()
            candidates = obj.get("ner_candidates") or obj.get("source_candidates") or obj.get("candidates")
            if not utter_id or not isinstance(candidates, list):
                stats["skipped_bad_row"] += 1
                continue
            variants: set[Tuple[str, ...]] = set()
            for cand in candidates:
                cand_text = str(cand or "").strip()
                if not cand_text:
                    continue
                for tokens in tokenize_text_variants(cand_text):
                    if tokens:
                        variants.add(tuple(tokens))
            if not variants:
                stats["skipped_no_variants"] += 1
                continue
            allowed.setdefault(utter_id, set()).update(variants)
            stats["rows_loaded"] += 1
            stats["candidate_variants_loaded"] += len(variants)
    if not allowed:
        raise ValueError(f"No usable source candidates loaded from {path}")
    stats["utterances_loaded"] = len(allowed)
    return allowed, dict(stats)


def iter_source_occurrences(
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
                dedupe_key = (start_idx, width, str(entry.get("term_key")))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                yield {
                    "term": str(entry.get("term") or "").strip(),
                    "token_tuple": list(tokens),
                    "token_key": str(entry.get("token_key") or " ".join(tokens)),
                    "term_key": str(entry.get("term_key") or entry.get("term") or ""),
                    "mfa_text": raw_text,
                    "mfa_start": span_start,
                    "mfa_end": span_end,
                }


def _chunk_mfa_text(words: Sequence[WordInterval], chunk_start: float, chunk_end: float) -> str:
    return " ".join(
        w.word
        for w in words
        if max(w.start, chunk_start) < min(w.end, chunk_end)
    )


def _task_key(task: Mapping[str, Any], *, lang_code: str) -> str:
    return _sha_json({
        "v": 1,
        "lang": lang_code,
        "utter_id": task["utter_id"],
        "chunk_idx": task["chunk_idx"],
        "term": task["term"],
        "mfa_text": task["mfa_text"],
        "source_chunk_hash": hashlib.sha256(str(task["source_chunk_text"]).encode("utf-8")).hexdigest(),
        "future_hash": hashlib.sha256(str(task["assistant_future_text"]).encode("utf-8")).hexdigest(),
    })


def collect_tasks(args: argparse.Namespace, glossary_by_tokens, max_term_words: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mfa = GigaSpeechMFA(args.sqlite_index, args.textgrid_dir)
    audio_map = load_audio_map_tsv(args.audio_map_tsv)
    source_allowlist, source_allowlist_stats = load_source_candidate_allowlist(args.source_candidate_jsonl)
    stats = Counter()
    tasks: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    row_sources: Dict[int, Dict[str, Any]] = {}

    for lineno, obj in iter_jsonl(args.input_jsonl):
        stats["rows_seen"] += 1
        if args.limit_rows > 0 and stats["rows_seen"] > args.limit_rows:
            break
        try:
            messages = obj.get("messages")
            audios = obj.get("audios")
            if not isinstance(messages, list) or not isinstance(audios, list) or not audios:
                raise ValueError("missing messages or audios")
            audio_idxs = audio_user_indices(messages)
            if len(audio_idxs) != len(audios):
                raise ValueError(f"audio user messages={len(audio_idxs)} audios={len(audios)}")
            utter_id = str(obj.get("utter_id") or "").strip()
            clip_id = _extract_utter_id_from_audio(str(audios[0]))
            if not utter_id:
                utter_id = clip_id
            if not utter_id:
                raise ValueError("missing utter_id/clip_id")
            allowed_tokens = source_allowlist.get(utter_id) or source_allowlist.get(clip_id)
            if source_allowlist and allowed_tokens is None:
                stats["rows_missing_source_candidate_allowlist"] += 1
            legacy_term_map_translations = (
                _parse_legacy_term_map_translations(messages)
                if args.prefilter_reference_spans_from_input_term_map
                else {}
            )
            if args.prefilter_reference_spans_from_input_term_map:
                stats["rows_with_legacy_term_map_translation_lexicon"] += 1 if legacy_term_map_translations else 0

            align_info, words = mfa.row_words(utter_id, candidate_limit=args.overlap_query_limit)
            if align_info is None and clip_id in audio_map:
                opus, start_samples, end_samples = audio_map[clip_id]
                align_info = (opus, start_samples, end_samples)
                words = mfa_words_for_span(
                    mfa,
                    cache_key=clip_id,
                    opus=opus,
                    start_samples=start_samples,
                    end_samples=end_samples,
                    candidate_limit=args.overlap_query_limit,
                )
                stats["mfa_audio_map_lookup_used"] += 1
            elif align_info is not None:
                stats["mfa_direct_align_id_lookup_used"] += 1
            if align_info is None:
                raise ValueError(f"missing MFA align row and audio map entry for utter_id={utter_id} clip_id={clip_id}")
            if not words:
                raise ValueError(f"no MFA words for utter_id={utter_id} clip_id={clip_id}")
            _opus, align_start_samples, _align_end_samples = align_info
            align_start_sec = align_start_samples / SAMPLE_RATE

            durations = [wav_duration_sec(str(p)) for p in audios]
            chunk_bounds = []
            cursor = align_start_sec
            for duration in durations:
                chunk_bounds.append((cursor, cursor + duration))
                cursor += duration
                stats[f"chunk_duration_sec_{round(duration, 2)}"] += 1

            source_full_mfa_text = " ".join(w.word for w in words)
            source_chunk_texts: List[str] = []
            row_occurrences = list(iter_source_occurrences(words, glossary_by_tokens, max_term_words=max_term_words))
            stats["mfa_source_occurrences_total"] += len(row_occurrences)

            for chunk_idx, ((chunk_start, chunk_end), msg_idx) in enumerate(zip(chunk_bounds, audio_idxs)):
                source_chunk_text = _chunk_mfa_text(words, chunk_start, chunk_end)
                source_chunk_texts.append(source_chunk_text)
                future_text = assistant_future_text(messages, msg_idx)
                assistant_chunk = _assistant_msg_for_chunk(messages, msg_idx)
                chunk_candidates = []
                seen = set()
                for occ in row_occurrences:
                    if not term_matches_chunk(occ, chunk_start, chunk_end, args.chunk_assignment_policy):
                        continue
                    stats["mfa_chunk_source_candidates_total"] += 1
                    if source_allowlist:
                        token_tuple = tuple(str(x) for x in occ.get("token_tuple") or str(occ.get("token_key") or "").split())
                        if allowed_tokens is None or token_tuple not in allowed_tokens:
                            stats["candidate_drop_not_in_source_candidate_allowlist"] += 1
                            continue
                        stats["candidate_kept_by_source_candidate_allowlist"] += 1
                    term_key = str(occ.get("term_key") or occ.get("term") or "").casefold()
                    if term_key in seen:
                        stats["deduped_same_chunk_term"] += 1
                        continue
                    seen.add(term_key)
                    future_prompt_text = future_text[: args.max_future_chars]
                    reference_span_hint = ""
                    if args.prefilter_reference_spans_from_input_term_map:
                        translations = legacy_term_map_translations.get(occ["term_key"]) or []
                        if not translations:
                            stats["candidate_drop_no_legacy_termmap_translation"] += 1
                            continue
                        reference_span_hint = _find_exact_translation_span(
                            future_prompt_text,
                            translations,
                            require_boundaries=args.require_text_boundaries,
                        ) or ""
                        if not reference_span_hint:
                            stats["candidate_drop_no_legacy_termmap_future_exact_span"] += 1
                            continue
                        stats["candidate_kept_by_legacy_termmap_future_exact_span"] += 1
                    task = {
                        "row_line": lineno,
                        "utter_id": utter_id,
                        "chunk_idx": chunk_idx,
                        "audio_msg_idx": msg_idx,
                        "term": occ["term"],
                        "term_key": occ["term_key"],
                        "token_key": occ["token_key"],
                        "mfa_text": occ["mfa_text"],
                        "mfa_start": round(float(occ["mfa_start"]) - align_start_sec, 4),
                        "mfa_end": round(float(occ["mfa_end"]) - align_start_sec, 4),
                        "mfa_chunk_start": round(chunk_start - align_start_sec, 4),
                        "mfa_chunk_end": round(chunk_end - align_start_sec, 4),
                        "source_chunk_text": source_chunk_text[: args.max_source_chars],
                        "assistant_chunk_text": assistant_chunk[: args.max_assistant_chunk_chars],
                        "assistant_future_text": future_prompt_text,
                    }
                    if reference_span_hint:
                        task["reference_span_hint"] = reference_span_hint
                        task["legacy_termmap_translation"] = reference_span_hint
                    task["cache_key"] = _task_key(task, lang_code=args.lang_code)
                    chunk_candidates.append(task)
                    if float(occ["mfa_start"]) < chunk_start or float(occ["mfa_end"]) > chunk_end:
                        stats["boundary_overlap_candidates"] += 1
                chunk_candidates.sort(key=lambda x: (float(x["mfa_start"]), -len(str(x["term"]).split()), str(x["term"]).casefold()))
                if args.max_candidates_per_chunk > 0 and len(chunk_candidates) > args.max_candidates_per_chunk:
                    stats["candidate_cap_dropped"] += len(chunk_candidates) - args.max_candidates_per_chunk
                    chunk_candidates = chunk_candidates[: args.max_candidates_per_chunk]
                tasks.extend(chunk_candidates)
                if len(samples) < args.sample_count and chunk_candidates:
                    samples.append({
                        "row_line": lineno,
                        "utter_id": utter_id,
                        "chunk_idx": chunk_idx,
                        "source_chunk_text": source_chunk_text,
                        "assistant_chunk_text": assistant_chunk,
                        "candidates": [
                            {k: t[k] for k in ("term", "mfa_text", "mfa_start", "mfa_end")}
                            for t in chunk_candidates[:20]
                        ],
                    })
            row_sources[lineno] = {
                "utter_id": utter_id,
                "source_chunk_mfa_text_by_chunk": source_chunk_texts,
                "source_full_mfa_text": source_full_mfa_text,
            }
            stats["rows_collect_ok"] += 1
        except Exception as exc:
            stats["rows_collect_failed"] += 1
            stats[f"collect_error:{str(exc).splitlines()[0][:160]}"] += 1
            if not args.drop_bad_rows:
                raise RuntimeError(f"Failed collecting {args.input_jsonl}:{lineno}: {exc}") from exc
    stats["tasks_total"] = len(tasks)
    if source_allowlist:
        stats["source_candidate_allowlist_enabled"] = 1
        for key, value in source_allowlist_stats.items():
            stats[f"source_candidate_allowlist:{key}"] = value
    return tasks, {"stats": dict(stats), "samples": samples, "row_sources": row_sources}


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"OpenAI cache must be a JSON object: {path}")
    return {str(k): dict(v) for k, v in obj.items() if isinstance(v, Mapping)}


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _openai_rewrite_batch(
    *,
    api_key: str,
    model: str,
    lang_code: str,
    batch: List[Dict[str, Any]],
    timeout: int,
    max_retries: int,
) -> List[Dict[str, Any]]:
    lang_name = {"de": "German", "ja": "Japanese", "zh": "Chinese"}.get(lang_code, lang_code)
    system = (
        f"You prepare {lang_name} terminology supervision for speech translation SFT. "
        "For each item, find an exact substring from assistant_future_text that translates the English source_term, "
        f"then propose a fluent but less common {lang_name} target translation for that same source term. "
        "The reference_span must be copied exactly from assistant_future_text. "
        "Return strict JSON only."
    )
    user = {
        "schema": {
            "items": [
                {
                    "id": "string",
                    "status": "ok | no_span",
                    "reference_span": "exact substring from assistant_future_text if ok",
                    "target_translation": f"uncommon but correct {lang_name} terminology translation if ok",
                }
            ]
        },
        "constraints": [
            "Do not invent reference_span. It must be an exact substring copied from assistant_future_text.",
            "If reference_span_hint is non-empty, use it as reference_span unless it clearly does not translate source_term.",
            "target_translation must be different from reference_span when possible.",
            "Do not include XML, explanations, equals signs, or newline characters.",
            "If no exact assistant substring translates the source term, status must be no_span.",
        ],
        "items": [
            {
                "id": str(i),
                "source_term": item["term"],
                "mfa_text": item["mfa_text"],
                "source_chunk_text": item["source_chunk_text"],
                "assistant_chunk_text": item["assistant_chunk_text"],
                "assistant_future_text": item["assistant_future_text"],
                "reference_span_hint": item.get("reference_span_hint", ""),
            }
            for i, item in enumerate(batch)
        ],
    }
    payload = {
        "model": model,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            content = json.loads(raw)["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            items = parsed.get("items")
            if not isinstance(items, list):
                raise ValueError("OpenAI JSON missing items list")
            return [dict(x) for x in items if isinstance(x, Mapping)]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))
    raise RuntimeError(f"OpenAI rewrite batch failed after retries: {last_error}") from last_error


def fill_openai_cache(args: argparse.Namespace, tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cache = _load_cache(args.openai_cache_json)
    if args.use_legacy_termmap_span_as_target_without_openai:
        seeded = 0
        missing_hint = 0
        for task in tasks:
            key = task["cache_key"]
            hint = _clean_text(str(task.get("reference_span_hint") or ""))
            if not hint:
                missing_hint += 1
                cache[key] = {
                    "status": "invalid",
                    "reason": "missing_reference_span_hint",
                    "source_term": task["term"],
                    "reference_span": "",
                    "target_translation": "",
                    "model": "legacy_termmap_exact_no_openai",
                }
                continue
            cache[key] = {
                "status": "ok",
                "reason": "legacy_termmap_future_exact_span",
                "source_term": task["term"],
                "mfa_text": task["mfa_text"],
                "reference_span": hint,
                "target_translation": hint,
                "model": "legacy_termmap_exact_no_openai",
            }
            seeded += 1
        _write_json_atomic(args.openai_cache_json, cache)
        print(json.dumps({
            "openai_cache_json": str(args.openai_cache_json),
            "mode": "legacy_termmap_exact_no_openai",
            "seeded": seeded,
            "missing_hint": missing_hint,
            "cache_total": len(cache),
        }, ensure_ascii=False), flush=True)
        return cache
    pending = []
    seen = set()
    for task in tasks:
        key = task["cache_key"]
        if key in cache:
            continue
        if key in seen:
            continue
        seen.add(key)
        pending.append(task)
    if args.max_api_items > 0:
        pending = pending[: args.max_api_items]
    if not pending:
        return cache
    if args.cache_only:
        raise RuntimeError(f"OpenAI cache missing {len(pending)} items and --cache-only was set")
    if args.dry_run:
        for task in pending:
            span = task["assistant_chunk_text"][: max(1, min(8, len(task["assistant_chunk_text"])))]
            cache[task["cache_key"]] = {
                "status": "invalid",
                "reason": "dry_run_no_real_span",
                "source_term": task["term"],
                "reference_span": span,
                "target_translation": span + "_TERM",
                "model": "dry_run",
            }
        _write_json_atomic(args.openai_cache_json, cache)
        return cache
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run or --cache-only with complete cache")
    for start in range(0, len(pending), args.openai_batch_size):
        chunk = pending[start : start + args.openai_batch_size]
        results = _openai_rewrite_batch(
            api_key=api_key,
            model=args.openai_model,
            lang_code=args.lang_code,
            batch=chunk,
            timeout=args.openai_timeout,
            max_retries=args.openai_max_retries,
        )
        by_id = {str(x.get("id")): x for x in results}
        for i, task in enumerate(chunk):
            raw = by_id.get(str(i), {})
            status = str(raw.get("status") or "").strip()
            ref_span = str(raw.get("reference_span") or "").strip()
            target = _clean_text(str(raw.get("target_translation") or ""))
            ok, reason = _validate_target_translation(str(task["term"]), ref_span, target)
            if status != "ok":
                ok = False
                reason = status or "not_ok"
            if ref_span and ref_span not in str(task["assistant_future_text"]):
                ok = False
                reason = "reference_span_not_in_prompt_future_text"
            cache[task["cache_key"]] = {
                "status": "ok" if ok else "invalid",
                "reason": reason,
                "source_term": task["term"],
                "mfa_text": task["mfa_text"],
                "reference_span": ref_span,
                "target_translation": target,
                "model": args.openai_model,
            }
        _write_json_atomic(args.openai_cache_json, cache)
        print(json.dumps({
            "openai_cache_json": str(args.openai_cache_json),
            "generated": min(start + len(chunk), len(pending)),
            "pending_total": len(pending),
            "cache_total": len(cache),
        }, ensure_ascii=False), flush=True)
        time.sleep(args.openai_sleep_sec)
    return cache


def apply_cache(
    args: argparse.Namespace,
    task_by_row: Mapping[int, List[Dict[str, Any]]],
    row_sources: Mapping[int, Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    stats = Counter()
    samples: List[Dict[str, Any]] = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for lineno, obj in iter_jsonl(args.input_jsonl):
            stats["rows_seen"] += 1
            if args.limit_rows > 0 and stats["rows_seen"] > args.limit_rows:
                break
            messages = obj.get("messages")
            audios = obj.get("audios")
            if not isinstance(messages, list) or not isinstance(audios, list):
                raise ValueError(f"missing messages/audios at row {lineno}")
            audio_idxs = audio_user_indices(messages)
            if len(audio_idxs) != len(audios):
                raise ValueError(f"audio user messages={len(audio_idxs)} audios={len(audios)} at row {lineno}")
            utter_id = str(obj.get("utter_id") or "").strip() or _extract_utter_id_from_audio(str(audios[0]))
            obj["utter_id"] = utter_id
            _set_audio_term_map_none(messages)
            gt_by_chunk: List[List[Dict[str, Any]]] = [[] for _ in audios]
            row_events: List[Dict[str, Any]] = []
            source_info = row_sources.get(lineno, {})
            source_chunk_texts = list(source_info.get("source_chunk_mfa_text_by_chunk") or [])
            if len(source_chunk_texts) != len(audios):
                source_chunk_texts = ["" for _ in audios]
            row_tasks = task_by_row.get(lineno, [])
            for task in row_tasks:
                chunk_idx = int(task["chunk_idx"])
                if 0 <= chunk_idx < len(source_chunk_texts):
                    source_chunk_texts[chunk_idx] = str(task.get("source_chunk_text") or "")
                cached = cache.get(str(task["cache_key"]))
                if not cached or cached.get("status") != "ok":
                    stats[f"openai_drop:{(cached or {}).get('reason', 'missing_cache')}"] += 1
                    continue
                reference_span = str(cached.get("reference_span") or "")
                target = _clean_text(str(cached.get("target_translation") or ""))
                msg_idx = int(task["audio_msg_idx"])
                if reference_span not in assistant_future_text(messages, msg_idx):
                    stats["drop_reference_span_missing_after_previous_rewrite"] += 1
                    continue
                replaced_idx = _replace_future_assistant(
                    messages,
                    start_idx=msg_idx + 1,
                    reference_span=reference_span,
                    target_translation=target,
                    require_boundaries=args.require_text_boundaries,
                )
                if replaced_idx is None:
                    stats["drop_rewrite_boundary_violation_or_missing"] += 1
                    continue
                item = {
                    "term": task["term"],
                    args.lang_code: target,
                    "translation": target,
                    "reference_span": reference_span,
                    "mfa_text": task["mfa_text"],
                    "mfa_start": task["mfa_start"],
                    "mfa_end": task["mfa_end"],
                    "mfa_chunk_start": task["mfa_chunk_start"],
                    "mfa_chunk_end": task["mfa_chunk_end"],
                    "openai_model": cached.get("model"),
                }
                gt_by_chunk[chunk_idx].append(item)
                row_events.append({
                    "chunk_idx": chunk_idx,
                    "assistant_msg_idx": replaced_idx,
                    "term": task["term"],
                    "reference_span": reference_span,
                    "target_translation": target,
                })
                stats["kept_gt_terms"] += 1
                if float(task["mfa_start"]) < float(task["mfa_chunk_start"]) or float(task["mfa_end"]) > float(task["mfa_chunk_end"]):
                    stats["kept_boundary_overlap_gt_terms"] += 1
            for chunk_terms in gt_by_chunk:
                chunk_terms.sort(key=lambda x: (float(x["mfa_start"]), -len(str(x["term"]).split()), str(x["term"]).casefold()))
                if chunk_terms:
                    stats["chunks_with_gt"] += 1
            stats["chunks_total"] += len(audios)
            obj["gt_terms_by_chunk"] = gt_by_chunk
            obj["source_chunk_mfa_text_by_chunk"] = source_chunk_texts
            obj["source_full_mfa_text"] = str(source_info.get("source_full_mfa_text") or "")
            obj["mfa_openai_rewrite_gt_policy"] = {
                "version": "mfa_openai_rewrite_v1",
                "source": "GigaSpeech MFA TextGrid word timestamps",
                "glossary_json": str(args.glossary_json),
                "sqlite_index": str(args.sqlite_index),
                "textgrid_dir": str(args.textgrid_dir),
                "source_match": "MFA normalized source word n-gram exact match",
                "source_candidate_jsonl": str(args.source_candidate_jsonl) if args.source_candidate_jsonl else None,
                "source_candidate_filter": "utterance-level NER/noun-chunk/proper-noun allowlist" if args.source_candidate_jsonl else None,
                "legacy_term_map_reference_span_prefilter": bool(args.prefilter_reference_spans_from_input_term_map),
                "legacy_term_map_span_as_target_without_openai": bool(args.use_legacy_termmap_span_as_target_without_openai),
                "chunk_assignment_policy": args.chunk_assignment_policy,
                "openai_model": args.openai_model,
                "reference_span_policy": "exact substring in assistant future text",
                "assistant_rewritten": True,
                "term_map_output_policy": "none",
                "events": row_events[:80],
            }
            if len(samples) < args.sample_count and row_events:
                samples.append({
                    "row_line": lineno,
                    "utter_id": utter_id,
                    "events": row_events[:12],
                    "source_chunk_mfa_text_by_chunk": source_chunk_texts[:6],
                })
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["rows_written"] += 1
    stats["chunks_with_gt_rate"] = stats["chunks_with_gt"] / stats["chunks_total"] if stats["chunks_total"] else 0.0
    stats["avg_gt_terms_per_chunk"] = stats["kept_gt_terms"] / stats["chunks_total"] if stats["chunks_total"] else 0.0
    return {"stats": dict(stats), "samples": samples}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--glossary-json", type=Path, required=True)
    parser.add_argument("--source-candidate-jsonl", type=Path, default=None)
    parser.add_argument("--prefilter-reference-spans-from-input-term-map", action="store_true")
    parser.add_argument("--use-legacy-termmap-span-as-target-without-openai", action="store_true")
    parser.add_argument("--openai-cache-json", type=Path, required=True)
    parser.add_argument("--sqlite-index", type=Path, default=Path("/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite"))
    parser.add_argument("--textgrid-dir", type=Path, default=Path("/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"))
    parser.add_argument("--audio-map-tsv", type=Path, default=None, help="Optional TSV with id/audio columns used only to map clip ids to opus sample spans.")
    parser.add_argument("--lang-code", choices=["de", "ja", "zh"], required=True)
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument("--min-norm-chars", type=int, default=3)
    parser.add_argument("--exclude-source-tokens", default=SOURCE_TOKEN_EXCLUDE_DEFAULT)
    parser.add_argument("--chunk-assignment-policy", choices=["overlap", "contained", "end_in_chunk", "midpoint"], default="overlap")
    parser.add_argument("--overlap-query-limit", type=int, default=128)
    parser.add_argument("--max-candidates-per-chunk", type=int, default=16)
    parser.add_argument("--max-source-chars", type=int, default=600)
    parser.add_argument("--max-assistant-chunk-chars", type=int, default=600)
    parser.add_argument("--max-future-chars", type=int, default=1800)
    parser.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--openai-batch-size", type=int, default=4)
    parser.add_argument("--openai-timeout", type=int, default=120)
    parser.add_argument("--openai-max-retries", type=int, default=4)
    parser.add_argument("--openai-sleep-sec", type=float, default=0.2)
    parser.add_argument("--max-api-items", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--require-text-boundaries", action="store_true")
    parser.add_argument("--drop-bad-rows", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    for path in [args.input_jsonl, args.glossary_json, args.sqlite_index, args.textgrid_dir]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.audio_map_tsv is not None and not args.audio_map_tsv.exists():
        raise FileNotFoundError(args.audio_map_tsv)
    if args.source_candidate_jsonl is not None and not args.source_candidate_jsonl.exists():
        raise FileNotFoundError(args.source_candidate_jsonl)
    return args


def main() -> None:
    args = parse_args()
    glossary_by_tokens, glossary_stats = load_source_glossary(
        args.glossary_json,
        max_words=args.max_words,
        min_norm_chars=args.min_norm_chars,
        exclude_tokens=_parse_token_set(args.exclude_source_tokens),
    )
    tasks, collect = collect_tasks(args, glossary_by_tokens, int(glossary_stats["max_term_words"]))
    task_by_row: Dict[int, List[Dict[str, Any]]] = {}
    for task in tasks:
        task_by_row.setdefault(int(task["row_line"]), []).append(task)
    cache = fill_openai_cache(args, tasks)
    applied = apply_cache(args, task_by_row, collect.get("row_sources", {}), cache)
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "glossary_json": str(args.glossary_json),
        "source_candidate_jsonl": str(args.source_candidate_jsonl) if args.source_candidate_jsonl else None,
        "legacy_term_map_reference_span_prefilter": bool(args.prefilter_reference_spans_from_input_term_map),
        "legacy_term_map_span_as_target_without_openai": bool(args.use_legacy_termmap_span_as_target_without_openai),
        "openai_cache_json": str(args.openai_cache_json),
        "lang_code": args.lang_code,
        "glossary": glossary_stats,
        "collect": collect["stats"],
        "apply": applied["stats"],
    }
    args.stats_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.sample_json.write_text(json.dumps({
        "collect_samples": collect["samples"],
        "apply_samples": applied["samples"],
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_jsonl": str(args.output_jsonl),
        "rows_written": applied["stats"].get("rows_written", 0),
        "tasks_total": collect["stats"].get("tasks_total", 0),
        "kept_gt_terms": applied["stats"].get("kept_gt_terms", 0),
        "chunks_with_gt_rate": applied["stats"].get("chunks_with_gt_rate", 0.0),
        "avg_gt_terms_per_chunk": applied["stats"].get("avg_gt_terms_per_chunk", 0.0),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
