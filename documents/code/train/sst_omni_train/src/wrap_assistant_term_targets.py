#!/usr/bin/env python3
"""Wrap GT target translations in assistant outputs.

This is a supervision-side salience ablation for Speech LLM SFT.  It keeps the
user prompt and term_map unchanged, reads ``gt_terms_by_chunk``, then wraps the
first exact future assistant occurrence of each GT target translation with a
configurable tag such as ``<term>...</term>``.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


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
    return [
        i
        for i, msg in enumerate(messages)
        if msg.get("role") == "user" and str(msg.get("content") or "").startswith("<audio>")
    ]


def _extract_translation(entry: Mapping[str, Any], lang_code: str) -> str:
    value = entry.get("translation") or entry.get("target_translation") or entry.get(lang_code)
    if value is None and isinstance(entry.get("target_translations"), Mapping):
        value = entry["target_translations"].get(lang_code)
    return str(value or "").strip()


def _norm_len(text: str) -> int:
    return len("".join(str(text or "").split()))


def _source_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", str(text or "").lower())


def _parse_token_set(text: str) -> set[str]:
    return {tok.strip().lower() for tok in str(text or "").split(",") if tok.strip()}


def _template_parts(tag_template: str) -> Tuple[str, str]:
    marker = "{translation}"
    if tag_template.count(marker) != 1:
        raise ValueError("--tag-template must contain exactly one {translation}")
    prefix, suffix = tag_template.split(marker)
    if not prefix and not suffix:
        raise ValueError("--tag-template must add a non-empty prefix or suffix")
    return prefix, suffix


def _replace_once_unwrapped(
    text: str,
    old: str,
    new: str,
    prefix: str,
    suffix: str,
    *,
    require_text_boundaries: bool,
) -> Tuple[str, bool, str]:
    start = 0
    while True:
        pos = text.find(old, start)
        if pos < 0:
            return text, False, "not_found"
        end = pos + len(old)
        if not _span_is_unwrapped(text, pos, end, prefix, suffix):
            start = pos + len(old)
            continue
        if require_text_boundaries and not _has_safe_text_boundaries(text, pos, end, old):
            start = pos + len(old)
            continue
        return text[:pos] + new + text[end:], True, "ok"


def _tag_ranges(text: str, prefix: str, suffix: str) -> List[Tuple[int, int]]:
    if not prefix or not suffix:
        return []
    ranges: List[Tuple[int, int]] = []
    search = 0
    while True:
        start = text.find(prefix, search)
        if start < 0:
            break
        end = text.find(suffix, start + len(prefix))
        if end < 0:
            ranges.append((start, len(text)))
            break
        end += len(suffix)
        ranges.append((start, end))
        search = end
    return ranges


def _span_overlaps_tagged_region(text: str, start: int, end: int, prefix: str, suffix: str) -> bool:
    for tag_start, tag_end in _tag_ranges(text, prefix, suffix):
        if start < tag_end and end > tag_start:
            return True
    return False


def _is_latin_alnum(ch: str) -> bool:
    return bool(ch) and ch.isalnum() and bool(re.match(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", ch))


def _has_safe_text_boundaries(text: str, start: int, end: int, replacement_text: str) -> bool:
    if start < 0 or end > len(text) or start >= end:
        return False
    if not replacement_text:
        return False
    first = replacement_text[0]
    last = replacement_text[-1]
    if start > 0 and _is_latin_alnum(text[start - 1]) and _is_latin_alnum(first):
        return False
    if end < len(text) and _is_latin_alnum(text[end]) and _is_latin_alnum(last):
        return False
    return True


def _span_is_unwrapped(text: str, start: int, end: int, prefix: str, suffix: str) -> bool:
    before = text[:start]
    after = text[end:]
    if prefix and before.endswith(prefix):
        return False
    if suffix and after.startswith(suffix):
        return False
    if prefix in text[start:end] or suffix in text[start:end]:
        return False
    if _span_overlaps_tagged_region(text, start, end, prefix, suffix):
        return False
    return True


def _remove_previous_assistant_suffix(
    messages: Sequence[MutableMapping[str, Any]],
    *,
    before_idx: int,
    suffix_text: str,
    min_chars: int,
    prefix: str,
    suffix: str,
) -> Optional[Dict[str, Any]]:
    if _norm_len(suffix_text) < min_chars:
        return None
    for msg_idx in range(before_idx - 1, -1, -1):
        msg = messages[msg_idx]
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        if not content.endswith(suffix_text):
            return None
        start = len(content) - len(suffix_text)
        end = len(content)
        if not _span_is_unwrapped(content, start, end, prefix, suffix):
            return None
        msg["content"] = content[:start]
        return {
            "assistant_msg_idx": msg_idx,
            "removed_prefix": suffix_text,
            "start": start,
            "end": end,
        }
    return None


def _best_rewrite_span(
    text: str,
    translation: str,
    *,
    prefix: str,
    suffix: str,
    min_target_chars: int,
    min_score: float,
    min_coverage: float,
    max_span_ratio: float,
    max_span_extra_chars: int,
    require_text_boundaries: bool,
) -> Optional[Dict[str, Any]]:
    if _norm_len(translation) < min_target_chars or not text:
        return None

    blocks = [
        block
        for block in SequenceMatcher(None, translation, text, autojunk=False).get_matching_blocks()
        if block.size > 0
    ]
    if not blocks:
        return None

    target_len = len(translation)
    max_span = max(target_len + max_span_extra_chars, int(target_len * max_span_ratio) + max_span_extra_chars)
    best: Optional[Dict[str, Any]] = None
    for left in range(len(blocks)):
        matched = 0
        start = blocks[left].b
        end = blocks[left].b + blocks[left].size
        for right in range(left, len(blocks)):
            block = blocks[right]
            matched += block.size
            start = min(start, block.b)
            end = max(end, block.b + block.size)
            span_len = end - start
            if span_len <= 0 or span_len > max_span:
                continue
            if not _span_is_unwrapped(text, start, end, prefix, suffix):
                continue
            if require_text_boundaries and not _has_safe_text_boundaries(text, start, end, translation):
                continue
            coverage = matched / max(1, target_len)
            score = (2.0 * matched) / max(1, target_len + span_len)
            if coverage < min_coverage or score < min_score:
                continue
            candidate = {
                "start": start,
                "end": end,
                "span": text[start:end],
                "matched_chars": matched,
                "coverage": coverage,
                "score": score,
                "target_start": min(blocks[k].a for k in range(left, right + 1)),
                "target_end": max(blocks[k].a + blocks[k].size for k in range(left, right + 1)),
            }
            if best is None:
                best = candidate
                continue
            key = (candidate["score"], candidate["matched_chars"], -len(candidate["span"]))
            best_key = (best["score"], best["matched_chars"], -len(best["span"]))
            if key > best_key:
                best = candidate
    return best


def _wrap_future_assistant(
    messages: Sequence[MutableMapping[str, Any]],
    *,
    start_idx: int,
    translation: str,
    wrapped: str,
    prefix: str,
    suffix: str,
    require_text_boundaries: bool,
) -> Optional[int]:
    for msg_idx in range(start_idx, len(messages)):
        msg = messages[msg_idx]
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        new_content, changed, _ = _replace_once_unwrapped(
            content,
            translation,
            wrapped,
            prefix,
            suffix,
            require_text_boundaries=require_text_boundaries,
        )
        if changed:
            msg["content"] = new_content
            return msg_idx
    return None


def _rewrite_future_assistant(
    messages: Sequence[MutableMapping[str, Any]],
    *,
    start_idx: int,
    translation: str,
    wrapped: str,
    prefix: str,
    suffix: str,
    min_target_chars: int,
    min_score: float,
    min_coverage: float,
    max_span_ratio: float,
    max_span_extra_chars: int,
    avoid_boundary_overlap: bool,
    delay_boundary_prefix: bool,
    delay_boundary_min_prefix_chars: int,
    require_text_boundaries: bool,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    for msg_idx in range(start_idx, len(messages)):
        msg = messages[msg_idx]
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        span = _best_rewrite_span(
            content,
            translation,
            prefix=prefix,
            suffix=suffix,
            min_target_chars=min_target_chars,
            min_score=min_score,
            min_coverage=min_coverage,
            max_span_ratio=max_span_ratio,
            max_span_extra_chars=max_span_extra_chars,
            require_text_boundaries=require_text_boundaries,
        )
        if span is None:
            continue
        if avoid_boundary_overlap:
            target_start = int(span.get("target_start") or 0)
            target_end = int(span.get("target_end") or len(translation))
            missing_prefix = translation[:target_start]
            missing_suffix = translation[target_end:]
            previous_assistant = "".join(
                str(m.get("content") or "")
                for m in messages[:msg_idx]
                if m.get("role") == "assistant"
            )
            following_assistant = "".join(
                str(m.get("content") or "")
                for m in messages[msg_idx + 1 :]
                if m.get("role") == "assistant"
            )
            if _norm_len(missing_prefix) >= 2 and missing_prefix in previous_assistant:
                if delay_boundary_prefix:
                    delayed = _remove_previous_assistant_suffix(
                        messages,
                        before_idx=msg_idx,
                        suffix_text=missing_prefix,
                        min_chars=delay_boundary_min_prefix_chars,
                        prefix=prefix,
                        suffix=suffix,
                    )
                    if delayed is None:
                        continue
                    span["delayed_boundary_prefix"] = delayed
                    span["boundary_delay_applied"] = True
                else:
                    continue
            if _norm_len(missing_suffix) >= 2 and missing_suffix in following_assistant:
                continue
        msg["content"] = content[: span["start"]] + wrapped + content[span["end"] :]
        return msg_idx, span
    return None


def _rewrite_boundary_only_future_assistant(
    messages: Sequence[MutableMapping[str, Any]],
    *,
    start_idx: int,
    translation: str,
    wrapped: str,
    prefix: str,
    suffix: str,
    delay_boundary_min_prefix_chars: int,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    """Repair terms split exactly across adjacent assistant boundaries only.

    This intentionally avoids the broader SequenceMatcher fallback used by
    ``_rewrite_future_assistant``.  It is for cases such as previous assistant
    ending in ``富兰克林`` and current assistant starting with ``·罗斯福``.
    """
    if _norm_len(translation) < 2:
        return None
    for msg_idx in range(start_idx, len(messages)):
        msg = messages[msg_idx]
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        if not content:
            continue
        # Prefer longer previous-side prefixes.  This keeps the current chunk
        # start as close to a real suffix as possible.
        for split in range(len(translation) - 1, 0, -1):
            missing_prefix = translation[:split]
            current_suffix = translation[split:]
            if _norm_len(missing_prefix) < delay_boundary_min_prefix_chars:
                continue
            if _norm_len(current_suffix) < 1:
                continue
            if not content.startswith(current_suffix):
                continue
            if not _span_is_unwrapped(content, 0, len(current_suffix), prefix, suffix):
                continue
            if not _has_safe_text_boundaries(content, 0, len(current_suffix), current_suffix):
                continue
            delayed = _remove_previous_assistant_suffix(
                messages,
                before_idx=msg_idx,
                suffix_text=missing_prefix,
                min_chars=delay_boundary_min_prefix_chars,
                prefix=prefix,
                suffix=suffix,
            )
            if delayed is None:
                continue
            msg["content"] = wrapped + content[len(current_suffix) :]
            span = {
                "start": 0,
                "end": len(current_suffix),
                "span": current_suffix,
                "matched_chars": len(current_suffix),
                "coverage": len(current_suffix) / max(1, len(translation)),
                "score": (2.0 * len(current_suffix)) / max(1, len(translation) + len(current_suffix)),
                "target_start": split,
                "target_end": len(translation),
                "delayed_boundary_prefix": delayed,
                "boundary_delay_applied": True,
                "boundary_only": True,
            }
            return msg_idx, span
    return None


def process_row(
    obj: Dict[str, Any],
    *,
    lineno: int,
    lang_code: str,
    tag_template: str,
    prefix: str,
    suffix: str,
    min_target_chars: int,
    max_tags_per_row: int,
    missing_gt_policy: str,
    enable_local_rewrite: bool,
    rewrite_min_target_chars: int,
    rewrite_min_score: float,
    rewrite_min_coverage: float,
    rewrite_max_span_ratio: float,
    rewrite_max_span_extra_chars: int,
    rewrite_avoid_boundary_overlap: bool,
    rewrite_delay_boundary_prefix: bool,
    rewrite_delay_boundary_min_prefix_chars: int,
    rewrite_require_text_boundaries: bool,
    exact_require_text_boundaries: bool,
    rewrite_boundary_only: bool,
    exclude_source_tokens: set[str],
    stats: Counter,
    samples: List[Dict[str, Any]],
    sample_count: int,
) -> Dict[str, Any]:
    messages = obj.get("messages")
    audios = obj.get("audios")
    gt_terms_by_chunk = obj.get("gt_terms_by_chunk")
    if not isinstance(messages, list):
        raise ValueError(f"Missing messages list at row {lineno}")
    if not isinstance(audios, list):
        raise ValueError(f"Missing audios list at row {lineno}")
    if not isinstance(gt_terms_by_chunk, list):
        if missing_gt_policy != "keep_unchanged":
            raise ValueError(f"Missing gt_terms_by_chunk list at row {lineno}")
        stats["rows_seen"] += 1
        stats["rows_missing_gt_terms_by_chunk"] += 1
        stats["chunks_missing_gt_terms_by_chunk"] += len(audios)
        obj["assistant_term_target_tagging"] = {
            "version": "v1",
            "source": "gt_terms_by_chunk",
            "lang_code": lang_code,
            "tag_template": tag_template,
            "min_target_chars": min_target_chars,
            "max_tags_per_row": max_tags_per_row,
            "tags_in_row": 0,
            "events": [],
            "user_input_unchanged": True,
            "term_map_unchanged": True,
            "missing_gt_terms_by_chunk": True,
            "missing_gt_policy": missing_gt_policy,
            "local_rewrite_enabled": enable_local_rewrite,
            "rewrite_avoid_boundary_overlap": rewrite_avoid_boundary_overlap,
            "rewrite_delay_boundary_prefix": rewrite_delay_boundary_prefix,
            "rewrite_delay_boundary_min_prefix_chars": rewrite_delay_boundary_min_prefix_chars,
            "rewrite_require_text_boundaries": rewrite_require_text_boundaries,
            "exact_require_text_boundaries": exact_require_text_boundaries,
            "rewrite_boundary_only": rewrite_boundary_only,
            "exclude_source_tokens": sorted(exclude_source_tokens),
        }
        return obj

    user_indices = _audio_user_indices(messages)
    if len(user_indices) != len(audios):
        raise ValueError(f"Row {lineno}: user audio messages={len(user_indices)} audios={len(audios)}")
    if len(gt_terms_by_chunk) != len(audios):
        raise ValueError(f"Row {lineno}: gt_terms_by_chunk={len(gt_terms_by_chunk)} audios={len(audios)}")

    stats["rows_seen"] += 1
    stats["chunks_total"] += len(audios)
    row_tags = 0
    row_events: List[Dict[str, Any]] = []

    for chunk_idx, user_idx in enumerate(user_indices):
        gt_terms = gt_terms_by_chunk[chunk_idx]
        if not isinstance(gt_terms, list):
            raise ValueError(f"Row {lineno} chunk {chunk_idx}: gt_terms_by_chunk entry must be a list")
        if gt_terms:
            stats["chunks_with_gt_terms"] += 1
        for term_pos, term_obj in enumerate(gt_terms):
            if not isinstance(term_obj, Mapping):
                continue
            term = str(term_obj.get("term") or term_obj.get("source") or "").strip()
            translation = _extract_translation(term_obj, lang_code)
            if not term or not translation:
                stats["skipped_missing_term_or_translation"] += 1
                continue
            stats["raw_gt_terms"] += 1
            source_tokens = set(_source_tokens(term))
            if exclude_source_tokens and source_tokens.intersection(exclude_source_tokens):
                stats["skipped_excluded_source_token"] += 1
                continue
            stats["candidate_gt_terms"] += 1
            if _norm_len(translation) < min_target_chars:
                stats["skipped_short_target"] += 1
                continue
            if max_tags_per_row > 0 and row_tags >= max_tags_per_row:
                stats["skipped_row_cap"] += 1
                continue

            wrapped = tag_template.format(translation=translation)
            assistant_idx = _wrap_future_assistant(
                messages,
                start_idx=user_idx + 1,
                translation=translation,
                wrapped=wrapped,
                prefix=prefix,
                suffix=suffix,
                require_text_boundaries=exact_require_text_boundaries,
            )
            replacement_type = "exact"
            rewrite_span: Optional[Dict[str, Any]] = None
            if assistant_idx is None and enable_local_rewrite:
                if rewrite_boundary_only:
                    rewritten = _rewrite_boundary_only_future_assistant(
                        messages,
                        start_idx=user_idx + 1,
                        translation=translation,
                        wrapped=wrapped,
                        prefix=prefix,
                        suffix=suffix,
                        delay_boundary_min_prefix_chars=rewrite_delay_boundary_min_prefix_chars,
                    )
                    if rewritten is None:
                        stats["skipped_missing_boundary_split"] += 1
                else:
                    rewritten = _rewrite_future_assistant(
                        messages,
                        start_idx=user_idx + 1,
                        translation=translation,
                        wrapped=wrapped,
                        prefix=prefix,
                        suffix=suffix,
                        min_target_chars=rewrite_min_target_chars,
                        min_score=rewrite_min_score,
                        min_coverage=rewrite_min_coverage,
                        max_span_ratio=rewrite_max_span_ratio,
                        max_span_extra_chars=rewrite_max_span_extra_chars,
                        avoid_boundary_overlap=rewrite_avoid_boundary_overlap,
                        delay_boundary_prefix=rewrite_delay_boundary_prefix,
                        delay_boundary_min_prefix_chars=rewrite_delay_boundary_min_prefix_chars,
                        require_text_boundaries=rewrite_require_text_boundaries,
                    )
                if rewritten is not None:
                    assistant_idx, rewrite_span = rewritten
                    replacement_type = "rewrite"
                    stats["assistant_tag_rewrite_replacements"] += 1
                    if rewrite_span.get("boundary_delay_applied"):
                        stats["assistant_tag_boundary_delay_replacements"] += 1
                    if rewrite_span.get("boundary_only"):
                        stats["assistant_tag_boundary_only_replacements"] += 1
                    else:
                        stats["assistant_tag_global_fuzzy_replacements"] += 1

            if assistant_idx is None:
                stats["skipped_missing_future_reference_exact"] += 1
                if enable_local_rewrite:
                    stats["skipped_missing_rewrite_span"] += 1
                continue

            row_tags += 1
            stats["assistant_tag_replacements"] += 1
            if replacement_type == "exact":
                stats["assistant_tag_exact_replacements"] += 1
            event = {
                "chunk_idx": chunk_idx,
                "term_pos": term_pos,
                "term": term,
                "translation": translation,
                "assistant_msg_idx": assistant_idx,
                "wrapped": wrapped,
                "replacement_type": replacement_type,
            }
            if rewrite_span is not None:
                event["rewrite_span"] = rewrite_span
            row_events.append(event)
            if len(samples) < sample_count:
                samples.append({"row_line": lineno, "utter_id": obj.get("utter_id"), **event})

    if row_tags:
        stats["rows_with_assistant_tags"] += 1
    obj["assistant_term_target_tagging"] = {
        "version": "v1",
        "source": "gt_terms_by_chunk",
        "lang_code": lang_code,
        "tag_template": tag_template,
        "min_target_chars": min_target_chars,
        "max_tags_per_row": max_tags_per_row,
        "tags_in_row": row_tags,
        "events": row_events[:80],
        "user_input_unchanged": True,
        "term_map_unchanged": True,
        "local_rewrite_enabled": enable_local_rewrite,
        "rewrite_min_target_chars": rewrite_min_target_chars,
        "rewrite_min_score": rewrite_min_score,
        "rewrite_min_coverage": rewrite_min_coverage,
        "rewrite_max_span_ratio": rewrite_max_span_ratio,
        "rewrite_max_span_extra_chars": rewrite_max_span_extra_chars,
        "rewrite_avoid_boundary_overlap": rewrite_avoid_boundary_overlap,
        "rewrite_delay_boundary_prefix": rewrite_delay_boundary_prefix,
        "rewrite_delay_boundary_min_prefix_chars": rewrite_delay_boundary_min_prefix_chars,
        "rewrite_require_text_boundaries": rewrite_require_text_boundaries,
        "exact_require_text_boundaries": exact_require_text_boundaries,
        "rewrite_boundary_only": rewrite_boundary_only,
        "exclude_source_tokens": sorted(exclude_source_tokens),
    }
    return obj


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, default=None)
    parser.add_argument("--lang-code", default="zh")
    parser.add_argument("--tag-template", default="<term>{translation}</term>")
    parser.add_argument("--min-target-chars", type=int, default=2)
    parser.add_argument("--max-tags-per-row", type=int, default=16)
    parser.add_argument("--enable-local-rewrite", action="store_true")
    parser.add_argument("--rewrite-min-target-chars", type=int, default=4)
    parser.add_argument("--rewrite-min-score", type=float, default=0.58)
    parser.add_argument("--rewrite-min-coverage", type=float, default=0.40)
    parser.add_argument("--rewrite-max-span-ratio", type=float, default=1.60)
    parser.add_argument("--rewrite-max-span-extra-chars", type=int, default=4)
    parser.add_argument("--rewrite-avoid-boundary-overlap", action="store_true")
    parser.add_argument("--rewrite-delay-boundary-prefix", action="store_true")
    parser.add_argument("--rewrite-delay-boundary-min-prefix-chars", type=int, default=2)
    parser.add_argument(
        "--rewrite-require-text-boundaries",
        action="store_true",
        help="Only rewrite spans whose replacement will not split Latin-letter/digit word boundaries.",
    )
    parser.add_argument(
        "--exact-require-text-boundaries",
        action="store_true",
        help=(
            "Only exact-wrap target translations when the matched span is not "
            "inside a Latin-letter/digit word. This prevents tags such as "
            "Mäuse</term>n."
        ),
    )
    parser.add_argument(
        "--rewrite-boundary-only",
        action="store_true",
        help=(
            "When exact wrapping fails, only repair translations split exactly "
            "across adjacent assistant boundaries.  This disables general "
            "SequenceMatcher local fuzzy rewriting."
        ),
    )
    parser.add_argument(
        "--exclude-source-tokens",
        default="",
        help="Comma-separated source-side tokens; GT terms containing any of them are skipped.",
    )
    parser.add_argument(
        "--missing-gt-policy",
        choices=["error", "keep_unchanged"],
        default="error",
        help="How to handle legacy rows without gt_terms_by_chunk.",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=80)
    args = parser.parse_args()

    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if args.min_target_chars < 1:
        raise ValueError("--min-target-chars must be >= 1")
    exclude_source_tokens = _parse_token_set(args.exclude_source_tokens)
    prefix, suffix = _template_parts(args.tag_template)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    samples: List[Dict[str, Any]] = []

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(args.output_jsonl.parent),
        prefix=args.output_jsonl.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            for lineno, obj in _iter_jsonl(args.input_jsonl):
                if args.max_rows > 0 and stats["rows_written"] >= args.max_rows:
                    break
                out_obj = process_row(
                    obj,
                    lineno=lineno,
                    lang_code=args.lang_code,
                    tag_template=args.tag_template,
                    prefix=prefix,
                    suffix=suffix,
                    min_target_chars=args.min_target_chars,
                    max_tags_per_row=args.max_tags_per_row,
                    missing_gt_policy=args.missing_gt_policy,
                    enable_local_rewrite=args.enable_local_rewrite,
                    rewrite_min_target_chars=args.rewrite_min_target_chars,
                    rewrite_min_score=args.rewrite_min_score,
                    rewrite_min_coverage=args.rewrite_min_coverage,
                    rewrite_max_span_ratio=args.rewrite_max_span_ratio,
                    rewrite_max_span_extra_chars=args.rewrite_max_span_extra_chars,
                    rewrite_avoid_boundary_overlap=args.rewrite_avoid_boundary_overlap,
                    rewrite_delay_boundary_prefix=args.rewrite_delay_boundary_prefix,
                    rewrite_delay_boundary_min_prefix_chars=args.rewrite_delay_boundary_min_prefix_chars,
                    rewrite_require_text_boundaries=args.rewrite_require_text_boundaries,
                    exact_require_text_boundaries=args.exact_require_text_boundaries,
                    rewrite_boundary_only=args.rewrite_boundary_only,
                    exclude_source_tokens=exclude_source_tokens,
                    stats=stats,
                    samples=samples,
                    sample_count=args.sample_count,
                )
                tmp.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    tmp_path.replace(args.output_jsonl)

    stats_dict: Dict[str, Any] = dict(stats)
    stats_dict.update(
        {
            "input_jsonl": str(args.input_jsonl),
            "output_jsonl": str(args.output_jsonl),
            "lang_code": args.lang_code,
            "tag_template": args.tag_template,
            "min_target_chars": args.min_target_chars,
            "max_tags_per_row": args.max_tags_per_row,
            "enable_local_rewrite": args.enable_local_rewrite,
            "rewrite_min_target_chars": args.rewrite_min_target_chars,
            "rewrite_min_score": args.rewrite_min_score,
            "rewrite_min_coverage": args.rewrite_min_coverage,
            "rewrite_max_span_ratio": args.rewrite_max_span_ratio,
            "rewrite_max_span_extra_chars": args.rewrite_max_span_extra_chars,
            "rewrite_avoid_boundary_overlap": args.rewrite_avoid_boundary_overlap,
            "rewrite_delay_boundary_prefix": args.rewrite_delay_boundary_prefix,
            "rewrite_delay_boundary_min_prefix_chars": args.rewrite_delay_boundary_min_prefix_chars,
            "rewrite_require_text_boundaries": args.rewrite_require_text_boundaries,
            "exact_require_text_boundaries": args.exact_require_text_boundaries,
            "rewrite_boundary_only": args.rewrite_boundary_only,
            "exclude_source_tokens": sorted(exclude_source_tokens),
            "candidate_gt_terms_after_min_len": stats["candidate_gt_terms"] - stats["skipped_short_target"],
            "assistant_tag_rate_over_gt_terms": _rate(stats["assistant_tag_replacements"], stats["candidate_gt_terms"]),
            "assistant_tag_rate_after_min_len": _rate(
                stats["assistant_tag_replacements"],
                stats["candidate_gt_terms"] - stats["skipped_short_target"],
            ),
            "rows_with_tag_rate": _rate(stats["rows_with_assistant_tags"], stats["rows_seen"]),
        }
    )
    args.stats_json.write_text(
        json.dumps(stats_dict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.sample_json:
        args.sample_json.write_text(
            json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(stats_dict, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
