#!/usr/bin/env python3
"""Derive chunk-level GT terms from existing term_map entries.

This is for legacy Speech LLM SFT JSONL files that have user-side ``term_map``
entries but no ``gt_terms_by_chunk``.  For each audio chunk, we keep a term-map
entry as chunk-level GT only when its target translation is supported by the
current or future assistant text via exact substring match or an optional local
fuzzy match.  The output keeps messages unchanged and only adds
``gt_terms_by_chunk`` plus provenance metadata.
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


_TERM_MAP_MARKER = "term_map:"


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
        idx
        for idx, msg in enumerate(messages)
        if msg.get("role") == "user" and str(msg.get("content") or "").startswith("<audio>")
    ]


def _norm_len(text: str) -> int:
    return len("".join(str(text or "").split()))


def _source_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", str(text or "").lower())


def _parse_token_set(text: str) -> set[str]:
    return {tok.strip().lower() for tok in str(text or "").split(",") if tok.strip()}


def _parse_term_map(content: str) -> List[Dict[str, str]]:
    content = str(content or "")
    marker_idx = content.find(_TERM_MAP_MARKER)
    if marker_idx < 0 or "term_map:NONE" in content:
        return []
    body = content[marker_idx + len(_TERM_MAP_MARKER) :].strip()
    if not body:
        return []
    entries: List[Dict[str, str]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        source, target = line.split("=", 1)
        source = source.strip()
        target = target.strip()
        if source and target:
            entries.append({"term": source, "translation": target})
    return entries


def _future_assistant_text(messages: Sequence[Mapping[str, Any]], *, start_idx: int) -> str:
    return "".join(
        str(msg.get("content") or "")
        for msg in messages[start_idx:]
        if msg.get("role") == "assistant"
    )


def _best_fuzzy_score(text: str, target: str, *, max_span_ratio: float, max_span_extra_chars: int) -> float:
    if not text or not target:
        return 0.0
    blocks = [
        block
        for block in SequenceMatcher(None, target, text, autojunk=False).get_matching_blocks()
        if block.size > 0
    ]
    if not blocks:
        return 0.0
    target_len = len(target)
    max_span = max(target_len + max_span_extra_chars, int(target_len * max_span_ratio) + max_span_extra_chars)
    best = 0.0
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
            score = (2.0 * matched) / max(1, target_len + span_len)
            if score > best:
                best = score
    return best


def process_row(
    obj: MutableMapping[str, Any],
    *,
    lineno: int,
    lang_code: str,
    min_target_chars: int,
    exclude_source_tokens: set[str],
    fuzzy_match: bool,
    fuzzy_min_target_chars: int,
    fuzzy_min_score: float,
    fuzzy_max_span_ratio: float,
    fuzzy_max_span_extra_chars: int,
    max_terms_per_chunk: int,
    stats: Counter,
    samples: List[Dict[str, Any]],
    sample_count: int,
) -> MutableMapping[str, Any]:
    messages = obj.get("messages")
    audios = obj.get("audios")
    if not isinstance(messages, list):
        raise ValueError(f"Missing messages list at row {lineno}")
    if not isinstance(audios, list):
        raise ValueError(f"Missing audios list at row {lineno}")
    user_indices = _audio_user_indices(messages)
    if len(user_indices) != len(audios):
        raise ValueError(f"Row {lineno}: user audio messages={len(user_indices)} audios={len(audios)}")

    row_gt: List[List[Dict[str, Any]]] = []
    row_exact = 0
    row_fuzzy = 0
    row_termmap_entries = 0
    for chunk_idx, user_idx in enumerate(user_indices):
        msg = messages[user_idx]
        entries = _parse_term_map(str(msg.get("content") or ""))
        row_termmap_entries += len(entries)
        stats["chunks_total"] += 1
        stats["termmap_entries"] += len(entries)
        if entries:
            stats["chunks_with_termmap"] += 1
        future_text = _future_assistant_text(messages, start_idx=user_idx + 1)
        chunk_terms: List[Dict[str, Any]] = []
        seen = set()
        for entry in entries:
            source = entry["term"]
            target = entry["translation"]
            if _norm_len(target) < min_target_chars:
                stats["skipped_short_target"] += 1
                continue
            source_token_set = set(_source_tokens(source))
            if exclude_source_tokens and source_token_set.intersection(exclude_source_tokens):
                stats["skipped_excluded_source_token"] += 1
                continue
            key = (source.casefold(), target)
            if key in seen:
                stats["skipped_duplicate_in_chunk"] += 1
                continue
            match_type = ""
            fuzzy_score = 0.0
            if target in future_text:
                match_type = "exact"
                row_exact += 1
                stats["selected_exact"] += 1
            elif fuzzy_match and _norm_len(target) >= fuzzy_min_target_chars:
                fuzzy_score = _best_fuzzy_score(
                    future_text,
                    target,
                    max_span_ratio=fuzzy_max_span_ratio,
                    max_span_extra_chars=fuzzy_max_span_extra_chars,
                )
                if fuzzy_score >= fuzzy_min_score:
                    match_type = "fuzzy"
                    row_fuzzy += 1
                    stats["selected_fuzzy"] += 1
            if not match_type:
                stats["skipped_no_future_match"] += 1
                continue
            item: Dict[str, Any] = {
                "term": source,
                lang_code: target,
                "translation": target,
                "target_translation": target,
                "match_type": match_type,
            }
            if match_type == "fuzzy":
                item["fuzzy_score"] = round(float(fuzzy_score), 6)
            chunk_terms.append(item)
            seen.add(key)
            if max_terms_per_chunk > 0 and len(chunk_terms) >= max_terms_per_chunk:
                stats["hit_max_terms_per_chunk"] += 1
                break
        if chunk_terms:
            stats["chunks_with_gt_terms"] += 1
            stats["gt_terms_total"] += len(chunk_terms)
            if len(samples) < sample_count:
                samples.append(
                    {
                        "row_line": lineno,
                        "utter_id": obj.get("utter_id"),
                        "chunk_idx": chunk_idx,
                        "gt_terms": chunk_terms[:8],
                        "termmap_entries": len(entries),
                    }
                )
        row_gt.append(chunk_terms)

    stats["rows_seen"] += 1
    stats["audio_chunks"] += len(audios)
    stats["rows_with_any_gt"] += int(any(row_gt))
    stats["row_termmap_entries"] += row_termmap_entries
    stats["row_selected_exact"] += row_exact
    stats["row_selected_fuzzy"] += row_fuzzy

    obj["gt_terms_by_chunk"] = row_gt
    obj["derived_gt_terms_by_chunk_policy"] = {
        "version": "termmap_future_assistant_match_v1",
        "source": "existing user term_map entries",
        "lang_code": lang_code,
        "target_match_policy": "future_assistant_exact_or_fuzzy" if fuzzy_match else "future_assistant_exact",
        "target_match_reference": "assistant messages from current audio response through conversation end",
        "target_match_exact_substring": True,
        "fuzzy_match": fuzzy_match,
        "fuzzy_min_score": fuzzy_min_score,
        "fuzzy_max_span_ratio": fuzzy_max_span_ratio,
        "fuzzy_max_span_extra_chars": fuzzy_max_span_extra_chars,
        "max_terms_per_chunk": max_terms_per_chunk,
        "term_map_unchanged": True,
    }
    return obj


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--stats-json", type=Path, required=True)
    ap.add_argument("--sample-json", type=Path, default=None)
    ap.add_argument("--lang-code", required=True)
    ap.add_argument("--min-target-chars", type=int, default=2)
    ap.add_argument("--exclude-source-tokens", default="")
    ap.add_argument("--fuzzy-match", action="store_true")
    ap.add_argument("--fuzzy-min-target-chars", type=int, default=4)
    ap.add_argument("--fuzzy-min-score", type=float, default=0.58)
    ap.add_argument("--fuzzy-max-span-ratio", type=float, default=1.60)
    ap.add_argument("--fuzzy-max-span-extra-chars", type=int, default=4)
    ap.add_argument("--max-terms-per-chunk", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--sample-count", type=int, default=100)
    args = ap.parse_args()

    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if args.lang_code not in {"zh", "ja", "de"}:
        raise ValueError(f"Unsupported --lang-code: {args.lang_code}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    exclude_source_tokens = _parse_token_set(args.exclude_source_tokens)
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
                    min_target_chars=args.min_target_chars,
                    exclude_source_tokens=exclude_source_tokens,
                    fuzzy_match=args.fuzzy_match,
                    fuzzy_min_target_chars=args.fuzzy_min_target_chars,
                    fuzzy_min_score=args.fuzzy_min_score,
                    fuzzy_max_span_ratio=args.fuzzy_max_span_ratio,
                    fuzzy_max_span_extra_chars=args.fuzzy_max_span_extra_chars,
                    max_terms_per_chunk=args.max_terms_per_chunk,
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
            "min_target_chars": args.min_target_chars,
            "exclude_source_tokens": sorted(exclude_source_tokens),
            "fuzzy_match": args.fuzzy_match,
            "fuzzy_min_target_chars": args.fuzzy_min_target_chars,
            "fuzzy_min_score": args.fuzzy_min_score,
            "fuzzy_max_span_ratio": args.fuzzy_max_span_ratio,
            "fuzzy_max_span_extra_chars": args.fuzzy_max_span_extra_chars,
            "max_terms_per_chunk": args.max_terms_per_chunk,
            "selected_terms_total": stats["selected_exact"] + stats["selected_fuzzy"],
            "selected_over_termmap_rate": _rate(stats["selected_exact"] + stats["selected_fuzzy"], stats["termmap_entries"]),
            "gt_chunk_rate": _rate(stats["chunks_with_gt_terms"], stats["chunks_total"]),
            "avg_gt_terms_per_chunk": _rate(stats["gt_terms_total"], stats["chunks_total"]),
            "avg_termmap_entries_per_chunk": _rate(stats["termmap_entries"], stats["chunks_total"]),
            "rows_with_any_gt_rate": _rate(stats["rows_with_any_gt"], stats["rows_seen"]),
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
