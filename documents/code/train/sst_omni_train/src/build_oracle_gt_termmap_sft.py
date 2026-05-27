#!/usr/bin/env python3
"""Build Speech LLM SFT JSONL with oracle GT-only term_map entries.

This converts an existing streaming SFT JSONL that contains
``gt_terms_by_chunk`` into a clean oracle term-map dataset:

* chunks with GT terms get exactly those terms in ``term_map``;
* chunks without GT terms get ``term_map:NONE`` by default;
* existing noisy/retriever term maps in user messages are overwritten.

The script is intentionally fail-fast because this dataset is used as an
upper-bound control for whether the Speech LLM can learn to use terminology.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


SYSTEM_PROMPT_BY_LANG = {
    "zh": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate "
        "the audio into Chinese text. Use the 'term_map' as a reference for "
        "terminology if provided."
    ),
    "de": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate "
        "the audio into German text. Use the 'term_map' as a reference for "
        "terminology if provided."
    ),
    "ja": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate "
        "the audio into Japanese text. Use the 'term_map' as a reference for "
        "terminology if provided."
    ),
}


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


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def _extract_translation(term_entry: Mapping[str, Any], lang_code: str) -> str:
    value = term_entry.get(lang_code)
    if value is None:
        value = term_entry.get("translation") or term_entry.get("target_translation")
    if value is None and isinstance(term_entry.get("target_translations"), Mapping):
        value = term_entry["target_translations"].get(lang_code)
    return str(value or "").strip()


def _normalize_gt_terms(raw_terms: Any, lang_code: str) -> List[Dict[str, str]]:
    if raw_terms is None:
        return []
    if not isinstance(raw_terms, list):
        raise ValueError(f"gt_terms_by_chunk entry must be a list, got {type(raw_terms).__name__}")

    out: List[Dict[str, str]] = []
    seen = set()
    for idx, item in enumerate(raw_terms):
        if not isinstance(item, Mapping):
            raise ValueError(f"GT term entry {idx} is not an object")
        term = str(item.get("term") or item.get("source") or "").strip()
        translation = _extract_translation(item, lang_code)
        if not term or not translation:
            raise ValueError(f"GT term entry {idx} missing term or {lang_code} translation: {item}")
        key = _term_key(term)
        if key in seen:
            continue
        seen.add(key)
        out.append({"term": term, "translation": translation})
    return out


def _format_term_map(terms: List[Dict[str, str]], no_gt_mode: str) -> str:
    if not terms:
        if no_gt_mode == "term_map_none":
            return "<audio>\n\nterm_map:NONE"
        if no_gt_mode == "audio_only":
            return "<audio>"
        raise ValueError(f"Unsupported no_gt_mode={no_gt_mode}")
    lines = ["<audio>", "", "term_map:"]
    for item in terms:
        lines.append(f"{item['term']}={item['translation']}")
    return "\n".join(lines)


def _is_audio_user_message(msg: Mapping[str, Any]) -> bool:
    return msg.get("role") == "user" and "<audio>" in str(msg.get("content") or "")


def _count_termmap_entries(content: str) -> int:
    if "term_map:" not in content:
        return 0
    tail = content.split("term_map:", 1)[1].strip()
    if not tail or tail.upper() == "NONE":
        return 0
    return sum(1 for line in tail.splitlines() if line.strip() and "=" in line)


def _percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((p / 100.0) * (len(values) - 1)))
    idx = max(0, min(idx, len(values) - 1))
    return float(values[idx])


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "lang_code": args.lang_code,
        "no_gt_mode": args.no_gt_mode,
        "max_conversations": args.max_conversations,
        "input_rows_seen": 0,
        "rows": 0,
        "dropped_rows": 0,
        "dropped_missing_gt_terms_by_chunk": 0,
        "dropped_mismatched_gt_chunk_count": 0,
        "audio_user_chunks": 0,
        "gt_chunks": 0,
        "no_gt_chunks": 0,
        "gt_terms_total": 0,
        "rows_with_mismatched_gt_chunk_count": 0,
        "rows_with_missing_gt_terms_by_chunk": 0,
    }
    termmap_sizes: List[int] = []
    samples: List[Dict[str, Any]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as f_out:
        for _lineno, obj in _iter_jsonl(args.input_jsonl):
            if 0 < args.max_conversations <= stats["rows"]:
                break
            stats["input_rows_seen"] += 1

            messages = obj.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Row {stats['rows']} missing non-empty messages list")

            gt_by_chunk = obj.get("gt_terms_by_chunk")
            if gt_by_chunk is None:
                stats["rows_with_missing_gt_terms_by_chunk"] += 1
                if args.drop_missing_gt_rows:
                    stats["dropped_rows"] += 1
                    stats["dropped_missing_gt_terms_by_chunk"] += 1
                    continue
                raise ValueError(
                    f"Row {stats['rows']} missing gt_terms_by_chunk; "
                    "oracle GT SFT data cannot be built from this source"
                )
            if not isinstance(gt_by_chunk, list):
                raise ValueError(f"Row {stats['rows']} gt_terms_by_chunk is not a list")

            if messages[0].get("role") == "system":
                messages[0]["content"] = SYSTEM_PROMPT_BY_LANG[args.lang_code]

            audio_user_indices = [
                idx for idx, msg in enumerate(messages)
                if isinstance(msg, Mapping) and _is_audio_user_message(msg)
            ]
            if len(audio_user_indices) != len(gt_by_chunk):
                stats["rows_with_mismatched_gt_chunk_count"] += 1
                if args.drop_mismatched_gt_rows:
                    stats["dropped_rows"] += 1
                    stats["dropped_mismatched_gt_chunk_count"] += 1
                    continue
                raise ValueError(
                    f"Row {stats['rows']} has {len(audio_user_indices)} audio user chunks "
                    f"but {len(gt_by_chunk)} gt_terms_by_chunk entries"
                )

            for chunk_idx, msg_idx in enumerate(audio_user_indices):
                terms = _normalize_gt_terms(gt_by_chunk[chunk_idx], args.lang_code)
                messages[msg_idx]["content"] = _format_term_map(terms, args.no_gt_mode)
                stats["audio_user_chunks"] += 1
                stats["gt_terms_total"] += len(terms)
                if terms:
                    stats["gt_chunks"] += 1
                else:
                    stats["no_gt_chunks"] += 1
                termmap_sizes.append(_count_termmap_entries(messages[msg_idx]["content"]))

            if len(samples) < args.sample_count:
                first_user = next((messages[i]["content"] for i in audio_user_indices), "")
                first_gt_idx = next(
                    (i for i, msg_idx in enumerate(audio_user_indices)
                     if _count_termmap_entries(messages[msg_idx]["content"]) > 0),
                    None,
                )
                samples.append({
                    "row": stats["rows"],
                    "utter_id": obj.get("utter_id"),
                    "audio_chunks": len(audio_user_indices),
                    "gt_chunk_count": sum(1 for x in gt_by_chunk if x),
                    "first_user": first_user,
                    "first_gt_user": (
                        messages[audio_user_indices[first_gt_idx]]["content"]
                        if first_gt_idx is not None else ""
                    ),
                })

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["rows"] += 1

    nonzero_sizes = [x for x in termmap_sizes if x > 0]
    stats.update({
        "gt_chunk_ratio": stats["gt_chunks"] / stats["audio_user_chunks"]
        if stats["audio_user_chunks"] else 0.0,
        "avg_gt_terms_per_audio_chunk": stats["gt_terms_total"] / stats["audio_user_chunks"]
        if stats["audio_user_chunks"] else 0.0,
        "avg_term_map_entries_nonempty": sum(nonzero_sizes) / len(nonzero_sizes)
        if nonzero_sizes else 0.0,
        "p50_term_map_entries_nonempty": _percentile(nonzero_sizes, 50),
        "p90_term_map_entries_nonempty": _percentile(nonzero_sizes, 90),
        "max_term_map_entries": max(termmap_sizes) if termmap_sizes else 0,
        "term_map_size_hist_top20": dict(Counter(termmap_sizes).most_common(20)),
        "policy": {
            "has_gt_chunk": "inject all unique gt_terms_by_chunk entries only",
            "no_gt_chunk": args.no_gt_mode,
            "existing_user_term_map": "overwritten",
            "system_prompt": "language-specific term_map-aware simultaneous interpreter prompt",
            "row_filtering": {
                "drop_missing_gt_rows": bool(args.drop_missing_gt_rows),
                "drop_mismatched_gt_rows": bool(args.drop_mismatched_gt_rows),
                "reason": "oracle GT term_map requires row-level gt_terms_by_chunk aligned to audio user chunks",
            },
        },
    })
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sample_json:
        args.sample_json.write_text(
            json.dumps(samples, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, default=None)
    parser.add_argument("--lang-code", choices=sorted(SYSTEM_PROMPT_BY_LANG), default="zh")
    parser.add_argument(
        "--no-gt-mode",
        choices=["term_map_none", "audio_only"],
        default="term_map_none",
    )
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument(
        "--drop-missing-gt-rows",
        action="store_true",
        help="Filter rows missing gt_terms_by_chunk and report the count.",
    )
    parser.add_argument(
        "--drop-mismatched-gt-rows",
        action="store_true",
        help="Filter rows where gt_terms_by_chunk length does not match audio user chunks.",
    )
    args = parser.parse_args()

    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    stats = build_dataset(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
