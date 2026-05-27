#!/usr/bin/env python3
"""Build dense top-k retriever term_map SFT data for Speech LLM.

Input is a JSONL already containing per-chunk retriever ``term_map`` entries,
typically produced by ``build_retriever_timeline_termmap_sft.py`` with
``--score-threshold`` disabled.  This script reshapes that raw top-k stream into
the two current curricula:

* V9: keep retriever top-k terms, override matched GT translations, and backfill
  exact-reference GT terms up to a hard cap.
* V10: same as V9, but wrap GT target translations with random marker strings in
  both ``term_map`` and assistant targets, forcing the SFT objective to learn
  literal term-map adoption.

The exact-reference check is a Python substring check against the full assistant
target text.  If the target translation is not an exact substring, it is not
trusted as GT/backfill supervision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SYSTEM_PROMPT_BY_LANG = {
    "zh": (
        "You are a professional simultaneous interpreter. You will be given "
        "chunks of English audio and you need to translate the audio into "
        "Chinese text. Use the 'term_map' as a reference for terminology if "
        "provided. Use only terms that are supported by the speech, and ignore "
        "irrelevant or unsupported term_map entries."
    ),
}

MARKER_PAIRS = [
    ("@@A", "A@@"),
    ("##B", "B##"),
    ("%%C", "C%%"),
    ("&&D", "D&&"),
    ("[[E", "E]]"),
    ("{{F", "F}}"),
    ("<<G", "G>>"),
    ("__H", "H__"),
]


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{lineno}")
            yield lineno, obj


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def _extract_translation(entry: Mapping[str, Any], lang_code: str) -> str:
    value = entry.get("translation") or entry.get("target_translation") or entry.get(lang_code)
    if value is None and isinstance(entry.get("target_translations"), Mapping):
        value = entry["target_translations"].get(lang_code)
    return str(value or "").strip()


def _audio_user_indices(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    return [
        idx for idx, msg in enumerate(messages)
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or "")
    ]


def _assistant_target_text(messages: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(
        str(msg.get("content") or "")
        for msg in messages
        if msg.get("role") == "assistant"
    )


def _parse_term_map(content: str) -> List[Dict[str, str]]:
    if "term_map:" not in content:
        return []
    tail = content.split("term_map:", 1)[1].strip()
    if not tail or tail.upper() == "NONE":
        return []
    out: List[Dict[str, str]] = []
    seen = set()
    for raw in tail.splitlines():
        line = raw.strip()
        if not line or line.upper() == "NONE":
            continue
        line = re.sub(r"^\[TERM\]\s*", "", line)
        line = re.sub(r"\s*\[/TERM\]$", "", line)
        line = re.sub(r"^<term>\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\s*</term>$", "", line, flags=re.IGNORECASE)
        if "=>" in line:
            term, translation = line.split("=>", 1)
        elif "=" in line:
            term, translation = line.split("=", 1)
        else:
            continue
        term = term.strip()
        translation = translation.strip()
        key = _term_key(term)
        if not term or not translation or not key or key in seen:
            continue
        seen.add(key)
        out.append({"term": term, "translation": translation, "key": key})
    return out


def _normalize_gt_terms(raw_terms: Any, lang_code: str) -> List[Dict[str, str]]:
    if raw_terms is None:
        return []
    if not isinstance(raw_terms, list):
        raise ValueError(f"gt_terms_by_chunk entry must be a list, got {type(raw_terms).__name__}")
    out: List[Dict[str, str]] = []
    seen = set()
    for item in raw_terms:
        if not isinstance(item, Mapping):
            raise ValueError("gt_terms_by_chunk term entry must be an object")
        term = str(item.get("term") or item.get("source") or "").strip()
        translation = _extract_translation(item, lang_code)
        key = _term_key(term)
        if not term or not translation or not key or key in seen:
            continue
        seen.add(key)
        out.append({"term": term, "translation": translation, "key": key})
    return out


def _filter_exact_ref_gt(
    gt_terms: Sequence[Mapping[str, str]],
    *,
    full_target_text: str,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for item in gt_terms:
        translation = str(item.get("translation") or "").strip()
        key = str(item.get("key") or "")
        if translation and translation in full_target_text and key and key not in seen:
            seen.add(key)
            out.append(dict(item))
    return out


def _marker_for(seed: int, row_key: str, term_key: str, translation: str) -> Tuple[str, str]:
    h = hashlib.sha256(f"{seed}|{row_key}|{term_key}|{translation}".encode("utf-8")).hexdigest()
    return MARKER_PAIRS[int(h[:8], 16) % len(MARKER_PAIRS)]


def _format_term_map(items: Sequence[Mapping[str, str]]) -> str:
    if not items:
        return "<audio>\n\nterm_map:NONE"
    lines = ["<audio>", "", "term_map:"]
    seen = set()
    for item in items:
        term = str(item.get("term") or "").replace("\n", " ").strip()
        translation = str(item.get("translation") or "").replace("\n", " ").strip()
        key = _term_key(term)
        if not term or not translation or not key or key in seen:
            continue
        seen.add(key)
        lines.append(f"{term}={translation}")
    if len(lines) == 3:
        return "<audio>\n\nterm_map:NONE"
    return "\n".join(lines)


def _build_dense_items(
    *,
    retrieved: Sequence[Mapping[str, str]],
    gt_terms: Sequence[Mapping[str, str]],
    max_terms: int,
) -> Tuple[List[Dict[str, str]], int, int]:
    gt_by_key = {str(x["key"]): dict(x) for x in gt_terms}
    combined: List[Dict[str, str]] = []
    seen = set()
    retrieved_gt = 0
    for item in retrieved:
        key = str(item.get("key") or _term_key(str(item.get("term") or "")))
        if not key or key in seen:
            continue
        seen.add(key)
        cur = {
            "term": str(item.get("term") or "").strip(),
            "translation": str(item.get("translation") or "").strip(),
            "key": key,
        }
        if key in gt_by_key:
            cur["translation"] = gt_by_key[key]["translation"]
            cur["is_gt"] = "1"
            retrieved_gt += 1
        else:
            cur["is_gt"] = "0"
        if cur["term"] and cur["translation"]:
            combined.append(cur)

    backfilled = 0
    for gt in gt_terms:
        key = str(gt["key"])
        if key in seen:
            continue
        seen.add(key)
        combined.append({
            "term": str(gt["term"]),
            "translation": str(gt["translation"]),
            "key": key,
            "is_gt": "1",
            "backfilled_gt": "1",
        })
        backfilled += 1

    if max_terms > 0 and len(combined) > max_terms:
        gt_items = [x for x in combined if x.get("is_gt") == "1"]
        non_gt = [x for x in combined if x.get("is_gt") != "1"]
        combined = gt_items[:max_terms] + non_gt[: max(0, max_terms - len(gt_items))]
    return combined, retrieved_gt, backfilled


def _apply_marker_curriculum(
    *,
    items_by_chunk: Sequence[List[Dict[str, str]]],
    messages: Sequence[Dict[str, Any]],
    row_key: str,
    seed: int,
) -> Tuple[int, int, int]:
    translation_to_marked: Dict[str, str] = {}
    marker_terms = 0
    for items in items_by_chunk:
        for item in items:
            if item.get("is_gt") != "1":
                continue
            translation = str(item.get("translation") or "")
            term_key = str(item.get("key") or _term_key(str(item.get("term") or "")))
            if not translation or not term_key:
                continue
            if translation not in translation_to_marked:
                prefix, suffix = _marker_for(seed, row_key, term_key, translation)
                translation_to_marked[translation] = f"{prefix}{translation}{suffix}"
            item["translation"] = translation_to_marked[translation]
            item["marker_augmented_gt"] = "1"
            marker_terms += 1

    replacements = 0
    replaced_messages = 0
    # Longer translations first prevents partial replacement of nested terms.
    ordered = sorted(translation_to_marked.items(), key=lambda kv: len(kv[0]), reverse=True)
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        new_content = content
        for original, marked in ordered:
            count = new_content.count(original)
            if count:
                replacements += count
                new_content = new_content.replace(original, marked)
        if new_content != content:
            replaced_messages += 1
            msg["content"] = new_content
    return marker_terms, replacements, replaced_messages


def build(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "variant": args.variant,
        "lang_code": args.lang_code,
        "max_terms": args.max_terms,
        "seed": args.seed,
        "rows_seen": 0,
        "rows_written": 0,
        "chunks": 0,
        "raw_gt_terms_total": 0,
        "exact_ref_gt_terms_total": 0,
        "exact_ref_gt_chunks": 0,
        "retrieved_entries_total": 0,
        "term_map_entries_total": 0,
        "gt_terms_in_term_map": 0,
        "retrieved_gt_terms": 0,
        "backfilled_gt_terms": 0,
        "non_gt_entries_total": 0,
        "nonempty_term_map_chunks": 0,
        "no_gt_chunks": 0,
        "no_gt_nonempty_term_map_chunks": 0,
        "marker_augmented_gt_entries": 0,
        "assistant_marker_replacements": 0,
        "assistant_messages_replaced": 0,
        "term_map_size_hist": Counter(),
        "dropped_rows": 0,
        "dropped_reasons": Counter(),
    }
    samples: List[Dict[str, Any]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for lineno, obj_in in _iter_jsonl(args.input_jsonl):
            stats["rows_seen"] += 1
            try:
                obj = copy.deepcopy(obj_in)
                messages = obj.get("messages")
                audios = obj.get("audios")
                gt_by_chunk = obj.get("gt_terms_by_chunk")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("missing non-empty messages")
                if not isinstance(audios, list) or not audios:
                    raise ValueError("missing non-empty audios")
                if not isinstance(gt_by_chunk, list):
                    raise ValueError("missing list gt_terms_by_chunk")
                audio_idxs = _audio_user_indices(messages)
                if len(audio_idxs) != len(audios):
                    raise ValueError(f"audio user messages={len(audio_idxs)} audios={len(audios)}")
                if len(gt_by_chunk) != len(audios):
                    raise ValueError(f"gt_terms_by_chunk={len(gt_by_chunk)} audios={len(audios)}")

                if messages[0].get("role") == "system":
                    messages[0]["content"] = SYSTEM_PROMPT_BY_LANG[args.lang_code]

                row_key = str(obj.get("utter_id") or lineno)
                full_target_text = _assistant_target_text(messages)
                new_gt_by_chunk: List[List[Dict[str, str]]] = []
                built_items_by_chunk: List[List[Dict[str, str]]] = []

                for chunk_idx, msg_idx in enumerate(audio_idxs):
                    retrieved = _parse_term_map(str(messages[msg_idx].get("content") or ""))
                    raw_gt = _normalize_gt_terms(gt_by_chunk[chunk_idx], args.lang_code)
                    exact_gt = _filter_exact_ref_gt(raw_gt, full_target_text=full_target_text)
                    items, retrieved_gt, backfilled = _build_dense_items(
                        retrieved=retrieved,
                        gt_terms=exact_gt,
                        max_terms=args.max_terms,
                    )
                    built_items_by_chunk.append(items)
                    new_gt_by_chunk.append([
                        {"term": x["term"], args.lang_code: x["translation"]}
                        for x in exact_gt
                    ])

                    gt_keys = {x["key"] for x in exact_gt}
                    item_keys = {x["key"] for x in items}
                    gt_hits = len(gt_keys & item_keys)
                    map_size = len(items)
                    has_gt = bool(exact_gt)

                    stats["chunks"] += 1
                    stats["raw_gt_terms_total"] += len(raw_gt)
                    stats["exact_ref_gt_terms_total"] += len(exact_gt)
                    stats["retrieved_entries_total"] += len(retrieved)
                    stats["term_map_entries_total"] += map_size
                    stats["gt_terms_in_term_map"] += gt_hits
                    stats["retrieved_gt_terms"] += retrieved_gt
                    stats["backfilled_gt_terms"] += backfilled
                    stats["non_gt_entries_total"] += max(0, map_size - gt_hits)
                    stats["term_map_size_hist"][str(map_size)] += 1
                    if map_size:
                        stats["nonempty_term_map_chunks"] += 1
                    if has_gt:
                        stats["exact_ref_gt_chunks"] += 1
                    else:
                        stats["no_gt_chunks"] += 1
                        if map_size:
                            stats["no_gt_nonempty_term_map_chunks"] += 1

                    if len(samples) < args.sample_count and (items or exact_gt):
                        samples.append({
                            "line": lineno,
                            "utter_id": obj.get("utter_id"),
                            "chunk_idx": chunk_idx,
                            "raw_gt": raw_gt[:8],
                            "exact_ref_gt": exact_gt[:8],
                            "retrieved": retrieved[:12],
                            "term_map": items[:20],
                        })

                if args.variant == "v10_marker":
                    marker_terms, repl, repl_msgs = _apply_marker_curriculum(
                        items_by_chunk=built_items_by_chunk,
                        messages=messages,
                        row_key=row_key,
                        seed=args.seed,
                    )
                    stats["marker_augmented_gt_entries"] += marker_terms
                    stats["assistant_marker_replacements"] += repl
                    stats["assistant_messages_replaced"] += repl_msgs

                for items, msg_idx in zip(built_items_by_chunk, audio_idxs):
                    messages[msg_idx]["content"] = _format_term_map(items)

                obj["gt_terms_by_chunk"] = new_gt_by_chunk
                obj["dense_topk_backfill_sft_policy"] = {
                    "version": args.variant,
                    "source": "retriever_timeline_top10_no_tau",
                    "top_k": 10,
                    "score_threshold": "disabled_upstream",
                    "max_terms": args.max_terms,
                    "gt_policy": "source exact match plus exact target substring in full assistant reference",
                    "gt_backfill": True,
                    "marker_augmented_targets": args.variant == "v10_marker",
                    "seed": args.seed,
                    "source_policy": obj.get("retriever_timeline_termmap_policy"),
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
            except Exception as exc:
                if args.drop_bad_rows:
                    stats["dropped_rows"] += 1
                    stats["dropped_reasons"][str(exc).splitlines()[0][:200]] += 1
                    continue
                raise RuntimeError(f"Failed processing {args.input_jsonl}:{lineno}: {exc}") from exc

    stats["term_map_size_hist"] = dict(stats["term_map_size_hist"])
    stats["dropped_reasons"] = dict(stats["dropped_reasons"])
    stats["exact_ref_gt_keep_rate"] = (
        stats["exact_ref_gt_terms_total"] / stats["raw_gt_terms_total"]
        if stats["raw_gt_terms_total"] else 0.0
    )
    stats["gt_term_in_term_map_rate"] = (
        stats["gt_terms_in_term_map"] / stats["exact_ref_gt_terms_total"]
        if stats["exact_ref_gt_terms_total"] else 0.0
    )
    stats["nonempty_term_map_rate"] = (
        stats["nonempty_term_map_chunks"] / stats["chunks"] if stats["chunks"] else 0.0
    )
    stats["no_gt_nonempty_term_map_rate"] = (
        stats["no_gt_nonempty_term_map_chunks"] / stats["no_gt_chunks"]
        if stats["no_gt_chunks"] else 0.0
    )
    stats["avg_term_map_entries_per_chunk"] = (
        stats["term_map_entries_total"] / stats["chunks"] if stats["chunks"] else 0.0
    )
    stats["avg_non_gt_entries_per_chunk"] = (
        stats["non_gt_entries_total"] / stats["chunks"] if stats["chunks"] else 0.0
    )
    stats["avg_retrieved_entries_per_chunk"] = (
        stats["retrieved_entries_total"] / stats["chunks"] if stats["chunks"] else 0.0
    )
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sample_json:
        args.sample_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path)
    parser.add_argument("--variant", choices=["v9", "v10_marker"], required=True)
    parser.add_argument("--lang-code", choices=sorted(SYSTEM_PROMPT_BY_LANG), default="zh")
    parser.add_argument("--max-terms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--drop-bad-rows", action="store_true")
    args = parser.parse_args()
    if args.max_terms <= 0:
        raise ValueError("--max-terms must be positive")
    return args


def main() -> None:
    args = parse_args()
    stats = build(args)
    print(json.dumps({
        "output_jsonl": str(args.output_jsonl),
        "variant": args.variant,
        "rows_written": stats["rows_written"],
        "chunks": stats["chunks"],
        "exact_ref_gt_keep_rate": stats["exact_ref_gt_keep_rate"],
        "gt_term_in_term_map_rate": stats["gt_term_in_term_map_rate"],
        "nonempty_term_map_rate": stats["nonempty_term_map_rate"],
        "no_gt_nonempty_term_map_rate": stats["no_gt_nonempty_term_map_rate"],
        "avg_retrieved_entries_per_chunk": stats["avg_retrieved_entries_per_chunk"],
        "avg_term_map_entries_per_chunk": stats["avg_term_map_entries_per_chunk"],
        "marker_augmented_gt_entries": stats["marker_augmented_gt_entries"],
        "assistant_marker_replacements": stats["assistant_marker_replacements"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
