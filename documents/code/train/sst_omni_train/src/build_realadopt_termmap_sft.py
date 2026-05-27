#!/usr/bin/env python3
"""Build real-adoption-oriented Speech LLM SFT data from retriever term maps.

Input must already contain per-chunk retriever ``term_map`` entries, ideally
from the deployed timeline retriever configuration.  This script does not
backfill missed GT terms.  A term is trusted as positive supervision only when:

1. the retriever put the source term in the chunk's term_map;
2. the source term is present in ``gt_terms_by_chunk``;
3. the target translation is an exact substring of the local assistant output.

This makes the loss depend on term-map adoption without inflating recall beyond
the real retriever behavior.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


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


def _extract_translation(item: Mapping[str, Any], lang_code: str) -> str:
    value = item.get("translation") or item.get("target_translation") or item.get(lang_code)
    if value is None and isinstance(item.get("target_translations"), Mapping):
        value = item["target_translations"].get(lang_code)
    return str(value or "").strip()


def _audio_user_indices(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    return [
        idx for idx, msg in enumerate(messages)
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or "")
    ]


def _assistant_idx_after(messages: Sequence[Mapping[str, Any]], user_idx: int) -> int | None:
    for idx in range(user_idx + 1, min(len(messages), user_idx + 3)):
        if messages[idx].get("role") == "assistant":
            return idx
    return None


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
        raise ValueError(f"gt_terms_by_chunk entry must be list, got {type(raw_terms).__name__}")
    out: List[Dict[str, str]] = []
    seen = set()
    for item in raw_terms:
        if not isinstance(item, Mapping):
            raise ValueError("gt_terms_by_chunk term entry must be an object")
        term = str(item.get("term") or item.get("source") or "").strip()
        trans = _extract_translation(item, lang_code)
        key = _term_key(term)
        if not term or not trans or not key or key in seen:
            continue
        seen.add(key)
        out.append({"term": term, "translation": trans, "key": key})
    return out


def _format_term_map(items: Sequence[Mapping[str, str]]) -> str:
    if not items:
        return "<audio>\n\nterm_map:NONE"
    lines = ["<audio>", "", "term_map:"]
    seen = set()
    for item in items:
        term = str(item.get("term") or "").replace("\n", " ").strip()
        trans = str(item.get("translation") or "").replace("\n", " ").strip()
        key = _term_key(term)
        if not term or not trans or not key or key in seen:
            continue
        seen.add(key)
        lines.append(f"{term}={trans}")
    if len(lines) == 3:
        return "<audio>\n\nterm_map:NONE"
    return "\n".join(lines)


def _rng(seed: int, row_key: str, chunk_idx: int) -> random.Random:
    h = hashlib.sha256(f"{seed}|{row_key}|{chunk_idx}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def _marker_for(seed: int, row_key: str, term_key: str, translation: str) -> Tuple[str, str]:
    h = hashlib.sha256(f"{seed}|marker|{row_key}|{term_key}|{translation}".encode("utf-8")).hexdigest()
    return MARKER_PAIRS[int(h[:8], 16) % len(MARKER_PAIRS)]


def _take_unique(items: Sequence[Mapping[str, str]], cap: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        term = str(item.get("term") or "").strip()
        trans = str(item.get("translation") or "").strip()
        key = str(item.get("key") or _term_key(term))
        if not term or not trans or not key or key in seen:
            continue
        seen.add(key)
        cur = dict(item)
        cur["key"] = key
        out.append(cur)
        if len(out) >= cap:
            break
    return out


def _build_items(
    *,
    retrieved: Sequence[Mapping[str, str]],
    gt_terms: Sequence[Mapping[str, str]],
    assistant_text: str,
    variant: str,
    rng: random.Random,
    max_terms: int,
    max_noise_with_gt: int,
    max_noise_without_gt: int,
    no_gt_keep_prob: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, int]]:
    gt_by_key = {x["key"]: dict(x) for x in gt_terms}
    exact_gt: List[Dict[str, str]] = []
    non_gt: List[Dict[str, str]] = []
    for item in retrieved:
        key = str(item.get("key") or _term_key(str(item.get("term") or "")))
        if not key:
            continue
        if key in gt_by_key and gt_by_key[key]["translation"] in assistant_text:
            cur = dict(gt_by_key[key])
            cur["is_gt"] = "1"
            exact_gt.append(cur)
        else:
            non_gt.append(dict(item))

    meta = {"exact_gt_retrieved": len(_take_unique(exact_gt, 9999)), "noise_kept": 0}
    if variant == "realistic":
        items: List[Dict[str, str]] = []
        gt_keys = {x["key"] for x in exact_gt}
        for item in retrieved:
            key = str(item.get("key") or _term_key(str(item.get("term") or "")))
            if key in gt_keys:
                gt = next(x for x in exact_gt if x["key"] == key)
                items.append(dict(gt))
            else:
                items.append(dict(item))
        items = _take_unique(items, max_terms)
        meta["noise_kept"] = sum(1 for x in items if x.get("key") not in gt_keys)
        return items, _take_unique(exact_gt, 9999), meta

    if exact_gt:
        gt_items = _take_unique(exact_gt, max_terms)
        shuffled_noise = [dict(x) for x in non_gt]
        rng.shuffle(shuffled_noise)
        noise = _take_unique(shuffled_noise, max_noise_with_gt)
        items = _take_unique(gt_items + noise, max_terms)
        meta["noise_kept"] = len(noise)
        return items, gt_items, meta

    if rng.random() > no_gt_keep_prob:
        return [], [], meta
    shuffled_noise = [dict(x) for x in non_gt]
    rng.shuffle(shuffled_noise)
    items = _take_unique(shuffled_noise, min(max_terms, max_noise_without_gt))
    meta["noise_kept"] = len(items)
    return items, [], meta


def _apply_markers(
    *,
    items: List[Dict[str, str]],
    exact_gt_items: Sequence[Mapping[str, str]],
    assistant_text: str,
    row_key: str,
    seed: int,
) -> Tuple[str, int, int]:
    marked_by_translation: Dict[str, str] = {}
    gt_keys = {str(x["key"]) for x in exact_gt_items}
    marker_terms = 0
    for item in items:
        key = str(item.get("key") or "")
        if key not in gt_keys:
            continue
        trans = str(item.get("translation") or "")
        if not trans:
            continue
        if trans not in marked_by_translation:
            prefix, suffix = _marker_for(seed, row_key, key, trans)
            marked_by_translation[trans] = f"{prefix}{trans}{suffix}"
        item["translation"] = marked_by_translation[trans]
        item["marker_augmented_gt"] = "1"
        marker_terms += 1

    replacements = 0
    new_text = assistant_text
    for original, marked in sorted(marked_by_translation.items(), key=lambda kv: len(kv[0]), reverse=True):
        count = new_text.count(original)
        if count:
            replacements += count
            new_text = new_text.replace(original, marked)
    return new_text, marker_terms, replacements


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
        "rows_seen": 0,
        "rows_written": 0,
        "chunks": 0,
        "retrieved_entries_total": 0,
        "raw_gt_terms_total": 0,
        "retrieved_exact_local_gt_terms": 0,
        "term_map_entries_total": 0,
        "gt_entries_in_term_map": 0,
        "non_gt_entries_in_term_map": 0,
        "chunks_with_exact_local_gt": 0,
        "chunks_with_gt_term_map": 0,
        "no_gt_chunks": 0,
        "no_gt_nonempty_term_map_chunks": 0,
        "marker_augmented_gt_entries": 0,
        "assistant_marker_replacements": 0,
        "term_map_size_hist": Counter(),
        "dropped_rows": 0,
        "dropped_reasons": Counter(),
    }
    samples: List[Dict[str, Any]] = []
    tmp_output = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")

    with tmp_output.open("w", encoding="utf-8") as out:
        for lineno, obj_in in _iter_jsonl(args.input_jsonl):
            stats["rows_seen"] += 1
            try:
                obj = copy.deepcopy(obj_in)
                messages = obj.get("messages")
                audios = obj.get("audios")
                gt_by_chunk = obj.get("gt_terms_by_chunk")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("missing messages")
                if not isinstance(audios, list) or not audios:
                    raise ValueError("missing audios")
                if not isinstance(gt_by_chunk, list):
                    raise ValueError("missing list gt_terms_by_chunk")
                user_idxs = _audio_user_indices(messages)
                if len(user_idxs) != len(audios):
                    raise ValueError(f"audio user messages={len(user_idxs)} audios={len(audios)}")
                if len(gt_by_chunk) != len(audios):
                    raise ValueError(f"gt_terms_by_chunk={len(gt_by_chunk)} audios={len(audios)}")
                if messages[0].get("role") == "system":
                    messages[0]["content"] = SYSTEM_PROMPT_BY_LANG[args.lang_code]

                row_key = str(obj.get("utter_id") or lineno)
                new_gt_by_chunk: List[List[Dict[str, str]]] = []
                for chunk_idx, user_idx in enumerate(user_idxs):
                    assistant_idx = _assistant_idx_after(messages, user_idx)
                    assistant_text = str(messages[assistant_idx].get("content") or "") if assistant_idx is not None else ""
                    retrieved = _parse_term_map(str(messages[user_idx].get("content") or ""))
                    raw_gt = _normalize_gt_terms(gt_by_chunk[chunk_idx], args.lang_code)
                    rng = _rng(args.seed, row_key, chunk_idx)
                    items, exact_gt_items, meta = _build_items(
                        retrieved=retrieved,
                        gt_terms=raw_gt,
                        assistant_text=assistant_text,
                        variant=args.variant,
                        rng=rng,
                        max_terms=args.max_terms,
                        max_noise_with_gt=args.max_noise_with_gt,
                        max_noise_without_gt=args.max_noise_without_gt,
                        no_gt_keep_prob=args.no_gt_keep_prob,
                    )
                    if args.variant == "marker" and assistant_idx is not None:
                        new_text, marker_terms, repl = _apply_markers(
                            items=items,
                            exact_gt_items=exact_gt_items,
                            assistant_text=assistant_text,
                            row_key=f"{row_key}:{chunk_idx}",
                            seed=args.seed,
                        )
                        messages[assistant_idx]["content"] = new_text
                        stats["marker_augmented_gt_entries"] += marker_terms
                        stats["assistant_marker_replacements"] += repl

                    messages[user_idx]["content"] = _format_term_map(items)
                    new_gt_by_chunk.append([
                        {"term": x["term"], args.lang_code: x["translation"]}
                        for x in exact_gt_items
                    ])

                    gt_keys = {x["key"] for x in exact_gt_items}
                    item_keys = {str(x.get("key") or _term_key(str(x.get("term") or ""))) for x in items}
                    gt_hits = len(gt_keys & item_keys)
                    stats["chunks"] += 1
                    stats["retrieved_entries_total"] += len(retrieved)
                    stats["raw_gt_terms_total"] += len(raw_gt)
                    stats["retrieved_exact_local_gt_terms"] += len(exact_gt_items)
                    stats["term_map_entries_total"] += len(items)
                    stats["gt_entries_in_term_map"] += gt_hits
                    stats["non_gt_entries_in_term_map"] += max(0, len(items) - gt_hits)
                    stats["term_map_size_hist"][str(len(items))] += 1
                    if exact_gt_items:
                        stats["chunks_with_exact_local_gt"] += 1
                    else:
                        stats["no_gt_chunks"] += 1
                        if items:
                            stats["no_gt_nonempty_term_map_chunks"] += 1
                    if gt_hits:
                        stats["chunks_with_gt_term_map"] += 1

                    if len(samples) < args.sample_count and (items or exact_gt_items):
                        samples.append({
                            "line": lineno,
                            "utter_id": obj.get("utter_id"),
                            "chunk_idx": chunk_idx,
                            "variant": args.variant,
                            "raw_gt": raw_gt[:12],
                            "exact_local_gt": exact_gt_items[:12],
                            "retrieved": retrieved[:12],
                            "term_map": items[:12],
                            "assistant_after": messages[assistant_idx]["content"][:300] if assistant_idx is not None else "",
                        })

                obj["gt_terms_by_chunk"] = new_gt_by_chunk
                obj["realadopt_termmap_sft_policy"] = {
                    "version": "v1",
                    "variant": args.variant,
                    "source": "timeline retriever tau-filtered term_map",
                    "positive_gt_definition": (
                        "retrieved source term intersects gt_terms_by_chunk and target translation "
                        "is an exact substring of the local assistant chunk"
                    ),
                    "gt_backfill": False,
                    "max_terms": args.max_terms,
                    "max_noise_with_gt": args.max_noise_with_gt,
                    "max_noise_without_gt": args.max_noise_without_gt,
                    "no_gt_keep_prob": args.no_gt_keep_prob,
                    "marker_augmented": args.variant == "marker",
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

    tmp_output.replace(args.output_jsonl)
    stats["term_map_size_hist"] = dict(stats["term_map_size_hist"])
    stats["dropped_reasons"] = dict(stats["dropped_reasons"])
    stats["avg_retrieved_entries_per_chunk"] = (
        stats["retrieved_entries_total"] / stats["chunks"] if stats["chunks"] else 0.0
    )
    stats["avg_term_map_entries_per_chunk"] = (
        stats["term_map_entries_total"] / stats["chunks"] if stats["chunks"] else 0.0
    )
    stats["avg_non_gt_entries_per_chunk"] = (
        stats["non_gt_entries_in_term_map"] / stats["chunks"] if stats["chunks"] else 0.0
    )
    stats["retrieved_exact_local_gt_rate_vs_raw_gt"] = (
        stats["retrieved_exact_local_gt_terms"] / stats["raw_gt_terms_total"]
        if stats["raw_gt_terms_total"] else 0.0
    )
    stats["gt_term_in_term_map_rate"] = (
        stats["gt_entries_in_term_map"] / stats["retrieved_exact_local_gt_terms"]
        if stats["retrieved_exact_local_gt_terms"] else 0.0
    )
    stats["chunks_with_exact_local_gt_rate"] = (
        stats["chunks_with_exact_local_gt"] / stats["chunks"] if stats["chunks"] else 0.0
    )
    stats["no_gt_nonempty_term_map_rate"] = (
        stats["no_gt_nonempty_term_map_chunks"] / stats["no_gt_chunks"]
        if stats["no_gt_chunks"] else 0.0
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
    parser.add_argument("--variant", choices=["realistic", "marker"], required=True)
    parser.add_argument("--lang-code", choices=sorted(SYSTEM_PROMPT_BY_LANG), default="zh")
    parser.add_argument("--max-terms", type=int, default=10)
    parser.add_argument("--max-noise-with-gt", type=int, default=2)
    parser.add_argument("--max-noise-without-gt", type=int, default=3)
    parser.add_argument("--no-gt-keep-prob", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--drop-bad-rows", action="store_true")
    args = parser.parse_args()
    if args.max_terms <= 0:
        raise ValueError("--max-terms must be positive")
    if not (0.0 <= args.no_gt_keep_prob <= 1.0):
        raise ValueError("--no-gt-keep-prob must be in [0,1]")
    return args


def main() -> None:
    stats = build(parse_args())
    print(json.dumps({
        "output_jsonl": stats["output_jsonl"],
        "variant": stats["variant"],
        "rows_written": stats["rows_written"],
        "chunks": stats["chunks"],
        "avg_retrieved_entries_per_chunk": stats["avg_retrieved_entries_per_chunk"],
        "avg_term_map_entries_per_chunk": stats["avg_term_map_entries_per_chunk"],
        "retrieved_exact_local_gt_rate_vs_raw_gt": stats["retrieved_exact_local_gt_rate_vs_raw_gt"],
        "gt_term_in_term_map_rate": stats["gt_term_in_term_map_rate"],
        "chunks_with_exact_local_gt_rate": stats["chunks_with_exact_local_gt_rate"],
        "no_gt_nonempty_term_map_rate": stats["no_gt_nonempty_term_map_rate"],
        "marker_augmented_gt_entries": stats["marker_augmented_gt_entries"],
        "assistant_marker_replacements": stats["assistant_marker_replacements"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
