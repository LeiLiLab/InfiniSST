#!/usr/bin/env python3
"""Cap embedded user-side term_map entries while preserving derived GT terms.

This is for legacy Speech LLM SFT JSONL rows whose user messages already contain
``term_map:`` blocks.  It rewrites each block to keep at most ``--max-terms``:

1. entries whose source term matches ``gt_terms_by_chunk`` for that audio chunk,
2. then the remaining original term-map entries in order.

The script raises on malformed row structure instead of silently skipping rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


TERM_MAP_MARKER = "term_map:"


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, MutableMapping[str, Any]]]:
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
                raise ValueError(f"Expected object on {path}:{lineno}")
            yield lineno, obj


def _audio_user_indices(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    return [
        idx
        for idx, msg in enumerate(messages)
        if msg.get("role") == "user" and str(msg.get("content") or "").startswith("<audio>")
    ]


def _parse_entries(body: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        term, translation = line.split("=", 1)
        term = term.strip()
        translation = translation.strip()
        if term and translation:
            entries.append({"term": term, "translation": translation})
    return entries


def _entry_line(entry: Mapping[str, str]) -> str:
    return f"{entry['term']}={entry['translation']}"


def _gt_terms_for_chunk(gt_by_chunk: Sequence[Any], chunk_idx: int) -> set[str]:
    if chunk_idx >= len(gt_by_chunk):
        return set()
    out: set[str] = set()
    for row in gt_by_chunk[chunk_idx] or []:
        if isinstance(row, Mapping):
            term = str(row.get("term") or "").strip().lower()
        else:
            term = str(row or "").strip().lower()
        if term:
            out.add(term)
    return out


def _cap_entries(entries: Sequence[Mapping[str, str]], gt_terms: set[str], max_terms: int) -> Tuple[List[Dict[str, str]], int]:
    seen: set[str] = set()
    gt_first: List[Dict[str, str]] = []
    rest: List[Dict[str, str]] = []
    missing_gt = set(gt_terms)
    for entry in entries:
        term = str(entry.get("term") or "").strip()
        translation = str(entry.get("translation") or "").strip()
        if not term or not translation:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        item = {"term": term, "translation": translation}
        if key in gt_terms:
            gt_first.append(item)
            missing_gt.discard(key)
        else:
            rest.append(item)
    capped = (gt_first + rest)[:max_terms]
    return capped, len(missing_gt)


def process_row(
    obj: MutableMapping[str, Any],
    *,
    lineno: int,
    max_terms: int,
    stats: Counter,
    samples: List[Dict[str, Any]],
    sample_count: int,
) -> MutableMapping[str, Any]:
    messages = obj.get("messages")
    audios = obj.get("audios")
    gt_by_chunk = obj.get("gt_terms_by_chunk")
    if not isinstance(messages, list) or not isinstance(audios, list) or not isinstance(gt_by_chunk, list):
        raise ValueError(f"Row {lineno}: expected messages, audios, and gt_terms_by_chunk lists")
    user_indices = _audio_user_indices(messages)
    if len(user_indices) != len(audios) or len(gt_by_chunk) != len(audios):
        raise ValueError(
            f"Row {lineno}: user/audio/gt mismatch {len(user_indices)}/{len(audios)}/{len(gt_by_chunk)}"
        )

    stats["rows"] += 1
    stats["chunks"] += len(user_indices)
    for chunk_idx, msg_idx in enumerate(user_indices):
        msg = messages[msg_idx]
        content = str(msg.get("content") or "")
        marker_idx = content.find(TERM_MAP_MARKER)
        if marker_idx < 0 or "term_map:NONE" in content:
            stats["chunks_without_termmap"] += 1
            continue
        prefix = content[: marker_idx + len(TERM_MAP_MARKER)]
        body = content[marker_idx + len(TERM_MAP_MARKER) :].strip()
        entries = _parse_entries(body)
        if not entries:
            stats["chunks_without_termmap"] += 1
            continue

        gt_terms = _gt_terms_for_chunk(gt_by_chunk, chunk_idx)
        capped, missing_gt = _cap_entries(entries, gt_terms, max_terms)
        stats["chunks_with_termmap"] += 1
        stats["entries_before"] += len(entries)
        stats["entries_after"] += len(capped)
        stats["max_entries_before"] = max(stats["max_entries_before"], len(entries))
        stats["max_entries_after"] = max(stats["max_entries_after"], len(capped))
        stats["chunks_over_cap_before"] += int(len(entries) > max_terms)
        stats["gt_terms_total"] += len(gt_terms)
        stats["gt_terms_missing_from_original_map"] += missing_gt
        if missing_gt:
            stats["chunks_with_missing_gt_from_original_map"] += 1
        if not capped:
            msg["content"] = content[:marker_idx].rstrip()
            stats["chunks_capped_to_empty"] += 1
        else:
            msg["content"] = prefix.rstrip() + "\n" + "\n".join(_entry_line(entry) for entry in capped)
        if len(samples) < sample_count and len(entries) > max_terms:
            samples.append(
                {
                    "lineno": lineno,
                    "chunk_idx": chunk_idx,
                    "before": len(entries),
                    "after": len(capped),
                    "gt_terms": sorted(gt_terms),
                    "kept": capped[:max_terms],
                }
            )
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, default=None)
    parser.add_argument("--max-terms", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    if args.max_terms < 1:
        raise ValueError("--max-terms must be >= 1")
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    samples: List[Dict[str, Any]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for idx, (lineno, obj) in enumerate(_iter_jsonl(args.input_jsonl), 1):
            if args.max_rows and idx > args.max_rows:
                break
            new_obj = process_row(
                obj,
                lineno=lineno,
                max_terms=args.max_terms,
                stats=stats,
                samples=samples,
                sample_count=args.sample_count,
            )
            out.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    summary = dict(stats)
    chunks = max(1, int(summary.get("chunks", 0)))
    term_chunks = max(1, int(summary.get("chunks_with_termmap", 0)))
    summary["chunk_termmap_rate"] = float(summary.get("chunks_with_termmap", 0)) / chunks
    summary["avg_entries_before"] = float(summary.get("entries_before", 0)) / term_chunks
    summary["avg_entries_after"] = float(summary.get("entries_after", 0)) / term_chunks
    summary["max_terms"] = args.max_terms
    args.stats_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sample_json:
        args.sample_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
