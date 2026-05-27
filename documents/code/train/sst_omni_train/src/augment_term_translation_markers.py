#!/usr/bin/env python3
"""Strengthen term-map adoption by perturbing GT target translations.

This script takes an existing speech-LLM SFT JSONL with retriever-produced
``term_map`` entries and ``gt_terms_by_chunk``.  For a configurable fraction of
retrieved GT terms, it replaces the canonical target translation with a
deterministic marked variant in both:

* the current chunk's ``term_map`` line;
* the corresponding ``gt_terms_by_chunk`` entry;
* the first exact target occurrence in assistant text from the current chunk
  onward.

The replacement is atomic.  If the GT term is not in the current term_map, or if
the target translation is not found exactly in the future assistant text, the
sample is left unchanged and counted in stats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
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


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def _extract_translation(entry: Mapping[str, Any], lang_code: str) -> str:
    value = entry.get("translation") or entry.get("target_translation") or entry.get(lang_code)
    if value is None and isinstance(entry.get("target_translations"), Mapping):
        value = entry["target_translations"].get(lang_code)
    return str(value or "").strip()


def _set_translation(entry: MutableMapping[str, Any], lang_code: str, value: str) -> None:
    if lang_code in entry:
        entry[lang_code] = value
    elif "translation" in entry:
        entry["translation"] = value
    elif "target_translation" in entry:
        entry["target_translation"] = value
    else:
        entry[lang_code] = value
    target_translations = entry.get("target_translations")
    if isinstance(target_translations, MutableMapping):
        target_translations[lang_code] = value


def _audio_user_indices(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    out = []
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or ""):
            out.append(idx)
    return out


def _parse_term_map(content: str) -> Dict[str, Tuple[int, str, str]]:
    """Return term-key -> (line index, surface term, translation)."""
    out: Dict[str, Tuple[int, str, str]] = {}
    for idx, line in enumerate(str(content or "").splitlines()):
        if not line or line.strip() == "term_map:" or line.startswith("<audio>"):
            continue
        if "=" not in line:
            continue
        term, translation = line.split("=", 1)
        term = term.strip()
        translation = translation.strip()
        key = _term_key(term)
        if term and translation and key and key not in out:
            out[key] = (idx, term, translation)
    return out


def _replace_term_map_line(content: str, line_idx: int, term: str, new_translation: str) -> str:
    lines = str(content or "").splitlines()
    if line_idx < 0 or line_idx >= len(lines):
        raise ValueError(f"term_map line index out of range: {line_idx}")
    lines[line_idx] = f"{term}={new_translation}"
    return "\n".join(lines)


def _find_and_replace_future_assistant(
    messages: Sequence[MutableMapping[str, Any]],
    start_idx: int,
    old: str,
    new: str,
) -> Optional[int]:
    for msg_idx in range(start_idx, len(messages)):
        msg = messages[msg_idx]
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        if old not in content:
            continue
        msg["content"] = content.replace(old, new, 1)
        return msg_idx
    return None


def _marker_code(seed_text: str, length: int) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    chars = []
    for b in digest:
        chars.append(alphabet[b % len(alphabet)])
        if len(chars) >= length:
            break
    return "".join(chars)


def _marked_translation(translation: str, code: str, template: str) -> str:
    return template.format(translation=translation, code=code)


def process_row(
    obj: Dict[str, Any],
    *,
    row_line: int,
    lang_code: str,
    augment_prob: float,
    rng: random.Random,
    marker_template: str,
    marker_len: int,
    max_augmented_terms_per_row: int,
    stats: Counter,
    samples: List[Dict[str, Any]],
    sample_count: int,
) -> Dict[str, Any]:
    messages = obj.get("messages")
    audios = obj.get("audios")
    gt_by_chunk = obj.get("gt_terms_by_chunk")
    if not isinstance(messages, list) or not isinstance(audios, list):
        raise ValueError("missing messages or audios")
    if not isinstance(gt_by_chunk, list):
        raise ValueError("missing gt_terms_by_chunk")
    audio_user_idxs = _audio_user_indices(messages)
    if len(audio_user_idxs) != len(audios):
        raise ValueError(f"audio user messages={len(audio_user_idxs)} audios={len(audios)}")
    if len(gt_by_chunk) != len(audios):
        raise ValueError(f"gt_terms_by_chunk={len(gt_by_chunk)} audios={len(audios)}")

    stats["rows_seen"] += 1
    stats["chunks_total"] += len(audios)
    row_augmented = 0
    row_events: List[Dict[str, Any]] = []

    for chunk_idx, msg_idx in enumerate(audio_user_idxs):
        raw_terms = gt_by_chunk[chunk_idx]
        if not isinstance(raw_terms, list):
            raise ValueError("gt_terms_by_chunk entry must be list")
        term_map = _parse_term_map(str(messages[msg_idx].get("content") or ""))
        if term_map:
            stats["chunks_with_nonempty_term_map"] += 1
        if raw_terms:
            stats["chunks_with_gt_terms"] += 1

        for term_pos, term_obj in enumerate(raw_terms):
            if not isinstance(term_obj, MutableMapping):
                raise ValueError("gt term entry must be object")
            term = str(term_obj.get("term") or term_obj.get("source") or "").strip()
            translation = _extract_translation(term_obj, lang_code)
            key = _term_key(term)
            if not term or not translation or not key:
                continue
            stats["candidate_gt_terms"] += 1
            if key not in term_map:
                stats["skipped_gt_not_in_current_term_map"] += 1
                continue
            stats["candidate_gt_terms_in_current_term_map"] += 1
            if max_augmented_terms_per_row > 0 and row_augmented >= max_augmented_terms_per_row:
                stats["skipped_row_cap"] += 1
                continue
            if rng.random() >= augment_prob:
                stats["skipped_by_probability"] += 1
                continue

            line_idx, map_term, map_translation = term_map[key]
            if map_translation != translation:
                stats["term_map_translation_differs_from_gt"] += 1

            code = _marker_code(f"{row_line}:{chunk_idx}:{term_pos}:{term}:{translation}", marker_len)
            new_translation = _marked_translation(translation, code, marker_template)
            if new_translation == translation:
                raise ValueError("marked translation equals canonical translation")

            replaced_msg_idx = _find_and_replace_future_assistant(
                messages,
                start_idx=msg_idx + 1,
                old=translation,
                new=new_translation,
            )
            if replaced_msg_idx is None:
                stats["skipped_missing_future_reference_exact"] += 1
                continue

            messages[msg_idx]["content"] = _replace_term_map_line(
                str(messages[msg_idx].get("content") or ""),
                line_idx=line_idx,
                term=map_term,
                new_translation=new_translation,
            )
            _set_translation(term_obj, lang_code, new_translation)
            row_augmented += 1
            stats["augmented_terms"] += 1
            stats["assistant_replacements"] += 1
            stats["term_map_replacements"] += 1
            event = {
                "chunk_idx": chunk_idx,
                "term": term,
                "old_translation": translation,
                "new_translation": new_translation,
                "assistant_msg_idx": replaced_msg_idx,
            }
            row_events.append(event)
            if len(samples) < sample_count:
                samples.append({
                    "row_line": row_line,
                    "utter_id": obj.get("utter_id"),
                    **event,
                })

    if row_augmented:
        stats["rows_with_augmentation"] += 1
    obj["term_translation_marker_augmentation"] = {
        "version": "v1",
        "lang_code": lang_code,
        "augment_prob": augment_prob,
        "marker_template": marker_template,
        "marker_len": marker_len,
        "max_augmented_terms_per_row": max_augmented_terms_per_row,
        "augmented_terms_in_row": row_augmented,
        "events": row_events[:50],
    }
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--sample-json", type=Path, default=None)
    parser.add_argument("--lang-code", default="zh")
    parser.add_argument("--augment-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1505)
    parser.add_argument("--marker-template", default="{translation}__tm{code}")
    parser.add_argument("--marker-len", type=int, default=4)
    parser.add_argument("--max-augmented-terms-per-row", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=80)
    args = parser.parse_args()

    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if not (0.0 <= args.augment_prob <= 1.0):
        raise ValueError("--augment-prob must be in [0, 1]")
    if "{translation}" not in args.marker_template or "{code}" not in args.marker_template:
        raise ValueError("--marker-template must contain {translation} and {code}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    samples: List[Dict[str, Any]] = []
    rng = random.Random(args.seed)

    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for row_line, obj in _iter_jsonl(args.input_jsonl):
            if args.max_rows > 0 and stats["rows_written"] >= args.max_rows:
                break
            out_obj = process_row(
                obj,
                row_line=row_line,
                lang_code=args.lang_code,
                augment_prob=args.augment_prob,
                rng=rng,
                marker_template=args.marker_template,
                marker_len=args.marker_len,
                max_augmented_terms_per_row=args.max_augmented_terms_per_row,
                stats=stats,
                samples=samples,
                sample_count=args.sample_count,
            )
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            stats["rows_written"] += 1

    total = stats["candidate_gt_terms"]
    in_map = stats["candidate_gt_terms_in_current_term_map"]
    augmented = stats["augmented_terms"]
    stats_dict: Dict[str, Any] = dict(stats)
    stats_dict.update({
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "lang_code": args.lang_code,
        "augment_prob": args.augment_prob,
        "seed": args.seed,
        "marker_template": args.marker_template,
        "marker_len": args.marker_len,
        "max_augmented_terms_per_row": args.max_augmented_terms_per_row,
        "gt_term_in_current_term_map_rate": in_map / total if total else 0.0,
        "augmented_over_gt_terms_rate": augmented / total if total else 0.0,
        "augmented_over_gt_in_map_rate": augmented / in_map if in_map else 0.0,
    })
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
