#!/usr/bin/env python3
"""Reshape Speech LLM SFT chunks to match deployed low-latency inference.

The historical SFT data samples ``merge_multiplier`` uniformly from 1..12.
For RAG-SST term-map training this creates two mismatches:

* chunks longer than the deployed low-latency regime;
* lm=1/2 examples whose retriever context is too short at the beginning unless
  we simulate the streaming buffer.

This script rewrites the SFT JSONL into effective chunks of 3..6 base units
(base unit = 0.96s).  Rows whose original chunks are longer than 6 units are
dropped.  Rows with lm=1/2 are grouped into consecutive buffered chunks before
the retriever is run.  No term_map is added here; downstream retriever data prep
will fill it.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf


UNIT_SEC = 0.96
EXPECTED_SR = 16000


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
                raise ValueError(f"Expected JSON object at {path}:{lineno}")
            yield lineno, obj


def _audio_user_indices(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    return [
        idx for idx, msg in enumerate(messages)
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or "")
    ]


def _assistant_after(messages: Sequence[Mapping[str, Any]], user_idx: int) -> str:
    for msg in messages[user_idx + 1 : user_idx + 3]:
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def _extract_translation(item: Mapping[str, Any], lang_code: str) -> str:
    value = item.get("translation") or item.get("target_translation") or item.get(lang_code)
    if value is None and isinstance(item.get("target_translations"), Mapping):
        value = item["target_translations"].get(lang_code)
    return str(value or "").strip()


def _dedupe_terms(chunks: Sequence[Sequence[Mapping[str, Any]]], lang_code: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for terms in chunks:
        if not isinstance(terms, list):
            continue
        for item in terms:
            if not isinstance(item, Mapping):
                continue
            term = str(item.get("term") or item.get("source") or "").strip()
            trans = _extract_translation(item, lang_code)
            key = _term_key(term)
            if not term or not trans or not key or key in seen:
                continue
            seen.add(key)
            out.append({"term": term, lang_code: trans})
    return out


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text[:120] or "row"


def _read_audio(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if sr != EXPECTED_SR:
        raise ValueError(f"unexpected sample rate {sr} for {path}; expected {EXPECTED_SR}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size <= 0:
        raise ValueError(f"empty audio: {path}")
    return audio, sr


def _duration_lm(path: str) -> int:
    info = sf.info(path)
    if info.samplerate != EXPECTED_SR:
        raise ValueError(f"unexpected sample rate {info.samplerate} for {path}; expected {EXPECTED_SR}")
    return max(1, int(round((info.frames / info.samplerate) / UNIT_SEC)))


def _make_groups(lms: Sequence[int], min_lm: int, max_lm: int) -> Tuple[List[List[int]], Counter]:
    groups: List[List[int]] = []
    reasons: Counter = Counter()
    cur: List[int] = []
    cur_lm = 0
    for idx, lm in enumerate(lms):
        if lm > max_lm:
            reasons["chunk_gt_max_lm"] += 1
            continue
        if lm >= min_lm:
            if cur:
                if groups and sum(lms[i] for i in groups[-1]) + cur_lm <= max_lm:
                    groups[-1].extend(cur)
                    reasons["residual_short_attached_prev"] += 1
                else:
                    reasons["residual_short_dropped"] += 1
                cur, cur_lm = [], 0
            groups.append([idx])
            continue
        if cur_lm + lm > max_lm:
            if cur_lm >= min_lm:
                groups.append(cur)
            else:
                reasons["short_group_dropped"] += 1
            cur, cur_lm = [idx], lm
        else:
            cur.append(idx)
            cur_lm += lm
            if cur_lm >= min_lm:
                groups.append(cur)
                cur, cur_lm = [], 0
    if cur:
        if groups and sum(lms[i] for i in groups[-1]) + cur_lm <= max_lm:
            groups[-1].extend(cur)
            reasons["tail_short_attached_prev"] += 1
        else:
            reasons["tail_short_dropped"] += 1
    return groups, reasons


def _write_group_audio(
    *,
    audio_paths: Sequence[str],
    group: Sequence[int],
    output_audio_dir: Path,
    row_key: str,
    group_idx: int,
) -> str:
    if len(group) == 1:
        return str(audio_paths[group[0]])
    row_dir = output_audio_dir / _safe_name(row_key)
    row_dir.mkdir(parents=True, exist_ok=True)
    out_path = row_dir / f"g{group_idx:04d}.wav"
    if out_path.exists():
        return str(out_path)
    parts = []
    for idx in group:
        audio, _ = _read_audio(str(audio_paths[idx]))
        parts.append(audio)
    merged = np.concatenate(parts).astype(np.float32)
    tmp_path = out_path.with_suffix(".tmp.wav")
    sf.write(tmp_path, merged, EXPECTED_SR)
    tmp_path.replace(out_path)
    return str(out_path)


def build(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_audio_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "output_audio_dir": str(args.output_audio_dir),
        "lang_code": args.lang_code,
        "min_lm": args.min_lm,
        "max_lm": args.max_lm,
        "rows_seen": 0,
        "rows_written": 0,
        "rows_dropped": 0,
        "drop_reasons": Counter(),
        "original_chunks": 0,
        "output_chunks": 0,
        "original_lm_hist": Counter(),
        "output_lm_hist": Counter(),
        "groups_per_row_hist": Counter(),
        "grouping_reasons": Counter(),
        "raw_gt_terms_total": 0,
        "output_gt_terms_total": 0,
    }
    samples: List[Dict[str, Any]] = []
    tmp_output = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")

    with tmp_output.open("w", encoding="utf-8") as fout:
        for lineno, row_in in _iter_jsonl(args.input_jsonl):
            stats["rows_seen"] += 1
            try:
                row = copy.deepcopy(row_in)
                messages = row.get("messages")
                audios = row.get("audios")
                gt_by_chunk = row.get("gt_terms_by_chunk")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("missing non-empty messages")
                if not isinstance(audios, list) or not audios:
                    raise ValueError("missing non-empty audios")
                if not isinstance(gt_by_chunk, list):
                    raise ValueError("missing list gt_terms_by_chunk")
                user_idxs = _audio_user_indices(messages)
                if len(user_idxs) != len(audios):
                    raise ValueError(f"audio user messages={len(user_idxs)} audios={len(audios)}")
                if len(gt_by_chunk) != len(audios):
                    raise ValueError(f"gt_terms_by_chunk={len(gt_by_chunk)} audios={len(audios)}")

                lms = [_duration_lm(str(p)) for p in audios]
                for lm in lms:
                    stats["original_lm_hist"][str(lm)] += 1
                stats["original_chunks"] += len(lms)
                if any(lm > args.max_lm for lm in lms):
                    raise ValueError("row_contains_chunk_gt_max_lm")

                groups, reasons = _make_groups(lms, args.min_lm, args.max_lm)
                stats["grouping_reasons"].update(reasons)
                if not groups:
                    raise ValueError("no_output_groups")

                new_messages: List[Dict[str, str]] = []
                if messages[0].get("role") == "system":
                    new_messages.append(dict(messages[0]))
                new_audios: List[str] = []
                new_gt: List[List[Dict[str, str]]] = []
                chunk_ranges: List[Dict[str, Any]] = []
                row_key = str(row.get("utter_id") or f"line{lineno}")

                for group_idx, group in enumerate(groups):
                    eff_lm = sum(lms[i] for i in group)
                    if eff_lm < args.min_lm or eff_lm > args.max_lm:
                        raise ValueError(f"invalid_effective_lm={eff_lm}")
                    out_audio = _write_group_audio(
                        audio_paths=[str(x) for x in audios],
                        group=group,
                        output_audio_dir=args.output_audio_dir,
                        row_key=row_key,
                        group_idx=group_idx,
                    )
                    assistant_parts = [_assistant_after(messages, user_idxs[i]) for i in group]
                    assistant_text = args.assistant_join.join(x for x in assistant_parts if x)
                    gt_terms = _dedupe_terms([gt_by_chunk[i] for i in group], args.lang_code)

                    new_audios.append(out_audio)
                    new_gt.append(gt_terms)
                    new_messages.append({"role": "user", "content": "<audio>"})
                    new_messages.append({"role": "assistant", "content": assistant_text})
                    chunk_ranges.append({
                        "group_idx": group_idx,
                        "source_chunk_indices": list(group),
                        "source_lms": [lms[i] for i in group],
                        "effective_lm": eff_lm,
                    })
                    stats["output_lm_hist"][str(eff_lm)] += 1
                    stats["raw_gt_terms_total"] += sum(
                        len(x) for x in (gt_by_chunk[i] for i in group) if isinstance(x, list)
                    )
                    stats["output_gt_terms_total"] += len(gt_terms)

                row["messages"] = new_messages
                row["audios"] = new_audios
                row["gt_terms_by_chunk"] = new_gt
                row["merge_multiplier"] = None
                row["effective_merge_multipliers"] = [x["effective_lm"] for x in chunk_ranges]
                row["realadopt_chunk_reshape_policy"] = {
                    "version": "v1",
                    "unit_sec": UNIT_SEC,
                    "min_lm": args.min_lm,
                    "max_lm": args.max_lm,
                    "drop_original_lm_gt_max": True,
                    "lm_lt_min_policy": "buffer_consecutive_chunks_until_min_lm",
                    "term_map": "stripped; downstream retriever fills term_map",
                    "chunk_ranges": chunk_ranges,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["rows_written"] += 1
                stats["output_chunks"] += len(new_audios)
                stats["groups_per_row_hist"][str(len(new_audios))] += 1
                if len(samples) < args.sample_count:
                    samples.append({
                        "line": lineno,
                        "utter_id": row.get("utter_id"),
                        "original_lms": lms[:20],
                        "effective_lms": row["effective_merge_multipliers"][:20],
                        "chunk_ranges": chunk_ranges[:12],
                        "first_user": new_messages[1]["content"] if len(new_messages) > 1 else "",
                        "first_assistant": new_messages[2]["content"] if len(new_messages) > 2 else "",
                    })
            except Exception as exc:
                if args.drop_bad_rows:
                    stats["rows_dropped"] += 1
                    stats["drop_reasons"][str(exc).splitlines()[0][:200]] += 1
                    continue
                raise RuntimeError(f"Failed processing {args.input_jsonl}:{lineno}: {exc}") from exc

    tmp_output.replace(args.output_jsonl)
    stats["drop_reasons"] = dict(stats["drop_reasons"])
    stats["original_lm_hist"] = dict(sorted(stats["original_lm_hist"].items(), key=lambda kv: int(kv[0])))
    stats["output_lm_hist"] = dict(sorted(stats["output_lm_hist"].items(), key=lambda kv: int(kv[0])))
    stats["groups_per_row_hist"] = dict(stats["groups_per_row_hist"].most_common(80))
    stats["grouping_reasons"] = dict(stats["grouping_reasons"])
    stats["row_keep_rate"] = stats["rows_written"] / stats["rows_seen"] if stats["rows_seen"] else 0.0
    stats["avg_output_chunks_per_row"] = (
        stats["output_chunks"] / stats["rows_written"] if stats["rows_written"] else 0.0
    )
    stats["avg_gt_terms_per_output_chunk"] = (
        stats["output_gt_terms_total"] / stats["output_chunks"] if stats["output_chunks"] else 0.0
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
    parser.add_argument("--output-audio-dir", type=Path, required=True)
    parser.add_argument("--lang-code", default="zh")
    parser.add_argument("--min-lm", type=int, default=3)
    parser.add_argument("--max-lm", type=int, default=6)
    parser.add_argument("--assistant-join", default="")
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--drop-bad-rows", action="store_true")
    args = parser.parse_args()
    if args.min_lm <= 0 or args.max_lm < args.min_lm:
        raise ValueError("invalid min/max lm")
    if not args.input_jsonl.exists():
        raise FileNotFoundError(args.input_jsonl)
    return args


def main() -> None:
    stats = build(parse_args())
    print(json.dumps({
        "rows_seen": stats["rows_seen"],
        "rows_written": stats["rows_written"],
        "rows_dropped": stats["rows_dropped"],
        "row_keep_rate": stats["row_keep_rate"],
        "original_lm_hist": stats["original_lm_hist"],
        "output_lm_hist": stats["output_lm_hist"],
        "avg_output_chunks_per_row": stats["avg_output_chunks_per_row"],
        "avg_gt_terms_per_output_chunk": stats["avg_gt_terms_per_output_chunk"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
