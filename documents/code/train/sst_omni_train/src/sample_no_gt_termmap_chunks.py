#!/usr/bin/env python3
"""Deterministically sample term_map entries on chunks without GT terms.

This repair keeps chunks with GT terms unchanged. For chunks where
``gt_terms_by_chunk[i]`` is empty, it samples retrieved term_map entries with a
stable hash. Empty sampled no-GT chunks are rewritten to ``term_map:NONE``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


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


def _row_key(obj: Mapping[str, Any], lineno: int) -> str:
    utter_id = str(obj.get("utter_id") or "").strip()
    if utter_id:
        return utter_id
    audios = obj.get("audios")
    if isinstance(audios, list) and audios:
        return str(audios[0])
    return f"line:{lineno}"


def _stable_float(seed: str, row_key: str, chunk_idx: int) -> float:
    payload = f"{seed}\t{row_key}\t{chunk_idx}".encode("utf-8", errors="replace")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _stable_entry_float(seed: str, row_key: str, chunk_idx: int, entry_idx: int, entry: str) -> float:
    payload = f"{seed}\t{row_key}\t{chunk_idx}\t{entry_idx}\t{entry}".encode(
        "utf-8",
        errors="replace",
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _parse_term_map_entries(content: str) -> List[str]:
    if "term_map:NONE" in content:
        return []
    marker = "term_map:"
    idx = content.find(marker)
    if idx < 0:
        return []
    body = content[idx + len(marker) :]
    entries: List[str] = []
    invalid: List[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper() == "NONE":
            continue
        if "=" not in line:
            invalid.append(line)
            continue
        src, tgt = line.split("=", 1)
        if not src.strip() or not tgt.strip():
            invalid.append(line)
            continue
        entries.append(line)
    if invalid:
        raise ValueError(f"Malformed term_map lines: {invalid[:5]!r}")
    return entries


def _format_audio_term_map(entries: Sequence[str]) -> str:
    if not entries:
        return "<audio>\n\nterm_map:NONE"
    return "<audio>\n\nterm_map:\n" + "\n".join(entries)


def _sample_entries(
    entries: Sequence[str],
    *,
    keep_prob: float,
    max_terms: int,
    sample_unit: str,
    seed: str,
    row_key: str,
    chunk_idx: int,
) -> Tuple[List[str], bool, str, int]:
    if not entries:
        return [], False, "empty_before", 0
    if sample_unit == "chunk":
        keep = _stable_float(seed, row_key, chunk_idx) < keep_prob
        if not keep:
            return [], False, "sampled_empty", 0
        sampled = list(entries)
        sampled_before_cap = len(sampled)
    elif sample_unit == "term":
        sampled = [
            entry
            for entry_idx, entry in enumerate(entries)
            if _stable_entry_float(seed, row_key, chunk_idx, entry_idx, entry) < keep_prob
        ]
        sampled_before_cap = len(sampled)
        if not sampled:
            return [], False, "sampled_empty", sampled_before_cap
    else:
        raise ValueError(f"Unsupported sample_unit={sample_unit!r}")
    if max_terms > 0:
        sampled = sampled[:max_terms]
    return sampled, True, "sampled_keep", sampled_before_cap


def _convert_row(
    obj: MutableMapping[str, Any],
    *,
    lineno: int,
    keep_prob: float,
    max_terms: int,
    sample_unit: str,
    seed: str,
) -> Tuple[MutableMapping[str, Any], Dict[str, int]]:
    messages = obj.get("messages")
    gt_terms_by_chunk = obj.get("gt_terms_by_chunk")
    if not isinstance(messages, list):
        raise ValueError(f"Missing messages list at row {lineno}")
    if not isinstance(gt_terms_by_chunk, list):
        raise ValueError(f"Missing gt_terms_by_chunk list at row {lineno}")

    user_indices = _audio_user_indices(messages)
    if len(user_indices) != len(gt_terms_by_chunk):
        raise ValueError(
            f"Row {lineno} user audio messages ({len(user_indices)}) "
            f"!= gt_terms_by_chunk ({len(gt_terms_by_chunk)})"
        )

    stats = {
        "chunks": 0,
        "gt_chunks": 0,
        "no_gt_chunks": 0,
        "entries_before": 0,
        "entries_after": 0,
        "nonempty_before": 0,
        "nonempty_after": 0,
        "gt_entries_before": 0,
        "gt_entries_after": 0,
        "gt_nonempty_before": 0,
        "gt_nonempty_after": 0,
        "gt_terms_total": 0,
        "no_gt_entries_before": 0,
        "no_gt_entries_after": 0,
        "no_gt_nonempty_before": 0,
        "no_gt_nonempty_after": 0,
        "no_gt_sampled_keep": 0,
        "no_gt_sampled_empty": 0,
        "no_gt_empty_before": 0,
        "no_gt_entries_removed": 0,
        "no_gt_entries_capped": 0,
        "no_gt_terms_sampled_before_cap": 0,
    }

    row_key = _row_key(obj, lineno)
    for chunk_idx, msg_idx in enumerate(user_indices):
        msg = messages[msg_idx]
        content = str(msg.get("content") or "")
        if not content.startswith("<audio>"):
            raise ValueError(f"Row {lineno} chunk {chunk_idx}: unexpected user content prefix")
        entries = _parse_term_map_entries(content)
        before = len(entries)
        gt_terms = gt_terms_by_chunk[chunk_idx] or []
        has_gt = bool(gt_terms)

        stats["chunks"] += 1
        stats["entries_before"] += before
        stats["nonempty_before"] += int(before > 0)

        if has_gt:
            stats["gt_chunks"] += 1
            stats["gt_entries_before"] += before
            stats["gt_entries_after"] += before
            stats["gt_nonempty_before"] += int(before > 0)
            stats["gt_nonempty_after"] += int(before > 0)
            stats["gt_terms_total"] += len(gt_terms) if isinstance(gt_terms, list) else 0
            after = before
        else:
            sampled, kept, reason, sampled_before_cap = _sample_entries(
                entries,
                keep_prob=keep_prob,
                max_terms=max_terms,
                sample_unit=sample_unit,
                seed=seed,
                row_key=row_key,
                chunk_idx=chunk_idx,
            )
            msg["content"] = _format_audio_term_map(sampled)
            after = len(sampled)
            stats["no_gt_chunks"] += 1
            stats["no_gt_entries_before"] += before
            stats["no_gt_entries_after"] += after
            stats["no_gt_nonempty_before"] += int(before > 0)
            stats["no_gt_nonempty_after"] += int(after > 0)
            stats["no_gt_sampled_keep"] += int(kept)
            stats["no_gt_sampled_empty"] += int(reason == "sampled_empty")
            stats["no_gt_empty_before"] += int(reason == "empty_before")
            stats["no_gt_entries_removed"] += before - after
            stats["no_gt_entries_capped"] += int(kept and max_terms > 0 and before > max_terms)
            stats["no_gt_terms_sampled_before_cap"] += sampled_before_cap

        stats["entries_after"] += after
        stats["nonempty_after"] += int(after > 0)

    obj["no_gt_termmap_sampling_policy"] = {
        "version": "v1",
        "source": "gt_terms_by_chunk",
        "rule": "if gt_terms_by_chunk[i] is empty, deterministically sample retrieved term_map entries according to sample_unit and keep_prob; empty sampled chunks become term_map:NONE; GT chunks unchanged",
        "sample_unit": sample_unit,
        "keep_prob": keep_prob,
        "max_no_gt_terms": max_terms,
        "seed": seed,
    }
    return obj, stats


def _add_stats(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + int(value)


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--stats-json", type=Path, required=True)
    ap.add_argument("--sample-json", type=Path, default=None)
    ap.add_argument("--keep-prob", type=float, required=True)
    ap.add_argument("--max-no-gt-terms", type=int, default=0)
    ap.add_argument(
        "--sample-unit",
        choices=["term", "chunk"],
        default="term",
        help="term samples each no-GT term_map entry independently; chunk samples whole no-GT term_map blocks.",
    )
    ap.add_argument("--seed", default="20260525_sample50")
    ap.add_argument("--max-samples", type=int, default=20)
    args = ap.parse_args()

    if not 0.0 <= args.keep_prob <= 1.0:
        raise ValueError("--keep-prob must be in [0, 1]")
    if args.max_no_gt_terms < 0:
        raise ValueError("--max-no-gt-terms must be >= 0")
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    for out in [args.output_jsonl, args.stats_json, args.sample_json]:
        if out and out.exists():
            raise FileExistsError(out)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)

    totals: Dict[str, int] = {"rows": 0}
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
                converted, row_stats = _convert_row(
                    obj,
                    lineno=lineno,
                    keep_prob=args.keep_prob,
                    max_terms=args.max_no_gt_terms,
                    sample_unit=args.sample_unit,
                    seed=args.seed,
                )
                totals["rows"] += 1
                _add_stats(totals, row_stats)
                if len(samples) < args.max_samples and row_stats["no_gt_entries_removed"] > 0:
                    samples.append({
                        "lineno": lineno,
                        "utter_id": converted.get("utter_id"),
                        "row_stats": row_stats,
                    })
                tmp.write(json.dumps(converted, ensure_ascii=False) + "\n")
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(args.output_jsonl)

    stats = dict(totals)
    stats.update({
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "keep_prob": args.keep_prob,
        "max_no_gt_terms": args.max_no_gt_terms,
        "sample_unit": args.sample_unit,
        "seed": args.seed,
        "avg_entries_per_chunk_before": _rate(totals.get("entries_before", 0), totals.get("chunks", 0)),
        "avg_entries_per_chunk_after": _rate(totals.get("entries_after", 0), totals.get("chunks", 0)),
        "nonempty_rate_before": _rate(totals.get("nonempty_before", 0), totals.get("chunks", 0)),
        "nonempty_rate_after": _rate(totals.get("nonempty_after", 0), totals.get("chunks", 0)),
        "avg_no_gt_entries_before": _rate(totals.get("no_gt_entries_before", 0), totals.get("no_gt_chunks", 0)),
        "avg_no_gt_entries_after": _rate(totals.get("no_gt_entries_after", 0), totals.get("no_gt_chunks", 0)),
        "no_gt_nonempty_rate_before": _rate(totals.get("no_gt_nonempty_before", 0), totals.get("no_gt_chunks", 0)),
        "no_gt_nonempty_rate_after": _rate(totals.get("no_gt_nonempty_after", 0), totals.get("no_gt_chunks", 0)),
        "no_gt_entry_keep_rate": _rate(totals.get("no_gt_entries_after", 0), totals.get("no_gt_entries_before", 0)),
        "gt_nonempty_rate_before": _rate(totals.get("gt_nonempty_before", 0), totals.get("gt_chunks", 0)),
        "gt_nonempty_rate_after": _rate(totals.get("gt_nonempty_after", 0), totals.get("gt_chunks", 0)),
    })
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.sample_json:
        args.sample_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
