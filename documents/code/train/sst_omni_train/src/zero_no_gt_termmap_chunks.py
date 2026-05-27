#!/usr/bin/env python3
"""Zero term_map entries on chunks without GT terms.

This is a narrow Speech LLM data ablation.  It keeps the original streaming
chunking, audio paths, assistant targets, and all with-GT term maps unchanged.
For chunks where ``gt_terms_by_chunk[i]`` is empty, the corresponding user
message is rewritten to ``<audio>\n\nterm_map:NONE``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def _term_map_entry_count(content: str) -> int:
    if "term_map:NONE" in content:
        return 0
    marker = "term_map:"
    idx = content.find(marker)
    if idx < 0:
        return 0
    body = content[idx + len(marker) :].strip()
    if not body:
        return 0
    return sum(1 for line in body.splitlines() if line.strip())


def _set_no_term_map(content: str) -> str:
    if not content.startswith("<audio>"):
        raise ValueError(f"Unexpected user audio message prefix: {content[:80]!r}")
    return "<audio>\n\nterm_map:NONE"


def _audio_user_count(messages: List[Any]) -> int:
    return sum(
        1
        for msg in messages
        if isinstance(msg, dict)
        and msg.get("role") == "user"
        and str(msg.get("content", "")).startswith("<audio>")
    )


def convert_row(
    obj: Dict[str, Any],
    *,
    lineno: int,
    missing_gt_policy: str = "error",
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    messages = obj.get("messages")
    gt_terms_by_chunk = obj.get("gt_terms_by_chunk")
    if not isinstance(messages, list):
        raise ValueError(f"Missing messages list at row {lineno}")
    if not isinstance(gt_terms_by_chunk, list):
        if missing_gt_policy == "keep_unchanged":
            chunks = _audio_user_count(messages)
            entry_counts = [
                _term_map_entry_count(str(msg.get("content", "")))
                for msg in messages
                if isinstance(msg, dict)
                and msg.get("role") == "user"
                and str(msg.get("content", "")).startswith("<audio>")
            ]
            entries = sum(entry_counts)
            nonempty = sum(1 for n in entry_counts if n > 0)
            obj["no_gt_termmap_zero_policy"] = {
                "version": "v1",
                "source": "gt_terms_by_chunk",
                "rule": "missing gt_terms_by_chunk; kept row unchanged because --missing-gt-policy=keep_unchanged",
            }
            return obj, {
                "chunks": chunks,
                "gt_chunks": 0,
                "no_gt_chunks": 0,
                "entries_before": entries,
                "entries_after": entries,
                "removed_entries": 0,
                "nonempty_before": nonempty,
                "nonempty_after": nonempty,
                "no_gt_entries_before": 0,
                "no_gt_entries_after": 0,
                "no_gt_nonempty_before": 0,
                "no_gt_nonempty_after": 0,
                "gt_entries_before": 0,
                "gt_entries_after": 0,
                "gt_nonempty_before": 0,
                "gt_nonempty_after": 0,
                "gt_terms_total": 0,
                "rows_missing_gt_terms_by_chunk": 1,
                "chunks_missing_gt_terms_by_chunk": chunks,
                "entries_kept_missing_gt_terms_by_chunk": entries,
            }
        raise ValueError(f"Missing gt_terms_by_chunk list at row {lineno}")

    user_indices = [
        i
        for i, msg in enumerate(messages)
        if isinstance(msg, dict)
        and msg.get("role") == "user"
        and str(msg.get("content", "")).startswith("<audio>")
    ]
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
        "removed_entries": 0,
        "nonempty_before": 0,
        "nonempty_after": 0,
        "no_gt_entries_before": 0,
        "no_gt_entries_after": 0,
        "no_gt_nonempty_before": 0,
        "no_gt_nonempty_after": 0,
        "gt_entries_before": 0,
        "gt_entries_after": 0,
        "gt_nonempty_before": 0,
        "gt_nonempty_after": 0,
        "gt_terms_total": 0,
        "rows_missing_gt_terms_by_chunk": 0,
        "chunks_missing_gt_terms_by_chunk": 0,
        "entries_kept_missing_gt_terms_by_chunk": 0,
    }

    for chunk_idx, msg_idx in enumerate(user_indices):
        msg = messages[msg_idx]
        content = str(msg.get("content", ""))
        before = _term_map_entry_count(content)
        gt_terms = gt_terms_by_chunk[chunk_idx] or []
        has_gt = bool(gt_terms)

        stats["chunks"] += 1
        stats["entries_before"] += before
        stats["nonempty_before"] += int(before > 0)
        stats["gt_terms_total"] += len(gt_terms) if isinstance(gt_terms, list) else 0

        if has_gt:
            after = before
            stats["gt_chunks"] += 1
            stats["gt_entries_before"] += before
            stats["gt_entries_after"] += after
            stats["gt_nonempty_before"] += int(before > 0)
            stats["gt_nonempty_after"] += int(after > 0)
        else:
            msg["content"] = _set_no_term_map(content)
            after = 0
            stats["no_gt_chunks"] += 1
            stats["no_gt_entries_before"] += before
            stats["no_gt_entries_after"] += after
            stats["no_gt_nonempty_before"] += int(before > 0)
            stats["no_gt_nonempty_after"] += int(after > 0)

        stats["entries_after"] += after
        stats["nonempty_after"] += int(after > 0)
        stats["removed_entries"] += before - after

    obj["no_gt_termmap_zero_policy"] = {
        "version": "v1",
        "source": "gt_terms_by_chunk",
        "rule": "if gt_terms_by_chunk[i] is empty, set user chunk i to term_map:NONE; otherwise keep original term_map",
    }
    return obj, stats


def _add_stats(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + int(v)


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--stats-json", type=Path, required=True)
    ap.add_argument("--sample-json", type=Path, default=None)
    ap.add_argument("--max-samples", type=int, default=5)
    ap.add_argument(
        "--missing-gt-policy",
        choices=["error", "keep_unchanged"],
        default="error",
        help="How to handle legacy rows without gt_terms_by_chunk. Default fails fast.",
    )
    args = ap.parse_args()

    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)
    if args.output_jsonl.exists():
        raise FileExistsError(args.output_jsonl)
    if args.stats_json.exists():
        raise FileExistsError(args.stats_json)
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
                new_obj, row_stats = convert_row(
                    obj,
                    lineno=lineno,
                    missing_gt_policy=args.missing_gt_policy,
                )
                totals["rows"] += 1
                _add_stats(totals, row_stats)
                if len(samples) < args.max_samples and row_stats["removed_entries"] > 0:
                    samples.append(new_obj)
                tmp.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    stats = dict(totals)
    stats.update(
        {
            "input_jsonl": str(args.input_jsonl),
            "output_jsonl": str(args.output_jsonl),
            "avg_entries_per_chunk_before": _rate(totals.get("entries_before", 0), totals.get("chunks", 0)),
            "avg_entries_per_chunk_after": _rate(totals.get("entries_after", 0), totals.get("chunks", 0)),
            "nonempty_rate_before": _rate(totals.get("nonempty_before", 0), totals.get("chunks", 0)),
            "nonempty_rate_after": _rate(totals.get("nonempty_after", 0), totals.get("chunks", 0)),
            "no_gt_nonempty_rate_before": _rate(totals.get("no_gt_nonempty_before", 0), totals.get("no_gt_chunks", 0)),
            "no_gt_nonempty_rate_after": _rate(totals.get("no_gt_nonempty_after", 0), totals.get("no_gt_chunks", 0)),
            "gt_nonempty_rate_before": _rate(totals.get("gt_nonempty_before", 0), totals.get("gt_chunks", 0)),
            "gt_nonempty_rate_after": _rate(totals.get("gt_nonempty_after", 0), totals.get("gt_chunks", 0)),
            "removed_entry_rate": _rate(totals.get("removed_entries", 0), totals.get("entries_before", 0)),
            "missing_gt_policy": args.missing_gt_policy,
            "missing_gt_row_rate": _rate(
                totals.get("rows_missing_gt_terms_by_chunk", 0),
                totals.get("rows", 0),
            ),
        }
    )
    tmp_path.replace(args.output_jsonl)
    args.stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.sample_json:
        args.sample_json.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
