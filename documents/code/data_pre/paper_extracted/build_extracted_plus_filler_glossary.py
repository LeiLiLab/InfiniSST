#!/usr/bin/env python3
"""Build a per-paper extracted glossary expanded with filler terms.

The output preserves all extracted entries first and then appends filler entries
until the requested size is reached.  Duplicate source terms are ignored with
the extracted glossary taking priority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _entries(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, dict):
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            out = dict(entry)
            out.setdefault("term", key)
            yield out
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                yield dict(entry)
    else:
        raise ValueError(f"unsupported glossary JSON type: {type(data).__name__}")


def _key(entry: Dict[str, Any]) -> str:
    return " ".join(str(entry.get("term") or "").split()).casefold()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted", required=True, type=Path)
    parser.add_argument("--filler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=10000)
    args = parser.parse_args()

    extracted_data = json.loads(args.extracted.read_text(encoding="utf-8"))
    filler_data = json.loads(args.filler.read_text(encoding="utf-8"))

    output: Dict[str, Dict[str, Any]] = {}
    skipped_missing_term = 0
    duplicate_filler = 0

    for entry in _entries(extracted_data):
        key = _key(entry)
        if not key:
            skipped_missing_term += 1
            continue
        entry.setdefault("source", "paper_extracted")
        output[key] = entry

    extracted_kept = len(output)
    for entry in _entries(filler_data):
        if len(output) >= args.target_size:
            break
        key = _key(entry)
        if not key:
            skipped_missing_term += 1
            continue
        if key in output:
            duplicate_filler += 1
            continue
        entry.setdefault("source", "filler")
        output[key] = entry

    if len(output) != args.target_size:
        raise SystemExit(
            f"could only build {len(output)} entries, target={args.target_size}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    stats = {
        "output": str(args.output),
        "target_size": args.target_size,
        "output_entries": len(output),
        "extracted_kept": extracted_kept,
        "filler_added": len(output) - extracted_kept,
        "duplicate_filler": duplicate_filler,
        "skipped_missing_term": skipped_missing_term,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
