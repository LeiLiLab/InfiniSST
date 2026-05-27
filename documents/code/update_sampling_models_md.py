#!/usr/bin/env python3
"""
Update the auto-generated sampling model table in documents/data/sst_omni_train_dataset.md.

This script only edits the table region between:
<!-- AUTO_SAMPLING_MODELS_START -->
...
<!-- AUTO_SAMPLING_MODELS_END -->
"""

from __future__ import annotations

import argparse
from typing import List, Tuple


START_MARK = "<!-- AUTO_SAMPLING_MODELS_START -->"
END_MARK = "<!-- AUTO_SAMPLING_MODELS_END -->"


def _find_region(lines: List[str]) -> Tuple[int, int]:
    try:
        s = next(i for i, ln in enumerate(lines) if START_MARK in ln)
        e = next(i for i, ln in enumerate(lines) if END_MARK in ln)
    except StopIteration as e:
        raise RuntimeError("Marker not found in markdown file") from e
    if e <= s:
        raise RuntimeError("Invalid marker order in markdown file")
    return s, e


def _is_table_row(line: str) -> bool:
    ln = line.strip()
    return ln.startswith("|") and ln.endswith("|") and len(ln.split("|")) >= 5


def _update_row(line: str, keep_ratio: str, hf_path: str) -> str:
    parts = [p.strip() for p in line.strip().split("|")]
    # parts: ["", keep_ratio, dataset_path, save_root, hf_model_path, ""]
    if len(parts) < 6:
        return line
    ratio = parts[1]
    if ratio != keep_ratio:
        return line
    parts[4] = hf_path
    return "| " + " | ".join(parts[1:-1]) + " |\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, help="Markdown file path to update")
    ap.add_argument("--keep-ratio", required=True, help='Keep ratio string, e.g. "0.5" or "1.0"')
    ap.add_argument("--hf-path", required=True, help="HF model output path to write into table")
    args = ap.parse_args()

    with open(args.md, "r", encoding="utf-8") as f:
        lines = f.readlines()

    s, e = _find_region(lines)
    updated = False
    for i in range(s + 1, e):
        if not _is_table_row(lines[i]):
            continue
        new_line = _update_row(lines[i], args.keep_ratio, args.hf_path)
        if new_line != lines[i]:
            lines[i] = new_line
            updated = True

    if not updated:
        raise RuntimeError(f"Row with keep_ratio={args.keep_ratio} not found in table region")

    with open(args.md, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("Done.")
    print(f"md={args.md}")
    print(f"keep_ratio={args.keep_ratio}")
    print(f"hf_path={args.hf_path}")


if __name__ == "__main__":
    main()







