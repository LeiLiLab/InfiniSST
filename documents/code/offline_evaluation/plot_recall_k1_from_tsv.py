#!/usr/bin/env python3

"""
Plot Recall@K1 saturation curve from an existing TSV file.

This script intentionally generates PDF only (no PNG).
All log messages are in English.
"""

from __future__ import annotations

# ======Configuration=====
DEFAULT_TSV_PATH = "/mnt/gemini/data2/jiaxuanluo/offline_eval_k1_saturation_gigaspeech_dev/recall_k1_saturation.tsv"
DEFAULT_PDF_PATH = "/mnt/gemini/data2/jiaxuanluo/offline_eval_k1_saturation_gigaspeech_dev/recall_k1_saturation.pdf"

TSV_DELIMITER = "\t"
REQUIRED_COLUMNS = ["k1", "recall"]

PLOT_TITLE = "GigaSpeech dev: Recall@K1 saturation (chunk=1.92s)"
X_LABEL = "K1"
Y_LABEL = "Recall@K1"

PLOT_FIGSIZE = (8.0, 4.8)
PLOT_LINEWIDTH = 2.0
PLOT_MARKER = "o"
PLOT_GRID_ALPHA = 0.3
PLOT_DPI = 300
# ======Configuration=====

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _err(msg: str) -> None:
    raise RuntimeError(msg)


def _read_tsv(tsv_path: Path) -> Tuple[List[int], List[float]]:
    if not tsv_path.exists():
        _err(f"TSV not found: {tsv_path}")

    rows: List[Dict[str, Any]] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=TSV_DELIMITER)
        if reader.fieldnames is None:
            _err(f"TSV has no header: {tsv_path}")
        for col in REQUIRED_COLUMNS:
            if col not in reader.fieldnames:
                _err(f"Missing required column {col!r}. Available columns: {reader.fieldnames}")
        for r in reader:
            rows.append(r)

    if not rows:
        _err(f"TSV is empty: {tsv_path}")

    xs: List[int] = []
    ys: List[float] = []
    for r in rows:
        k1 = int(str(r["k1"]).strip())
        recall = float(str(r["recall"]).strip())
        xs.append(k1)
        ys.append(recall)

    pairs = sorted(zip(xs, ys), key=lambda x: x[0])
    xs_sorted = [p[0] for p in pairs]
    ys_sorted = [p[1] for p in pairs]
    return xs_sorted, ys_sorted


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Recall@K1 curve from TSV and save as PDF.")
    parser.add_argument("--tsv", type=str, default=DEFAULT_TSV_PATH, help="Input TSV path (tab-separated).")
    parser.add_argument("--out_pdf", type=str, default=DEFAULT_PDF_PATH, help="Output PDF path.")
    args = parser.parse_args()

    tsv_path = Path(args.tsv)
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    xs, ys = _read_tsv(tsv_path)
    _log(f"Loaded points: {len(xs)}")
    _log(f"Input TSV: {tsv_path}")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        _err(f"Missing dependency: matplotlib. Error: {e}")

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.plot(xs, ys, marker=PLOT_MARKER, linewidth=PLOT_LINEWIDTH)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title(PLOT_TITLE)
    plt.grid(True, alpha=PLOT_GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=PLOT_DPI, format="pdf")
    plt.close()

    _log(f"Wrote PDF: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

