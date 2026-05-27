#!/usr/bin/env python3
"""Plot paper Figure 3 from a frozen paper-local TSV snapshot."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATA = SCRIPT_DIR / "data.tsv"
DEFAULT_OUT = SCRIPT_DIR / "rag_compute_rtf.pdf"

LMS: Sequence[int] = (1, 2, 3, 4)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def select_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_lm = {int(row["lm"]): row for row in rows}
    missing = [lm for lm in LMS if lm not in by_lm]
    if missing:
        raise ValueError(f"missing lm rows: {missing}")
    return [by_lm[lm] for lm in LMS]


def plot(rows: List[Dict[str, str]], out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    xs = [float(row["streamlaal_sec"]) * 1000.0 for row in rows]
    mean_rtf = [float(row["rag_mean_rtf_pct"]) for row in rows]
    median_ms = [float(row["rag_median_ms"]) for row in rows]

    metrics = [
        ("Retriever RTF (%)", mean_rtf, "#D6604D", "o"),
        ("Retriever Time\nper Call (ms)", median_ms, "#4393C3", "D"),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(len(metrics), 1, figsize=(4.2, 5.2), sharex=True)
    for ax, (ylabel, ys, color, marker) in zip(axes, metrics):
        ax.plot(
            xs,
            ys,
            marker=marker,
            linewidth=2.0,
            markersize=7.0,
            color=color,
        )
        ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel("StreamLAAL (ms)")
    fig.tight_layout(h_pad=1.0)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--update-paper",
        action="store_true",
        help="Also copy regenerated PDF/PNG into latex/figures.",
    )
    args = parser.parse_args()

    rows = select_rows(read_rows(args.data))
    plot(rows, args.out)
    print(f"[OK] wrote {args.out}")
    print(f"[OK] wrote {args.out.with_suffix('.png')}")
    if args.update_paper:
        figure_dir = PAPER_DIR / "latex/figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out, figure_dir / "rag_compute_rtf.pdf")
        shutil.copy2(
            args.out.with_suffix(".png"),
            figure_dir / "rag_compute_rtf.png",
        )
        print(f"[OK] updated {figure_dir / 'rag_compute_rtf.pdf'}")
        print(f"[OK] updated {figure_dir / 'rag_compute_rtf.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
