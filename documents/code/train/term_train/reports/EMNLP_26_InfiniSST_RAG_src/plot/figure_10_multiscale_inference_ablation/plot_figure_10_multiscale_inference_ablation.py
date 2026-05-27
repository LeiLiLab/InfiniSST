#!/usr/bin/env python3
"""Plot the paper multi-scale inference ablation from a frozen TSV snapshot."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATA = SCRIPT_DIR / "data.tsv"
DEFAULT_OUT = SCRIPT_DIR / "local" / "multiscale_inference_ablation_devraw.pdf"

PRIMARY_VARIANTS: Sequence[str] = (
    "multiscale_maxsim",
    "only_largest_maxsim_window",
    "dense_1p92_trained",
)

VARIANT_LABELS = {
    "multiscale_maxsim": "Multi-scale",
    "only_largest_maxsim_window": "Largest-infer",
    "dense_1p92_trained": "Largest-train",
}

RECALL_SERIES = (
    ("recall_at_10", "1K", "#4C78A8", ""),
    ("recall_at_10_gs10000", "10K", "#F58518", "//"),
    ("recall_at_10_gs100000", "100K", "#54A24B", "xx"),
)

def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def as_percent(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "NA", "nan"}:
        return math.nan
    return 100.0 * float(value)


def primary_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    by_variant = {row["variant"]: row for row in rows}
    missing = [variant for variant in PRIMARY_VARIANTS if variant not in by_variant]
    if missing:
        raise ValueError(f"missing primary variants: {missing}")
    return [by_variant[variant] for variant in PRIMARY_VARIANTS]


def grouped_bars(
    ax: plt.Axes,
    rows: Sequence[Dict[str, str]],
    series: Sequence[tuple[str, str, str, str]],
    width: float,
) -> None:
    xs = list(range(len(rows)))
    offsets = [width * (idx - (len(series) - 1) / 2.0) for idx in range(len(series))]
    for offset, (key, label, color, hatch) in zip(offsets, series):
        ys = [as_percent(row, key) for row in rows]
        ax.bar(
            [x + offset for x in xs],
            ys,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            hatch=hatch,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([VARIANT_LABELS[row["variant"]] for row in rows])
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)


def annotate_deltas(ax: plt.Axes, rows: Sequence[Dict[str, str]], key: str) -> None:
    baseline = as_percent(rows[0], key)
    for x, row in enumerate(rows[1:], start=1):
        value = as_percent(row, key)
        if math.isnan(value):
            continue
        ax.text(
            x,
            value + 0.6,
            f"{value - baseline:+.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot(rows: Sequence[Dict[str, str]], out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.4))
    grouped_bars(ax, rows, RECALL_SERIES, width=0.24)
    ax.set_ylabel("Recall@10 (%)")
    ax.set_ylim(90, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        bbox_to_anchor=(0.5, 0.02),
        borderaxespad=0.1,
    )
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

    rows = primary_rows(read_rows(args.data))
    plot(rows, args.out)
    print(f"[OK] wrote {args.out}")
    print(f"[OK] wrote {args.out.with_suffix('.png')}")
    if args.update_paper:
        figure_dir = PAPER_DIR / "latex/figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            src = args.out.with_suffix(suffix)
            dst = figure_dir / f"multiscale_inference_ablation_devraw{suffix}"
            shutil.copy2(src, dst)
            print(f"[OK] updated {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
