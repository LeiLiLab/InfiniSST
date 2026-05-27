#!/usr/bin/env python3
"""Plot paper Figure 7 from a frozen paper-local TSV snapshot."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATA = SCRIPT_DIR / "data.tsv"
DEFAULT_OUT = SCRIPT_DIR / "local" / "glossary_bank_ablation_zh_fixedraw.pdf"

BANKS: Sequence[str] = ("raw", "gs1k", "gs10k")
LMS: Sequence[int] = (1, 2, 3, 4)
COLORS = {"raw": "#D6604D", "gs1k": "#4393C3", "gs10k": "#5AAE61"}
LABELS = {"raw": "0.2K", "gs1k": "1K", "gs10k": "10k"}
MARKERS = {"raw": "*", "gs1k": "o", "gs10k": "D"}

NO_RETRIEVAL_STREAMLAAL: Sequence[float] = (1181.1470, 1765.7196, 2232.6733, 2616.3493)
NO_RETRIEVAL_BLEU: Sequence[float] = (40.6663, 45.8268, 46.7119, 47.3897)
NO_RETRIEVAL_TERM_ACC: Sequence[float] = (74.31, 76.55, 76.75, 77.54)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def tagged_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("dataset") == "Tagged ACL"
        and row.get("lang") == "zh"
        and row.get("runtime_bank") in BANKS
    ]
    expected = {(bank, str(lm)) for bank in BANKS for lm in LMS}
    seen = {(row["runtime_bank"], row["lm"]) for row in selected}
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"missing Tagged ACL rows: {missing}")
    return selected


def xy_values(
    rows: Sequence[Dict[str, str]],
    bank: str,
    y_key: str,
) -> Tuple[List[float], List[float]]:
    by_key = {(row["runtime_bank"], int(row["lm"])): row for row in rows}
    xs: List[float] = []
    ys: List[float] = []
    for lm in LMS:
        row = by_key[(bank, lm)]
        xs.append(float(row["StreamLAAL"]))
        ys.append(float(row[y_key]))
    return xs, ys


def plot(rows: Sequence[Dict[str, str]], out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("TERM_ACC", "Terminology\nAccuracy (%)"),
        ("BLEU", "BLEU Score"),
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
    no_retrieval_ys = {"TERM_ACC": NO_RETRIEVAL_TERM_ACC, "BLEU": NO_RETRIEVAL_BLEU}
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.plot(
            NO_RETRIEVAL_STREAMLAAL,
            no_retrieval_ys[metric],
            linestyle=":",
            linewidth=2.0,
            marker="x",
            markersize=6.0,
            color="black",
            label="w/o retrieval",
        )
        for bank in BANKS:
            xs, ys = xy_values(rows, bank, metric)
            ax.plot(
                xs,
                ys,
                marker=MARKERS[bank],
                linewidth=2.0,
                markersize=7.0 if MARKERS[bank] != '*' else 10,
                color=COLORS[bank],
                label=LABELS[bank],
            )
        ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel("StreamLAAL (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(h_pad=1.0)
    fig.subplots_adjust(bottom=0.24)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        bbox_to_anchor=(0.5, 0.0),
        borderaxespad=0.2,
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

    rows = tagged_rows(read_rows(args.data))
    plot(rows, args.out)
    print(f"[OK] wrote {args.out}")
    print(f"[OK] wrote {args.out.with_suffix('.png')}")
    if args.update_paper:
        figure_dir = PAPER_DIR / "latex/figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out, figure_dir / "glossary_bank_ablation_zh_fixedraw.pdf")
        shutil.copy2(
            args.out.with_suffix(".png"),
            figure_dir / "glossary_bank_ablation_zh_fixedraw.png",
        )
        print(f"[OK] updated {figure_dir / 'glossary_bank_ablation_zh_fixedraw.pdf'}")
        print(f"[OK] updated {figure_dir / 'glossary_bank_ablation_zh_fixedraw.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
