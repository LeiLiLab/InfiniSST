#!/usr/bin/env python3
"""Plot retriever training-data ablation from the frozen paper TSV."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATA = SCRIPT_DIR / "data.tsv"
DEFAULT_PDF = SCRIPT_DIR / "retriever_data_ablation_dev.pdf"
DEFAULT_PNG = SCRIPT_DIR / "retriever_data_ablation_dev.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 2:
        raise ValueError(f"expected exactly 2 ablation rows, found {len(rows)}")
    required = [
        "setting",
        "eval_dev_recall_at10_gs10000",
        "eval_dev_filtered_recall_tau_0p70_gs10000",
        "eval_dev_filtered_recall_tau_0p75_gs10000",
        "eval_dev_filtered_recall_tau_0p80_gs10000",
        "eval_dev_filtered_recall_tau_0p85_gs10000",
    ]
    for row in rows:
        for key in required:
            if key not in row or row[key] == "":
                raise ValueError(f"missing {key} for {row.get('setting', '<unknown>')}")
            if key != "setting" and not math.isfinite(float(row[key])):
                raise ValueError(f"non-finite {key} for {row['setting']}")
    return rows


def pct(value: str) -> float:
    return 100.0 * float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--update-paper", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.data)
    colors = {
        "Main": "#2563eb",
        "GigaSpeech only": "#9ca3af",
    }
    labels = {
        "Main": "GigaSpeech + Wiki",
        "GigaSpeech only": "GigaSpeech only",
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 18,
            "legend.fontsize": 17,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.0))

    tau_values = [0.70, 0.75, 0.80, 0.85]
    tau_keys = [
        "eval_dev_filtered_recall_tau_0p70_gs10000",
        "eval_dev_filtered_recall_tau_0p75_gs10000",
        "eval_dev_filtered_recall_tau_0p80_gs10000",
        "eval_dev_filtered_recall_tau_0p85_gs10000",
    ]
    for row in rows:
        ys = [pct(row[key]) for key in tau_keys]
        setting = row["setting"]
        ax.plot(
            tau_values,
            ys,
            marker="o",
            markersize=6,
            linewidth=2.4,
            color=colors[setting],
            label=labels[setting],
        )
        for x, y in zip(tau_values, ys):
            ax.text(x, y + 0.22, f"{y:.1f}", ha="center", va="bottom", fontsize=11, color=colors[setting])
    ax.set_xlabel(r"Retriever score threshold ($\tau$)")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(91.5, 99.4)
    ax.set_xticks(tau_values, [f"{x:.2f}" for x in tau_values])
    ax.grid(axis="both", color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower left", frameon=True, edgecolor="0.7")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if args.update_paper:
        paper_pdf = PAPER_DIR / "latex/figures/retriever_data_ablation_dev.pdf"
        paper_png = PAPER_DIR / "latex/figures/retriever_data_ablation_dev.png"
        shutil.copy2(args.out_pdf, paper_pdf)
        shutil.copy2(args.out_png, paper_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
