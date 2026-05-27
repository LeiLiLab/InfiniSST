#!/usr/bin/env python3
"""Plot paper Figure 5 from this package's frozen TSV snapshots."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parents[1]
DEFAULT_CURVES = SCRIPT_DIR / "data_multibank.tsv"
DEFAULT_DROP1 = SCRIPT_DIR / "data_drop1_points.tsv"
DEFAULT_PREFIX = SCRIPT_DIR / "retriever_dev_pr_fixedraw_hn_comparison"

BANKS: Sequence[Tuple[str, str]] = (
    ("raw (~1k)", "Glossary 1K"),
    ("gs10k", "Glossary 10K"),
    ("gs100k", "Glossary 100K"),
)
MODELS: Sequence[str] = ("no-HN", "HN256", "HN1024")
MODEL_DISPLAY = {
    "no-HN": r"$N=0$",
    "HN256": r"$N=256$",
    "HN512": r"$N=512$",
    "HN1024": r"$N=1024$",
}
MODEL_STYLES = {
    "no-HN": {"color": "#2b6cb0", "linestyle": ":", "linewidth": 2.6},
    "HN256": {"color": "#3a923a", "linestyle": "--", "linewidth": 2.4},
    "HN512": {"color": "#9467bd", "linestyle": "-.", "linewidth": 2.4},
    "HN1024": {"color": "#d62728", "linestyle": "-", "linewidth": 2.4},
}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def plot(
    curves: List[Dict[str, str]],
    drop1: List[Dict[str, str]],
    output_prefix: Path,
    models: Sequence[str] = MODELS,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, len(BANKS), figsize=(12.5, 4.6), sharey=True)
    handles: List[object] = []
    labels: List[str] = []

    drop1_by_bank = {row["bank"]: row for row in drop1}

    for col, (bank, title) in enumerate(BANKS):
        ax = axes[col]
        for model in models:
            xs = [
                float(row["precision"])
                for row in curves
                if row["bank"] == bank and row["model"] == model
            ]
            ys = [
                float(row["recall_drop_from_raw_ref"])
                for row in curves
                if row["bank"] == bank and row["model"] == model
            ]
            paired = sorted(zip(xs, ys), key=lambda p: p[0])
            xs = [p[0] for p in paired]
            ys = [p[1] for p in paired]
            display = MODEL_DISPLAY[model]
            line = ax.plot(xs, ys, label=display, **MODEL_STYLES[model])[0]
            if col == 0 and display not in labels:
                handles.append(line)
                labels.append(display)

        ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.2)
        if bank in drop1_by_bank:
            row = drop1_by_bank[bank]
            tau = float(row["tau"])
            tau_pos = {
                "raw (~1k)": (0.28, 0.42),
                "gs10k": (0.28, 0.42),
                "gs100k": (0.65, 0.58),
            }[bank]
            ax.text(
                tau_pos[0],
                tau_pos[1],
                rf"$\tau={tau:.3f}$",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#d62728",
                fontsize=15,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Precision (%)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax.set_xlim(10.0, 20.0)
        ax.set_ylim(2.0, 0.0)

    axes[0].set_ylabel(r"Recall drop from $\tau=0$" + "\n" + r"and glossary 1K (%)")

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            frameon=True,
            bbox_to_anchor=(0.5, 0.0),
            columnspacing=1.8,
            handlelength=2.4,
        )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0), w_pad=1.4)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--drop1", type=Path, default=DEFAULT_DROP1)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument(
        "--update-paper",
        action="store_true",
        help="Also copy regenerated PDF/PNG into latex/figures.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Comma-separated model order to plot; defaults to the paper-frozen set.",
    )
    args = parser.parse_args()

    curves = load_rows(args.curves)
    drop1 = load_rows(args.drop1)
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    plot(curves, drop1, args.out_prefix, models=models)
    print(f"wrote {args.out_prefix.with_suffix('.pdf')}")
    print(f"wrote {args.out_prefix.with_suffix('.png')}")

    if args.update_paper:
        figure_dir = PAPER_DIR / "latex/figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            args.out_prefix.with_suffix(".pdf"),
            figure_dir / "retriever_dev_pr_fixedraw_hn_comparison.pdf",
        )
        shutil.copy2(
            args.out_prefix.with_suffix(".png"),
            figure_dir / "retriever_dev_pr_fixedraw_hn_comparison.png",
        )
        print(f"updated {figure_dir / 'retriever_dev_pr_fixedraw_hn_comparison.pdf'}")
        print(f"updated {figure_dir / 'retriever_dev_pr_fixedraw_hn_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
