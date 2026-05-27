#!/usr/bin/env python3
"""Plot paper Figure 4 from this package's frozen TSV snapshot."""

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
DEFAULT_DATA = SCRIPT_DIR / "data.tsv"
DEFAULT_PDF = SCRIPT_DIR / "retriever_encoder_ablation_devraw.pdf"
DEFAULT_PNG = SCRIPT_DIR / "retriever_encoder_ablation_devraw.png"

BANKS: Sequence[Tuple[str, str]] = (
    ("raw", "1K"),
    ("gs10k", "10K"),
    ("gs100k", "100K"),
)
MODELS: Sequence[Tuple[str, str]] = (
    ("main", "Qwen3-Omni-AuT + BGE-M3"),
    ("text_e5", "Qwen3-Omni-AuT + mE5"),
    ("audio_wavlm", "WavLM + BGE-M3"),
)
MODEL_STYLES = {
    "main": {"color": "#2B6CB0", "marker": "o", "linestyle": "-"},
    "text_e5": {"color": "#6B7280", "marker": "s", "linestyle": "--"},
    "audio_wavlm": {"color": "#374151", "marker": "^", "linestyle": "-."},
}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def plot(rows: List[Dict[str, str]], out_pdf: Path, out_png: Path) -> None:
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

    x = list(range(len(BANKS)))
    bank_labels = [label for _, label in BANKS]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    all_y: List[float] = []
    for model_id, model_label in MODELS:
        by_bank = {
            row["bank_id"]: float(row["recall_pct"])
            for row in rows
            if row["model_id"] == model_id
        }
        ys = [by_bank[bank_id] for bank_id, _ in BANKS]
        all_y.extend(ys)
        ax.plot(
            x,
            ys,
            label=model_label,
            linewidth=2.4,
            markersize=8.0,
            **MODEL_STYLES[model_id],
        )

    ymin = max(88.0, min(all_y) - 1.6)
    ymax = min(100.4, max(all_y) + 0.9)

    ax.set_xticks(x, bank_labels)
    ax.set_xlabel("Glossary Size")
    ax.set_ylabel("Recall@10 (%)")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.25, len(BANKS) - 0.75)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    ax.legend(
        loc="lower left",
        frameon=True,
        edgecolor="0.7",
    )
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument(
        "--update-paper",
        action="store_true",
        help="Also copy regenerated PDF/PNG into latex/figures.",
    )
    args = parser.parse_args()

    rows = load_rows(args.data)
    plot(rows, args.out_pdf, args.out_png)
    print(f"wrote {args.out_pdf}")
    print(f"wrote {args.out_png}")

    if args.update_paper:
        figure_dir = PAPER_DIR / "latex/figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out_pdf, figure_dir / "retriever_encoder_ablation_devraw.pdf")
        shutil.copy2(args.out_png, figure_dir / "retriever_encoder_ablation_devraw.png")
        print(f"updated {figure_dir / 'retriever_encoder_ablation_devraw.pdf'}")
        print(f"updated {figure_dir / 'retriever_encoder_ablation_devraw.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
