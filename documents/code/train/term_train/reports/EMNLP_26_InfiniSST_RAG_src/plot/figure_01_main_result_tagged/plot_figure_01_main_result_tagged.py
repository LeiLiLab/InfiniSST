#!/usr/bin/env python3
"""Plot paper Figure 1 from this package's frozen TSV snapshot."""

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
DEFAULT_PREFIX = SCRIPT_DIR / "new_main_result_tagged"

DATASET = "acl_tagged_raw"
LANGS: Sequence[Tuple[str, str]] = (("zh", "En-Zh"), ("de", "En-De"), ("ja", "En-Ja"))
OFFLINE_METHODS = ("Offline ST", "Offline + GT terms")
PLOT_METHODS = (*OFFLINE_METHODS, "InfiniSST", "RASST")
METHOD_DISPLAY = {"Offline ST": "Offline"}

METHOD_STYLES = {
    "Offline ST": {"color": "#3a923a", "linestyle": "--", "linewidth": 2.8},
    "Offline + GT terms": {"color": "#805ad5", "linestyle": "-.", "linewidth": 2.8},
    "InfiniSST": {
        "color": "#2b6cb0",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.8,
        "markersize": 10.0,
    },
    "RASST": {
        "color": "#d62728",
        "marker": "*",
        "linestyle": "-",
        "linewidth": 3.0,
        "markersize": 14.0,
    },
}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def finite(value: str) -> float | None:
    if value in {"", "NA"}:
        return None
    return float(value)


def plot_dataset(rows: Sequence[Dict[str, str]], output_prefix: Path) -> None:
    data = [r for r in rows if r["dataset"] == DATASET]
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
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.6))
    handles: List[object] = []
    labels: List[str] = []

    for col, (lang, title) in enumerate(LANGS):
        lang_rows = [r for r in data if r["lang"] == lang]
        for row_idx, metric in enumerate(("TERM_ACC", "BLEU")):
            ax = axes[row_idx][col]
            x_values: List[float] = []
            y_values: List[float] = []
            for method in PLOT_METHODS:
                method_rows = [r for r in lang_rows if r["method"] == method]
                display = METHOD_DISPLAY.get(method, method)
                if method in OFFLINE_METHODS:
                    offline = next((r for r in method_rows if finite(r[metric]) is not None), None)
                    if offline:
                        y = finite(offline[metric])
                        assert y is not None
                        y_scaled = y * (100.0 if metric == "TERM_ACC" else 1.0)
                        line = ax.axhline(y_scaled, label=display, **METHOD_STYLES[method])
                        y_values.append(y_scaled)
                        if col == 0 and row_idx == 0:
                            handles.append(line)
                            labels.append(display)
                    continue
                points: List[Tuple[float, float]] = []
                for r in sorted(
                    method_rows,
                    key=lambda item: int(item["lm"]) if item["lm"].isdigit() else 99,
                ):
                    x = finite(r["StreamLAAL"])
                    y = finite(r[metric])
                    if x is None or y is None:
                        continue
                    points.append((x, y * (100.0 if metric == "TERM_ACC" else 1.0)))
                if not points:
                    continue
                line = ax.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    label=display,
                    **METHOD_STYLES[method],
                )[0]
                x_values.extend(p[0] for p in points)
                y_values.extend(p[1] for p in points)
                if col == 0 and row_idx == 0 and display not in labels:
                    handles.append(line)
                    labels.append(display)

            if x_values:
                x_low = min(x_values)
                x_high = max(x_values)
                pad = max((x_high - x_low) * 0.10, 90.0)
                ax.set_xlim(x_low - pad, x_high + pad)
            if y_values:
                y_low = min(y_values)
                y_high = max(y_values)
                pad = max((y_high - y_low) * 0.10, 1.2 if metric == "BLEU" else 1.8)
                ax.set_ylim(y_low - pad, y_high + pad)
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
            if row_idx == 0:
                ax.set_title(title, fontweight="bold")
            else:
                ax.set_xlabel("StreamLAAL (ms)")
            if col == 0:
                ax.set_ylabel(
                    "Terminology\nAccuracy (%)" if metric == "TERM_ACC" else "BLEU Score"
                )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(1, len(labels)),
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
            columnspacing=1.8,
            handlelength=2.4,
        )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), w_pad=1.4, h_pad=1.8)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument(
        "--update-paper",
        action="store_true",
        help="Also copy regenerated PDF/PNG into latex/figures.",
    )
    args = parser.parse_args()

    rows = load_rows(args.data)
    plot_dataset(rows, args.out_prefix)
    print(f"wrote {args.out_prefix.with_suffix('.pdf')}")
    print(f"wrote {args.out_prefix.with_suffix('.png')}")

    if args.update_paper:
        figure_dir = PAPER_DIR / "latex/figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out_prefix.with_suffix(".pdf"), figure_dir / "new_main_result_tagged.pdf")
        shutil.copy2(args.out_prefix.with_suffix(".png"), figure_dir / "new_main_result_tagged.png")
        print(f"updated {figure_dir / 'new_main_result_tagged.pdf'}")
        print(f"updated {figure_dir / 'new_main_result_tagged.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
