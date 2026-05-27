#!/usr/bin/env python3
"""Regenerate Figure 6: Speech LLM ablation."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATA = SCRIPT_DIR / "data.tsv"
DEFAULT_PREFIX = SCRIPT_DIR / "speechllm_ablation_de"
PAPER_STEM = "speechllm_ablation_de"

LANG_TITLES = {"zh": "En-Zh", "de": "En-De", "ja": "En-Ja"}

METHOD_ORDER: Sequence[str] = (
    "Offline ST",
    "Oracle term upper bound",
    "InfiniSST",
    "SLLM+RAG (no TM-SFT)",
    "RASST (LLM-generated TM SFT)",
    "RASST",
    "RASST (tau=0.78)",
    "RASST (tau=0.0)",
)

HLINE_METHODS = {"Offline ST", "Oracle term upper bound"}

METHOD_DISPLAY = {
    "Offline ST": "Offline ST",
    "Oracle term upper bound": "Offline + GT Terms",
    "SLLM+RAG (no TM-SFT)": "InfiniSST + RAG",
    "RASST (LLM-generated TM SFT)": "RASST w/ LLM generated HN",
    "RASST (tau=0.78)": r"RASST ($\tau=0.78$)",
    "RASST (tau=0.0)": r"RASST ($\tau=0.0$)",
}

METHOD_STYLES = {
    "Offline ST": {"color": "#2f855a", "linestyle": "--", "linewidth": 2.4},
    "Oracle term upper bound": {"color": "#2563eb", "linestyle": "-.", "linewidth": 2.4},
    "InfiniSST": {
        "color": "#6b7280",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.2,
        "markersize": 8.0,
    },
    "SLLM+RAG (no TM-SFT)": {
        "color": "#dd6b20",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.2,
        "markersize": 7.5,
    },
    "RASST (LLM-generated TM SFT)": {
        "color": "#8b5a2b",
        "marker": "D",
        "linestyle": "--",
        "linewidth": 2.2,
        "markersize": 7.5,
    },
    "RASST": {
        "color": "#d62728",
        "marker": "*",
        "linestyle": "-",
        "linewidth": 2.4,
        "markersize": 11.0,
    },
    "RASST (tau=0.78)": {
        "color": "#d62728",
        "marker": "*",
        "linestyle": "-",
        "linewidth": 2.4,
        "markersize": 11.0,
    },
    "RASST (tau=0.0)": {
        "color": "#7c3aed",
        "marker": "D",
        "linestyle": "--",
        "linewidth": 2.3,
        "markersize": 7.5,
    },
}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty data TSV: {path}")
    return [dict(row) for row in rows]


def finite(value: str) -> float | None:
    if value in {"", "NA", "N/A", None}:
        return None
    return float(value)


def validate(rows: Sequence[Mapping[str, str]]) -> None:
    seen = set()
    langs = {row["lang"] for row in rows}
    if len(langs) != 1:
        raise ValueError(f"Figure 6 expects one language only; got {sorted(langs)}")
    present_methods = {row["method"] for row in rows}
    unknown_methods = present_methods - set(METHOD_ORDER)
    if unknown_methods:
        raise ValueError(f"unexpected methods in {DEFAULT_DATA}: {sorted(unknown_methods)}")
    for row in rows:
        if row["lang"] not in LANG_TITLES:
            raise ValueError(f"unsupported language in {DEFAULT_DATA}: {row['lang']!r}")
        method = row["method"]
        key = (method, row["lm"])
        if key in seen:
            raise ValueError(f"duplicate method/lm row: {key}")
        seen.add(key)
        if method in HLINE_METHODS and row["lm"] != "NA":
            raise ValueError(f"horizontal reference row should use lm=NA: {row}")
        if method not in HLINE_METHODS and finite(row["StreamLAAL"]) is None:
            raise ValueError(f"series row is missing StreamLAAL: {row}")

    for method in [method for method in METHOD_ORDER if method in present_methods]:
        rows_for_method = [row for row in rows if row["method"] == method]
        if method in HLINE_METHODS:
            if len(rows_for_method) != 1:
                raise ValueError(f"expected one horizontal row for {method}, found {len(rows_for_method)}")
        elif len(rows_for_method) != 4:
            raise ValueError(f"expected four LM rows for {method}, found {len(rows_for_method)}")


def plot(
    rows: Sequence[Mapping[str, str]],
    output_prefix: Path,
    *,
    x_left_pad: float | None = None,
    x_right_pad: float | None = None,
) -> None:
    lang = rows[0]["lang"]
    title = LANG_TITLES[lang]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True)
    handles: List[object] = []
    labels: List[str] = []

    x_values = [
        finite(row["StreamLAAL"])
        for row in rows
        if row["method"] not in HLINE_METHODS and finite(row["StreamLAAL"]) is not None
    ]
    assert x_values
    x_low = min(x_values)
    x_high = max(x_values)
    x_pad = max((x_high - x_low) * 0.08, 95.0)
    left_pad = x_pad if x_left_pad is None else x_left_pad
    right_pad = x_pad if x_right_pad is None else x_right_pad
    present_methods = {row["method"] for row in rows}
    plot_methods = [method for method in METHOD_ORDER if method in present_methods]

    for ax, metric, ylabel in (
        (axes[0], "TERM_ACC", "Terminology\nAccuracy (%)"),
        (axes[1], "BLEU", "BLEU Score"),
    ):
        y_values: List[float] = []
        for method in plot_methods:
            method_rows = [row for row in rows if row["method"] == method]
            style = dict(METHOD_STYLES[method])
            display = METHOD_DISPLAY.get(method, method)
            if method in HLINE_METHODS:
                value = finite(method_rows[0][metric])
                if value is None:
                    continue
                y = value * 100.0 if metric == "TERM_ACC" else value
                line = ax.axhline(y, label=display, **style)
                y_values.append(y)
            else:
                points = []
                for row in sorted(method_rows, key=lambda item: int(item["lm"])):
                    x = finite(row["StreamLAAL"])
                    y_raw = finite(row[metric])
                    if x is None or y_raw is None:
                        continue
                    points.append((x, y_raw * 100.0 if metric == "TERM_ACC" else y_raw))
                line = ax.plot([p[0] for p in points], [p[1] for p in points], label=display, **style)[0]
                y_values.extend(p[1] for p in points)

            if metric == "TERM_ACC" and display not in labels:
                handles.append(line)
                labels.append(display)

        y_low = min(y_values)
        y_high = max(y_values)
        y_pad = max((y_high - y_low) * 0.10, 1.8 if metric == "TERM_ACC" else 1.2)
        ax.set_xlim(x_low - left_pad, x_high + right_pad)
        ax.set_ylim(y_low - y_pad, y_high + y_pad)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        ax.set_xlabel("StreamLAAL (ms)")
        ax.set_ylabel(ylabel)

    fig.suptitle(f"ACL 60/60 {title}", fontweight="bold")
    fig.tight_layout(w_pad=1.4)
    fig.subplots_adjust(bottom=0.30, top=0.88)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.6,
        handlelength=2.2,
        borderaxespad=0.2,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument(
        "--paper-stem",
        default=None,
        help="Paper-facing filename stem when --update-paper is set. Defaults to the output prefix stem.",
    )
    parser.add_argument(
        "--update-paper",
        action="store_true",
        help="Also copy regenerated PDF/PNG into latex/figures.",
    )
    parser.add_argument(
        "--x-left-pad",
        type=float,
        default=None,
        help="Optional left x-axis padding in StreamLAAL milliseconds.",
    )
    parser.add_argument(
        "--x-right-pad",
        type=float,
        default=None,
        help="Optional right x-axis padding in StreamLAAL milliseconds.",
    )
    args = parser.parse_args()

    rows = load_rows(args.data)
    validate(rows)
    plot(rows, args.out_prefix, x_left_pad=args.x_left_pad, x_right_pad=args.x_right_pad)
    print(f"wrote {args.out_prefix.with_suffix('.pdf')}")
    print(f"wrote {args.out_prefix.with_suffix('.png')}")

    if args.update_paper:
        figure_dir = PAPER_DIR / "latex/figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        paper_stem = args.paper_stem or args.out_prefix.stem or PAPER_STEM
        shutil.copy2(args.out_prefix.with_suffix(".pdf"), figure_dir / f"{paper_stem}.pdf")
        shutil.copy2(args.out_prefix.with_suffix(".png"), figure_dir / f"{paper_stem}.png")
        print(f"updated {figure_dir / f'{paper_stem}.pdf'}")
        print(f"updated {figure_dir / f'{paper_stem}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
