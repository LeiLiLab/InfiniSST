#!/usr/bin/env python3
"""Export and plot the retriever encoder ablation dev-bank readout."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


BANKS: Tuple[Tuple[str, str, str], ...] = (
    ("raw", "Raw", "eval_dev/recall@10"),
    ("gs10k", "GigaSpeech-10k", "eval_dev/recall@10_gs10000"),
    ("gs100k", "GigaSpeech-100k", "eval_dev/recall@10_gs100000"),
)


MODELS: Tuple[Tuple[str, str], ...] = (
    ("main", "Qwen3-Omni-AuT + BGE-M3"),
    ("text_e5", "Qwen3-Omni-AuT + mE5"),
    ("audio_wavlm", "WavLM + BGE-M3"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="qwen3_rag")
    parser.add_argument(
        "--input-tsv",
        default="",
        help="Optional cached TSV to plot without querying W&B.",
    )
    parser.add_argument("--main-run", default="")
    parser.add_argument("--text-e5-run", default="")
    parser.add_argument("--audio-wavlm-run", default="")
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--out-pdf", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--latex-pdf", default="")
    return parser.parse_args()


def wandb_show(project: str, run_id: str) -> Dict:
    cmd = [
        sys.executable,
        "documents/code/general/wandb_tool.py",
        "--project",
        project,
        "--json",
        "show",
        run_id,
        "--summary-regex",
        r"^eval_dev/",
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    return data


def require_float(summary: Dict, run_id: str, key: str) -> float:
    if key not in summary:
        raise SystemExit(f"missing required metric for {run_id}: {key}")
    value = summary[key]
    if value is None:
        raise SystemExit(f"metric is null for {run_id}: {key}")
    return float(value)


def collect_rows(project: str, run_map: Dict[str, str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for model_id, model_label in MODELS:
        run_id = run_map[model_id]
        data = wandb_show(project, run_id)
        summary = data.get("summary") or {}
        fixed = float(summary.get("eval_dev/fixed_metric_denominator", 0.0))
        if fixed != 1.0:
            raise SystemExit(
                f"{run_id} did not report eval_dev/fixed_metric_denominator=1.0"
            )
        for bank_id, bank_label, metric_key in BANKS:
            recall = require_float(summary, run_id, metric_key)
            rows.append(
                {
                    "model_id": model_id,
                    "model": model_label,
                    "run_id": run_id,
                    "bank_id": bank_id,
                    "bank": bank_label,
                    "metric_key": metric_key,
                    "recall": f"{recall:.8f}",
                    "recall_pct": f"{recall * 100.0:.4f}",
                    "fixed_metric_denominator": f"{fixed:.0f}",
                    "metrics_bank_terms": str(
                        int(float(summary.get("eval_dev/metrics_bank_terms", 0.0)))
                    ),
                }
            )
    if len(rows) != len(MODELS) * len(BANKS):
        raise SystemExit(f"expected 9 rows, got {len(rows)}")
    return rows


def write_tsv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != len(MODELS) * len(BANKS):
        raise SystemExit(f"expected 9 rows in {path}, got {len(rows)}")
    return rows


def plot(rows: List[Dict[str, str]], out_pdf: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 7.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {
        "main": "#2B6CB0",
        "text_e5": "#6B7280",
        "audio_wavlm": "#374151",
    }
    markers = {
        "main": "o",
        "text_e5": "s",
        "audio_wavlm": "^",
    }
    linestyles = {
        "main": "-",
        "text_e5": "--",
        "audio_wavlm": "-.",
    }
    plot_labels = {
        "main": "Qwen3-Omni-AuT + BGE-M3",
        "text_e5": "Qwen3-Omni-AuT + mE5",
        "audio_wavlm": "WavLM + BGE-M3",
    }
    x = list(range(len(BANKS)))
    bank_labels = [label for _, label, _ in BANKS]

    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    all_y: List[float] = []
    for model_id, model_label in MODELS:
        by_bank = {row["bank_id"]: float(row["recall_pct"]) for row in rows if row["model_id"] == model_id}
        ys = [by_bank[bank_id] for bank_id, _, _ in BANKS]
        if len(ys) != len(BANKS):
            raise SystemExit(f"missing plotted values for {model_id}")
        all_y.extend(ys)
        ax.plot(
            x,
            ys,
            label=plot_labels[model_id],
            color=colors[model_id],
            marker=markers[model_id],
            linestyle=linestyles[model_id],
            linewidth=1.55,
            markersize=4.2,
            markerfacecolor=colors[model_id] if model_id == "main" else "white",
            markeredgecolor="white",
            markeredgewidth=0.6,
            solid_capstyle="round",
            zorder=3,
        )
        ax.text(x[-1] + 0.04, ys[-1], f"{ys[-1]:.1f}", color=colors[model_id], fontsize=5.8, va="center")

    ymin = max(88.0, min(all_y) - 1.6)
    ymax = min(100.4, max(all_y) + 0.9)

    ax.set_xticks(x, bank_labels)
    ax.set_xlabel("Runtime candidate bank", labelpad=2.0)
    ax.set_ylabel("Recall@10 (%)")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.18, len(BANKS) - 0.58)
    ax.set_yticks([92, 94, 96, 98, 100])
    ax.grid(axis="y", color="#D9D9D9", linestyle=":", linewidth=0.55)
    ax.grid(axis="x", color="#ECECEC", linestyle=":", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="both", colors="#333333", length=2.5, width=0.6)
    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.94,
        edgecolor="#CFCFCF",
        borderpad=0.28,
        handlelength=1.55,
        labelspacing=0.25,
        prop={"size": 5.4},
    )
    fig.tight_layout(pad=0.45)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.input_tsv:
        rows = read_tsv(Path(args.input_tsv))
    else:
        run_map = {
            "main": args.main_run,
            "text_e5": args.text_e5_run,
            "audio_wavlm": args.audio_wavlm_run,
        }
        missing = [name for name, run_id in run_map.items() if not run_id]
        if missing:
            raise SystemExit(
                "missing run ids without --input-tsv: " + ", ".join(missing)
            )
        rows = collect_rows(args.project, run_map)
    out_tsv = Path(args.out_tsv)
    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png)
    write_tsv(out_tsv, rows)
    plot(rows, out_pdf, out_png)
    if args.latex_pdf:
        latex_pdf = Path(args.latex_pdf)
        latex_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_pdf, latex_pdf)
    print(f"wrote {out_tsv}")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if args.latex_pdf:
        print(f"copied {args.latex_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
