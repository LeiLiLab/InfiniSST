#!/usr/bin/env python3
"""Plot RAG retriever compute RTF against the vLLM generation cadence."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


UNIT_SEC = 0.96
LOOKBACK_SEC = 1.92
LMS = (1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-tsv",
        default="documents/code/simuleval/reports/20260525_glossary_bank_ablation_zh_fixedraw_data.tsv",
        help="Verified eval summary with source_path and StreamLAAL columns.",
    )
    parser.add_argument("--dataset", default="Medicine")
    parser.add_argument("--lang", default="zh")
    parser.add_argument("--runtime-bank", default="raw")
    parser.add_argument("--out-tsv", required=True)
    parser.add_argument("--out-pdf", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--latex-pdf", default="")
    parser.add_argument("--latex-png", default="")
    return parser.parse_args()


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def _latest_runtime_jsonl(output_dir: Path) -> Path:
    paths = sorted(output_dir.glob("runtime_omni_vllm_maxsim_rag_*.jsonl"))
    if not paths:
        raise SystemExit(f"missing runtime JSONL under {output_dir}")
    return paths[-1]


def _read_rag_seconds(runtime_path: Path) -> List[float]:
    seconds: List[float] = []
    with runtime_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{runtime_path}:{line_no}: invalid JSON") from exc
            if record.get("type") != "rag_window":
                continue
            if record.get("trigger") != "vllm_timeline":
                continue
            value = _float_or_none(record.get("rag_sec"))
            if value is not None:
                seconds.append(value)
    if not seconds:
        raise SystemExit(f"no timed vllm_timeline rag_window rows in {runtime_path}")
    return seconds


def collect_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    summary_path = Path(args.summary_tsv)
    with summary_path.open("r", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f, delimiter="\t"))

    selected: Dict[int, Dict[str, str]] = {}
    for row in raw_rows:
        if row.get("dataset") != args.dataset:
            continue
        if row.get("lang") != args.lang:
            continue
        if row.get("runtime_bank") != args.runtime_bank:
            continue
        lm = int(row["lm"])
        selected[lm] = row

    missing = [lm for lm in LMS if lm not in selected]
    if missing:
        raise SystemExit(
            "missing summary rows for lm=" + ",".join(str(lm) for lm in missing)
        )

    out: List[Dict[str, str]] = []
    for lm in LMS:
        source_path = Path(selected[lm]["source_path"])
        if not source_path.is_file():
            raise SystemExit(f"missing source eval TSV: {source_path}")
        runtime_path = _latest_runtime_jsonl(source_path.parent)
        rag_sec = _read_rag_seconds(runtime_path)
        cadence_sec = lm * UNIT_SEC
        input_span_sec = cadence_sec + LOOKBACK_SEC
        mean_sec = statistics.mean(rag_sec)
        median_sec = statistics.median(rag_sec)
        p90_sec = _percentile(rag_sec, 0.90)
        p95_sec = _percentile(rag_sec, 0.95)
        out.append(
            {
                "dataset": args.dataset,
                "lang": args.lang,
                "runtime_bank": args.runtime_bank,
                "lm": str(lm),
                "vllm_cadence_sec": f"{cadence_sec:.2f}",
                "retriever_input_span_sec": f"{input_span_sec:.2f}",
                "lookback_sec": f"{LOOKBACK_SEC:.2f}",
                "streamlaal_sec": f"{float(selected[lm]['StreamLAAL']) / 1000.0:.4f}",
                "rag_calls": str(len(rag_sec)),
                "rag_mean_ms": f"{mean_sec * 1000.0:.3f}",
                "rag_median_ms": f"{median_sec * 1000.0:.3f}",
                "rag_p90_ms": f"{p90_sec * 1000.0:.3f}",
                "rag_p95_ms": f"{p95_sec * 1000.0:.3f}",
                "rag_mean_rtf_pct": f"{mean_sec / cadence_sec * 100.0:.4f}",
                "rag_median_rtf_pct": f"{median_sec / cadence_sec * 100.0:.4f}",
                "rag_p90_rtf_pct": f"{p90_sec / cadence_sec * 100.0:.4f}",
                "runtime_jsonl": str(runtime_path),
                "eval_results_tsv": str(source_path),
            }
        )
    return out


def write_tsv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: List[Dict[str, str]], out_pdf: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = list(range(len(rows)))
    median_rtf = [float(row["rag_median_rtf_pct"]) for row in rows]
    mean_rtf = [float(row["rag_mean_rtf_pct"]) for row in rows]
    median_ms = [float(row["rag_median_ms"]) for row in rows]
    labels = [
        f"LM {row['lm']}\nLAAL {float(row['streamlaal_sec']):.2f}s\nspan {float(row['retriever_input_span_sec']):.2f}s"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    bar_color = "#267C8F"
    mean_color = "#8F2D56"
    line_color = "#4A4E69"

    bars = ax.bar(
        x,
        median_rtf,
        width=0.58,
        color=bar_color,
        edgecolor="#184B56",
        linewidth=0.6,
        label="Median RAG RTF",
        zorder=3,
    )
    ax.plot(
        x,
        mean_rtf,
        color=mean_color,
        marker="D",
        markersize=3.8,
        linewidth=1.4,
        label="Mean RAG RTF",
        zorder=4,
    )

    ax2 = ax.twinx()
    ax2.plot(
        x,
        median_ms,
        color=line_color,
        marker="o",
        markersize=4.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
        linewidth=1.5,
        label="Median retrieve ms",
        zorder=5,
    )

    for rect, value in zip(bars, median_rtf):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            max(0.16, rect.get_height() - 0.28),
            f"{value:.2f}%",
            ha="center",
            va="top",
            fontsize=6.2,
            color="white",
            fontweight="bold",
            zorder=6,
        )
    for xi, value in zip(x, median_ms):
        ax2.text(
            xi,
            value + 2.6,
            f"{value:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=5.9,
            color=line_color,
            zorder=6,
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel("Retriever compute RTF (%)")
    ax2.set_ylabel("Retriever time per call (ms)")
    ax.set_ylim(0, max(max(median_rtf), max(mean_rtf)) + 0.75)
    ax2.set_ylim(0, max(median_ms) + 22.0)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.7)
    ax.set_axisbelow(True)

    for spine in ["top"]:
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax2.spines["right"].set_color("#555555")
    ax.tick_params(axis="both", colors="#333333", length=2.5, width=0.6)
    ax2.tick_params(axis="y", colors="#333333", length=2.5, width=0.6)

    lines, line_labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines + lines2,
        line_labels + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.9,
        borderpad=0.1,
    )
    fig.tight_layout(pad=0.35)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = collect_rows(args)
    out_tsv = Path(args.out_tsv)
    out_pdf = Path(args.out_pdf)
    out_png = Path(args.out_png)
    write_tsv(out_tsv, rows)
    plot(rows, out_pdf, out_png)
    if args.latex_pdf:
        latex_pdf = Path(args.latex_pdf)
        latex_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_pdf, latex_pdf)
    if args.latex_png:
        latex_png = Path(args.latex_png)
        latex_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_png, latex_png)
    print(f"wrote {out_tsv}")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if args.latex_pdf:
        print(f"copied {args.latex_pdf}")
    if args.latex_png:
        print(f"copied {args.latex_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
