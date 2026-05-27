#!/usr/bin/env python3
"""Plot tagged ACL main results against StreamLAAL.

The current SLLM+RAG line is parsed from eval_results.tsv files. Historical
offline and InfiniSST numbers are fixed inputs supplied for this figure. The
RASST line defaults to previous fixed inputs, or can be replaced from an
LLM-generated term-map SFT summary markdown file.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LANG_ORDER: Sequence[Tuple[str, str]] = (("zh", "En-Zh"), ("de", "En-De"), ("ja", "En-Ja"))
LMS: Sequence[int] = (1, 2, 3, 4)
CURRENT_METHOD = "SLLM+RAG (no TM-SFT)"
RASST_METHOD = "RASST"


OFFLINE = {
    "zh": {"bleu": 49.6625, "term_acc": 0.7912},
    "ja": {"bleu": 32.9328, "term_acc": 0.6624},
    "de": {"bleu": 35.7641, "term_acc": 0.7057},
}

INFINISST = {
    "zh": {
        "lm": [1, 2, 3, 4],
        "bleu": [40.6663, 45.8268, 46.7119, 47.3897],
        "streamlaal": [1181.1470, 1765.7196, 2232.6733, 2616.3493],
        "term_acc": [0.7431, 0.7655, 0.7675, 0.7754],
    },
    "ja": {
        "lm": [1, 2, 3, 4],
        "bleu": [22.0137, 27.8786, 29.3039, 30.6042],
        "streamlaal": [1571.0, 2300.0, 2707.0, 3252.0],
        "term_acc": [0.6331, 0.6564, 0.6724, 0.6751],
    },
    "de": {
        "lm": [1, 2, 3, 4],
        "bleu": [27.4672, 31.6370, 31.7033, 32.6733],
        "streamlaal": [1124.0, 1773.0, 2383.0, 2808.0],
        "term_acc": [0.6496, 0.6744, 0.6864, 0.6881],
    },
}

RASST_PREVIOUS = {
    "zh": {
        "lm": [1, 2, 3, 4],
        "bleu": [44.2272, 48.7556, 49.1509, 49.6847],
        "streamlaal": [1225.8638, 1781.0892, 2258.0696, 2664.1809],
        "term_acc": [0.8241, 0.8393, 0.8577, 0.8551],
    },
    "ja": {
        "lm": [1, 2, 3, 4],
        "bleu": [20.1920, 28.2989, 31.8029, 32.5395],
        "streamlaal": [1309.0, 2092.0, 2592.0, 3071.0],
        "term_acc": [0.7705, 0.7919, 0.8239, 0.8266],
    },
    "de": {
        "lm": [1, 2, 3, 4],
        "bleu": [27.4309, 32.1915, 33.1950, 34.5998],
        "streamlaal": [1055.0, 1698.0, 2233.0, 2744.0],
        "term_acc": [0.7378, 0.8003, 0.7851, 0.8075],
    },
}

METHOD_STYLES = {
    "Offline ST": {
        "color": "#2ca02c",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "InfiniSST": {
        "color": "#1f77b4",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 6.5,
    },
    CURRENT_METHOD: {
        "color": "#ff7f0e",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 6.0,
    },
    RASST_METHOD: {
        "color": "#e41a1c",
        "marker": "*",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 9.0,
    },
}


def _float(row: Mapping[str, str], key: str, path: Path) -> float:
    raw = row.get(key, "")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {key} in {path}: {raw!r}") from exc


def _read_last_tsv_row(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty eval_results.tsv: {path}")
    return dict(rows[-1])


def _parse_markdown_table(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    header: List[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if not header:
            header = cells
            continue
        if len(cells) != len(header):
            raise ValueError(f"Malformed markdown table row in {path}: {raw_line}")
        rows.append(dict(zip(header, cells)))
    if not rows:
        raise ValueError(f"No markdown table rows found in {path}")
    return rows


def _glossary_from_dir(name: str) -> str:
    if "gacl6060_tagged_gt_raw_min_norm2" in name:
        return "raw"
    if "gacl6060_tagged_gt_union_gs10000_min_norm2_backfill" in name:
        return "gs10k"
    if "gacl6060_tagged_gt_union_gs1000_min_norm2_backfill" in name:
        return "gs1k"
    raise ValueError(f"Cannot identify glossary kind from directory: {name}")


def load_current_results(base_dir: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Current result directory does not exist: {base_dir}")

    observed: MutableMapping[Tuple[str, int, str], Dict[str, float]] = {}
    for path in sorted(base_dir.rglob("eval_results.tsv")):
        rel = path.relative_to(base_dir)
        if len(rel.parts) < 3:
            raise ValueError(f"Unexpected eval path layout: {path}")
        lang = rel.parts[0]
        setting_dir = path.parent.name
        if lang not in {code for code, _ in LANG_ORDER}:
            raise ValueError(f"Unexpected language directory in {path}: {lang}")
        lm_match = re.search(r"_lm(\d+)_", setting_dir)
        if not lm_match:
            raise ValueError(f"Missing latency multiplier in setting dir: {setting_dir}")
        lm = int(lm_match.group(1))
        glossary = _glossary_from_dir(setting_dir)
        key = (lang, lm, glossary)
        if key in observed:
            raise ValueError(f"Duplicate current result for {key}: {path}")
        row = _read_last_tsv_row(path)
        observed[key] = {
            "bleu": _float(row, "BLEU", path),
            "streamlaal": _float(row, "StreamLAAL", path),
            "term_acc": _float(row, "TERM_ACC", path),
        }

    expected = {(lang, lm, glossary) for lang, _ in LANG_ORDER for lm in LMS for glossary in ("raw", "gs1k", "gs10k")}
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(f"Current full sweep is incomplete or unexpected. missing={missing} extra={extra}")

    selected: Dict[str, Dict[int, Dict[str, float]]] = {}
    for lang, _ in LANG_ORDER:
        selected[lang] = {}
        for lm in LMS:
            selected[lang][lm] = observed[(lang, lm, "raw")]
    return selected


def _parse_tsv_table(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"No TSV table rows found in {path}")
    return [dict(row) for row in rows]


def load_summary_results(summary_path: Path, glossary: str, *, table_format: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    if not summary_path.exists():
        raise FileNotFoundError(f"RASST summary file does not exist: {summary_path}")
    if glossary not in {"raw", "gs1k", "gs10k"}:
        raise ValueError(f"Unsupported glossary for summary selection: {glossary}")
    if table_format == "markdown":
        rows = _parse_markdown_table(summary_path)
    elif table_format == "tsv":
        rows = _parse_tsv_table(summary_path)
    else:
        raise ValueError(f"Unsupported summary table format: {table_format}")

    observed: MutableMapping[Tuple[str, int, str], Dict[str, float]] = {}
    for row in rows:
        if row.get("gs") not in {"raw", "gs1k", "gs10k"}:
            continue
        lang = row.get("lang", "")
        if lang not in {code for code, _ in LANG_ORDER}:
            raise ValueError(f"Unexpected language in {summary_path}: {lang!r}")
        try:
            lm = int(row.get("lm", ""))
            bleu = float(row.get("BLEU", ""))
            term_acc_pct = float(row.get("TERM_ACC", ""))
            streamlaal = float(row.get("StreamLAAL", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric row in {summary_path}: {row}") from exc
        key = (lang, lm, row["gs"])
        if key in observed:
            raise ValueError(f"Duplicate summary result for {key} in {summary_path}")
        observed[key] = {
            "bleu": bleu,
            "streamlaal": streamlaal,
            "term_acc_pct": term_acc_pct,
        }

    expected = {(lang, lm, gs) for lang, _ in LANG_ORDER for lm in LMS for gs in ("raw", "gs1k", "gs10k")}
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(f"RASST summary grid is incomplete or unexpected. missing={missing} extra={extra}")

    selected: Dict[str, Dict[int, Dict[str, float]]] = {}
    for lang, _ in LANG_ORDER:
        selected[lang] = {}
        for lm in LMS:
            selected[lang][lm] = observed[(lang, lm, glossary)]
    return selected


def _series_from_static(data: Mapping[str, Mapping[str, Sequence[float]]], lang: str) -> Dict[str, List[float]]:
    item = data[lang]
    return {
        "lm": [int(x) for x in item["lm"]],
        "bleu": [float(x) for x in item["bleu"]],
        "streamlaal": [float(x) for x in item["streamlaal"]],
        "term_acc_pct": [float(x) * 100.0 for x in item["term_acc"]],
    }


def _series_from_percent_rows(data: Mapping[str, Mapping[int, Mapping[str, float]]], lang: str) -> Dict[str, List[float]]:
    rows = [data[lang][lm] for lm in LMS]
    return {
        "lm": list(LMS),
        "bleu": [row["bleu"] for row in rows],
        "streamlaal": [row["streamlaal"] for row in rows],
        "term_acc_pct": [row["term_acc_pct"] for row in rows],
    }


def _series_from_current(data: Mapping[str, Mapping[int, Mapping[str, float]]], lang: str) -> Dict[str, List[float]]:
    rows = [data[lang][lm] for lm in LMS]
    return {
        "lm": list(LMS),
        "bleu": [row["bleu"] for row in rows],
        "streamlaal": [row["streamlaal"] for row in rows],
        "term_acc_pct": [row["term_acc"] * 100.0 for row in rows],
    }


def build_methods(
    current: Mapping[str, Mapping[int, Mapping[str, float]]],
    rasst: Mapping[str, Mapping[int, Mapping[str, float]]] | None = None,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for lang, _ in LANG_ORDER:
        out[lang] = {
            "InfiniSST": _series_from_static(INFINISST, lang),
            CURRENT_METHOD: _series_from_current(current, lang),
            RASST_METHOD: _series_from_percent_rows(rasst, lang) if rasst else _series_from_static(RASST_PREVIOUS, lang),
        }
    return out


def write_plot_data(
    path: Path,
    methods: Mapping[str, Mapping[str, Mapping[str, List[float]]]],
    *,
    rasst_source: str,
    rasst_note: str,
) -> None:
    fields = [
        "method",
        "lang",
        "lm",
        "glossary",
        "bleu",
        "streamlaal_ms",
        "term_acc_pct",
        "source",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for lang, _ in LANG_ORDER:
            writer.writerow(
                {
                    "method": "Offline ST",
                    "lang": lang,
                    "lm": "",
                    "glossary": "tagged_raw",
                    "bleu": f"{OFFLINE[lang]['bleu']:.4f}",
                    "streamlaal_ms": "",
                    "term_acc_pct": f"{OFFLINE[lang]['term_acc'] * 100.0:.2f}",
                    "source": "user_supplied_offline_table",
                    "note": "horizontal reference line",
                }
            )
            for method in ("InfiniSST", CURRENT_METHOD, RASST_METHOD):
                data = methods[lang][method]
                source = {
                    "InfiniSST": "user_supplied_infinisst_table",
                    CURRENT_METHOD: "parsed_current_eval_results_raw_glossary",
                    RASST_METHOD: rasst_source,
                }[method]
                for idx, lm in enumerate(data["lm"]):
                    writer.writerow(
                        {
                            "method": method,
                            "lang": lang,
                            "lm": int(lm),
                            "glossary": "tagged_raw",
                            "bleu": f"{data['bleu'][idx]:.4f}",
                            "streamlaal_ms": f"{data['streamlaal'][idx]:.4f}",
                            "term_acc_pct": f"{data['term_acc_pct'][idx]:.2f}",
                            "source": source,
                            "note": rasst_note if method == RASST_METHOD else "",
                        }
                    )


def _axis_limits(values: Iterable[float], min_pad: float) -> Tuple[float, float]:
    vals = list(values)
    low = min(vals)
    high = max(vals)
    span = high - low
    pad = max(span * 0.10, min_pad)
    return low - pad, high + pad


def plot(methods: Mapping[str, Mapping[str, Mapping[str, List[float]]]], output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.9,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.9))
    legend_handles = []
    legend_labels = []

    for col, (lang, title) in enumerate(LANG_ORDER):
        for row_idx, metric in enumerate(("term_acc_pct", "bleu")):
            ax = axes[row_idx][col]
            offline_y = OFFLINE[lang]["term_acc"] * 100.0 if metric == "term_acc_pct" else OFFLINE[lang]["bleu"]
            offline_line = ax.axhline(offline_y, **METHOD_STYLES["Offline ST"], label="Offline ST")
            if col == 0 and row_idx == 0:
                legend_handles.append(offline_line)
                legend_labels.append("Offline ST")

            x_values: List[float] = []
            y_values: List[float] = [offline_y]
            for method in ("InfiniSST", CURRENT_METHOD, RASST_METHOD):
                data = methods[lang][method]
                style = METHOD_STYLES[method]
                line = ax.plot(
                    data["streamlaal"],
                    data[metric],
                    label=method,
                    **style,
                )[0]
                x_values.extend(data["streamlaal"])
                y_values.extend(data[metric])
                if col == 0 and row_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(method)

            ax.set_xlim(*_axis_limits(x_values, 90.0))
            ax.set_ylim(*_axis_limits(y_values, 1.2 if metric == "bleu" else 1.8))
            ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.65)
            if row_idx == 0:
                ax.set_title(title, fontweight="bold")
            else:
                ax.set_xlabel("StreamLAAL (ms)")
            if col == 0:
                ax.set_ylabel("Terminology\nAccuracy (%)" if metric == "term_acc_pct" else "BLEU Score")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.5,
        handlelength=2.0,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), w_pad=1.2, h_pad=1.6)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-result-dir",
        type=Path,
        default=Path(
            "/mnt/gemini/data2/jiaxuanluo/"
            "tagged_acl_origin_bsz4_tau073_baseline_20260520T1010_ragreset_full_mp3/full"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("documents/code/simuleval/reports/20260520_tagged_acl_main_results_fourline"),
    )
    parser.add_argument(
        "--plot-data-tsv",
        type=Path,
        default=Path("documents/code/simuleval/reports/20260520_tagged_acl_main_results_fourline_data.tsv"),
    )
    parser.add_argument(
        "--rasst-summary-md",
        type=Path,
        default=None,
        help="Optional LLM-generated term-map SFT summary markdown used to replace the RASST line.",
    )
    parser.add_argument(
        "--rasst-summary-tsv",
        type=Path,
        default=None,
        help="Optional LLM-generated term-map SFT summary TSV used to replace the RASST line with exact values.",
    )
    parser.add_argument(
        "--rasst-glossary",
        choices=("raw", "gs1k", "gs10k"),
        default="raw",
        help="Glossary row to plot from --rasst-summary-md.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rasst_summary_md and args.rasst_summary_tsv:
        raise ValueError("Use only one of --rasst-summary-md or --rasst-summary-tsv")
    current = load_current_results(args.current_result_dir)
    rasst_summary_path = args.rasst_summary_tsv or args.rasst_summary_md
    rasst_format = "tsv" if args.rasst_summary_tsv else "markdown"
    rasst = (
        load_summary_results(rasst_summary_path, args.rasst_glossary, table_format=rasst_format)
        if rasst_summary_path
        else None
    )
    methods = build_methods(current, rasst)
    rasst_source = (
        f"parsed_llmgen_sft_summary_{args.rasst_glossary}_glossary"
        if rasst_summary_path
        else "user_supplied_previous_rasst_table_pending_update"
    )
    rasst_note = (
        f"replaced from {rasst_summary_path}" if rasst_summary_path else "previous placeholder RASST data"
    )
    write_plot_data(args.plot_data_tsv, methods, rasst_source=rasst_source, rasst_note=rasst_note)
    plot(methods, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {args.plot_data_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
