#!/usr/bin/env python3
"""Build zh glossary-bank ablation data and figure.

The ablation keeps the metric denominator fixed to each task's raw glossary.
Only the runtime retrieval bank changes: raw, 1k, and 10k.
"""

from __future__ import annotations

import argparse
import csv
import io
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("/home/jiaxuanluo/InfiniSST")
REPORT_DIR = ROOT / "documents/code/simuleval/reports"
PAPER_FIG_DIR = (
    ROOT
    / "documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures"
)
PAPER_TABLE_DIR = (
    ROOT
    / "documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/tables"
)

TAGGED_TSV = REPORT_DIR / "20260524_tagged_acl_new_v9_hn1024_tau078_zh_raw_gs_fixedraw_data.tsv"
MAIN_RESULT_TSV = REPORT_DIR / "20260524_main_result_data.tsv"

MEDICINE_PSC_ROOTS = [
    "/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/outputs/"
    "medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/"
    "20260524T1837_retry5h_audio8trim_psc_med_newv9_hn1024_tau078_"
    "gs1k_gs10k_lm12_fixedraw_zh",
    "/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/outputs/"
    "medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/"
    "20260524T1345_retry_audio8trim_psc_med_newv9_hn1024_tau078_"
    "gs1k_gs10k_fixedraw_zh",
]
MEDICINE_GS_REEVAL_ROOT = Path(
    "/mnt/gemini/data1/jiaxuanluo/psc_medicine_gs_reposteval_fixedraw_20260525"
)

SSH_TARGET = "jluo7@bridges2.psc.edu"
SSH_CONTROL = "/home/jiaxuanluo/.ssh/sockets/jluo7@bridges2.psc.edu:22"

BANKS: Sequence[str] = ("raw", "gs1k", "gs10k")
LMS: Sequence[int] = (1, 2, 3, 4)
FIELDS = [
    "dataset",
    "lang",
    "lm",
    "runtime_bank",
    "metric_denominator",
    "BLEU",
    "StreamLAAL",
    "TERM_ACC",
    "REAL_TERM_ADOPT",
    "TERM_FCR",
    "TERM_CORRECT",
    "TERM_TOTAL",
    "source_path",
    "status",
]


@dataclass(frozen=True)
class AblationRow:
    dataset: str
    lm: int
    runtime_bank: str
    metric_denominator: str
    bleu: float
    streamlaal: float
    term_acc_pct: float
    real_adopt_pct: float
    term_fcr_pct: float
    term_correct: str
    term_total: str
    source_path: str
    status: str = "verified"

    def as_dict(self) -> Dict[str, str]:
        return {
            "dataset": self.dataset,
            "lang": "zh",
            "lm": str(self.lm),
            "runtime_bank": self.runtime_bank,
            "metric_denominator": self.metric_denominator,
            "BLEU": f"{self.bleu:.4f}",
            "StreamLAAL": f"{self.streamlaal:.4f}",
            "TERM_ACC": f"{self.term_acc_pct:.2f}",
            "REAL_TERM_ADOPT": f"{self.real_adopt_pct:.2f}",
            "TERM_FCR": f"{self.term_fcr_pct:.2f}",
            "TERM_CORRECT": self.term_correct,
            "TERM_TOTAL": self.term_total,
            "source_path": self.source_path,
            "status": self.status,
        }


def read_local_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def ssh_text(command: str) -> str:
    proc = subprocess.run(
        [
            "ssh",
            "-S",
            SSH_CONTROL,
            "-o",
            "BatchMode=yes",
            SSH_TARGET,
            command,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def read_remote_tsv(path: str) -> List[Dict[str, str]]:
    text = ssh_text(f"cat {shlex.quote(path)}")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def find_remote_eval(root: str, bank: str, lm: int) -> str | None:
    path = f"{root}/{bank}/lm{lm}/zh"
    cmd = (
        "for f in "
        f"{shlex.quote(path)}/*/eval_results.tsv; do "
        '[ -s "$f" ] && { printf "%s\\n" "$f"; break; }; '
        "done; true"
    )
    out = ssh_text(cmd).strip()
    return out or None


def read_eval_result(path: str) -> Mapping[str, str]:
    rows = read_remote_tsv(path) if path.startswith("/ocean/") else read_local_tsv(Path(path))
    if not rows:
        raise ValueError(f"empty eval TSV: {path}")
    return rows[-1]


def pct(value: str | float, *, already_pct: bool = False) -> float:
    v = float(value)
    return v if already_pct else v * 100.0


def collect_tagged() -> List[AblationRow]:
    out: List[AblationRow] = []
    for row in read_local_tsv(TAGGED_TSV):
        bank = row["runtime_glossary"].replace("tagged_", "")
        if bank not in BANKS:
            continue
        out.append(
            AblationRow(
                dataset="Tagged ACL",
                lm=int(row["lm"]),
                runtime_bank=bank,
                metric_denominator="tagged_raw_fixed",
                bleu=float(row["bleu"]),
                streamlaal=float(row["streamlaal_ms"]),
                term_acc_pct=float(row["term_acc_pct"]),
                real_adopt_pct=float(row["real_adopt_pct"]),
                term_fcr_pct=float(row["term_fcr_pct"]),
                term_correct="",
                term_total="",
                source_path=row.get("eval_tsv", ""),
            )
        )
    return out


def collect_medicine_raw() -> List[AblationRow]:
    out: List[AblationRow] = []
    for row in read_local_tsv(MAIN_RESULT_TSV):
        if (
            row.get("dataset") != "medicine_hardraw"
            or row.get("method") != "RASST"
            or row.get("lang") != "zh"
        ):
            continue
        lm_raw = row.get("lm", "")
        if lm_raw not in {"1", "2", "3", "4"}:
            continue
        eval_path = row.get("source_path", "")
        eval_row = read_eval_result(eval_path)
        out.append(
            AblationRow(
                dataset="Medicine",
                lm=int(lm_raw),
                runtime_bank="raw",
                metric_denominator="medicine_hardraw_fixed",
                bleu=float(eval_row["BLEU"]),
                streamlaal=float(eval_row["StreamLAAL"]),
                term_acc_pct=pct(eval_row["TERM_ACC"]),
                real_adopt_pct=pct(eval_row["REAL_TERM_ADOPT"]),
                term_fcr_pct=pct(eval_row["TERM_FCR"]),
                term_correct=eval_row.get("TERM_CORRECT", ""),
                term_total=eval_row.get("TERM_TOTAL", ""),
                source_path=eval_path,
            )
        )
    return out


def collect_medicine_gs() -> tuple[List[AblationRow], List[str]]:
    out: List[AblationRow] = []
    missing: List[str] = []
    for bank in ("gs1k", "gs10k"):
        for lm in LMS:
            eval_path = MEDICINE_GS_REEVAL_ROOT / f"{bank}_lm{lm}" / "eval_results.localraw.tsv"
            if not eval_path.is_file() or eval_path.stat().st_size == 0:
                missing.append(f"Medicine {bank} lm{lm}")
                continue
            row = read_eval_result(str(eval_path))
            out.append(
                AblationRow(
                    dataset="Medicine",
                    lm=lm,
                    runtime_bank=bank,
                    metric_denominator="medicine_hardraw_fixed",
                    bleu=float(row["BLEU"]),
                    streamlaal=float(row["StreamLAAL"]),
                    term_acc_pct=pct(row["TERM_ACC"]),
                    real_adopt_pct=pct(row["REAL_TERM_ADOPT"]),
                    term_fcr_pct=pct(row["TERM_FCR"]),
                    term_correct=row.get("TERM_CORRECT", ""),
                    term_total=row.get("TERM_TOTAL", ""),
                    source_path=str(eval_path),
                )
            )
    return out, missing


def expected_keys() -> set[tuple[str, str, int]]:
    return {(dataset, bank, lm) for dataset in ("Tagged ACL", "Medicine") for bank in BANKS for lm in LMS}


def validate_complete(rows: Sequence[AblationRow]) -> List[str]:
    seen = {(r.dataset, r.runtime_bank, r.lm) for r in rows}
    return [
        f"{dataset} {bank} lm{lm}"
        for dataset, bank, lm in sorted(expected_keys())
        if (dataset, bank, lm) not in seen
    ]


def write_tsv(rows: Sequence[AblationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.dataset, BANKS.index(r.runtime_bank), r.lm)):
            writer.writerow(row.as_dict())


def xy_values(
    rows: Sequence[AblationRow],
    dataset: str,
    bank: str,
    y_metric: str,
) -> tuple[List[float], List[float]]:
    by_lm = {(r.dataset, r.runtime_bank, r.lm): r for r in rows}
    xs: List[float] = []
    ys: List[float] = []
    for lm in LMS:
        row = by_lm.get((dataset, bank, lm))
        if row is None:
            continue
        xs.append(row.streamlaal)
        ys.append(float(getattr(row, y_metric)))
    return xs, ys


def make_figure(rows: Sequence[AblationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"raw": "#4C78A8", "gs1k": "#F58518", "gs10k": "#54A24B"}
    labels = {"raw": "Raw", "gs1k": "GS-1k", "gs10k": "GS-10k"}
    markers = {"raw": "o", "gs1k": "s", "gs10k": "^"}
    metrics = [
        ("bleu", "BLEU"),
        ("term_acc_pct", "TERM_ACC (%)"),
        ("term_fcr_pct", "FCR (%)"),
    ]
    dataset = "Tagged ACL"

    fig, axes = plt.subplots(len(metrics), 1, figsize=(3.35, 5.1), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for bank in BANKS:
            xs, ys = xy_values(rows, dataset, bank, metric)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                marker=markers[bank],
                linewidth=1.8,
                markersize=4.5,
                color=colors[bank],
                label=labels[bank],
            )
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("StreamLAAL (ms)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.0)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def write_appendix_table(rows: Sequence[AblationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Dataset & Bank & LM & BLEU & StreamLAAL & TERM\\_ACC & RealAdopt & FCR \\\\",
        "\\midrule",
    ]
    for dataset in ("Tagged ACL", "Medicine"):
        first_dataset_row = True
        for bank in BANKS:
            for lm in LMS:
                matches = [
                    row
                    for row in rows
                    if row.dataset == dataset and row.runtime_bank == bank and row.lm == lm
                ]
                if not matches:
                    continue
                row = matches[0]
                dataset_cell = dataset if first_dataset_row else ""
                first_dataset_row = False
                lines.append(
                    f"{dataset_cell} & {row.runtime_bank} & {row.lm} & "
                    f"{row.bleu:.2f} & {row.streamlaal:.0f} & "
                    f"{row.term_acc_pct:.2f} & {row.real_adopt_pct:.2f} & "
                    f"{row.term_fcr_pct:.2f} \\\\"
                )
        if dataset == "Tagged ACL":
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Full En-Zh runtime glossary-bank ablation. The runtime bank changes from the raw task glossary to GS-1k or GS-10k, while all terminology metrics use the corresponding fixed raw denominator. The main text plots the ACL rows; medicine rows are included as a stricter domain stress test.}",
            "\\label{tab:app_glossary_bank_ablation_full}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(rows: Sequence[AblationRow], missing: Sequence[str], path: Path) -> None:
    lines = [
        "# Glossary Bank Ablation, zh, Fixed Raw Denominator",
        "",
        "Runtime bank changes only the retrieval candidate bank. TERM metrics keep the raw task glossary denominator.",
        "",
        f"- rows collected: {len(rows)}/24",
    ]
    if missing:
        lines.append(f"- missing: {', '.join(missing)}")
    else:
        lines.append("- missing: none")
    lines.extend(["", "| dataset | bank | lm | BLEU | StreamLAAL | TERM_ACC | REAL_ADOPT | TERM_FCR |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in sorted(rows, key=lambda r: (r.dataset, BANKS.index(r.runtime_bank), r.lm)):
        lines.append(
            "| "
            f"{row.dataset} | {row.runtime_bank} | {row.lm} | "
            f"{row.bleu:.2f} | {row.streamlaal:.0f} | {row.term_acc_pct:.2f} | "
            f"{row.real_adopt_pct:.2f} | {row.term_fcr_pct:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--out-tsv",
        type=Path,
        default=REPORT_DIR / "20260525_glossary_bank_ablation_zh_fixedraw_data.tsv",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPORT_DIR / "20260525_glossary_bank_ablation_zh_fixedraw_summary.md",
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=PAPER_FIG_DIR / "glossary_bank_ablation_zh_fixedraw.pdf",
    )
    parser.add_argument(
        "--out-appendix-table",
        type=Path,
        default=PAPER_TABLE_DIR / "glossary_bank_ablation_zh_fixedraw_appendix.tex",
    )
    args = parser.parse_args()

    rows: List[AblationRow] = []
    rows.extend(collect_tagged())
    rows.extend(collect_medicine_raw())
    gs_rows, gs_missing = collect_medicine_gs()
    rows.extend(gs_rows)
    missing = sorted(set(validate_complete(rows) + gs_missing))
    if missing and not args.allow_partial:
        raise SystemExit("missing required rows: " + ", ".join(missing))
    write_tsv(rows, args.out_tsv)
    write_summary(rows, missing, args.out_summary)
    make_figure(rows, args.out_figure)
    write_appendix_table(rows, args.out_appendix_table)
    print(f"[OK] wrote {args.out_tsv}")
    print(f"[OK] wrote {args.out_summary}")
    print(f"[OK] wrote {args.out_figure}")
    print(f"[OK] wrote {args.out_appendix_table}")
    if missing:
        print("[WARN] missing rows: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
