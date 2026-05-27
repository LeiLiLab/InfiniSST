#!/usr/bin/env python3
"""Plot En-De lm=2 BLEU-vs-term-accuracy setting ablation.

The comparison intentionally mixes two provenance classes:

* historical paper-table references from the canonical main-result TSV;
* verified rerun metrics from event manifests.

The output TSV keeps those source classes explicit so the old InfiniSST point is
not mistaken for a newly reproduced metric.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = ROOT / "documents/code/simuleval/reports"
MANIFEST_DIR = ROOT / "documents/code/simuleval/manifests/2026/05"
PAPER_FIG_DIR = (
    ROOT
    / "documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures"
)
MAIN_TSV = REPORT_DIR / "20260524_main_result_data.tsv"

OUT_TSV = REPORT_DIR / "20260524_de_lm2_bleu_termacc_ablation.tsv"
OUT_PDF = PAPER_FIG_DIR / "de_lm2_bleu_termacc_ablation.pdf"
OUT_PNG = PAPER_FIG_DIR / "de_lm2_bleu_termacc_ablation.png"


@dataclass(frozen=True)
class Point:
    setting_id: str
    label: str
    display_label: str
    slm: str
    retriever: str
    term_training: str
    source_class: str
    source_path: str
    event_id: str
    wandb_run_id: str
    bleu: float
    streamlaal: str
    streamlaal_ca: str
    term_acc: float
    term_correct: str
    term_total: str
    status: str
    note: str


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main_result_row(*, dataset: str, method: str, lang: str, lm: str) -> dict[str, str]:
    for row in read_tsv(MAIN_TSV):
        if (
            row["dataset"] == dataset
            and row["method"] == method
            and row["lang"] == lang
            and row["lm"] == lm
        ):
            return row
    raise KeyError((dataset, method, lang, lm))


def load_manifest(event_id: str) -> dict:
    path = MANIFEST_DIR / f"{event_id}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("status") != "success":
        raise ValueError(f"{event_id} is not success: {data.get('status')}")
    return data


def artifact_path(manifest: Mapping, role: str) -> str:
    for artifact in manifest.get("artifacts", []):
        if artifact.get("role") == role:
            return str(artifact.get("path", ""))
    evidence = manifest.get("metadata", {}).get("completion_evidence", {})
    key = f"{role}_path"
    return str(evidence.get(key, ""))


def manifest_point(
    *,
    setting_id: str,
    label: str,
    display_label: str,
    slm: str,
    retriever: str,
    term_training: str,
    event_id: str,
    note: str,
) -> Point:
    manifest = load_manifest(event_id)
    metrics = manifest["metadata"]["metrics"]
    return Point(
        setting_id=setting_id,
        label=label,
        display_label=display_label,
        slm=slm,
        retriever=retriever,
        term_training=term_training,
        source_class="verified_manifest",
        source_path=artifact_path(manifest, "eval_results"),
        event_id=event_id,
        wandb_run_id=str(manifest.get("wandb_run_id") or ""),
        bleu=float(metrics["BLEU"]),
        streamlaal=f'{float(metrics["StreamLAAL"]):.4f}',
        streamlaal_ca=f'{float(metrics["StreamLAAL_CA"]):.4f}',
        term_acc=float(metrics["TERM_ACC"]),
        term_correct=str(metrics.get("TERM_CORRECT", "")),
        term_total=str(metrics.get("TERM_TOTAL", "")),
        status="verified",
        note=note,
    )


def tsv_point(
    *,
    setting_id: str,
    label: str,
    display_label: str,
    slm: str,
    retriever: str,
    term_training: str,
    row: Mapping[str, str],
    status: str,
    note: str,
) -> Point:
    return Point(
        setting_id=setting_id,
        label=label,
        display_label=display_label,
        slm=slm,
        retriever=retriever,
        term_training=term_training,
        source_class=row["source_type"],
        source_path=row["source_path"],
        event_id=row["event_id"],
        wandb_run_id=row["wandb_run_id"],
        bleu=float(row["BLEU"]),
        streamlaal=row["StreamLAAL"],
        streamlaal_ca=row["StreamLAAL_CA"],
        term_acc=float(row["TERM_ACC"]),
        term_correct=row["TERM_CORRECT"],
        term_total=row["TERM_TOTAL"],
        status=status,
        note=note,
    )


def build_points() -> list[Point]:
    old_infinisst = main_result_row(
        dataset="acl_tagged_raw", method="InfiniSST", lang="de", lm="2"
    )
    clean_rasst = main_result_row(
        dataset="acl_tagged_raw", method="RASST", lang="de", lm="2"
    )
    offline = main_result_row(
        dataset="acl_tagged_raw", method="Offline ST", lang="de", lm="NA"
    )

    return [
        tsv_point(
            setting_id="offline_st_reference",
            label="Offline ST",
            display_label="Offline ST",
            slm="offline cascade",
            retriever="none",
            term_training="none",
            row=offline,
            status="reference",
            note="Offline reference has no streaming latency; included only as an upper-reference point.",
        ),
        tsv_point(
            setting_id="old_infinisst_prompt_row",
            label="Old InfiniSST row",
            display_label="Old InfiniSST\n(prompt row)",
            slm="origin de SLM",
            retriever="none",
            term_training="none",
            row=old_infinisst,
            status="old_unreproduced_reference",
            note="Historical user-supplied row; not reproduced by the same-lm batch rerun.",
        ),
        manifest_point(
            setting_id="origin_norag_rerun",
            label="InfiniSST/no-RAG rerun",
            display_label="No-RAG rerun",
            slm="origin de SLM",
            retriever="none",
            term_training="none",
            event_id="20260524T2200__simuleval__tagged_acl_origin_norag_de_lm2_batch_max80_aries01",
            note="Same-lm batch rerun on Aries GPU 0,1 with max_new_tokens=80.",
        ),
        manifest_point(
            setting_id="origin_slm_hn1024",
            label="Origin SLM + HN1024",
            display_label="Origin SLM\n+ HN1024",
            slm="origin de SLM",
            retriever="HN1024 tau=0.78",
            term_training="none",
            event_id="20260524T2135__simuleval__tagged_acl_origin_de_lm2_hn1024_batch",
            note="No-TM-SFT original de SLM plus HN1024 retriever.",
        ),
        manifest_point(
            setting_id="tm_sft_slm_hn1024",
            label="TM-SFT + HN1024",
            display_label="TM-SFT SLM\n+ HN1024",
            slm="old TM-SFT de SLM",
            retriever="HN1024 tau=0.78",
            term_training="TM-SFT",
            event_id="20260524T2135__simuleval__tagged_acl_tmv4_de_lm2_hn1024_batch",
            note="Old TM-SFT de SLM plus HN1024 retriever.",
        ),
        tsv_point(
            setting_id="clean_newv9_slm_hn1024",
            label="Clean SLM (RASST) + HN1024",
            display_label="Clean NewV9 SLM\n+ HN1024",
            slm="clean NewV9 MFA/source-filtered de SLM",
            retriever="HN1024 tau=0.78",
            term_training="RASST",
            row=clean_rasst,
            status="verified",
            note=clean_rasst["note"],
        ),
    ]


def write_tsv(points: Iterable[Point], path: Path) -> None:
    fields = [
        "setting_id",
        "label",
        "slm",
        "retriever",
        "term_training",
        "BLEU",
        "StreamLAAL",
        "StreamLAAL_CA",
        "TERM_ACC",
        "TERM_CORRECT",
        "TERM_TOTAL",
        "source_class",
        "source_path",
        "event_id",
        "wandb_run_id",
        "status",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "setting_id": point.setting_id,
                    "label": point.label,
                    "slm": point.slm,
                    "retriever": point.retriever,
                    "term_training": point.term_training,
                    "BLEU": f"{point.bleu:.4f}",
                    "StreamLAAL": point.streamlaal,
                    "StreamLAAL_CA": point.streamlaal_ca,
                    "TERM_ACC": f"{point.term_acc:.4f}",
                    "TERM_CORRECT": point.term_correct,
                    "TERM_TOTAL": point.term_total,
                    "source_class": point.source_class,
                    "source_path": point.source_path,
                    "event_id": point.event_id,
                    "wandb_run_id": point.wandb_run_id,
                    "status": point.status,
                    "note": point.note,
                }
            )


def plot(points: list[Point], pdf_path: Path, png_path: Path) -> None:
    styles: Dict[str, dict] = {
        "offline_st_reference": {
            "marker": "^",
            "color": "#7A7A7A",
            "s": 110,
            "edgecolor": "black",
            "linewidth": 0.8,
            "alpha": 0.75,
        },
        "old_infinisst_prompt_row": {
            "marker": "X",
            "color": "#D62728",
            "s": 120,
            "edgecolor": "black",
            "linewidth": 0.8,
        },
        "origin_norag_rerun": {
            "marker": "o",
            "color": "#1F77B4",
            "s": 105,
            "edgecolor": "black",
            "linewidth": 0.8,
        },
        "origin_slm_hn1024": {
            "marker": "o",
            "color": "#2CA02C",
            "s": 105,
            "edgecolor": "black",
            "linewidth": 0.8,
        },
        "tm_sft_slm_hn1024": {
            "marker": "s",
            "color": "#9467BD",
            "s": 105,
            "edgecolor": "black",
            "linewidth": 0.8,
        },
        "clean_newv9_slm_hn1024": {
            "marker": "D",
            "color": "#FF7F0E",
            "s": 105,
            "edgecolor": "black",
            "linewidth": 0.8,
        },
    }
    offsets = {
        "offline_st_reference": (-30, -16),
        "old_infinisst_prompt_row": (-48, 10),
        "origin_norag_rerun": (8, -8),
        "origin_slm_hn1024": (8, -20),
        "tm_sft_slm_hn1024": (8, 8),
        "clean_newv9_slm_hn1024": (-52, -18),
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for point in points:
        style = styles[point.setting_id]
        ax.scatter(point.term_acc, point.bleu, label=point.label, **style)
        dx, dy = offsets[point.setting_id]
        color = "#D62728" if point.status == "old_unreproduced_reference" else "#222222"
        ax.annotate(
            point.display_label,
            xy=(point.term_acc, point.bleu),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 0.6,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    ax.set_title("En-De lm=2 Tagged-ACL Raw: BLEU vs Term Accuracy", fontsize=12)
    ax.set_xlabel("Term accuracy")
    ax.set_ylabel("BLEU")
    ax.grid(True, color="#DDDDDD", linewidth=0.7, alpha=0.9)
    ax.set_xlim(0.60, 0.885)
    ax.set_ylim(28.8, 36.5)
    ax.text(
        0.603,
        28.98,
        "Red X: historical InfiniSST row not reproduced by current batch rerun.",
        fontsize=7.8,
        color="#7A1F1F",
        ha="left",
        va="bottom",
    )
    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-tsv", type=Path, default=OUT_TSV)
    parser.add_argument("--out-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    points = build_points()
    write_tsv(points, args.out_tsv)
    plot(points, args.out_pdf, args.out_png)
    print(args.out_tsv)
    print(args.out_pdf)
    print(args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
