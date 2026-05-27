#!/usr/bin/env python3
"""Post-evaluate new ACL6060 DE/JA InfiniSST baselines and refresh Figure 1 data."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[4]
ACL_ROOT = ROOT / "documents/code/train/term_train/reports/figures/acl6060_de_ja_results"
REPORT_DIR = ROOT / "documents/code/simuleval/reports"
POSTEVAL_ROOT = REPORT_DIR / "20260525_acl6060_de_ja_baseline_posteval"
MAIN_TSV = REPORT_DIR / "20260524_main_result_data.tsv"
PAPER_ROOT = ROOT / "documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src"
FIGURE_DATASETS = [
    PAPER_ROOT / "plot/figure_01_main_result_tagged/data.tsv",
    PAPER_ROOT / "plot/figure_02_medicine_main_result/data.tsv",
]

OFFLINE_EVAL_SCRIPT = ROOT / "documents/code/offline_sst_eval/offline_streamlaal_eval.py"
DATA_ROOT = Path("/mnt/taurus/data/siqiouyang/datasets/acl6060")
FBK_FAIRSEQ_ROOT = Path("/mnt/taurus/home/jiaxuanluo/FBK-fairseq")
MWERSEGMENTER_ROOT = Path("/mnt/taurus/home/jiaxuanluo/mwerSegmenter")
PYTHON_BIN = Path("/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python")
GLOSSARY = Path("/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json")

SEG_TO_LM = {"seg960": 1, "seg1920": 2, "seg2880": 3, "seg3840": 4}
EVENT_ID = "20260525T2110__analysis__acl_tagged_de_ja_infinisst_baseline_refresh"
NOTE = (
    "new ACL6060 tagged raw InfiniSST baseline; BLEU/StreamLAAL/TERM from "
    "post-eval of acl6060_de_ja_results instances.log with tagged raw glossary"
)

MAIN_FIELDS = [
    "dataset",
    "method",
    "lang",
    "lm",
    "BLEU",
    "StreamLAAL",
    "StreamLAAL_CA",
    "TERM_ACC",
    "TERM_CORRECT",
    "TERM_TOTAL",
    "source_type",
    "source_path",
    "event_id",
    "wandb_run_id",
    "status",
    "note",
]


@dataclass(frozen=True)
class EvalPaths:
    lang: str
    seg: str
    lm: int
    instances_log: Path
    output_dir: Path
    raw_tsv: Path
    main_tsv: Path


def read_tsv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def single_row(path: Path) -> Dict[str, str]:
    rows = read_tsv_rows(path)
    if len(rows) != 1:
        raise ValueError(f"expected one data row in {path}, found {len(rows)}")
    return rows[0]


def fmt_float(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def eval_paths() -> List[EvalPaths]:
    paths: List[EvalPaths] = []
    for lang in ("de", "ja"):
        for seg, lm in SEG_TO_LM.items():
            setting_dir = ACL_ROOT / lang / seg
            out_dir = POSTEVAL_ROOT / f"{lang}_lm{lm}"
            paths.append(
                EvalPaths(
                    lang=lang,
                    seg=seg,
                    lm=lm,
                    instances_log=(setting_dir / "instances.log").resolve(),
                    output_dir=out_dir.resolve(),
                    raw_tsv=(out_dir / "eval_results.tsv").resolve(),
                    main_tsv=(out_dir / "eval_results.main.tsv").resolve(),
                )
            )
    return paths


def ensure_inputs(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))


def run_posteval(item: EvalPaths, *, force: bool) -> None:
    source_file = DATA_ROOT / "dev/text/txt/ACL.6060.dev.en-xx.en.txt"
    ref_file = DATA_ROOT / f"dev/text/txt/ACL.6060.dev.en-xx.{item.lang}.txt"
    audio_yaml = DATA_ROOT / "dev.yaml"
    ensure_inputs(
        [
            OFFLINE_EVAL_SCRIPT,
            FBK_FAIRSEQ_ROOT
            / "examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py",
            MWERSEGMENTER_ROOT,
            PYTHON_BIN,
            item.instances_log,
            source_file,
            ref_file,
            audio_yaml,
            GLOSSARY,
        ]
    )
    if item.raw_tsv.exists() and not force:
        return
    item.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MWERSEGMENTER_ROOT"] = str(MWERSEGMENTER_ROOT)
    env["PATH"] = f"{MWERSEGMENTER_ROOT}:{env.get('PATH', '')}"
    cmd = [
        str(PYTHON_BIN),
        str(OFFLINE_EVAL_SCRIPT),
        "--mode",
        "acl6060",
        "--instances-log",
        str(item.instances_log),
        "--lang-code",
        item.lang,
        "--data-root",
        str(DATA_ROOT),
        "--source-file",
        str(source_file),
        "--ref-file",
        str(ref_file),
        "--audio-yaml",
        str(audio_yaml),
        "--glossary-acl6060",
        str(GLOSSARY),
        "--fbk-fairseq-root",
        str(FBK_FAIRSEQ_ROOT),
        "--python-bin",
        str(PYTHON_BIN),
        "--strip-output-tags",
        "none",
        "--term-fcr-policy",
        "source_ref_negative_sentence",
        "--output-tsv",
        str(item.raw_tsv),
        "--output-log",
        str(item.output_dir / "eval_results.log"),
        "--work-dir",
        str(item.output_dir / "work"),
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def write_main_compatible_eval(item: EvalPaths) -> Dict[str, str]:
    raw = single_row(item.raw_tsv)
    raw["instances_log"] = str(item.instances_log)
    write_tsv(item.main_tsv, [raw], list(raw.keys()))
    return raw


def main_result_row(item: EvalPaths, raw: Mapping[str, str]) -> Dict[str, str]:
    return {
        "dataset": "acl_tagged_raw",
        "method": "InfiniSST",
        "lang": item.lang,
        "lm": str(item.lm),
        "BLEU": fmt_float(raw["BLEU"]),
        "StreamLAAL": fmt_float(raw["StreamLAAL"]),
        "StreamLAAL_CA": fmt_float(raw["StreamLAAL_CA"]),
        "TERM_ACC": fmt_float(raw["TERM_ACC"]),
        "TERM_CORRECT": str(int(float(raw["TERM_CORRECT"]))),
        "TERM_TOTAL": str(int(float(raw["TERM_TOTAL"]))),
        "source_type": "verified_acl6060_baseline_posteval",
        "source_path": str(item.main_tsv),
        "event_id": EVENT_ID,
        "wandb_run_id": "",
        "status": "verified",
        "note": NOTE,
    }


def validate_main_rows(rows: Sequence[Mapping[str, str]]) -> None:
    seen = set()
    for row in rows:
        key = (row["dataset"], row["method"], row["lang"], row["lm"])
        if key in seen:
            raise ValueError(f"duplicate main-result key: {key}")
        seen.add(key)
        explicit_placeholder = row["status"].startswith("placeholder")
        for field in ("BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL"):
            value = row[field]
            if value == "NA":
                if explicit_placeholder:
                    continue
                if row["method"].startswith("Offline") and field in {"StreamLAAL", "StreamLAAL_CA"}:
                    continue
                if row["source_type"] == "user_supplied_reusable" and field in {"TERM_CORRECT", "TERM_TOTAL"}:
                    continue
                raise ValueError(f"unexpected NA for {key} field {field}")
            float(value)


def update_main_tsv(replacements: Mapping[Tuple[str, int], Mapping[str, str]]) -> None:
    rows = read_tsv_rows(MAIN_TSV)
    updated = 0
    new_rows: List[Dict[str, str]] = []
    for row in rows:
        key = (row["lang"], int(row["lm"])) if row["lm"].isdigit() else None
        if row["dataset"] == "acl_tagged_raw" and row["method"] == "InfiniSST" and key in replacements:
            new_rows.append(dict(replacements[key]))
            updated += 1
        else:
            new_rows.append(row)
    if updated != len(replacements):
        raise ValueError(f"updated {updated} rows, expected {len(replacements)}")
    validate_main_rows(new_rows)
    write_tsv(MAIN_TSV, new_rows, MAIN_FIELDS)
    for snapshot in FIGURE_DATASETS:
        shutil.copy2(MAIN_TSV, snapshot)


def write_summary(rows: Sequence[Mapping[str, str]]) -> Path:
    summary_path = POSTEVAL_ROOT / "summary_acl6060_de_ja_infinisst_baseline.tsv"
    write_tsv(summary_path, rows, MAIN_FIELDS)
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recompute post-eval outputs.")
    parser.add_argument("--update-main-tsv", action="store_true")
    args = parser.parse_args()

    replacements: Dict[Tuple[str, int], Dict[str, str]] = {}
    summary_rows: List[Dict[str, str]] = []
    for item in eval_paths():
        run_posteval(item, force=args.force)
        raw = write_main_compatible_eval(item)
        row = main_result_row(item, raw)
        replacements[(item.lang, item.lm)] = row
        summary_rows.append(row)
        print(
            "\t".join(
                [
                    item.lang,
                    str(item.lm),
                    row["BLEU"],
                    row["StreamLAAL"],
                    row["StreamLAAL_CA"],
                    row["TERM_ACC"],
                    f'{row["TERM_CORRECT"]}/{row["TERM_TOTAL"]}',
                ]
            )
        )

    summary_path = write_summary(summary_rows)
    print(f"wrote {summary_path}")
    if args.update_main_tsv:
        update_main_tsv(replacements)
        print(f"updated {MAIN_TSV}")
        for snapshot in FIGURE_DATASETS:
            print(f"updated {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
