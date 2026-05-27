#!/usr/bin/env python3
"""Build En-Zh tau-comparison plotting data for the Speech LLM ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MAIN_DATA = SCRIPT_DIR.parent / "figure_01_main_result_tagged" / "data.tsv"
DEFAULT_TAU0_COMPARE = (
    SCRIPT_DIR.parents[5] / "simuleval/reports/20260526_tagged_acl_zh_tau000_vs_main_rasst.tsv"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "data_zh_tau_compare.tsv"

FIELDNAMES = [
    "method",
    "lang",
    "lm",
    "BLEU",
    "StreamLAAL",
    "TERM_ACC",
    "source_type",
    "source_path",
    "source_snapshot",
    "status",
    "note",
]


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def selected_main_rows(rows: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        if row["dataset"] != "acl_tagged_raw" or row["lang"] != "zh":
            continue
        method = row["method"]
        if method == "Offline + GT terms":
            plot_method = "Oracle term upper bound"
            note = "Offline full-context LLM with oracle/GT terms from the current main-result data."
        elif method == "RASST":
            plot_method = "RASST (tau=0.78)"
            note = "Current main RASST readout with HN1024 retriever threshold tau=0.78."
        elif method in {"Offline ST", "InfiniSST"}:
            plot_method = method
            note = row.get("note", "")
        else:
            continue

        out.append(
            {
                "method": plot_method,
                "lang": row["lang"],
                "lm": row["lm"],
                "BLEU": row["BLEU"],
                "StreamLAAL": row["StreamLAAL"],
                "TERM_ACC": row["TERM_ACC"],
                "source_type": row["source_type"],
                "source_path": row["source_path"],
                "source_snapshot": "figure_01_main_result_tagged/data.tsv",
                "status": row["status"],
                "note": note,
            }
        )
    return out


def selected_tau0_rows(rows: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        if row["dataset"] != "acl_tagged_raw" or row["lang"] != "zh":
            continue
        out.append(
            {
                "method": "RASST (tau=0.0)",
                "lang": row["lang"],
                "lm": row["lm"],
                "BLEU": row["tau0_BLEU"],
                "StreamLAAL": row["tau0_StreamLAAL"],
                "TERM_ACC": row["tau0_TERM_ACC"],
                "source_type": "verified_eval_results",
                "source_path": row["tau0_source_path"],
                "source_snapshot": "20260526_tagged_acl_zh_tau000_vs_main_rasst.tsv",
                "status": "verified",
                "note": "HN1024 raw tagged ACL En-Zh readout with retrieval threshold tau=0.0.",
            }
        )
    return out


def sort_key(row: Mapping[str, str]) -> tuple[int, int]:
    order = {
        "Offline ST": 0,
        "Oracle term upper bound": 1,
        "InfiniSST": 2,
        "RASST (tau=0.78)": 3,
        "RASST (tau=0.0)": 4,
    }
    lm = -1 if row["lm"] == "NA" else int(row["lm"])
    return order[row["method"]], lm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-data", type=Path, default=DEFAULT_MAIN_DATA)
    parser.add_argument("--tau0-compare", type=Path, default=DEFAULT_TAU0_COMPARE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = selected_main_rows(read_tsv(args.main_data))
    rows.extend(selected_tau0_rows(read_tsv(args.tau0_compare)))
    rows = sorted(rows, key=sort_key)
    expected_count = 14
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} plotting rows, got {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
