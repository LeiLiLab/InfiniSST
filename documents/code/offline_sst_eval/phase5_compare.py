#!/usr/bin/env python3
"""Phase 5 A/B comparison: control (d5_cap) vs experiment (d5_cap_adv).

Reads each variant's per-paper combined eval TSV, prints a compact A/B table,
and writes a machine-readable JSON decision gate result so the orchestrator
can decide whether to trigger Phase 6.

Decision gate (Sub-problem B diagnostic, per plan):
  experiment's TERM_FCR > TERM_FCR_THRESHOLD  -> trigger Phase 6
  (also report delta vs control's TERM_FCR for context)

All user-facing strings are in English.
"""

from __future__ import annotations

# ======Configuration=====
TERM_FCR_THRESHOLD = 0.05  # 5%: per plan's Phase 5 decision gate
# ======Configuration=====

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


TSV_COLS = [
    "mode",
    "lang_code",
    "BLEU",
    "StreamLAAL",
    "StreamLAAL_CA",
    "TERM_ACC",
    "TERM_CORRECT",
    "TERM_TOTAL",
    "TCR",
    "TCR_ADOPTED",
    "TCR_TOTAL",
    "TERM_FCR",
    "FALSE_COPY",
    "NEG_TOTAL",
    "instances_log",
]


def read_tsv_last_row(path: Path) -> Dict[str, str]:
    assert path.is_file(), f"TSV not found: {path}"
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    assert len(lines) >= 1, f"TSV is empty: {path}"
    row = lines[-1].split("\t")
    assert len(row) == len(TSV_COLS), (
        f"Unexpected TSV width {len(row)} vs expected {len(TSV_COLS)}: {path}\n"
        f"row={row}"
    )
    return dict(zip(TSV_COLS, row))


def fmt_row(d: Dict[str, str]) -> str:
    def f(x: str, k: int = 4) -> str:
        try:
            return f"{float(x):.{k}f}"
        except Exception:
            return str(x)
    return (
        f"BLEU={f(d['BLEU'], 2):>6}  "
        f"StreamLAAL={f(d['StreamLAAL'], 1):>8}  "
        f"TERM_ACC={f(d['TERM_ACC'])}  "
        f"({d['TERM_CORRECT']}/{d['TERM_TOTAL']})  "
        f"TCR={f(d['TCR'])}  "
        f"({d['TCR_ADOPTED']}/{d['TCR_TOTAL']})  "
        f"TERM_FCR={f(d['TERM_FCR'])}  "
        f"({d['FALSE_COPY']}/{d['NEG_TOTAL']})"
    )


def diff_stat(name: str, a_str: str, b_str: str, pct_points: bool = False) -> str:
    a = float(a_str)
    b = float(b_str)
    d = b - a
    if pct_points:
        return f"  {name:>12}: control={a:.4f}  experiment={b:.4f}  delta={d:+.4f} ({d*100:+.2f}pp)"
    return f"  {name:>12}: control={a:.4f}  experiment={b:.4f}  delta={d:+.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-tsv", required=True,
                    help="path to control variant's eval_results_by_paper.tsv")
    ap.add_argument("--experiment-tsv", required=True,
                    help="path to experiment variant's eval_results_by_paper.tsv")
    ap.add_argument("--decision-json", required=True,
                    help="output JSON with machine-readable decision")
    ap.add_argument("--term-fcr-threshold", type=float, default=TERM_FCR_THRESHOLD,
                    help=f"gate threshold on experiment TERM_FCR (default {TERM_FCR_THRESHOLD})")
    args = ap.parse_args()

    control_row = read_tsv_last_row(Path(args.control_tsv))
    experiment_row = read_tsv_last_row(Path(args.experiment_tsv))

    print("=" * 80)
    print("Phase 5 A/B comparison")
    print("=" * 80)
    print(f"Control    (d5_cap    ): {fmt_row(control_row)}")
    print(f"Experiment (d5_cap_adv): {fmt_row(experiment_row)}")
    print()
    print("Delta (experiment - control):")
    print(diff_stat("BLEU",      control_row["BLEU"],      experiment_row["BLEU"]))
    print(diff_stat("StreamLAAL", control_row["StreamLAAL"], experiment_row["StreamLAAL"]))
    print(diff_stat("TERM_ACC",  control_row["TERM_ACC"],  experiment_row["TERM_ACC"],  pct_points=True))
    print(diff_stat("TCR",       control_row["TCR"],       experiment_row["TCR"],       pct_points=True))
    print(diff_stat("TERM_FCR",  control_row["TERM_FCR"],  experiment_row["TERM_FCR"],  pct_points=True))

    experiment_fcr = float(experiment_row["TERM_FCR"])
    trigger_phase6 = experiment_fcr > args.term_fcr_threshold
    reason = (
        f"experiment TERM_FCR {experiment_fcr:.4f} > threshold {args.term_fcr_threshold}"
        if trigger_phase6 else
        f"experiment TERM_FCR {experiment_fcr:.4f} <= threshold {args.term_fcr_threshold}"
    )

    print()
    print("=" * 80)
    print(f"Phase 6 decision gate: TRIGGER={trigger_phase6}")
    print(f"  Reason: {reason}")
    print("=" * 80)

    payload = {
        "control": control_row,
        "experiment": experiment_row,
        "threshold_term_fcr": args.term_fcr_threshold,
        "experiment_term_fcr": experiment_fcr,
        "trigger_phase6": bool(trigger_phase6),
        "reason": reason,
    }
    os.makedirs(os.path.dirname(args.decision_json) or ".", exist_ok=True)
    with open(args.decision_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Decision JSON: {args.decision_json}")


if __name__ == "__main__":
    main()
