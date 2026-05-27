#!/usr/bin/env python3
"""Sweep top-k threshold metrics from a saved retriever top10 dump.

The input format matches `acl6060_gs10000_top10_dump.jsonl` produced by
`audit_acl_boundary_samples.py`: one JSON object per chunk with `has_term`,
`top10`, and per-candidate `score` / `is_gt` fields.

Selection rule:
1. Use `--baseline-tau` as the current operating point.
2. Keep any tau whose filtered recall is within `--recall-tolerance` of the
   baseline tau's filtered recall.
3. Among eligible taus, pick the one with the lowest no-term noise.
4. Tie-break by higher micro precision, then higher tau.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if "top10" not in row:
                raise ValueError(f"Missing top10 field on line {line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def _compute_metrics(rows: List[Dict[str, Any]], tau: float) -> Dict[str, Any]:
    has_term_rows = [row for row in rows if bool(row.get("has_term", False))]
    no_term_rows = [row for row in rows if not bool(row.get("has_term", False))]

    tp = 0
    total_kept = 0
    pass_rows = 0
    kept_sum_passing = 0
    macro_prec_sum = 0.0

    for row in has_term_rows:
        kept = [
            cand
            for cand in row.get("top10", [])
            if float(cand.get("score", float("-inf"))) >= tau
        ]
        kept_count = len(kept)
        gt_kept = any(bool(cand.get("is_gt", False)) for cand in kept)
        if gt_kept:
            tp += 1
        total_kept += kept_count
        if kept_count > 0:
            pass_rows += 1
            kept_sum_passing += kept_count
            if gt_kept:
                macro_prec_sum += 1.0 / kept_count

    filtered_recall = tp / len(has_term_rows) if has_term_rows else 0.0
    micro_precision = tp / total_kept if total_kept > 0 else 0.0
    macro_precision = macro_prec_sum / pass_rows if pass_rows > 0 else 0.0
    avg_kept_if_pass = kept_sum_passing / pass_rows if pass_rows > 0 else 0.0

    noise_sum = 0
    for row in no_term_rows:
        kept_noise = sum(
            1
            for cand in row.get("top10", [])
            if float(cand.get("score", float("-inf"))) >= tau
        )
        noise_sum += kept_noise
    noterm_noise = noise_sum / len(no_term_rows) if no_term_rows else 0.0

    return {
        "tau": tau,
        "filtered_recall": filtered_recall,
        "precision_micro": micro_precision,
        "precision_macro": macro_precision,
        "avg_kept_if_pass": avg_kept_if_pass,
        "pass_rate": pass_rows / len(has_term_rows) if has_term_rows else 0.0,
        "noterm_noise": noterm_noise,
        "term_rows": len(has_term_rows),
        "noterm_rows": len(no_term_rows),
        "tp": tp,
        "total_kept": total_kept,
    }


def _select_tau(
    metrics: List[Dict[str, Any]],
    baseline_tau: float,
    recall_tolerance: float,
) -> Dict[str, Any]:
    baseline = min(metrics, key=lambda row: abs(row["tau"] - baseline_tau))
    recall_floor = baseline["filtered_recall"] - recall_tolerance
    eligible = [
        row for row in metrics
        if row["filtered_recall"] >= recall_floor
    ]
    if not eligible:
        return baseline
    return min(
        eligible,
        key=lambda row: (
            row["noterm_noise"],
            -row["precision_micro"],
            -row["tau"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_jsonl", required=True, type=Path)
    parser.add_argument("--taus", nargs="+", type=float, required=True)
    parser.add_argument("--baseline-tau", type=float, default=0.80)
    parser.add_argument("--recall-tolerance", type=float, default=0.005)
    parser.add_argument("--out_tsv", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_rows(args.dump_jsonl)
    metrics = [_compute_metrics(rows, float(tau)) for tau in args.taus]
    metrics.sort(key=lambda row: row["tau"])
    selected = _select_tau(metrics, args.baseline_tau, args.recall_tolerance)
    baseline = min(metrics, key=lambda row: abs(row["tau"] - args.baseline_tau))

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8") as f:
        f.write(
            "tau\tfiltered_recall\tprecision_micro\tprecision_macro\t"
            "avg_kept_if_pass\tpass_rate\tnoterm_noise\tterm_rows\t"
            "noterm_rows\ttp\ttotal_kept\n"
        )
        for row in metrics:
            f.write(
                f"{row['tau']:.2f}\t{row['filtered_recall']:.6f}\t"
                f"{row['precision_micro']:.6f}\t{row['precision_macro']:.6f}\t"
                f"{row['avg_kept_if_pass']:.6f}\t{row['pass_rate']:.6f}\t"
                f"{row['noterm_noise']:.6f}\t{row['term_rows']}\t"
                f"{row['noterm_rows']}\t{row['tp']}\t{row['total_kept']}\n"
            )

    payload = {
        "input": str(args.dump_jsonl),
        "baseline_tau": args.baseline_tau,
        "recall_tolerance": args.recall_tolerance,
        "baseline": baseline,
        "selected": selected,
        "metrics": metrics,
    }
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        "[TAU_SELECT] baseline_tau="
        f"{baseline['tau']:.2f} baseline_recall={baseline['filtered_recall']:.4f} "
        f"baseline_noise={baseline['noterm_noise']:.2f}"
    )
    print(
        "[TAU_SELECT] selected_tau="
        f"{selected['tau']:.2f} recall={selected['filtered_recall']:.4f} "
        f"precision={selected['precision_micro']:.4f} "
        f"noise={selected['noterm_noise']:.2f}"
    )
    print(f"[TAU_SELECT] wrote {args.out_tsv}")
    print(f"[TAU_SELECT] wrote {args.out_json}")


if __name__ == "__main__":
    main()
