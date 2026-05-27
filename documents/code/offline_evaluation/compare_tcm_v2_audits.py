#!/usr/bin/env python3
"""Compare TCM-v2 audit outputs against the tys70s0y baseline.

Each audit directory is expected to contain:
  - `dense_tau_sweep.json`
  - `acl6060_boundary_summary.json`
  - `acl6060_boundary_samples.jsonl`

The script evaluates candidates at a fixed locked tau and applies the plan's
acceptance rule:
  - no recall collapse versus baseline (within tolerance),
  - lower no-term noise,
  - fewer clear-noise boundary failures, especially in
    `gt_missing_or_outranked`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_boundary_labels(boundary_jsonl: Path) -> Dict[str, Any]:
    overall: Dict[str, int] = {}
    per_subgroup: Dict[str, Dict[str, int]] = {}
    with boundary_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("boundary_label", "unknown")
            overall[label] = overall.get(label, 0) + 1
            for subgroup in row.get("subgroups", []):
                bucket = per_subgroup.setdefault(subgroup, {})
                bucket[label] = bucket.get(label, 0) + 1
    return {"overall": overall, "per_subgroup": per_subgroup}


def _metric_at_tau(dense_tau_json: Dict[str, Any], tau: float) -> Dict[str, Any]:
    for row in dense_tau_json.get("metrics", []):
        if abs(float(row["tau"]) - tau) < 1e-9:
            return row
    raise ValueError(f"tau={tau:.2f} not found in dense_tau_sweep.json")


def _load_audit_dir(path: Path, tau: float) -> Dict[str, Any]:
    dense = _load_json(path / "dense_tau_sweep.json")
    summary = _load_json(path / "acl6060_boundary_summary.json")
    boundary_counts = _count_boundary_labels(path / "acl6060_boundary_samples.jsonl")
    return {
        "path": str(path),
        "dense": dense,
        "summary": summary,
        "tau_metrics": _metric_at_tau(dense, tau),
        "boundary_counts": boundary_counts,
    }


def _accepted(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    recall_tolerance: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    base_tau = baseline["tau_metrics"]
    cand_tau = candidate["tau_metrics"]

    if cand_tau["filtered_recall"] < base_tau["filtered_recall"] - recall_tolerance:
        reasons.append("recall_collapse")
    if cand_tau["noterm_noise"] >= base_tau["noterm_noise"]:
        reasons.append("noise_not_improved")

    base_clear = baseline["boundary_counts"]["overall"].get("clear_noise", 0)
    cand_clear = candidate["boundary_counts"]["overall"].get("clear_noise", 0)
    if cand_clear >= base_clear:
        reasons.append("overall_clear_noise_not_reduced")

    base_miss_clear = (
        baseline["boundary_counts"]["per_subgroup"]
        .get("gt_missing_or_outranked", {})
        .get("clear_noise", 0)
    )
    cand_miss_clear = (
        candidate["boundary_counts"]["per_subgroup"]
        .get("gt_missing_or_outranked", {})
        .get("clear_noise", 0)
    )
    if cand_miss_clear >= base_miss_clear:
        reasons.append("gt_missing_clear_noise_not_reduced")

    return len(reasons) == 0, reasons


def _candidate_row(label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    tau = payload["tau_metrics"]
    overall_clear = payload["boundary_counts"]["overall"].get("clear_noise", 0)
    miss_clear = (
        payload["boundary_counts"]["per_subgroup"]
        .get("gt_missing_or_outranked", {})
        .get("clear_noise", 0)
    )
    return {
        "label": label,
        "filtered_recall": tau["filtered_recall"],
        "precision_micro": tau["precision_micro"],
        "noterm_noise": tau["noterm_noise"],
        "boundary_clear_noise": overall_clear,
        "gt_missing_clear_noise": miss_clear,
        "path": payload["path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in the form label=/abs/path/to/audit_dir",
    )
    parser.add_argument("--locked_tau", type=float, default=0.80)
    parser.add_argument("--recall_tolerance", type=float, default=0.005)
    parser.add_argument("--out_json", required=True, type=Path)
    parser.add_argument("--out_md", required=True, type=Path)
    args = parser.parse_args()

    baseline = _load_audit_dir(args.baseline_dir, args.locked_tau)
    baseline_row = _candidate_row("baseline", baseline)

    candidates: Dict[str, Dict[str, Any]] = {}
    for raw in args.candidate:
        label, path_str = raw.split("=", 1)
        candidates[label] = _load_audit_dir(Path(path_str), args.locked_tau)

    accepted_rows: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {
        "locked_tau": args.locked_tau,
        "recall_tolerance": args.recall_tolerance,
        "baseline": baseline_row,
        "candidates": {},
        "accepted": [],
        "winner": None,
    }

    for label, payload in candidates.items():
        row = _candidate_row(label, payload)
        accepted, reasons = _accepted(baseline, payload, args.recall_tolerance)
        row["accepted"] = accepted
        row["reject_reasons"] = reasons
        results["candidates"][label] = row
        if accepted:
            accepted_rows.append(row)

    if accepted_rows:
        winner = min(
            accepted_rows,
            key=lambda row: (
                -row["filtered_recall"],
                row["noterm_noise"],
                row["gt_missing_clear_noise"],
                row["boundary_clear_noise"],
            ),
        )
        results["accepted"] = [row["label"] for row in accepted_rows]
        results["winner"] = winner["label"]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# TCM-v2 Audit Comparison",
        "",
        f"- locked tau: `{args.locked_tau:.2f}`",
        f"- recall tolerance: `{args.recall_tolerance:.3f}`",
        "",
        "## Baseline",
        "",
        (
            f"- `filtered_recall={baseline_row['filtered_recall']:.4f}`, "
            f"`precision_micro={baseline_row['precision_micro']:.4f}`, "
            f"`noterm_noise={baseline_row['noterm_noise']:.2f}`, "
            f"`clear_noise={baseline_row['boundary_clear_noise']}`, "
            f"`gt_missing_clear_noise={baseline_row['gt_missing_clear_noise']}`"
        ),
        "",
        "## Candidates",
        "",
    ]
    for label, row in results["candidates"].items():
        status = "ACCEPT" if row["accepted"] else f"REJECT ({', '.join(row['reject_reasons'])})"
        lines.append(
            f"- `{label}`: {status}; "
            f"`filtered_recall={row['filtered_recall']:.4f}`, "
            f"`precision_micro={row['precision_micro']:.4f}`, "
            f"`noterm_noise={row['noterm_noise']:.2f}`, "
            f"`clear_noise={row['boundary_clear_noise']}`, "
            f"`gt_missing_clear_noise={row['gt_missing_clear_noise']}`"
        )
    lines.extend(["", "## Winner", ""])
    if results["winner"] is None:
        lines.append("- No candidate met the acceptance rule.")
    else:
        lines.append(f"- Accepted winner: `{results['winner']}`")

    with args.out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[COMPARE] wrote {args.out_json}")
    print(f"[COMPARE] wrote {args.out_md}")
    if results["winner"] is None:
        print("[COMPARE] no candidate met the acceptance rule")
    else:
        print(f"[COMPARE] winner={results['winner']}")


if __name__ == "__main__":
    main()
