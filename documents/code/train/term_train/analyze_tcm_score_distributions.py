#!/usr/bin/env python3
"""Analyze TCM-off score dumps and propose anchored TCM scout settings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


QUANTILES = (1, 5, 10, 15, 25, 50, 75, 90, 95, 99)


def _round_threshold(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return round(math.floor(float(value) / step + 0.5) * step, 4)


def _quantile_map(values: np.ndarray) -> Dict[str, float]:
    clean = np.asarray(values, dtype=np.float32)
    clean = clean[np.isfinite(clean)]
    return {f"p{q:02d}": float(np.percentile(clean, q)) for q in QUANTILES}


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _bank_size(data: Dict[str, np.ndarray]) -> int:
    raw = data.get("bank_size")
    if raw is None:
        return -1
    return int(np.asarray(raw).reshape(-1)[0])


def _summarize_npz(path: Path) -> Dict[str, Any]:
    data = _load_npz(path)
    pos = np.asarray(data["pos_sim"], dtype=np.float32)
    neg_max = np.asarray(data["neg_sim_max"], dtype=np.float32)
    neg_mean = np.asarray(data["neg_sim_mean"], dtype=np.float32)
    neg_top = np.asarray(data.get("neg_top_sim", np.empty((len(pos), 0))), dtype=np.float32)
    neg_top1 = neg_top[:, 0] if neg_top.ndim == 2 and neg_top.shape[1] else neg_max
    gap = pos - neg_max
    return {
        "path": str(path),
        "bank_size": _bank_size(data),
        "n": int(len(pos)),
        "pos": _quantile_map(pos),
        "neg_max": _quantile_map(neg_max),
        "neg_top1": _quantile_map(neg_top1),
        "neg_mean": _quantile_map(neg_mean),
        "gap_pos_minus_neg_max": _quantile_map(gap),
        "_arrays": {
            "pos": pos,
            "neg_max": neg_max,
        },
    }


def _pick_summary(summaries: Sequence[Dict[str, Any]], bank_size: int) -> Dict[str, Any]:
    exact = [s for s in summaries if int(s["bank_size"]) == bank_size]
    if exact:
        return exact[-1]
    candidates = [s for s in summaries if int(s["bank_size"]) > 0]
    if not candidates:
        raise ValueError("No glossary-scale NPZ files with bank_size metadata found")
    return min(candidates, key=lambda s: abs(int(s["bank_size"]) - bank_size))


def _pressure(summary: Dict[str, Any], beta: float, alpha: float) -> Dict[str, float]:
    pos = summary["_arrays"]["pos"]
    neg = summary["_arrays"]["neg_max"]
    pos_viol = pos < beta
    neg_viol = neg > alpha
    pos_gap = np.maximum(beta - pos, 0.0)
    neg_excess = np.maximum(neg - alpha, 0.0)
    pos_rate = float(pos_viol.mean())
    neg_rate = float(neg_viol.mean())
    pos_gap_mean = float(pos_gap[pos_viol].mean()) if pos_viol.any() else 0.0
    neg_excess_mean = float(neg_excess[neg_viol].mean()) if neg_viol.any() else 0.0
    # Rate is the main pressure; magnitude breaks ties when rates are close.
    pos_pressure = pos_rate * (1.0 + pos_gap_mean / 0.05)
    neg_pressure = neg_rate * (1.0 + neg_excess_mean / 0.05)
    return {
        "pos_viol_rate": pos_rate,
        "neg_viol_rate": neg_rate,
        "pos_gap_mean": pos_gap_mean,
        "neg_excess_mean": neg_excess_mean,
        "pos_pressure": float(pos_pressure),
        "neg_pressure": float(neg_pressure),
    }


def _threshold_pairs(
    summary: Dict[str, Any], round_step: float, margin: float
) -> List[Dict[str, Any]]:
    pos_q = summary["pos"]
    neg_q = summary["neg_max"]
    specs = [
        ("conservative", pos_q["p05"], neg_q["p99"]),
        ("center", pos_q["p10"], neg_q["p95"]),
        ("strict", pos_q["p15"], neg_q["p90"]),
    ]
    pairs: List[Dict[str, Any]] = []
    for role, raw_beta, raw_alpha in specs:
        beta = _round_threshold(raw_beta, round_step)
        alpha = _round_threshold(min(raw_alpha, beta - margin), round_step)
        pairs.append(
            {
                "role": role,
                "tcm_pos_threshold": beta,
                "tcm_neg_threshold": alpha,
                "raw_pos_anchor": float(raw_beta),
                "raw_neg_anchor": float(raw_alpha),
                "margin_cap_applied": bool(raw_alpha > beta - margin),
            }
        )
    return pairs


def _choose_weight_pairs(
    primary_pressure: Dict[str, float],
    stress_pressure: Optional[Dict[str, float]],
) -> List[Tuple[int, int]]:
    pos = primary_pressure["pos_pressure"]
    neg = primary_pressure["neg_pressure"]
    if stress_pressure is not None:
        pos += 0.5 * stress_pressure["pos_pressure"]
        neg += 0.5 * stress_pressure["neg_pressure"]
    if pos > 1.2 * max(neg, 1e-8):
        return [(2, 1), (4, 1), (4, 2), (2, 2), (1, 1), (2, 4)]
    if neg > 1.2 * max(pos, 1e-8):
        return [(1, 2), (1, 4), (2, 4), (2, 2), (1, 1), (4, 2)]
    return [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2), (2, 4)]


def _has_noterm_data(path: Path) -> bool:
    with np.load(path, allow_pickle=True) as data:
        return "noterm_topk_sim" in data.files


def _has_term_score_data(path: Path) -> bool:
    with np.load(path, allow_pickle=True) as data:
        return "pos_sim" in data.files and "pos_rank_topk" in data.files


def _tau_tag(tau: float) -> str:
    return f"{float(tau):.2f}".replace(".", "p")


def _compute_frontier(
    score_data: Dict[str, np.ndarray],
    tau_min: float,
    tau_max: float,
    tau_step: float,
) -> List[Dict[str, float]]:
    pos = np.asarray(score_data["pos_sim"], dtype=np.float32)
    rank = np.asarray(score_data.get("pos_rank_topk", np.ones_like(pos) * 999999), dtype=np.int64)
    rank_k = int(np.asarray(score_data.get("rank_k", [10])).reshape(-1)[0])
    noterm_topk = np.asarray(score_data["noterm_topk_sim"], dtype=np.float32)
    taus = np.arange(tau_min, tau_max + 0.5 * tau_step, tau_step)
    out: List[Dict[str, float]] = []
    in_topk = rank <= rank_k
    for tau in taus:
        kept = noterm_topk >= tau
        out.append(
            {
                "tau": float(round(float(tau), 4)),
                "filtered_recall_at_k": float(((pos >= tau) & in_topk).mean()),
                "noterm_avg_emitted": float(kept.sum(axis=1).mean()) if len(noterm_topk) else 0.0,
                "noterm_pass_rate": float(kept.any(axis=1).mean()) if len(noterm_topk) else 0.0,
            }
        )
    return out


def _select_tau_interval(
    frontier: Sequence[Dict[str, float]],
    min_filtered_recall: float,
    center_noise_budget: float,
    down_noise_budget: float,
) -> Dict[str, Any]:
    feasible = [
        row for row in frontier
        if row["filtered_recall_at_k"] >= min_filtered_recall
    ]
    center_candidates = [
        row for row in feasible
        if row["noterm_avg_emitted"] <= center_noise_budget
    ]
    if center_candidates:
        tau_center = min(center_candidates, key=lambda row: row["tau"])
    elif feasible:
        tau_center = min(feasible, key=lambda row: row["noterm_avg_emitted"])
    else:
        tau_center = min(frontier, key=lambda row: row["noterm_avg_emitted"])

    down_candidates = [
        row for row in feasible
        if row["tau"] <= tau_center["tau"]
        and row["noterm_avg_emitted"] <= down_noise_budget
    ]
    tau_down = min(down_candidates, key=lambda row: row["tau"]) if down_candidates else tau_center
    return {
        "tau_center": tau_center,
        "tau_down": tau_down,
        "feasible_count": len(feasible),
        "used_fallback_center": not bool(center_candidates),
        "used_same_down_as_center": tau_down is tau_center,
    }


def _make_bracket(
    score_data: Dict[str, np.ndarray],
    tau_interval: Dict[str, Any],
    round_step: float,
    alpha_margin: float,
    beta_margin: float,
) -> Dict[str, float]:
    pos = np.asarray(score_data["pos_sim"], dtype=np.float32)
    tau_down = tau_interval["tau_down"]["tau"]
    tau_center = tau_interval["tau_center"]["tau"]
    pos_p10 = float(np.percentile(pos, 10))
    pos_p15 = float(np.percentile(pos, 15))
    alpha = _round_threshold(tau_down - alpha_margin, round_step)
    beta = _round_threshold(min(pos_p15, tau_center + beta_margin), round_step)
    if beta <= alpha:
        beta = _round_threshold(max(pos_p10, alpha + 2 * round_step), round_step)
    return {
        "tcm_pos_threshold": beta,
        "tcm_neg_threshold": alpha,
        "pos_p10": pos_p10,
        "pos_p15": pos_p15,
        "tau_down": tau_down,
        "tau_center": tau_center,
    }


def _write_noterm_scout_tsv(
    path: Path,
    bracket: Dict[str, float],
    weight_pairs: Sequence[Tuple[int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("variant\tthreshold_role\tpos_threshold\tneg_threshold\tpos_weight\tneg_weight\n")
        for pos_w, neg_w in weight_pairs:
            variant = f"ntcm_t{_tau_tag(bracket['tau_center'])}_p{pos_w}n{neg_w}"
            f.write(
                f"{variant}\tnoterm\t{bracket['tcm_pos_threshold']:.4f}\t"
                f"{bracket['tcm_neg_threshold']:.4f}\t{pos_w}\t{neg_w}\n"
            )


def _discover_inputs(dump_dir: Optional[Path], inputs: Sequence[Path]) -> List[Path]:
    paths = list(inputs)
    if dump_dir is not None:
        paths.extend(sorted(dump_dir.rglob("*.npz")))
    seen = set()
    uniq: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            uniq.append(resolved)
    if not uniq:
        raise ValueError("No NPZ inputs provided or discovered")
    return uniq


def _write_scout_tsv(
    path: Path,
    center_pair: Dict[str, Any],
    weight_pairs: Iterable[Tuple[int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("variant\tthreshold_role\tpos_threshold\tneg_threshold\tpos_weight\tneg_weight\n")
        for pos_w, neg_w in weight_pairs:
            variant = f"tcmanc_c_p{pos_w}n{neg_w}"
            f.write(
                f"{variant}\tcenter\t{center_pair['tcm_pos_threshold']:.4f}\t"
                f"{center_pair['tcm_neg_threshold']:.4f}\t{pos_w}\t{neg_w}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_dir", type=Path, default=None)
    parser.add_argument("--inputs", type=Path, nargs="*", default=[])
    parser.add_argument("--primary_bank_size", type=int, default=10000)
    parser.add_argument("--stress_bank_size", type=int, default=100000)
    parser.add_argument("--round_step", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--tau_min", type=float, default=0.50)
    parser.add_argument("--tau_max", type=float, default=0.99)
    parser.add_argument("--tau_step", type=float, default=0.01)
    parser.add_argument("--min_filtered_recall", type=float, default=0.90)
    parser.add_argument("--center_noise_budget", type=float, default=0.20)
    parser.add_argument("--down_noise_budget", type=float, default=0.50)
    parser.add_argument("--alpha_margin", type=float, default=0.02)
    parser.add_argument("--beta_margin", type=float, default=0.02)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_scout_tsv", type=Path, default=None)
    args = parser.parse_args()

    inputs = _discover_inputs(args.dump_dir, args.inputs)
    noterm_inputs = [path for path in inputs if _has_noterm_data(path)]
    if noterm_inputs:
        noterm_path = noterm_inputs[-1]
        noterm_data = _load_npz(noterm_path)
        bank = _bank_size(noterm_data)
        term_candidates = [
            path for path in inputs
            if _has_term_score_data(path) and _bank_size(_load_npz(path)) == bank
        ]
        if not term_candidates:
            raise ValueError(
                "No term score NPZ with matching bank_size found for no-term frontier"
            )
        term_path = term_candidates[-1]
        score_data = _load_npz(term_path)
        score_data.update(noterm_data)
        frontier = _compute_frontier(
            score_data,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            tau_step=args.tau_step,
        )
        tau_interval = _select_tau_interval(
            frontier,
            min_filtered_recall=args.min_filtered_recall,
            center_noise_budget=args.center_noise_budget,
            down_noise_budget=args.down_noise_budget,
        )
        bracket = _make_bracket(
            score_data,
            tau_interval,
            round_step=args.round_step,
            alpha_margin=args.alpha_margin,
            beta_margin=args.beta_margin,
        )
        weight_pairs = [(1, 2), (1, 4), (2, 4), (1, 1)]
        result = {
            "mode": "noterm_frontier",
            "inputs": [str(path) for path in inputs],
            "selected_noterm_input": str(noterm_path),
            "selected_term_input": str(term_path),
            "bank_size": _bank_size(score_data),
            "n_no_term": int(np.asarray(score_data["noterm_topk_sim"]).shape[0]),
            "tau_interval": tau_interval,
            "tcm_bracket": bracket,
            "frontier": frontier,
            "weight_pairs": [
                {"tcm_pos_loss_weight": p, "tcm_neg_loss_weight": n}
                for p, n in weight_pairs
            ],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        if args.out_scout_tsv is not None:
            _write_noterm_scout_tsv(args.out_scout_tsv, bracket, weight_pairs)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    summaries = [_summarize_npz(path) for path in inputs]
    primary = _pick_summary(summaries, args.primary_bank_size)
    stress = _pick_summary(summaries, args.stress_bank_size)

    pairs = _threshold_pairs(primary, args.round_step, args.margin)
    center = next(pair for pair in pairs if pair["role"] == "center")
    primary_pressure = _pressure(
        primary, center["tcm_pos_threshold"], center["tcm_neg_threshold"]
    )
    stress_pressure = _pressure(
        stress, center["tcm_pos_threshold"], center["tcm_neg_threshold"]
    )
    weight_pairs = _choose_weight_pairs(primary_pressure, stress_pressure)

    result: Dict[str, Any] = {
        "inputs": [str(path) for path in inputs],
        "primary_bank_size": args.primary_bank_size,
        "stress_bank_size": args.stress_bank_size,
        "primary_summary": {k: v for k, v in primary.items() if k != "_arrays"},
        "stress_summary": {k: v for k, v in stress.items() if k != "_arrays"},
        "threshold_pairs": pairs,
        "center_pressure": {
            "primary": primary_pressure,
            "stress": stress_pressure,
        },
        "weight_pairs": [
            {"tcm_pos_loss_weight": p, "tcm_neg_loss_weight": n}
            for p, n in weight_pairs
        ],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.out_scout_tsv is not None:
        _write_scout_tsv(args.out_scout_tsv, center, weight_pairs)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

