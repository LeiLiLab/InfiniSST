#!/usr/bin/env python3
"""Summarize optimizer/scheduler state in a qwen3_glossary_neg_train checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch


def summarize_optimizer_state(optimizer_state: Dict[str, Any]) -> Dict[str, Any]:
    state = optimizer_state.get("state", {}) if isinstance(optimizer_state, dict) else {}
    param_groups = optimizer_state.get("param_groups", []) if isinstance(optimizer_state, dict) else []
    groups = []
    for group in param_groups:
        groups.append(
            {
                "name": group.get("name", ""),
                "lr": group.get("lr"),
                "initial_lr": group.get("initial_lr"),
                "betas": list(group.get("betas", [])),
                "eps": group.get("eps"),
                "weight_decay": group.get("weight_decay"),
                "param_count": len(group.get("params", [])),
            }
        )

    state_values = list(state.values())
    has_exp_avg = any(isinstance(v, dict) and "exp_avg" in v for v in state_values)
    has_exp_avg_sq = any(isinstance(v, dict) and "exp_avg_sq" in v for v in state_values)
    step_values = [
        int(v["step"].item() if hasattr(v.get("step"), "item") else v.get("step"))
        for v in state_values
        if isinstance(v, dict) and "step" in v
    ]
    return {
        "param_groups": groups,
        "state_param_count": len(state),
        "has_exp_avg": has_exp_avg,
        "has_exp_avg_sq": has_exp_avg_sq,
        "max_moment_step": max(step_values) if step_values else None,
        "min_moment_step": min(step_values) if step_values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    assert os.path.isfile(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    summary = {
        "checkpoint": args.checkpoint,
        "global_step": ckpt.get("global_step"),
        "epoch": ckpt.get("epoch"),
        "has_optimizer_state_dict": "optimizer_state_dict" in ckpt,
        "has_scheduler_state_dict": "scheduler_state_dict" in ckpt,
        "has_scaler_state_dict": "scaler_state_dict" in ckpt,
        "best_metric_key": ckpt.get("best_metric_key"),
        "best_metric_value": ckpt.get("best_metric_value"),
        "best_metric_secondary_value": ckpt.get("best_metric_secondary_value"),
    }
    if "optimizer_state_dict" in ckpt:
        summary["optimizer"] = summarize_optimizer_state(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler_state = ckpt["scheduler_state_dict"]
        summary["scheduler"] = {
            "class": type(scheduler_state).__name__,
            "keys": sorted(scheduler_state.keys()) if isinstance(scheduler_state, dict) else [],
            "last_epoch": scheduler_state.get("last_epoch") if isinstance(scheduler_state, dict) else None,
            "_step_count": scheduler_state.get("_step_count") if isinstance(scheduler_state, dict) else None,
            "_last_lr": scheduler_state.get("_last_lr") if isinstance(scheduler_state, dict) else None,
        }
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        kept_args = {
            key: ckpt["args"].get(key)
            for key in (
                "train_jsonl",
                "dev_jsonl",
                "lr",
                "text_lr",
                "epochs",
                "scheduler_epochs",
                "batch_size",
                "hard_neg_k_per_sample",
                "tcm_pos_loss_weight",
                "tcm_neg_loss_weight",
                "tcm_pos_threshold",
                "tcm_neg_threshold",
                "best_metric",
                "best_metric_secondary",
            )
        }
        summary["checkpoint_args"] = kept_args

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
