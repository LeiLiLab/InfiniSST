#!/usr/bin/env python3
"""Thin wrapper that pushes simuleval / offline-eval results into WandB.

One CLI invocation opens a single `simuleval_eval` run per call, scans the
known output directory layout for `eval_results.tsv` / `eval_results_by_paper.tsv`,
writes the headline metrics (BLEU, StreamLAAL, StreamLAAL_CA, TERM_ACC, TCR,
TERM_FCR) into `run.summary`, links back to the training run via
`config.trained_from_run`, and finishes.

Usage (from `run_one_density_eval.sh` or similar):

    python3 documents/code/offline_evaluation/wandb_eval_logger.py \
        --project simuleval_eval \
        --run-name "d5_cap__lms1-4__k10__zh__20260423" \
        --experiment-family sst_density_ablation \
        --data-tag extracted_by_paper \
        --notes-file "${NOTES_FILE}" \
        --trained-from-run "${TRAINED_FROM_RUN}" \
        --baseline-run-ids ${BASELINE_RUN_IDS} \
        --density 5 --rag-top-k 10 \
        --output-base /mnt/gemini/data2/jiaxuanluo/density_eval_maxsim \
        --lang-code zh \
        --latency-multipliers 1 2 3 4 \
        --glossary-tag glossary_acl6060 \
        --verdict "tentative: filled by agent after review"

This script enforces the `.cursor/rules/experiment_tracking.mdc` schema:
structured tags, validated run notes, `save_code=True`, `status:running` flipped
to `status:success|failed` at the end, `verdict` written into `run.summary`.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GENERAL_DIR = Path(__file__).resolve().parents[1] / "general"
if str(GENERAL_DIR) not in sys.path:
    sys.path.insert(0, str(GENERAL_DIR))

from wandb_tags import prepare_wandb_tags

REQUIRED_NOTES_SECTIONS = (
    "## Hypothesis",
    "## Background / Motivation",
    "## What changed vs baseline",
    "## Expected metrics",
    "## Verdict",
)

METRIC_NAMES = (
    "BLEU",
    "StreamLAAL",
    "StreamLAAL_CA",
    "TERM_ACC",
    "TERM_ADOPTION",
    "TERM_ADOPTION_MICRO",
    "REAL_TERM_ADOPT",
    "REAL_TERM_ADOPT_MICRO",
    "TERM_FCR",
)


def load_and_validate_run_notes(notes_path: str) -> str:
    if not notes_path:
        raise ValueError(
            "experiment_tracking: --notes-file is required. "
            "See documents/code/_templates/run_notes_template.md."
        )
    if not os.path.isfile(notes_path):
        raise FileNotFoundError(
            f"experiment_tracking: --notes-file not found: {notes_path}"
        )
    with open(notes_path, "r", encoding="utf-8") as f:
        text = f.read()
    missing = [s for s in REQUIRED_NOTES_SECTIONS if s not in text]
    if missing:
        raise ValueError(
            f"experiment_tracking: notes file {notes_path} is missing "
            f"required sections: {missing}."
        )
    for i, section in enumerate(REQUIRED_NOTES_SECTIONS[:-1]):
        start = text.index(section) + len(section)
        next_start = len(text)
        for later in REQUIRED_NOTES_SECTIONS[i + 1 :]:
            idx = text.find(later, start)
            if idx != -1:
                next_start = min(next_start, idx)
        body = text[start:next_start]
        body_stripped = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL).strip()
        if not body_stripped:
            raise ValueError(
                f"experiment_tracking: section '{section}' in "
                f"{notes_path} is empty (HTML comments don't count)."
            )
    return text


def parse_tsv_last_row(tsv_path: str) -> Optional[Dict[str, float]]:
    """Parse the last data row; return headline metrics or None if missing."""
    if not os.path.isfile(tsv_path):
        return None
    try:
        with open(tsv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    except OSError:
        return None
    if not rows:
        return None
    last = rows[-1]
    out: Dict[str, float] = {}
    for name in METRIC_NAMES:
        raw = (last.get(name) or "").strip()
        try:
            out[name] = float(raw)
        except ValueError:
            # Non-numeric (e.g. "N/A"): skip silently; WandB will show what
            # we did log. We do not silently substitute 0.
            continue
    return out or None


def scan_outputs(
    output_base: str,
    lang_code: str,
    density: str,
    rag_top_k: str,
    latency_multipliers: List[int],
    glossary_tag: str,
    rag_score_threshold: str = "",
    paper_id: str = "",
    oracle_term_map: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Collect per-lm, per-mode metrics. Returns flat dict keyed like
    'tagged/lm1/TERM_ACC'."""
    base = Path(output_base) / lang_code
    collected: Dict[str, Dict[str, float]] = {}
    for lm in latency_multipliers:
        mode_part = "_oraclegt" if oracle_term_map else ""
        if paper_id:
            threshold_part = f"_th{rag_score_threshold}" if rag_score_threshold else ""
            onepaper_dir = base / f"d{density}{mode_part}_lm{lm}_k{rag_top_k}{threshold_part}_g{glossary_tag}_pp{paper_id}"
            onepaper_tsv = onepaper_dir / "eval_results.tsv"
            onepaper_metrics = parse_tsv_last_row(str(onepaper_tsv))
            if onepaper_metrics is not None:
                collected[f"by_paper/lm{lm}/{glossary_tag}/{paper_id}"] = onepaper_metrics
            continue

        threshold_part = f"_th{rag_score_threshold}" if rag_score_threshold else ""
        tagged_dir = base / f"d{density}{mode_part}_lm{lm}_k{rag_top_k}{threshold_part}_g{glossary_tag}"
        tagged_tsv = tagged_dir / "eval_results.tsv"
        tagged_metrics = parse_tsv_last_row(str(tagged_tsv))
        if tagged_metrics is not None:
            collected[f"tagged/lm{lm}"] = tagged_metrics

        combined_dir = base / f"d{density}_lm{lm}_k{rag_top_k}_per_paper_combined"
        by_paper_tsv = combined_dir / "eval_results_by_paper.tsv"
        by_paper_metrics = parse_tsv_last_row(str(by_paper_tsv))
        if by_paper_metrics is not None:
            collected[f"by_paper/lm{lm}"] = by_paper_metrics
    return collected


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="simuleval_eval")
    p.add_argument("--run-name", required=True)
    p.add_argument("--experiment-family", required=True,
                   help="e.g. sst_density_ablation, retriever_threshold_sweep")
    p.add_argument("--data-tag", required=True,
                   help="e.g. extracted_by_paper, acl6060_tagged")
    p.add_argument("--task-tag", default="eval", choices=["eval", "smoke"])
    p.add_argument("--notes-file", required=True,
                   help="Markdown run notes (validated against template schema).")
    p.add_argument("--trained-from-run", default="",
                   help="WandB run id of the training run that produced the model.")
    p.add_argument("--baseline-run-ids", nargs="*", default=[])
    p.add_argument("--extra-tags", nargs="*", default=[])
    p.add_argument("--density", required=True)
    p.add_argument("--rag-top-k", required=True)
    p.add_argument("--output-base", required=True)
    p.add_argument("--lang-code", default="zh")
    p.add_argument("--latency-multipliers", nargs="+", type=int, required=True)
    p.add_argument("--glossary-tag", required=True)
    p.add_argument("--rag-score-threshold", default="",
                   help="Optional RAG tau used by one-paper output dirs.")
    p.add_argument("--oracle-term-map", action="store_true",
                   help="Scan eval_density_unified.sh directories with the _oraclegt suffix.")
    p.add_argument("--paper-id", default="",
                   help="Optional ACL paper id for one-paper eval output dirs.")
    p.add_argument("--model-name", default="")
    p.add_argument("--rag-model-path", default="")
    p.add_argument("--verdict", default="")
    p.add_argument("--mode", default="finalize", choices=["finalize"],
                   help="Reserved for future sub-commands (e.g. incremental log).")
    p.add_argument("--allow-empty", action="store_true",
                   help="If set, do not fail when no TSVs are found (useful for smoke).")
    args = p.parse_args()

    run_notes = load_and_validate_run_notes(args.notes_file)

    try:
        import wandb
    except ImportError as exc:
        print(f"[wandb_eval_logger] wandb not installed: {exc}", file=sys.stderr)
        return 2

    tags: List[str] = [
        f"family:{args.experiment_family}",
        f"task:{args.task_tag}",
        f"data:{args.data_tag}",
        "status:running",
    ]
    for t in args.extra_tags:
        if t and t not in tags:
            tags.append(t)
    tags, tag_compressions = prepare_wandb_tags(tags)

    config = {
        "density": args.density,
        "rag_top_k": args.rag_top_k,
        "lang_code": args.lang_code,
        "latency_multipliers": args.latency_multipliers,
        "glossary_tag": args.glossary_tag,
        "rag_score_threshold": args.rag_score_threshold,
        "oracle_term_map": args.oracle_term_map,
        "paper_id": args.paper_id,
        "output_base": args.output_base,
        "model_name": args.model_name,
        "rag_model_path": args.rag_model_path,
        "trained_from_run": args.trained_from_run,
        "baseline_run_ids": args.baseline_run_ids,
        "notes_file": args.notes_file,
        "wandb_tag_compressions": [
            {"original": old, "compressed": new}
            for old, new in tag_compressions
        ],
    }
    for old, new in tag_compressions:
        print(
            f"[wandb_eval_logger] compressed WandB tag: {old!r} -> {new!r}",
            file=sys.stderr,
        )

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        config=config,
        tags=tags,
        notes=run_notes,
        save_code=True,
    )

    metrics = scan_outputs(
        output_base=args.output_base,
        lang_code=args.lang_code,
        density=args.density,
        rag_top_k=args.rag_top_k,
        latency_multipliers=args.latency_multipliers,
        glossary_tag=args.glossary_tag,
        rag_score_threshold=args.rag_score_threshold,
        paper_id=args.paper_id,
        oracle_term_map=args.oracle_term_map,
    )

    if not metrics and not args.allow_empty:
        run.summary["verdict"] = "FAILED: no eval TSVs found to log"
        # Flip tag to failed; fail loudly.
        new_tags = [t for t in (run.tags or []) if not t.startswith("status:")]
        new_tags.append("status:failed")
        run.tags = tuple(prepare_wandb_tags(new_tags)[0])
        run.finish()
        print(
            f"[wandb_eval_logger] No eval TSVs found under {args.output_base}. "
            "Pass --allow-empty to suppress this failure (e.g. smoke tests).",
            file=sys.stderr,
        )
        return 3

    for section, per_metric in metrics.items():
        for metric_name, value in per_metric.items():
            run.summary[f"{section}/{metric_name}"] = value

    success = bool(metrics)
    verdict = args.verdict or "pending - awaiting agent fill (see run notes)"
    run.summary["verdict"] = verdict

    new_tags = [t for t in (run.tags or []) if not t.startswith("status:")]
    new_tags.append("status:success" if success else "status:failed")
    run.tags = tuple(prepare_wandb_tags(new_tags)[0])
    run.finish()
    print(
        f"[wandb_eval_logger] Logged {len(metrics)} section(s) to "
        f"{run.project}/{run.id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
