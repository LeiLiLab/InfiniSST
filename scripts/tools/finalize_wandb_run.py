#!/usr/bin/env python3
"""Poll a WandB training run until finished, then back-fill tags, notes, and a
verdict summary line per `documents/code/.cursor/rules/experiment_tracking.mdc`.

Designed for two scenarios:

1. Legacy / unmanaged runs (e.g. 43848 `zv28ve3q`) that were launched before
   the mandatory schema flags were threaded through; fills in all four
   schema tags (family, task, data, status) retroactively and writes
   `run.summary["verdict"]` + `## Verdict` in the notes markdown.

2. New runs launched via a schema-compliant launcher: the training script
   already wrote `family:`, `task:`, `data:`, `status:running` at init; this
   finalizer only flips `status:running` -> `status:success|partial|failed`
   and writes the verdict.

Decision rule (configurable via CLI thresholds):

    success  : best/metric_value >= primary_threshold
               AND best_secondary/metric_value >= secondary_threshold
               AND noise@tau_0p80_gs10000 <= noise_threshold
    partial  : best/metric_value >= primary_threshold only
    failed   : otherwise

The verdict sentence is also written into the `## Verdict` section of the
notes markdown file (replacing any placeholder).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("finalize_wandb_run")


NOISE_METRIC_KEY = "eval_acl6060/noterm_noise@top10_tau_0p80_gs10000"
TCM_POS_VIOL_KEY = "train/tcm_pos_viol_rate"
TCM_NEG_VIOL_KEY = "train/tcm_neg_viol_rate"
POS_SIM_KEY = "train/pos_sim"
PRIMARY_RECALL_KEY = "eval_acl6060/recall@10_gs1000"
SECONDARY_RECALL_KEY = "eval_acl6060/recall@10_gs10000"
FILT_PRIMARY_KEY = (
    "eval_acl6060/topk10_chunk_any_positive_filtered_recall@tau_0p80_gs1000"
)
FILT_SECONDARY_KEY = (
    "eval_acl6060/topk10_chunk_any_positive_filtered_recall@tau_0p80_gs10000"
)
FILT_PRIMARY_LEGACY_KEY = "eval_acl6060/topk10_filtered_recall@tau_0p80_gs1000"
FILT_SECONDARY_LEGACY_KEY = "eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000"

# The complete eval/train bundle we snapshot at each best checkpoint step.
# Rule §F forbids quoting these from `run.summary[...]` (last-step snapshot);
# the finalizer and `wandb_tool.py compare --at-best-step` are the only two
# readers that obey the at-best-step discipline.
BUNDLE_METRIC_KEYS: Tuple[str, ...] = (
    PRIMARY_RECALL_KEY,
    SECONDARY_RECALL_KEY,
    FILT_PRIMARY_KEY,
    FILT_SECONDARY_KEY,
    FILT_PRIMARY_LEGACY_KEY,
    FILT_SECONDARY_LEGACY_KEY,
    NOISE_METRIC_KEY,
    TCM_POS_VIOL_KEY,
    TCM_NEG_VIOL_KEY,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="luojiaxuan1215-johns-hopkins-university")
    p.add_argument("--project", default="qwen3_rag")
    p.add_argument("--run_id", required=True, help="WandB run id, e.g. zv28ve3q")
    p.add_argument("--notes_file", required=True,
                   help="Local markdown file; its contents overwrite run.notes "
                        "and the ## Verdict section is rewritten in place.")
    p.add_argument("--family", required=True,
                   help="experiment_family tag value, e.g. sst_ood_hardneg")
    p.add_argument("--data_tag", required=True,
                   help="data tag value, e.g. 3variant_1m_mfa")
    p.add_argument("--task_tag", default="train")
    p.add_argument("--variant_tag", required=True,
                   help="variant:<value> tag appended to run.tags")
    p.add_argument("--launcher", default="",
                   help="Launcher script basename for config.launcher")
    p.add_argument("--baseline_run_ids", nargs="*", default=[],
                   help="WandB run ids this run is compared against")
    p.add_argument("--primary_threshold", type=float, required=True,
                   help="Pass threshold for best/metric_value (primary)")
    p.add_argument("--secondary_threshold", type=float, required=True,
                   help="Pass threshold for best_secondary/metric_value")
    p.add_argument("--noise_threshold", type=float, default=2.0,
                   help="Max acceptable noterm_noise@top10_tau_0p80_gs10000")
    p.add_argument("--poll_interval_sec", type=int, default=60)
    p.add_argument("--max_wait_sec", type=int, default=28 * 3600,
                   help="Hard cap on how long to poll before aborting")
    p.add_argument("--finished_states", nargs="*",
                   default=["finished", "failed", "crashed", "killed"],
                   help="Run states that end polling")
    return p.parse_args()


def _wait_for_finish(api: wandb.Api, path: str,
                     poll_interval: int, max_wait: int,
                     finished_states: Iterable[str]):
    """Block until run.state in finished_states or timeout; return final run."""
    t0 = time.time()
    finished = set(finished_states)
    while True:
        run = api.run(path)
        state = run.state
        elapsed = int(time.time() - t0)
        logger.info(
            f"[poll] run={path} state={state} elapsed={elapsed}s "
            f"(waiting for {finished})"
        )
        if state in finished:
            return run
        if elapsed >= max_wait:
            raise TimeoutError(
                f"finalizer timed out after {elapsed}s waiting for {path} "
                f"(last state: {state})"
            )
        time.sleep(poll_interval)


def _scan_metric_at_step(run, metric_key: str, target_step: Optional[int]) -> Optional[float]:
    """Walk run.scan_history() and return metric_key closest to target_step.

    If target_step is None, returns the last non-null value.
    """
    best_val: Optional[float] = None
    best_step_delta: Optional[int] = None
    try:
        hist = run.scan_history(keys=[metric_key, "_step"], page_size=500)
    except Exception as exc:
        logger.warning(f"scan_history failed: {exc}")
        return None
    for row in hist:
        v = row.get(metric_key)
        if v is None:
            continue
        s = row.get("_step")
        if target_step is None:
            best_val = v
            continue
        if s is None:
            continue
        delta = abs(int(s) - int(target_step))
        if best_step_delta is None or delta < best_step_delta:
            best_val = v
            best_step_delta = delta
    return best_val


def _scan_bundle_at_step(
    run, metric_keys: Iterable[str], target_step: Optional[int]
) -> Dict[str, Optional[float]]:
    """Return {metric: value} at `target_step` for all `metric_keys` in ONE scan.

    - Exact-step match wins; otherwise each metric picks its nearest-step row
      independently (train/* may log at slightly different global_steps than
      eval_*). Values are read only from `run.scan_history` — never from
      `run.summary`, which carries last-logged snapshots and would silently
      contaminate the bundle with a different checkpoint's numbers.
    """
    keys_list = [k for k in metric_keys]
    out: Dict[str, Optional[float]] = {k: None for k in keys_list}
    if target_step is None:
        return out
    try:
        target_int = int(target_step)
    except Exception:
        return out
    # Fetch per key so that the "rows where ALL keys are non-null" quirk of
    # scan_history(keys=[...many...]) doesn't silently drop data.
    for k in keys_list:
        exact_val: Optional[float] = None
        nearest_val: Optional[float] = None
        nearest_delta: Optional[int] = None
        try:
            for row in run.scan_history(keys=[k, "_step"], page_size=500):
                v = row.get(k)
                s = row.get("_step")
                if v is None or s is None:
                    continue
                try:
                    s_int = int(s)
                except Exception:
                    continue
                if s_int == target_int:
                    exact_val = float(v)
                    break
                delta = abs(s_int - target_int)
                if nearest_delta is None or delta < nearest_delta:
                    nearest_val = float(v)
                    nearest_delta = delta
        except Exception as exc:
            logger.warning(f"scan_history[{k}] failed: {exc}")
            continue
        out[k] = exact_val if exact_val is not None else nearest_val
    return out


def _decide_status(primary: Optional[float], secondary: Optional[float],
                   noise: Optional[float],
                   primary_thr: float, secondary_thr: float,
                   noise_thr: float) -> str:
    primary_ok = primary is not None and primary >= primary_thr
    secondary_ok = secondary is not None and secondary >= secondary_thr
    noise_ok = noise is None or noise <= noise_thr
    if primary_ok and secondary_ok and noise_ok:
        return "status:success"
    if primary_ok or secondary_ok:
        return "status:partial"
    return "status:failed"


def _fmt_float(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _build_verdict_sentence(
    *,
    status_tag: str,
    best_step: Optional[int],
    bundle_primary: Dict[str, Optional[float]],
    best_secondary_step: Optional[int],
    bundle_secondary: Dict[str, Optional[float]],
    baseline_run_ids: list,
    baseline_primary: Optional[float] = None,
    baseline_secondary: Optional[float] = None,
) -> str:
    """Construct the one-line verdict anchored on the primary best-step bundle.

    All reported values come from `bundle_primary` (scan_history row at
    `best/step`), NEVER from `run.summary[...]` — see rule §F. We include the
    secondary best-step value (gs10000 recall) for reference since the
    secondary checkpoint is exported too.
    """
    primary = bundle_primary.get(PRIMARY_RECALL_KEY)
    secondary_at_primary = bundle_primary.get(SECONDARY_RECALL_KEY)
    secondary_at_secondary = bundle_secondary.get(SECONDARY_RECALL_KEY)
    filt_primary = bundle_primary.get(FILT_PRIMARY_KEY)
    filt_secondary = bundle_primary.get(FILT_SECONDARY_KEY)
    noise = bundle_primary.get(NOISE_METRIC_KEY)
    tcm_pos_viol = bundle_primary.get(TCM_POS_VIOL_KEY)
    tcm_neg_viol = bundle_primary.get(TCM_NEG_VIOL_KEY)

    parts = [
        f"{status_tag.upper()}",
        f"best@step={best_step}",
        f"recall@10_gs1000={_fmt_float(primary)}",
        f"recall@10_gs10000@primary={_fmt_float(secondary_at_primary)}",
        f"recall@10_gs10000@secondary(step={best_secondary_step})={_fmt_float(secondary_at_secondary)}",
        f"filt@tau0.80_gs1000={_fmt_float(filt_primary)}",
        f"filt@tau0.80_gs10000={_fmt_float(filt_secondary)}",
        f"noise@tau0.80_gs10000={_fmt_float(noise, 2)}",
        f"tcm_viol(pos/neg)={_fmt_float(tcm_pos_viol, 3)}/{_fmt_float(tcm_neg_viol, 3)}",
    ]
    if baseline_run_ids:
        b = ",".join(baseline_run_ids)
        parts.append(f"baseline={b}")
        if baseline_primary is not None and primary is not None:
            parts.append(f"delta_gs1000={primary - baseline_primary:+.4f}")
        if baseline_secondary is not None and secondary_at_secondary is not None:
            parts.append(
                f"delta_gs10000={secondary_at_secondary - baseline_secondary:+.4f}"
            )
    return "; ".join(parts)


def _fetch_baseline_best(api: wandb.Api, entity: str, project: str,
                         run_id: str) -> tuple:
    try:
        r = api.run(f"{entity}/{project}/{run_id}")
        s = dict(r.summary)
        return (s.get("best/metric_value"), s.get("best_secondary/metric_value"))
    except Exception as exc:
        logger.warning(f"baseline fetch {run_id} failed: {exc}")
        return (None, None)


def _rewrite_verdict_section(md_text: str, verdict: str) -> str:
    """Replace everything after `## Verdict` with the verdict body."""
    marker = "## Verdict"
    idx = md_text.find(marker)
    if idx == -1:
        return md_text.rstrip() + f"\n\n{marker}\n\n{verdict}\n"
    body = f"\n\n{verdict}\n"
    return md_text[: idx + len(marker)] + body


def main() -> int:
    args = _parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be set in the environment")

    api = wandb.Api()
    path = f"{args.entity}/{args.project}/{args.run_id}"

    logger.info(f"[finalize] target={path} notes_file={args.notes_file}")
    logger.info(f"[finalize] thresholds primary>={args.primary_threshold} "
                f"secondary>={args.secondary_threshold} noise<={args.noise_threshold}")

    run = _wait_for_finish(
        api, path,
        poll_interval=args.poll_interval_sec,
        max_wait=args.max_wait_sec,
        finished_states=args.finished_states,
    )
    logger.info(f"[finalize] run reached state={run.state}")

    summary = dict(run.summary)
    # best/metric_value is the ONLY `eval_*` summary key we trust — the
    # training script writes it alongside `best/step` at the exact moment a
    # new best checkpoint is saved, so the two always match (rule §F).
    primary = summary.get("best/metric_value")
    secondary_best = summary.get("best_secondary/metric_value")
    best_step = summary.get("best/step")
    best_secondary_step = summary.get("best_secondary/step")

    logger.info(
        f"[finalize] scanning history for bundle at best/step={best_step} "
        f"and best_secondary/step={best_secondary_step} "
        f"(keys={list(BUNDLE_METRIC_KEYS)})"
    )
    bundle_primary = _scan_bundle_at_step(run, BUNDLE_METRIC_KEYS, best_step)
    bundle_secondary = _scan_bundle_at_step(
        run, BUNDLE_METRIC_KEYS, best_secondary_step
    )

    # Prefer the at-best-step bundle value over the summary fallback so the
    # pass/fail decision reflects the checkpoint we would actually export.
    primary_for_status = bundle_primary.get(PRIMARY_RECALL_KEY)
    if primary_for_status is None:
        primary_for_status = primary
    secondary_for_status = bundle_secondary.get(SECONDARY_RECALL_KEY)
    if secondary_for_status is None:
        secondary_for_status = secondary_best
    noise_for_status = bundle_primary.get(NOISE_METRIC_KEY)

    if run.state == "finished":
        status_tag = _decide_status(
            primary_for_status, secondary_for_status, noise_for_status,
            args.primary_threshold, args.secondary_threshold, args.noise_threshold,
        )
    else:
        status_tag = "status:failed"

    baseline_primary = None
    baseline_secondary = None
    if args.baseline_run_ids:
        baseline_primary, baseline_secondary = _fetch_baseline_best(
            api, args.entity, args.project, args.baseline_run_ids[0],
        )

    verdict = _build_verdict_sentence(
        status_tag=status_tag,
        best_step=int(best_step) if best_step is not None else None,
        bundle_primary=bundle_primary,
        best_secondary_step=(
            int(best_secondary_step) if best_secondary_step is not None else None
        ),
        bundle_secondary=bundle_secondary,
        baseline_run_ids=args.baseline_run_ids,
        baseline_primary=baseline_primary,
        baseline_secondary=baseline_secondary,
    )
    logger.info(f"[finalize] verdict: {verdict}")

    verdict_metrics_blob: Dict[str, Any] = {
        "primary": {
            "step": int(best_step) if best_step is not None else None,
            "metrics": {k: v for k, v in bundle_primary.items()},
        },
        "secondary": {
            "step": (
                int(best_secondary_step) if best_secondary_step is not None else None
            ),
            "metrics": {k: v for k, v in bundle_secondary.items()},
        },
        "status_tag": status_tag,
    }

    with open(args.notes_file, "r", encoding="utf-8") as f:
        md_text = f.read()
    md_text_new = _rewrite_verdict_section(md_text, verdict)
    if md_text_new != md_text:
        with open(args.notes_file, "w", encoding="utf-8") as f:
            f.write(md_text_new)
        logger.info(f"[finalize] rewrote ## Verdict section in {args.notes_file}")

    new_tags = [
        f"family:{args.family}",
        f"task:{args.task_tag}",
        f"data:{args.data_tag}",
        status_tag,
        f"variant:{args.variant_tag}",
    ]
    existing = list(run.tags or [])
    preserved = [
        t for t in existing
        if not any(t.startswith(p) for p in ("family:", "task:", "data:", "status:", "variant:"))
    ]
    merged = []
    for t in new_tags + preserved:
        if t and t not in merged:
            merged.append(t)
    run.tags = merged
    run.notes = md_text_new
    try:
        run.summary["verdict"] = verdict
        # Persist the at-best-step bundles as a JSON string; rule §F allows
        # later agents to read run.summary["verdict_metrics"] directly
        # without re-scanning history.
        run.summary["verdict_metrics"] = json.dumps(
            verdict_metrics_blob, default=lambda o: None, sort_keys=True,
        )
        run.summary.update()
    except Exception as exc:
        logger.warning(f"summary.update() failed: {exc}")

    cfg_update = {}
    if args.baseline_run_ids:
        cfg_update["baseline_run_ids"] = list(args.baseline_run_ids)
    if args.launcher:
        cfg_update["launcher"] = args.launcher
    if cfg_update:
        try:
            run.config.update(cfg_update)
        except Exception as exc:
            logger.warning(f"config.update() failed: {exc}")

    try:
        run.update()
    except Exception as exc:
        logger.error(f"run.update() failed: {exc}")
        return 2

    logger.info(f"[finalize] done: tags={run.tags} verdict_set=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
