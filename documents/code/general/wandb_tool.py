#!/usr/bin/env python3
"""WandB CLI helper for InfiniSST.

Packages the repeated WandB query patterns from `.cursor/rules/experiment_tracking.mdc`
(pre-flight baseline discovery, matched-step / peak metric comparison, legacy-run
debt backfill) into a small set of subcommands, so an agent session can pull
authoritative run data in one shot instead of five ad-hoc `wandb.Api()` python
snippets.

Design constraints:
  - Reads `WANDB_ENTITY` / `WANDB_PROJECT` env or falls back to `api.default_entity`
    and the three rule-sanctioned projects (`qwen3_rag`, `sst_omni`, `simuleval_eval`).
  - NEVER invents metrics. Presets below mirror the exact key names that the
    existing training / eval scripts log.
  - All write-side commands (`annotate`, `flip-status`) require `--yes` to guard
    against accidental overwrites of prod WandB state.

Subcommands:
  list-projects  List projects in the entity.
  find           Find runs by id(s), name substring, family tag, config filter.
  show           Dump metadata / config / tags / notes / selected summary keys.
  history        Export run.history as TSV (supports metric presets + regex).
  snapshot       Rich per-step eval view with structured sweep tables + trends.
  compare        Side-by-side delta table across N runs (peak / matched / both).
  topn           Pre-flight baseline discovery: top-N runs per family, by metric.
  db-sync        Sync WandB-derived run metadata/config/notes/metrics into SQLite.
  db-find        Search the generated SQLite experiment index.
  db-show        Inspect one generated-index run entity.
  db-compare     Compare cached at-best bundles, optionally refreshing from WandB.
  db-doctor      Audit the generated SQLite index for missing pieces.
  annotate       Add/replace tags, notes, summary keys, config on a run.
  flip-status    Replace status:* tag (success | failed | deprecated | baseline).

See `.cursor/skills/wandb-query/SKILL.md` for usage patterns.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import wandb
except ImportError as exc:
    print(f"[wandb_tool] wandb not installed: {exc}", file=sys.stderr)
    sys.exit(2)

try:
    from experiment_db import ExperimentDB, default_db_path
except ImportError:
    ExperimentDB = None  # type: ignore[assignment]
    default_db_path = None  # type: ignore[assignment]

try:
    from wandb_tags import compress_wandb_tag, prepare_wandb_tags
except ImportError as exc:
    print(f"[wandb_tool] failed to import wandb_tags: {exc}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Metric presets — edited together with reference.md.
# Add new presets here when a new training / eval script lands.
# ---------------------------------------------------------------------------

METRIC_PRESETS: Dict[str, List[str]] = {
    # Retriever eval (qwen3_rag_AuT_BGE_M3_train_lora.py and siblings).
    "retriever_eval": [
        "eval_acl6060/step",
        "eval_acl6060/recall@10",
        "eval_acl6060/recall@10_gs1000",
        "eval_acl6060/recall@10_gs10000",
        "eval_acl6060/topk10_filtered_recall@tau_0p70",
        "eval_acl6060/topk10_filtered_recall@tau_0p70_gs10000",
        "eval_acl6060/topk10_filtered_recall@tau_0p75",
        "eval_acl6060/topk10_filtered_recall@tau_0p75_gs10000",
        "eval_acl6060/topk10_filtered_recall@tau_0p80",
        "eval_acl6060/topk10_filtered_recall@tau_0p80_gs1000",
        "eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000",
        "eval_acl6060/topk10_filtered_recall@tau_0p85",
        "eval_acl6060/topk10_filtered_recall@tau_0p85_gs10000",
        "eval_acl6060/noterm_noise@top10_tau_0p75_gs10000",
        "eval_acl6060/noterm_noise@top10_tau_0p80_gs10000",
        "eval_dev/recall@10",
        "eval_dev/recall@10_gs10000",
        "eval_dev/topk10_filtered_recall@tau_0p70",
        "eval_dev/topk10_filtered_recall@tau_0p70_gs10000",
        "eval_dev/topk10_filtered_recall@tau_0p75",
        "eval_dev/topk10_filtered_recall@tau_0p75_gs10000",
        "eval_dev/topk10_filtered_recall@tau_0p80",
        "eval_dev/topk10_filtered_recall@tau_0p80_gs10000",
        "eval_dev/topk10_filtered_recall@tau_0p85",
        "eval_dev/topk10_filtered_recall@tau_0p85_gs10000",
    ],
    # Full sweep view — every threshold × gallery size logged by the
    # retriever training script.  Used by `snapshot` as fallback when
    # auto-discovery is not available.
    "retriever_eval_full": [
        "eval_acl6060/step",
        "eval_acl6060/loss",
        "eval_acl6060/top1",
        "eval_acl6060/recall@10",
        "eval_acl6060/bank_terms",
        # base-bank sweep (tau = 0.50 .. 0.80)
        "eval_acl6060/topk10_filtered_recall@tau_0p50",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p50",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p50",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p50",
        "eval_acl6060/noterm_noise@top10_tau_0p50",
        "eval_acl6060/topk10_filtered_recall@tau_0p60",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p60",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p60",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p60",
        "eval_acl6060/noterm_noise@top10_tau_0p60",
        "eval_acl6060/topk10_filtered_recall@tau_0p70",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p70",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p70",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p70",
        "eval_acl6060/noterm_noise@top10_tau_0p70",
        "eval_acl6060/topk10_filtered_recall@tau_0p80",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p80",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p80",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p80",
        "eval_acl6060/noterm_noise@top10_tau_0p80",
        # gs1000
        "eval_acl6060/recall@10_gs1000",
        "eval_acl6060/topk10_filtered_recall@tau_0p50_gs1000",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p50_gs1000",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p50_gs1000",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p50_gs1000",
        "eval_acl6060/noterm_noise@top10_tau_0p50_gs1000",
        "eval_acl6060/topk10_filtered_recall@tau_0p60_gs1000",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p60_gs1000",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p60_gs1000",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p60_gs1000",
        "eval_acl6060/noterm_noise@top10_tau_0p60_gs1000",
        "eval_acl6060/topk10_filtered_recall@tau_0p70_gs1000",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p70_gs1000",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p70_gs1000",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p70_gs1000",
        "eval_acl6060/noterm_noise@top10_tau_0p70_gs1000",
        "eval_acl6060/topk10_filtered_recall@tau_0p80_gs1000",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p80_gs1000",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p80_gs1000",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p80_gs1000",
        "eval_acl6060/noterm_noise@top10_tau_0p80_gs1000",
        # gs10000
        "eval_acl6060/recall@10_gs10000",
        "eval_acl6060/topk10_filtered_recall@tau_0p50_gs10000",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p50_gs10000",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p50_gs10000",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p50_gs10000",
        "eval_acl6060/noterm_noise@top10_tau_0p50_gs10000",
        "eval_acl6060/topk10_filtered_recall@tau_0p60_gs10000",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p60_gs10000",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p60_gs10000",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p60_gs10000",
        "eval_acl6060/noterm_noise@top10_tau_0p60_gs10000",
        "eval_acl6060/topk10_filtered_recall@tau_0p70_gs10000",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p70_gs10000",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p70_gs10000",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p70_gs10000",
        "eval_acl6060/noterm_noise@top10_tau_0p70_gs10000",
        "eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000",
        "eval_acl6060/topk10_filtered_precision_micro@tau_0p80_gs10000",
        "eval_acl6060/topk10_filtered_precision_macro@tau_0p80_gs10000",
        "eval_acl6060/topk10_avg_kept_if_pass@tau_0p80_gs10000",
        "eval_acl6060/noterm_noise@top10_tau_0p80_gs10000",
    ],
    # Retriever training signal.
    "retriever_train": [
        "train/step",
        "train/epoch",
        "train/loss",
        "train/loss_infonce",
        "train/loss_tcm_pos",
        "train/loss_tcm_neg",
        "train/tcm_pos_viol_rate",
        "train/tcm_neg_viol_rate",
        "train/pos_sim_mean",
        "train/neg_sim_mean",
        "train/pos_sim",
        "train/neg_sim",
        "train/logit_scale",
        "train/temperature",
        "train/lr",
    ],
    # Offline simuleval (wandb_eval_logger.py writes these via scan_outputs).
    "simuleval_eval": [
        "by_paper/lm1/TERM_ACC",
        "by_paper/lm1/TERM_FCR",
        "by_paper/lm1/BLEU",
        "by_paper/lm1/StreamLAAL",
        "by_paper/lm1/StreamLAAL_CA",
        "by_paper/lm1/TCR",
        "by_paper/lm2/TERM_ACC",
        "by_paper/lm2/TERM_FCR",
        "by_paper/lm2/BLEU",
        "by_paper/lm2/StreamLAAL",
        "by_paper/lm3/TERM_ACC",
        "by_paper/lm3/TERM_FCR",
        "by_paper/lm3/BLEU",
        "by_paper/lm3/StreamLAAL",
        "by_paper/lm4/TERM_ACC",
        "by_paper/lm4/TERM_FCR",
        "by_paper/lm4/BLEU",
        "by_paper/lm4/StreamLAAL",
    ],
}


def _with_chunk_any_positive_aliases(keys: Sequence[str]) -> List[str]:
    """Add the renamed chunk-any-positive recall key beside legacy keys."""
    out: List[str] = []
    for key in keys:
        out.append(key)
        if "_filtered_recall@" in key:
            out.append(
                key.replace(
                    "_filtered_recall@",
                    "_chunk_any_positive_filtered_recall@",
                )
            )
    return out


for _preset_name in ("retriever_eval", "retriever_eval_full"):
    METRIC_PRESETS[_preset_name] = _with_chunk_any_positive_aliases(
        METRIC_PRESETS[_preset_name]
    )
    _medicine_keys = [
        key.replace("eval_acl6060/", "eval_medicine/")
        for key in METRIC_PRESETS[_preset_name]
        if key.startswith("eval_acl6060/")
    ]
    _tagged_acl_keys = [
        key.replace("eval_acl6060/", "eval_tagged_acl/")
        for key in METRIC_PRESETS[_preset_name]
        if key.startswith("eval_acl6060/")
    ]
    for _key in _medicine_keys + _tagged_acl_keys:
        if _key not in METRIC_PRESETS[_preset_name]:
            METRIC_PRESETS[_preset_name].append(_key)

RULE_PROJECTS = ("qwen3_rag", "sst_omni", "simuleval_eval")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_entity(arg_entity: Optional[str]) -> str:
    if arg_entity:
        return arg_entity
    env = os.environ.get("WANDB_ENTITY")
    if env:
        return env
    api = wandb.Api()
    return api.default_entity


def _resolve_projects(arg_project: Optional[str], arg_all_rule_projects: bool) -> List[str]:
    if arg_project:
        return [arg_project]
    if arg_all_rule_projects:
        return list(RULE_PROJECTS)
    env = os.environ.get("WANDB_PROJECT")
    if env:
        return [env]
    return list(RULE_PROJECTS)


def _expand_keys(args_keys: Optional[List[str]], args_presets: Optional[List[str]]) -> List[str]:
    keys: List[str] = []
    for preset in args_presets or []:
        if preset not in METRIC_PRESETS:
            raise SystemExit(
                f"[wandb_tool] unknown preset '{preset}'. Available: "
                f"{sorted(METRIC_PRESETS)}"
            )
        keys.extend(METRIC_PRESETS[preset])
    for k in args_keys or []:
        keys.append(k)
    # dedupe, preserve order
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


_API_SINGLETON: Optional["wandb.Api"] = None


def _get_api() -> "wandb.Api":
    global _API_SINGLETON
    if _API_SINGLETON is None:
        _API_SINGLETON = wandb.Api()
    return _API_SINGLETON


def _run_summary_dict(run: "wandb.apis.public.Run") -> Dict[str, Any]:
    """Return a plain-dict snapshot of `run.summary`.

    `api.runs(...)` (filter queries) returns Run objects whose
    `_attrs['summaryMetrics']` is `None`, while `api.run(<id>)` (direct lookup)
    populates it fully. `run.summary.get()` / `.keys()` has a known bug on
    string-typed keys like our `verdict` (raises
    `AttributeError: 'str' object has no attribute 'keys'`), so we cannot rely
    on it. Instead, when `_attrs` is empty, we re-fetch the run by id.
    """
    raw = run._attrs.get("summaryMetrics")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            pass
    if isinstance(raw, dict):
        return dict(raw)
    # Fallback: re-fetch by id (costs one extra GraphQL call per run).
    try:
        fresh = _get_api().run(f"{run.entity}/{run.project}/{run.id}")
        raw2 = fresh._attrs.get("summaryMetrics")
        if isinstance(raw2, str):
            raw2 = json.loads(raw2)
        if isinstance(raw2, dict):
            return dict(raw2)
    except Exception:
        pass
    return {}


def _run_config_dict(run: "wandb.apis.public.Run") -> Dict[str, Any]:
    """Return a plain config dict, refetching direct run objects if needed."""
    raw = getattr(run, "config", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if not str(k).startswith("_")}
    try:
        fresh = _get_api().run(f"{run.entity}/{run.project}/{run.id}")
        raw2 = getattr(fresh, "config", {})
        if isinstance(raw2, str):
            raw2 = json.loads(raw2)
        if isinstance(raw2, dict):
            return {k: v for k, v in raw2.items() if not str(k).startswith("_")}
    except Exception:
        pass
    return {}


def _match_runs(
    api: "wandb.Api",
    entity: str,
    projects: Sequence[str],
    ids: Sequence[str] = (),
    name_contains: Optional[str] = None,
    family: Optional[str] = None,
    tag_filters: Sequence[str] = (),
    config_filters: Sequence[str] = (),
    state: Optional[str] = None,
    limit: int = 200,
) -> List["wandb.apis.public.Run"]:
    """Return runs matching all of the supplied filters."""
    out: List["wandb.apis.public.Run"] = []
    if ids:
        # Direct id lookup across candidate projects.
        for rid in ids:
            found = None
            for p in projects:
                try:
                    found = api.run(f"{entity}/{p}/{rid}")
                    break
                except Exception:
                    continue
            if found is None:
                print(
                    f"[wandb_tool] WARNING: id '{rid}' not found in projects "
                    f"{list(projects)} under entity '{entity}'.",
                    file=sys.stderr,
                )
            else:
                out.append(found)
        return out

    # Server-side filter via MongoDB-style selector.
    selector: Dict[str, Any] = {}
    if name_contains:
        selector["display_name"] = {"$regex": re.escape(name_contains)}
    if state:
        selector["state"] = state
    tags_and: List[str] = list(tag_filters)
    if family:
        tags_and.append(f"family:{family}")
    if tags_and:
        selector["tags"] = {"$all": tags_and}
    for cf in config_filters:
        if "=" not in cf:
            raise SystemExit(f"[wandb_tool] --config filter must be key=value (got '{cf}').")
        k, v = cf.split("=", 1)
        # Best-effort type coercion.
        try:
            parsed: Any = json.loads(v)
        except Exception:
            parsed = v
        selector[f"config.{k}.value"] = parsed

    for p in projects:
        try:
            runs = api.runs(f"{entity}/{p}", filters=selector or None, per_page=min(limit, 200))
            for r in runs:
                out.append(r)
                if len(out) >= limit:
                    return out
        except Exception as exc:
            print(f"[wandb_tool] filter query failed on {entity}/{p}: {exc}", file=sys.stderr)
    return out


def _print_run_row(run: "wandb.apis.public.Run", wide: bool = False) -> None:
    tags = ",".join(run.tags or []) or "-"
    url = run.url if hasattr(run, "url") else f"https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}"
    name = run.name if wide else (run.name[:80] + "…" if len(run.name) > 80 else run.name)
    runtime = _run_summary_dict(run).get("_runtime")
    rt = f"{int(runtime)}s" if isinstance(runtime, (int, float)) else "-"
    print(f"{run.id:<10} {run.state:<10} {rt:<8} [{tags}]  {name}  {url}")


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_list_projects(args: argparse.Namespace) -> int:
    api = wandb.Api()
    entity = _resolve_entity(args.entity)
    projects = [p.name for p in api.projects(entity=entity)]
    if args.json:
        print(json.dumps({"entity": entity, "projects": projects}, indent=2))
    else:
        print(f"entity: {entity}")
        for p in projects:
            marker = " (rule-sanctioned)" if p in RULE_PROJECTS else ""
            print(f"  - {p}{marker}")
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    api = wandb.Api()
    entity = _resolve_entity(args.entity)
    projects = _resolve_projects(args.project, args.all_rule_projects)

    runs = _match_runs(
        api, entity, projects,
        ids=args.ids or (),
        name_contains=args.name_contains,
        family=args.family,
        tag_filters=args.tag or (),
        config_filters=args.config or (),
        state=args.state,
        limit=args.limit,
    )
    if args.json:
        payload = [
            {
                "id": r.id,
                "project": r.project,
                "name": r.name,
                "state": r.state,
                "tags": list(r.tags or []),
                "runtime_s": _run_summary_dict(r).get("_runtime"),
                "created_at": str(r.created_at),
                "url": getattr(r, "url", None),
            }
            for r in runs
        ]
        print(json.dumps(payload, indent=2, default=str))
    else:
        if not runs:
            print("(no runs matched)")
            return 0
        print(f"# {len(runs)} run(s) in entity={entity} project(s)={projects}")
        print(f"{'id':<10} {'state':<10} {'runtime':<8} {'tags':<30}  name  url")
        for r in runs:
            _print_run_row(r, wide=args.wide)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    api = wandb.Api()
    entity = _resolve_entity(args.entity)
    projects = _resolve_projects(args.project, args.all_rule_projects)
    runs = _match_runs(api, entity, projects, ids=[args.run_id])
    if not runs:
        return 4
    run = runs[0]

    summary = _run_summary_dict(run)
    if args.summary_prefix:
        summary = {k: v for k, v in summary.items() if k.startswith(args.summary_prefix)}
    if args.summary_regex:
        pat = re.compile(args.summary_regex)
        summary = {k: v for k, v in summary.items() if pat.search(k)}

    out = {
        "id": run.id,
        "project": run.project,
        "entity": run.entity,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "runtime_s": _run_summary_dict(run).get("_runtime"),
        "tags": list(run.tags or []),
        "url": getattr(run, "url", None),
        "notes": run.notes or "",
        "config": _run_config_dict(run),
        "summary": summary,
    }
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"id        : {out['id']}")
        print(f"project   : {out['entity']}/{out['project']}")
        print(f"name      : {out['name']}")
        print(f"state     : {out['state']}   runtime: {out['runtime_s']}s")
        print(f"created   : {out['created_at']}")
        print(f"url       : {out['url']}")
        print(f"tags      : {out['tags']}")
        print(f"notes     : {(out['notes'] or '')[:300]!r}{'…' if len(out['notes'] or '') > 300 else ''}")
        print("config    :")
        for k in sorted(out["config"]):
            print(f"  {k} = {out['config'][k]!r}")
        print(f"summary ({len(out['summary'])} keys shown):")
        for k in sorted(out["summary"]):
            v = out["summary"][k]
            if isinstance(v, float):
                vs = f"{v:.4f}"
            else:
                vs = str(v)
                if len(vs) > 80:
                    vs = vs[:77] + "..."
            print(f"  {k} = {vs}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    api = wandb.Api()
    entity = _resolve_entity(args.entity)
    projects = _resolve_projects(args.project, args.all_rule_projects)
    runs = _match_runs(api, entity, projects, ids=[args.run_id])
    if not runs:
        return 4
    run = runs[0]

    keys = _expand_keys(args.keys, args.preset)
    if not keys:
        raise SystemExit("[wandb_tool] history needs --keys or --preset.")

    df = run.history(keys=keys, pandas=True, samples=args.samples)
    if args.eval_rows_only:
        # Prefer whichever eval/*/step key is present.
        step_cols = [c for c in df.columns if c.endswith("/step") and c != "train/step"]
        if step_cols:
            df = df.dropna(subset=step_cols, how="all").sort_values(step_cols[0])

    if args.out:
        df.to_csv(args.out, sep="\t", index=False)
        print(f"[wandb_tool] wrote {len(df)} rows to {args.out}")
    else:
        try:
            import pandas as pd
            pd.set_option("display.width", 200)
            pd.set_option("display.max_columns", 50)
            print(df.round(4).to_string(index=False))
        except Exception:
            print(df.head(50).to_csv(sep="\t", index=False))
    return 0


def _step_col(df) -> Optional[str]:
    for c in ("eval_acl6060/step", "eval_dev/step", "train/step", "_step"):
        if c in df.columns:
            return c
    return None


def _row_at_step(
    df, step_col: Optional[str], metric_keys: Iterable[str], target_step: Optional[int]
) -> Dict[str, Optional[float]]:
    """Return {metric_key: value} from `df` at the row whose `step_col` equals
    `target_step` (exact match first; nearest-step fallback).

    Reads exclusively from `df` (run.history output); never from `run.summary`.
    This is the primitive used by the AT-BEST-STEP compare view and by the
    finalizer — both need the same eval bundle taken from a single checkpoint
    step, not a mix of per-metric argmaxes or last-logged snapshots.
    """
    out: Dict[str, Optional[float]] = {k: None for k in metric_keys}
    if step_col is None or target_step is None or step_col not in df.columns:
        return out
    try:
        target_int = int(target_step)
    except Exception:
        return out
    sub = df[df[step_col] == target_int]
    if sub.empty:
        deltas = (df[step_col] - target_int).abs()
        nearest = deltas.dropna().sort_values().index[:1]
        sub = df.loc[nearest] if len(nearest) else df.iloc[0:0]
    if sub.empty:
        return out
    for k in metric_keys:
        if k not in sub.columns:
            continue
        v = sub[k].iloc[0]
        # NaN check without requiring pandas/np import here.
        try:
            if v != v:
                continue
        except Exception:
            continue
        try:
            out[k] = float(v)
        except Exception:
            out[k] = None
    return out


def _anchor_steps(summary: Dict[str, Any], which: str) -> List[Tuple[str, Optional[int]]]:
    """Return [(label, step)] pairs for the requested anchor(s).

    `which` is one of 'primary', 'secondary', 'both'. Steps are read from
    `run.summary["best/step"]` / `run.summary["best_secondary/step"]`, the only
    two summary keys documented by the training script to mark checkpoint
    steps (`.cursor/rules/experiment_tracking.mdc` §F).
    """
    prim = summary.get("best/step")
    sec = summary.get("best_secondary/step")
    try:
        prim = int(prim) if prim is not None else None
    except Exception:
        prim = None
    try:
        sec = int(sec) if sec is not None else None
    except Exception:
        sec = None
    if which == "primary":
        return [("primary", prim)]
    if which == "secondary":
        return [("secondary", sec)]
    return [("primary", prim), ("secondary", sec)]


def _history_merged_for_run(
    run: "wandb.apis.public.Run",
    metric_keys: Sequence[str],
    samples: int,
):
    """Fetch per-key history and outer-join on `_step`.

    This mirrors `cmd_compare`'s at-best-step read path so DB sync stores the
    same WandB-history-derived bundles that chat comparisons are allowed to
    quote under rule §F.
    """
    import pandas as pd

    per_key: Dict[str, "pd.DataFrame"] = {}
    value_keys = [k for k in metric_keys if not k.endswith("/step")]
    for k in value_keys:
        try:
            dk = run.history(keys=[k], pandas=True, samples=samples)
        except Exception as exc:
            print(f"[wandb_tool] history fetch failed {run.id} {k}: {exc}", file=sys.stderr)
            continue
        if dk is None or dk.empty or k not in dk.columns:
            continue
        if k.startswith("eval_acl6060/") and "eval_acl6060/step" in dk.columns:
            step_col_in_frame = "eval_acl6060/step"
        elif k.startswith("eval_dev/") and "eval_dev/step" in dk.columns:
            step_col_in_frame = "eval_dev/step"
        elif k.startswith("train/") and "train/step" in dk.columns:
            step_col_in_frame = "train/step"
        elif "_step" in dk.columns:
            step_col_in_frame = "_step"
        else:
            continue
        sub = dk[[step_col_in_frame, k]].dropna(subset=[k]).copy()
        if step_col_in_frame != "_step":
            sub = sub.rename(columns={step_col_in_frame: "_step"})
        sub["_step"] = sub["_step"].astype("Int64")
        sub = sub.dropna(subset=["_step"]).drop_duplicates(subset=["_step"])
        per_key[k] = sub
    if per_key:
        merged = None
        for sub in per_key.values():
            merged = sub if merged is None else merged.merge(sub, on="_step", how="outer")
        return merged.sort_values("_step").reset_index(drop=True)
    return pd.DataFrame(columns=["_step"])


def _sync_run_to_db(
    db: "ExperimentDB",
    run: "wandb.apis.public.Run",
    *,
    best_bundles: bool,
    metric_keys: Sequence[str],
    samples: int,
    command: str,
) -> Dict[str, Any]:
    summary = _run_summary_dict(run)
    config = _run_config_dict(run)
    db.upsert_run(
        run_id=run.id,
        entity=run.entity,
        project=run.project,
        name=run.name,
        url=getattr(run, "url", f"https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}"),
        state=run.state,
        created_at=str(run.created_at),
        runtime_s=summary.get("_runtime") if isinstance(summary.get("_runtime"), (int, float)) else None,
        tags=list(run.tags or []),
        config=config,
        notes=run.notes or "",
        summary=summary,
    )

    bundle_counts: Dict[str, int] = {}
    if best_bundles:
        df = _history_merged_for_run(run, metric_keys, samples=samples)
        sc = _step_col(df)
        metric_value_keys = [k for k in metric_keys if not k.endswith("/step")]
        for anchor, step in _anchor_steps(summary, "both"):
            bundle = _row_at_step(df, sc, metric_value_keys, step)
            db.upsert_metric_bundle(
                run.id,
                anchor,
                step,
                bundle,
                source="wandb_history_at_best_step",
            )
            bundle_counts[anchor] = sum(1 for v in bundle.values() if v is not None)

    db.log_event(
        run_id=run.id,
        project=run.project,
        source="wandb",
        command=command,
        status="success",
        message=f"best_bundles={best_bundles} counts={bundle_counts}",
    )
    return {"run_id": run.id, "project": run.project, "best_bundle_counts": bundle_counts}


def cmd_compare(args: argparse.Namespace) -> int:
    import pandas as pd

    api = wandb.Api()
    entity = _resolve_entity(args.entity)
    projects = _resolve_projects(args.project, args.all_rule_projects)
    runs = _match_runs(api, entity, projects, ids=args.run_ids)
    if len(runs) != len(args.run_ids):
        print(f"[wandb_tool] only matched {len(runs)}/{len(args.run_ids)} runs.", file=sys.stderr)
        if not runs:
            return 4

    metric_keys = _expand_keys(args.keys, args.preset)
    if not metric_keys:
        raise SystemExit("[wandb_tool] compare needs --keys or --preset.")

    # History per run. `summaries[r.id]` is the plain-dict snapshot used for
    # the AT-BEST-STEP block (`best/step`, `best_secondary/step`) — never for
    # quoting `eval_*` / `train/*` values directly (see rule §F).
    #
    # IMPORTANT: `r.history(keys=[k1, k2, ...])` returns rows where ALL keys
    # are non-null, which is empty whenever `train/*` and `eval_*/*` are
    # logged at different global_steps (the normal case). We fetch per-key
    # and outer-join on `_step` so downstream argmax / row lookup works.
    histories: Dict[str, "pd.DataFrame"] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for r in runs:
        summaries[r.id] = _run_summary_dict(r)
        per_key: Dict[str, "pd.DataFrame"] = {}
        # Step-only keys are helpers, not metrics — skip in per-key fetch.
        value_keys = [k for k in metric_keys if not k.endswith("/step")]
        for k in value_keys:
            try:
                dk = r.history(keys=[k], pandas=True, samples=args.samples)
            except Exception as exc:
                print(f"[wandb_tool] history fetch failed {r.id} {k}: {exc}", file=sys.stderr)
                continue
            if dk is None or dk.empty or k not in dk.columns:
                continue
            # Prefer namespace-specific step column; fall back to global `_step`.
            if k.startswith("eval_acl6060/") and "eval_acl6060/step" in dk.columns:
                step_col_in_frame = "eval_acl6060/step"
            elif k.startswith("eval_dev/") and "eval_dev/step" in dk.columns:
                step_col_in_frame = "eval_dev/step"
            elif k.startswith("train/") and "train/step" in dk.columns:
                step_col_in_frame = "train/step"
            elif "_step" in dk.columns:
                step_col_in_frame = "_step"
            else:
                continue
            sub = dk[[step_col_in_frame, k]].dropna(subset=[k]).copy()
            if step_col_in_frame != "_step":
                sub = sub.rename(columns={step_col_in_frame: "_step"})
            sub["_step"] = sub["_step"].astype("Int64")
            sub = sub.dropna(subset=["_step"]).drop_duplicates(subset=["_step"])
            per_key[k] = sub
        if per_key:
            merged = None
            for k, sub in per_key.items():
                merged = sub if merged is None else merged.merge(sub, on="_step", how="outer")
            merged = merged.sort_values("_step").reset_index(drop=True)
        else:
            merged = pd.DataFrame(columns=["_step"])
        histories[r.id] = merged

    # PEAK view.
    peak_rows: List[Dict[str, Any]] = []
    for k in metric_keys:
        if k.endswith("/step"):
            continue
        row: Dict[str, Any] = {"metric": k}
        for r in runs:
            df = histories[r.id]
            if k not in df.columns or df[k].dropna().empty:
                row[f"{r.id}_best"] = None
                row[f"{r.id}_step"] = None
                continue
            step_col = _step_col(df)
            agg_idx = df[k].idxmax() if args.direction == "max" else df[k].idxmin()
            row[f"{r.id}_best"] = float(df.loc[agg_idx, k])
            if step_col:
                try:
                    row[f"{r.id}_step"] = int(df.loc[agg_idx, step_col])
                except Exception:
                    row[f"{r.id}_step"] = None
        peak_rows.append(row)

    # MATCHED-STEP view: intersect eval steps across runs.
    matched_rows: List[Dict[str, Any]] = []
    step_col_name = None
    for r in runs:
        sc = _step_col(histories[r.id])
        if sc is not None:
            step_col_name = sc
            break
    common_steps: Optional[set] = None
    if step_col_name:
        for r in runs:
            df = histories[r.id]
            if step_col_name not in df.columns:
                common_steps = set()
                break
            steps = set(df[step_col_name].dropna().astype(int).tolist())
            common_steps = steps if common_steps is None else common_steps & steps
    if common_steps:
        for step in sorted(common_steps):
            for k in metric_keys:
                if k.endswith("/step"):
                    continue
                row: Dict[str, Any] = {"step": step, "metric": k}
                for r in runs:
                    df = histories[r.id]
                    sub = df[df[step_col_name] == step] if step_col_name in df.columns else df.iloc[0:0]
                    if sub.empty or k not in df.columns:
                        row[r.id] = None
                    else:
                        val = sub[k].iloc[0]
                        row[r.id] = float(val) if val == val else None  # NaN check
                matched_rows.append(row)

    # AT-BEST-STEP view: for each run, pick the anchor step(s) from
    # summary["best/step"] and/or summary["best_secondary/step"], then pull
    # EVERY preset metric at that step via history (never from summary).
    at_best: Dict[str, Any] = {}
    if getattr(args, "at_best_step", False):
        anchor_choice = getattr(args, "anchor_metric", "both")
        at_best = {
            "anchor": anchor_choice,
            "runs": {},  # {run_id: {"primary": {...}, "secondary": {...}}}
        }
        metric_value_keys = [k for k in metric_keys if not k.endswith("/step")]
        for r in runs:
            df = histories[r.id]
            sc = _step_col(df)
            s = summaries[r.id]
            anchors = _anchor_steps(s, anchor_choice)
            run_block: Dict[str, Any] = {
                "state": r.state,
                "current_step": s.get("_step"),
                "bundles": {},
            }
            for label, step in anchors:
                metric_bundle = _row_at_step(df, sc, metric_value_keys, step)
                run_block["bundles"][label] = {
                    "anchor_step": step,
                    "metrics": metric_bundle,
                }
            at_best["runs"][r.id] = run_block

    report = {
        "runs": [{"id": r.id, "name": r.name, "state": r.state} for r in runs],
        "peak": peak_rows,
        "matched_step_rows": matched_rows,
        "step_column_used": step_col_name,
        "at_best_step": at_best or None,
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    print(f"# compare: runs={[r.id for r in runs]}  entity={entity}")
    print("\n## PEAK (each run's best step per metric)\n")
    hdr = ["metric"] + [f"{r.id}_best({r.id}_step)" for r in runs]
    if len(runs) == 2:
        hdr.append(f"Δ({runs[1].id}-{runs[0].id})")
    print("  | ".join(hdr))
    for row in peak_rows:
        vals = [row["metric"]]
        for r in runs:
            v = row.get(f"{r.id}_best")
            s = row.get(f"{r.id}_step")
            vals.append(f"{v:.4f}@{s}" if v is not None else "-")
        if len(runs) == 2:
            a = row.get(f"{runs[0].id}_best")
            b = row.get(f"{runs[1].id}_best")
            vals.append(f"{b-a:+.4f}" if (a is not None and b is not None) else "-")
        print("  | ".join(vals))

    if matched_rows:
        print(f"\n## MATCHED STEP ({step_col_name})  common_steps={sorted(common_steps)}\n")
        by_step: Dict[int, List[Dict[str, Any]]] = {}
        for row in matched_rows:
            by_step.setdefault(row["step"], []).append(row)
        for step in sorted(by_step):
            print(f"  -- step {step} --")
            for row in by_step[step]:
                parts = [row["metric"]]
                for r in runs:
                    v = row.get(r.id)
                    parts.append(f"{r.id}={v:.4f}" if v is not None else f"{r.id}=-")
                if len(runs) == 2:
                    a, b = row.get(runs[0].id), row.get(runs[1].id)
                    if a is not None and b is not None:
                        parts.append(f"Δ={b-a:+.4f}")
                print("    " + "  |  ".join(parts))

    if at_best:
        anchor_choice = at_best["anchor"]
        anchor_labels = (
            ["primary", "secondary"] if anchor_choice == "both" else [anchor_choice]
        )
        print(
            f"\n## AT-BEST-STEP (anchor={anchor_choice}; metrics pulled from history "
            f"at each run's best/step and/or best_secondary/step — rule §F)\n"
        )
        # Header line: label each run with state + anchor step(s).
        for r in runs:
            block = at_best["runs"][r.id]
            anchor_str = ", ".join(
                f"{lbl}@{block['bundles'].get(lbl, {}).get('anchor_step')}"
                for lbl in anchor_labels
            )
            state_tag = (
                f"[running@step={block.get('current_step')} — not final]"
                if r.state == "running"
                else f"[{r.state}]"
            )
            print(f"  {r.id} {state_tag} anchor=({anchor_str})")
        print()

        metric_value_keys = [k for k in metric_keys if not k.endswith("/step")]
        for label in anchor_labels:
            print(f"  ---- anchor: {label} ----")
            hdr = ["metric"] + [f"{r.id}" for r in runs]
            if len(runs) == 2:
                hdr.append(f"Δ({runs[1].id}-{runs[0].id})")
            print("    " + "  | ".join(hdr))
            for k in metric_value_keys:
                vals = [k]
                raw_vals: List[Optional[float]] = []
                for r in runs:
                    v = at_best["runs"][r.id]["bundles"].get(label, {}).get(
                        "metrics", {}
                    ).get(k)
                    raw_vals.append(v)
                    vals.append(f"{v:.4f}" if v is not None else "-")
                if len(runs) == 2:
                    a, b = raw_vals[0], raw_vals[1]
                    vals.append(
                        f"{b-a:+.4f}" if (a is not None and b is not None) else "-"
                    )
                print("    " + "  | ".join(vals))
            print()
    return 0


def _snapshot_discover_keys(
    run: "wandb.apis.public.Run", prefix: str
) -> List[str]:
    """Return all history keys that start with `prefix/`."""
    try:
        all_keys = run.history(pandas=False, samples=1, keys=None)
        if all_keys and isinstance(all_keys, list) and isinstance(all_keys[0], dict):
            return sorted(k for k in all_keys[0].keys() if k.startswith(f"{prefix}/"))
    except Exception:
        pass
    try:
        hk = run.historyKeys
        if isinstance(hk, dict):
            candidates = list(hk.get("keys", {}).keys()) + list(hk.get("lastStep", {}).keys())
        elif isinstance(hk, list):
            candidates = list(hk)
        else:
            candidates = []
        return sorted(set(k for k in candidates if k.startswith(f"{prefix}/")))
    except Exception:
        pass
    return []


_TAU_RE = re.compile(r"tau_(\d+)p(\d+)")
_GS_RE = re.compile(r"_gs(\d+)$")


def _parse_tau_tag(tag: str) -> Optional[float]:
    """'tau_0p80' -> 0.80"""
    m = _TAU_RE.search(tag)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return None


def _snapshot_group_keys(
    keys: List[str], prefix: str
) -> Tuple[
    List[str],  # header_keys (loss, top1, bank_terms, recall, step, epoch)
    Dict[str, List[str]],  # sweep_keys[gs_suffix] -> sorted list of tau tags
    Dict[str, str],  # recall_keys[gs_suffix] -> recall key
]:
    """Classify discovered keys into header, per-gs recall, per-gs sweep."""
    header_keys: List[str] = []
    sweep_taus: Dict[str, set] = {}  # gs_suffix -> set of tau tags
    recall_keys: Dict[str, str] = {}

    header_pats = ("loss", "top1", "bank_terms", "step", "epoch")

    for k in keys:
        short = k[len(prefix) + 1:]  # strip "eval_acl6060/"

        if any(short == hp or short.startswith(hp + "_") for hp in header_pats):
            header_keys.append(k)
            continue

        gs_m = _GS_RE.search(short)
        gs_suffix = gs_m.group(0) if gs_m else ""  # "" = base bank

        if short.startswith("recall@"):
            recall_keys[gs_suffix] = k
            continue

        tau_m = _TAU_RE.search(short)
        if tau_m:
            tau_tag = tau_m.group(0)
            sweep_taus.setdefault(gs_suffix, set()).add(tau_tag)

    sweep_keys_sorted: Dict[str, List[str]] = {}
    for gs, taus in sweep_taus.items():
        sweep_keys_sorted[gs] = sorted(taus, key=lambda t: _parse_tau_tag(t) or 0)

    return header_keys, sweep_keys_sorted, recall_keys


def _fmt(v: Any, width: int = 7) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "-".rjust(width)
    if isinstance(v, float):
        if abs(v) < 0.0001 and v != 0:
            return f"{v:.2e}".rjust(width)
        return f"{v:.4f}"[:width].rjust(width)
    return str(v)[:width].rjust(width)


def _delta_str(cur: Any, prev: Any, width: int = 7) -> str:
    if (
        cur is None or prev is None
        or (isinstance(cur, float) and cur != cur)
        or (isinstance(prev, float) and prev != prev)
    ):
        return "".rjust(width)
    try:
        d = float(cur) - float(prev)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"[:width].rjust(width)
    except Exception:
        return "".rjust(width)


def _get_val(row: Dict[str, Any], key: str) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return None
    try:
        fv = float(v)
        return None if fv != fv else fv
    except Exception:
        return None


def _chunk_any_positive_recall_key(prefix: str, topk: int, tau_tag: str, gs_key: str) -> str:
    return f"{prefix}/topk{topk}_chunk_any_positive_filtered_recall@{tau_tag}{gs_key}"


def _legacy_filtered_recall_key(prefix: str, topk: int, tau_tag: str, gs_key: str) -> str:
    return f"{prefix}/topk{topk}_filtered_recall@{tau_tag}{gs_key}"


def _get_chunk_any_positive_recall(
    row: Dict[str, Any],
    prefix: str,
    topk: int,
    tau_tag: str,
    gs_key: str,
) -> Optional[float]:
    new_key = _chunk_any_positive_recall_key(prefix, topk, tau_tag, gs_key)
    val = _get_val(row, new_key)
    if val is not None:
        return val
    return _get_val(row, _legacy_filtered_recall_key(prefix, topk, tau_tag, gs_key))


def _print_sweep_table(
    row: Dict[str, Any],
    prev_row: Optional[Dict[str, Any]],
    prefix: str,
    gs_suffix: str,
    tau_tags: List[str],
    topk: int = 10,
    indent: str = "    ",
) -> None:
    """Print a threshold-sweep mini-table for one gallery size at one step."""
    hdr = f"{'tau':>5}  {'R':>7}  {'P_mic':>7}  {'P_mac':>7}  {'kept':>7}  {'noise':>7}"
    if prev_row is not None:
        hdr += f"  {'ΔR':>7}  {'Δnoise':>7}"
    print(f"{indent}{hdr}")
    print(f"{indent}{'-' * len(hdr)}")

    for tau_tag in tau_tags:
        tau_val = _parse_tau_tag(tau_tag)
        tau_label = f"{tau_val:.2f}" if tau_val else tau_tag

        gs_key = f"_{gs_suffix.lstrip('_')}" if gs_suffix else ""
        pmic_key = f"{prefix}/topk{topk}_filtered_precision_micro@{tau_tag}{gs_key}"
        pmac_key = f"{prefix}/topk{topk}_filtered_precision_macro@{tau_tag}{gs_key}"
        kept_key = f"{prefix}/topk{topk}_avg_kept_if_pass@{tau_tag}{gs_key}"
        # noise key uses different naming: noterm_noise@top10_tau_...
        noise_key = f"{prefix}/noterm_noise@top{topk}_{tau_tag}{gs_key}"

        r = _get_chunk_any_positive_recall(row, prefix, topk, tau_tag, gs_key)
        pmic = _get_val(row, pmic_key)
        pmac = _get_val(row, pmac_key)
        kept = _get_val(row, kept_key)
        noise = _get_val(row, noise_key)

        line = (
            f"{tau_label:>5}  {_fmt(r)}  {_fmt(pmic)}  {_fmt(pmac)}  "
            f"{_fmt(kept)}  {_fmt(noise)}"
        )

        if prev_row is not None:
            prev_r = _get_chunk_any_positive_recall(
                prev_row, prefix, topk, tau_tag, gs_key
            )
            prev_noise = _get_val(prev_row, noise_key)
            line += f"  {_delta_str(r, prev_r)}  {_delta_str(noise, prev_noise)}"

        print(f"{indent}{line}")


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Rich per-step eval snapshot — structured sweep tables + trends."""
    import pandas as pd

    api = wandb.Api()
    entity = _resolve_entity(args.entity)
    projects = _resolve_projects(args.project, args.all_rule_projects)
    runs = _match_runs(api, entity, projects, ids=[args.run_id])
    if not runs:
        print(f"[wandb_tool] run '{args.run_id}' not found.", file=sys.stderr)
        return 4
    run = runs[0]

    prefix = args.prefix  # e.g. "eval_acl6060"

    # --- Step 1: discover all keys with this prefix ---
    discovered = _snapshot_discover_keys(run, prefix)
    if not discovered:
        # Fallback to the full preset
        preset_name = "retriever_eval_full"
        fallback = [k for k in METRIC_PRESETS.get(preset_name, []) if k.startswith(f"{prefix}/")]
        if fallback:
            discovered = fallback
            print(f"[wandb_tool] auto-discovery empty, using preset '{preset_name}' ({len(fallback)} keys).",
                  file=sys.stderr)
        else:
            print(f"[wandb_tool] no keys found for prefix '{prefix}'.", file=sys.stderr)
            return 4

    # --- Step 2: classify keys ---
    header_keys, sweep_keys, recall_keys = _snapshot_group_keys(discovered, prefix)

    # Sort gallery sizes: "" (base) first, then numerically
    gs_order = sorted(sweep_keys.keys(), key=lambda s: (0 if s == "" else 1, int(_GS_RE.search(s).group(1)) if _GS_RE.search(s) else 0))
    if "" not in gs_order and recall_keys.get("") is not None:
        gs_order = [""] + gs_order

    # Detect topk from key names
    topk = 10
    for k in discovered:
        m = re.search(r"topk(\d+)_", k)
        if m:
            topk = int(m.group(1))
            break

    # --- Step 3: fetch history (per-key outer join, same as compare) ---
    value_keys = [k for k in discovered if not k.endswith("/step")]
    per_key_dfs: Dict[str, "pd.DataFrame"] = {}
    step_key = f"{prefix}/step"

    for k in value_keys:
        try:
            dk = run.history(keys=[k], pandas=True, samples=args.samples)
        except Exception:
            continue
        if dk is None or dk.empty or k not in dk.columns:
            continue
        if step_key in dk.columns:
            sc = step_key
        elif "_step" in dk.columns:
            sc = "_step"
        else:
            continue
        sub = dk[[sc, k]].dropna(subset=[k]).copy()
        if sc != "_step":
            sub = sub.rename(columns={sc: "_step"})
        sub["_step"] = sub["_step"].astype("Int64")
        sub = sub.dropna(subset=["_step"]).drop_duplicates(subset=["_step"])
        per_key_dfs[k] = sub

    if not per_key_dfs:
        print(f"[wandb_tool] no data returned for prefix '{prefix}'.", file=sys.stderr)
        return 4

    merged = None
    for k, sub in per_key_dfs.items():
        merged = sub if merged is None else merged.merge(sub, on="_step", how="outer")
    merged = merged.sort_values("_step").reset_index(drop=True)

    # Apply tail/step filters
    if args.last:
        merged = merged.tail(args.last).reset_index(drop=True)
    if args.from_step is not None:
        merged = merged[merged["_step"] >= args.from_step].reset_index(drop=True)
    if args.to_step is not None:
        merged = merged[merged["_step"] <= args.to_step].reset_index(drop=True)

    n_steps = len(merged)
    if n_steps == 0:
        print("[wandb_tool] no eval steps in the requested range.", file=sys.stderr)
        return 4

    # --- Step 4: JSON output ---
    if args.json:
        out_rows = []
        for i, row in merged.iterrows():
            entry: Dict[str, Any] = {"_step": int(row["_step"])}
            for k in discovered:
                if k in row:
                    v = row[k]
                    if isinstance(v, float) and v != v:
                        continue
                    entry[k] = v
            out_rows.append(entry)
        print(json.dumps({
            "run_id": run.id,
            "run_name": run.name,
            "prefix": prefix,
            "n_steps": n_steps,
            "keys_discovered": len(discovered),
            "rows": out_rows,
        }, indent=2, default=str))
        return 0

    # --- Step 5: structured text output ---
    summary = _run_summary_dict(run)
    best_step = summary.get("best/step")
    best_sec_step = summary.get("best_secondary/step")

    print(f"# snapshot: {run.id} | {run.name}")
    print(f"# prefix={prefix}  keys={len(discovered)}  eval_steps={n_steps}")
    print(f"# best/step={best_step}  best_secondary/step={best_sec_step}")
    print(f"# url: {getattr(run, 'url', '')}")
    print()

    prev_row: Optional[Dict[str, Any]] = None
    for idx, pd_row in merged.iterrows():
        row = dict(pd_row)
        step = int(row["_step"])

        markers = []
        try:
            if best_step is not None and int(best_step) == step:
                markers.append("★ best")
        except Exception:
            pass
        try:
            if best_sec_step is not None and int(best_sec_step) == step:
                markers.append("★ best_secondary")
        except Exception:
            pass
        marker_str = f"  [{', '.join(markers)}]" if markers else ""

        # Header line
        loss = _get_val(row, f"{prefix}/loss")
        top1 = _get_val(row, f"{prefix}/top1")
        bank = _get_val(row, f"{prefix}/bank_terms")

        header_parts = [f"step={step}"]
        if loss is not None:
            header_parts.append(f"loss={loss:.4f}")
        if top1 is not None:
            header_parts.append(f"top1={top1:.4f}")
        if bank is not None:
            header_parts.append(f"bank={int(bank)}")

        print(f"{'='*72}")
        print(f"  {' '.join(header_parts)}{marker_str}")
        print(f"{'='*72}")

        for gs_suffix in gs_order:
            if gs_suffix == "":
                label = "Base bank"
            else:
                gs_num = _GS_RE.search(gs_suffix)
                label = f"gs{gs_num.group(1)}" if gs_num else gs_suffix

            recall_key = recall_keys.get(gs_suffix)
            recall_val = _get_val(row, recall_key) if recall_key else None
            prev_recall = _get_val(prev_row, recall_key) if (prev_row and recall_key) else None

            recall_str = f"recall@{topk}={recall_val:.4f}" if recall_val is not None else ""
            if prev_recall is not None and recall_val is not None:
                recall_str += f" ({_delta_str(recall_val, prev_recall).strip()})"

            print(f"\n  {label}: {recall_str}")

            tau_tags = sweep_keys.get(gs_suffix, [])
            if tau_tags:
                _print_sweep_table(
                    row, prev_row if prev_row is not None else None,
                    prefix, gs_suffix, tau_tags,
                    topk=topk, indent="    ",
                )

        # Also show recall keys for gs sizes that have no sweep
        for gs_suffix, rk in sorted(recall_keys.items()):
            if gs_suffix in gs_order:
                continue
            rv = _get_val(row, rk)
            if rv is not None:
                gs_m = _GS_RE.search(gs_suffix)
                label = f"gs{gs_m.group(1)}" if gs_m else gs_suffix
                print(f"\n  {label}: recall@{topk}={rv:.4f}")

        print()
        prev_row = row

    # --- Summary: trend across all steps for key metrics ---
    if n_steps >= 2:
        print(f"{'='*72}")
        print(f"  TREND SUMMARY (first -> last of {n_steps} steps)")
        print(f"{'='*72}")
        first_row = dict(merged.iloc[0])
        last_row = dict(merged.iloc[-1])
        trend_keys = []
        for gs in gs_order:
            rk = recall_keys.get(gs)
            if rk:
                trend_keys.append(rk)
            for tau_tag in sweep_keys.get(gs, []):
                gs_key = f"_{gs.lstrip('_')}" if gs else ""
                trend_keys.append(
                    _chunk_any_positive_recall_key(prefix, topk, tau_tag, gs_key)
                )
                trend_keys.append(
                    _legacy_filtered_recall_key(prefix, topk, tau_tag, gs_key)
                )
                trend_keys.append(f"{prefix}/noterm_noise@top{topk}_{tau_tag}{gs_key}")

        for tk in trend_keys:
            first_v = _get_val(first_row, tk)
            last_v = _get_val(last_row, tk)
            if first_v is not None and last_v is not None:
                short = tk[len(prefix) + 1:]
                print(f"  {short:<55} {first_v:.4f} -> {last_v:.4f}  {_delta_str(last_v, first_v).strip()}")
        print()

    return 0


def cmd_topn(args: argparse.Namespace) -> int:
    api = wandb.Api()
    entity = _resolve_entity(args.entity)
    projects = _resolve_projects(args.project, args.all_rule_projects)
    runs = _match_runs(
        api, entity, projects,
        family=args.family,
        tag_filters=args.tag or (),
        state=args.state,
        limit=args.scan_limit,
    )
    scored: List[Tuple[Optional[float], "wandb.apis.public.Run"]] = []
    for r in runs:
        # r.summary.get() has a known bug on nested/non-dict children in WandB SDK;
        # go through the raw json dict instead.
        raw = _run_summary_dict(r)
        v = raw.get(args.metric)
        if not isinstance(v, (int, float)):
            continue
        scored.append((float(v), r))
    reverse = args.direction == "max"
    scored.sort(key=lambda t: t[0], reverse=reverse)
    topn = scored[: args.top]

    if args.json:
        out = [
            {
                "id": r.id,
                "project": r.project,
                "name": r.name,
                "state": r.state,
                "tags": list(r.tags or []),
                "metric": args.metric,
                "value": v,
                "verdict": _run_summary_dict(r).get("verdict"),
                "url": getattr(r, "url", None),
            }
            for v, r in topn
        ]
        print(json.dumps(out, indent=2, default=str))
    else:
        print(
            f"# top-{args.top} in {entity}, projects={projects}, family={args.family}, "
            f"sorted by {args.metric} ({args.direction})"
        )
        for i, (v, r) in enumerate(topn, 1):
            verdict = _run_summary_dict(r).get("verdict") or ""
            print(f"{i:>2}. {args.metric}={v:.4f}  id={r.id}  state={r.state}")
            print(f"     name: {r.name}")
            print(f"     tags: {list(r.tags or [])}")
            if verdict:
                print(f"     verdict: {verdict}")
            print(f"     url: {getattr(r, 'url', '')}")
    if not scored:
        print(
            "[wandb_tool] WARNING: no runs in the filter had a numeric value for "
            f"'{args.metric}'. Check --family/--tag or the metric name.",
            file=sys.stderr,
        )
    return 0


# --- local SQLite experiment index -----------------------------------------


def _open_experiment_db(args: argparse.Namespace) -> "ExperimentDB":
    if ExperimentDB is None:
        raise SystemExit("[wandb_tool] experiment_db.py could not be imported.")
    db_path = getattr(args, "db_path", None)
    db = ExperimentDB(db_path)
    db.init_schema()
    return db


def cmd_db_sync(args: argparse.Namespace) -> int:
    api = wandb.Api()
    entity = _resolve_entity(args.entity)
    projects = _resolve_projects(args.project, args.all_rule_projects)
    run_ids = args.runs or args.ids or ()
    runs = _match_runs(
        api, entity, projects,
        ids=run_ids,
        name_contains=args.name_contains,
        family=args.family,
        tag_filters=args.tag or (),
        config_filters=args.config or (),
        state=args.state,
        limit=args.limit,
    )
    db = _open_experiment_db(args)
    metric_keys = _expand_keys(args.keys, args.preset)
    if args.best_bundles and not metric_keys:
        metric_keys = _expand_keys(None, ["retriever_eval", "retriever_train"])

    synced: List[Dict[str, Any]] = []
    try:
        for run in runs:
            try:
                synced.append(
                    _sync_run_to_db(
                        db,
                        run,
                        best_bundles=args.best_bundles,
                        metric_keys=metric_keys,
                        samples=args.samples,
                        command="db-sync",
                    )
                )
            except Exception as exc:
                db.log_event(
                    run_id=getattr(run, "id", None),
                    project=getattr(run, "project", None),
                    source="wandb",
                    command="db-sync",
                    status="failed",
                    message=str(exc),
                )
                raise
    finally:
        db.close()

    if args.json:
        print(json.dumps({"db_path": str(args.db_path or default_db_path()), "synced": synced}, indent=2))
    else:
        print(f"[wandb_tool] db-sync wrote {len(synced)} run(s) to {args.db_path or default_db_path()}")
        for item in synced:
            counts = item.get("best_bundle_counts") or {}
            count_str = f" bundles={counts}" if counts else ""
            print(f"  - {item['project']}/{item['run_id']}{count_str}")
    return 0


def _db_run_row(row: Any, wide: bool = False) -> str:
    name = row["name"] or ""
    if not wide and len(name) > 80:
        name = name[:80] + "…"
    return (
        f"{row['run_id']:<10} {row['project']:<14} {str(row['status_tag'] or '-'): <10} "
        f"{str(row['family'] or '-'): <22} {str(row['variant_tag'] or '-'): <34} "
        f"{name}  {row['url']}"
    )


def cmd_db_find(args: argparse.Namespace) -> int:
    db = _open_experiment_db(args)
    try:
        rows = db.find_runs(
            run_ids=args.runs or (),
            family=args.family,
            project=args.db_project or args.project,
            status=args.status,
            data_tag=args.data_tag,
            variant_contains=args.variant_contains,
            name_contains=args.name_contains,
            config_filters=args.config or (),
            limit=args.limit,
        )
    finally:
        db.close()
    if args.json:
        print(json.dumps([dict(r) for r in rows], indent=2, default=str))
    else:
        if not rows:
            print("(no db runs matched)")
            return 0
        print(f"# {len(rows)} db run(s) from {args.db_path or default_db_path()}")
        print(f"{'run_id':<10} {'project':<14} {'status':<10} {'family':<22} {'variant':<34} name  url")
        for row in rows:
            print(_db_run_row(row, wide=args.wide))
    return 0


def cmd_db_show(args: argparse.Namespace) -> int:
    db = _open_experiment_db(args)
    try:
        row = db.get_run(args.run_id)
        if row is None:
            print(f"[wandb_tool] db run not found: {args.run_id}", file=sys.stderr)
            return 4
        config = db.config_for(args.run_id) if args.config else {}
        notes = db.notes_for(args.run_id) if args.notes else {}
        baselines = db.baselines_for(args.run_id)
        metrics = db.metrics_for(args.run_id, anchor=args.anchor)
    finally:
        db.close()

    if args.json:
        print(json.dumps({
            "run": dict(row),
            "baselines": baselines,
            "config": config,
            "notes_sections": notes,
            "metrics_at_best": [dict(m) for m in metrics],
        }, indent=2, default=str))
        return 0

    print(f"id        : {row['run_id']}")
    print(f"project   : {row['entity']}/{row['project']}")
    print(f"name      : {row['name']}")
    print(f"state     : {row['state']}  status:{row['status_tag']}")
    print(f"family    : {row['family']}  data:{row['data_tag']}  variant:{row['variant_tag']}")
    print(f"notes     : {row['notes_path']}")
    print(f"verdict   : {(row['summary_verdict'] or '')[:240]}")
    print(f"baselines : {baselines}")
    print(f"url       : {row['url']}")
    if args.config:
        print("config    :")
        for key in sorted(config):
            print(f"  {key} = {config[key]!r}")
    if args.notes:
        print("notes sections:")
        for section, content in notes.items():
            preview = content.replace("\n", " ")[:240]
            print(f"  ## {section}: {preview}")
    if metrics:
        print("metrics_at_best:")
        for m in metrics:
            val = m["metric_value"]
            val_s = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"  {m['anchor']}@{m['anchor_step']} {m['metric_key']} = {val_s}")
    return 0


def cmd_db_compare(args: argparse.Namespace) -> int:
    if args.refresh:
        # Refresh from WandB in the same process before reading the DB cache.
        sync_args = argparse.Namespace(**vars(args))
        sync_args.runs = list(args.run_ids)
        sync_args.ids = None
        sync_args.name_contains = None
        sync_args.family = None
        sync_args.tag = None
        sync_args.config = None
        sync_args.state = None
        sync_args.limit = len(args.run_ids)
        sync_args.best_bundles = True
        sync_args.samples = args.samples
        cmd_db_sync(sync_args)

    metric_keys = _expand_keys(args.keys, args.preset)
    if not metric_keys:
        metric_keys = _expand_keys(None, ["retriever_eval", "retriever_train"])
    metric_keys = [k for k in metric_keys if not k.endswith("/step")]
    anchors = ["primary", "secondary"] if args.anchor_metric == "both" else [args.anchor_metric]

    db = _open_experiment_db(args)
    try:
        rows = {rid: db.get_run(rid) for rid in args.run_ids}
        metrics_by_run_anchor: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for rid in args.run_ids:
            for anchor in anchors:
                metrics_by_run_anchor[(rid, anchor)] = {
                    m["metric_key"]: m for m in db.metrics_for(rid, anchor=anchor)
                }
    finally:
        db.close()

    missing_runs = [rid for rid, row in rows.items() if row is None]
    if missing_runs:
        print(f"[wandb_tool] missing db runs: {missing_runs}. Run db-sync first.", file=sys.stderr)
        return 4

    if args.json:
        payload: Dict[str, Any] = {"runs": [], "anchors": {}}
        for rid in args.run_ids:
            payload["runs"].append(dict(rows[rid]))
        for anchor in anchors:
            payload["anchors"][anchor] = {}
            for rid in args.run_ids:
                payload["anchors"][anchor][rid] = {
                    key: dict(row) for key, row in metrics_by_run_anchor[(rid, anchor)].items()
                }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(
        f"# db-compare: runs={list(args.run_ids)}  "
        f"db={args.db_path or default_db_path()}  refresh={args.refresh}"
    )
    for anchor in anchors:
        print(f"\n  ---- anchor: {anchor} ----")
        hdr = ["metric"] + list(args.run_ids)
        if len(args.run_ids) == 2:
            hdr.append(f"Δ({args.run_ids[1]}-{args.run_ids[0]})")
        print("    " + "  | ".join(hdr))
        for metric in metric_keys:
            vals: List[Optional[float]] = []
            parts = [metric]
            for rid in args.run_ids:
                row = metrics_by_run_anchor[(rid, anchor)].get(metric)
                val = row["metric_value"] if row else None
                vals.append(val)
                parts.append(f"{val:.4f}" if val is not None else "-")
            if len(args.run_ids) == 2:
                a, b = vals[0], vals[1]
                parts.append(f"{b-a:+.4f}" if a is not None and b is not None else "-")
            print("    " + "  | ".join(parts))
    return 0


def cmd_db_doctor(args: argparse.Namespace) -> int:
    db = _open_experiment_db(args)
    try:
        issues = db.doctor()
    finally:
        db.close()
    if args.json:
        print(json.dumps(issues, indent=2, default=str))
        return 0
    print(f"# db-doctor: {args.db_path or default_db_path()}")
    total = 0
    for name, values in issues.items():
        total += len(values)
        print(f"{name}: {len(values)}")
        for item in values[: args.limit]:
            print(f"  - {item}")
        if len(values) > args.limit:
            print(f"  ... {len(values) - args.limit} more")
    return 1 if args.fail_on_issues and total else 0


# --- write-side: mutate live WandB runs ------------------------------------


def _require_confirm(args: argparse.Namespace, action: str) -> None:
    if not args.yes:
        raise SystemExit(
            f"[wandb_tool] refusing to {action} without --yes (write-side guard)."
        )


def cmd_annotate(args: argparse.Namespace) -> int:
    _require_confirm(args, "annotate")
    api = wandb.Api()
    entity = _resolve_entity(args.entity)
    projects = _resolve_projects(args.project, args.all_rule_projects)
    runs = _match_runs(api, entity, projects, ids=[args.run_id])
    if not runs:
        return 4
    run = runs[0]

    # Tags: replace vs add/remove.
    current_tags = list(run.tags or [])
    new_tags = list(current_tags)
    if args.replace_tags:
        new_tags = list(args.replace_tags)
    else:
        for t in args.add_tags or ():
            if t not in new_tags:
                new_tags.append(t)
        for t in args.remove_tags or ():
            new_tags = [x for x in new_tags if x != t]
    new_tags, tag_changes = prepare_wandb_tags(new_tags)
    for old, new in tag_changes:
        print(f"[wandb_tool] compressed WandB tag: {old!r} -> {new!r}")
    if tuple(new_tags) != tuple(current_tags):
        run.tags = new_tags

    # Notes.
    if args.notes_file:
        if not os.path.isfile(args.notes_file):
            raise SystemExit(f"[wandb_tool] notes file not found: {args.notes_file}")
        with open(args.notes_file, "r", encoding="utf-8") as f:
            run.notes = f.read()
    elif args.notes is not None:
        run.notes = args.notes

    # Summary.
    if args.verdict is not None:
        run.summary["verdict"] = args.verdict
    for kv in args.set_summary or ():
        if "=" not in kv:
            raise SystemExit(f"[wandb_tool] --set-summary must be key=value (got '{kv}').")
        k, v = kv.split("=", 1)
        try:
            parsed: Any = json.loads(v)
        except Exception:
            parsed = v
        run.summary[k] = parsed

    # Config.
    cfg_updates: Dict[str, Any] = {}
    for kv in args.set_config or ():
        if "=" not in kv:
            raise SystemExit(f"[wandb_tool] --set-config must be key=value (got '{kv}').")
        k, v = kv.split("=", 1)
        try:
            parsed = json.loads(v)
        except Exception:
            # allow comma-separated list shorthand for baseline_run_ids etc.
            if "," in v:
                parsed = [x.strip() for x in v.split(",") if x.strip()]
            else:
                parsed = v
        cfg_updates[k] = parsed
    if cfg_updates:
        run.config.update(cfg_updates)

    run.update()
    print(
        f"[wandb_tool] annotated {run.entity}/{run.project}/{run.id}\n"
        f"  tags   : {list(run.tags or [])}\n"
        f"  notes  : {(run.notes or '')[:120]!r}…\n"
        f"  verdict: {_run_summary_dict(run).get('verdict')!r}\n"
        f"  cfg+   : {list(cfg_updates)}"
    )
    return 0


def cmd_flip_status(args: argparse.Namespace) -> int:
    _require_confirm(args, "flip-status")
    api = wandb.Api()
    entity = _resolve_entity(args.entity)
    projects = _resolve_projects(args.project, args.all_rule_projects)
    runs = _match_runs(api, entity, projects, ids=[args.run_id])
    if not runs:
        return 4
    run = runs[0]
    new_status = args.status
    if new_status not in ("running", "success", "failed", "deprecated", "baseline"):
        raise SystemExit(f"[wandb_tool] unknown status '{new_status}'.")
    new_tags = [t for t in (run.tags or []) if not t.startswith("status:")]
    new_tags.append(f"status:{new_status}")
    new_tags, tag_changes = prepare_wandb_tags(new_tags)
    for old, new in tag_changes:
        print(f"[wandb_tool] compressed WandB tag: {old!r} -> {new!r}")
    run.tags = new_tags
    run.update()
    print(f"[wandb_tool] {run.id} tags now: {list(run.tags or [])}")
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wandb_tool",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Global connection knobs reused across subcommands.
    p.add_argument("--entity", default=None, help="WandB entity (default: $WANDB_ENTITY or api.default_entity).")
    p.add_argument("--project", default=None, help="WandB project (default: $WANDB_PROJECT).")
    p.add_argument(
        "--all-rule-projects", action="store_true",
        help=f"Search across rule-sanctioned projects {RULE_PROJECTS}.",
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # list-projects
    sp = sub.add_parser("list-projects", help="List projects under the entity.")
    sp.set_defaults(func=cmd_list_projects)

    # find
    sp = sub.add_parser("find", help="Find runs by id / name / family / tag / config.")
    sp.add_argument("--ids", nargs="+", default=None, help="Run ids (exact).")
    sp.add_argument("--name-contains", default=None, help="Substring of display_name.")
    sp.add_argument("--family", default=None, help="Shortcut for tag=family:<x>.")
    sp.add_argument("--tag", action="append", default=None, help="Tag filter (repeatable, ANDed).")
    sp.add_argument("--config", action="append", default=None,
                    help="Config filter key=value (repeatable, ANDed).")
    sp.add_argument("--state", default=None,
                    help="finished | running | crashed | failed (exact).")
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--wide", action="store_true", help="Do not truncate run name column.")
    sp.set_defaults(func=cmd_find)

    # show
    sp = sub.add_parser("show", help="Show run metadata / config / tags / notes / summary.")
    sp.add_argument("run_id")
    sp.add_argument("--summary-prefix", default=None)
    sp.add_argument("--summary-regex", default=None)
    sp.set_defaults(func=cmd_show)

    # history
    sp = sub.add_parser("history", help="Dump run.history for selected keys.")
    sp.add_argument("run_id")
    sp.add_argument("--keys", nargs="+", default=None)
    sp.add_argument("--preset", nargs="+", default=None,
                    choices=sorted(METRIC_PRESETS.keys()),
                    help="Metric preset(s); see reference.md.")
    sp.add_argument("--samples", type=int, default=2000)
    sp.add_argument("--eval-rows-only", action="store_true",
                    help="Drop rows where eval_*/step is NaN.")
    sp.add_argument("--out", default=None, help="TSV output path.")
    sp.set_defaults(func=cmd_history)

    # snapshot
    sp = sub.add_parser(
        "snapshot",
        help="Rich per-step eval view: structured sweep tables + trends.",
    )
    sp.add_argument("run_id")
    sp.add_argument(
        "--prefix", default="eval_acl6060",
        help="Key prefix to scan (default: eval_acl6060).",
    )
    sp.add_argument("--last", type=int, default=None,
                    help="Show only the last N eval steps.")
    sp.add_argument("--from-step", type=int, default=None,
                    help="Show steps >= this value.")
    sp.add_argument("--to-step", type=int, default=None,
                    help="Show steps <= this value.")
    sp.add_argument("--samples", type=int, default=5000,
                    help="WandB history sample points (increase for long runs).")
    sp.set_defaults(func=cmd_snapshot)

    # compare
    sp = sub.add_parser("compare", help="Side-by-side delta table across N runs.")
    sp.add_argument("run_ids", nargs="+")
    sp.add_argument("--keys", nargs="+", default=None)
    sp.add_argument("--preset", nargs="+", default=None,
                    choices=sorted(METRIC_PRESETS.keys()))
    sp.add_argument("--samples", type=int, default=2000)
    sp.add_argument("--direction", choices=("max", "min"), default="max",
                    help="'max' for recall/acc/bleu, 'min' for loss/FCR.")
    sp.add_argument(
        "--at-best-step", action="store_true",
        help="Emit an AT-BEST-STEP block: all preset metrics read from history "
             "at each run's best/step (and/or best_secondary/step). Rule §F "
             "requires this for any cross-run table quoted in chat.",
    )
    sp.add_argument(
        "--anchor-metric", choices=("primary", "secondary", "both"), default="both",
        help="Anchor step source for --at-best-step: 'primary' = best/step, "
             "'secondary' = best_secondary/step, 'both' = two sub-blocks.",
    )
    sp.set_defaults(func=cmd_compare)

    # topn (pre-flight baseline discovery)
    sp = sub.add_parser("topn", help="Top-N runs per family, by summary metric (rule §A).")
    sp.add_argument("--family", required=True)
    sp.add_argument("--metric", required=True, help="Summary key to rank by.")
    sp.add_argument("--top", type=int, default=3)
    sp.add_argument("--direction", choices=("max", "min"), default="max")
    sp.add_argument("--tag", action="append", default=None, help="Extra tag filter (ANDed).")
    sp.add_argument("--state", default=None)
    sp.add_argument("--scan-limit", type=int, default=200,
                    help="How many runs to scan before ranking locally.")
    sp.set_defaults(func=cmd_topn)

    # db-sync
    sp = sub.add_parser("db-sync", help="Sync WandB runs into the local SQLite experiment index.")
    sp.add_argument("--db-path", default=None, help="SQLite path (default: EXPERIMENT_DB_PATH or documents/code/.cache/experiments.sqlite).")
    sp.add_argument("--runs", nargs="+", default=None, help="Run ids to sync (alias for --ids).")
    sp.add_argument("--ids", nargs="+", default=None, help="Run ids to sync.")
    sp.add_argument("--name-contains", default=None)
    sp.add_argument("--family", default=None)
    sp.add_argument("--tag", action="append", default=None)
    sp.add_argument("--config", action="append", default=None)
    sp.add_argument("--state", default=None)
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--best-bundles", action="store_true",
                    help="Also cache at-best-step primary/secondary metric bundles from WandB history.")
    sp.add_argument("--keys", nargs="+", default=None)
    sp.add_argument("--preset", nargs="+", default=["retriever_eval", "retriever_train"],
                    choices=sorted(METRIC_PRESETS.keys()))
    sp.add_argument("--samples", type=int, default=2000)
    sp.set_defaults(func=cmd_db_sync)

    # db-find
    sp = sub.add_parser("db-find", help="Find runs in the local SQLite experiment index.")
    sp.add_argument("--db-path", default=None)
    sp.add_argument("--runs", nargs="+", default=None)
    sp.add_argument("--db-project", default=None, help="Filter DB project without using the global WandB --project.")
    sp.add_argument("--family", default=None)
    sp.add_argument("--status", default=None, help="running | success | failed | deprecated | baseline")
    sp.add_argument("--data-tag", default=None)
    sp.add_argument("--variant-contains", default=None)
    sp.add_argument("--name-contains", default=None)
    sp.add_argument("--config", action="append", default=None,
                    help="Config filter key=value (repeatable, exact string match).")
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--wide", action="store_true")
    sp.set_defaults(func=cmd_db_find)

    # db-show
    sp = sub.add_parser("db-show", help="Show one run from the local SQLite experiment index.")
    sp.add_argument("--db-path", default=None)
    sp.add_argument("run_id")
    sp.add_argument("--config", action="store_true")
    sp.add_argument("--notes", action="store_true")
    sp.add_argument("--anchor", choices=("primary", "secondary"), default=None)
    sp.set_defaults(func=cmd_db_show)

    # db-compare
    sp = sub.add_parser("db-compare", help="Compare cached at-best metric bundles from SQLite.")
    sp.add_argument("--db-path", default=None)
    sp.add_argument("run_ids", nargs="+")
    sp.add_argument("--keys", nargs="+", default=None)
    sp.add_argument("--preset", nargs="+", default=["retriever_eval", "retriever_train"],
                    choices=sorted(METRIC_PRESETS.keys()))
    sp.add_argument("--anchor-metric", choices=("primary", "secondary", "both"), default="both")
    sp.add_argument("--refresh", action="store_true",
                    help="Refresh the requested run ids from WandB before comparing.")
    sp.add_argument("--samples", type=int, default=2000)
    sp.set_defaults(func=cmd_db_compare)

    # db-doctor
    sp = sub.add_parser("db-doctor", help="Audit the local SQLite experiment index for missing schema pieces.")
    sp.add_argument("--db-path", default=None)
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--fail-on-issues", action="store_true")
    sp.set_defaults(func=cmd_db_doctor)

    # annotate
    sp = sub.add_parser("annotate", help="Mutate tags / notes / summary / config on a run.")
    sp.add_argument("run_id")
    sp.add_argument("--add-tags", nargs="+", default=None)
    sp.add_argument("--remove-tags", nargs="+", default=None)
    sp.add_argument("--replace-tags", nargs="+", default=None,
                    help="Replace ALL tags (mutually exclusive with --add/--remove).")
    sp.add_argument("--notes", default=None)
    sp.add_argument("--notes-file", default=None, help="Path to a markdown notes file.")
    sp.add_argument("--verdict", default=None, help="Set summary.verdict.")
    sp.add_argument("--set-summary", nargs="+", default=None,
                    help="Extra summary keys key=value (JSON-parsed).")
    sp.add_argument("--set-config", nargs="+", default=None,
                    help="Config keys key=value (JSON-parsed; comma list if not JSON).")
    sp.add_argument("--yes", action="store_true", help="Required to actually mutate.")
    sp.set_defaults(func=cmd_annotate)

    # flip-status
    sp = sub.add_parser("flip-status", help="Replace status:* tag on a run.")
    sp.add_argument("run_id")
    sp.add_argument("status",
                    choices=("running", "success", "failed", "deprecated", "baseline"))
    sp.add_argument("--yes", action="store_true")
    sp.set_defaults(func=cmd_flip_status)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[wandb_tool] ERROR: {exc}", file=sys.stderr)
        if os.environ.get("WANDB_TOOL_DEBUG"):
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
