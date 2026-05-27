#!/usr/bin/env python3
"""Wait for a newly-launched WandB run matching a variant tag, then exec
scripts/tools/finalize_wandb_run.py with the resolved run_id.

Blocks until the matching run reaches a finished state."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import wandb


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="luojiaxuan1215-johns-hopkins-university")
    p.add_argument("--project", default="qwen3_rag")
    p.add_argument("--variant_tag", required=True, help="e.g. variant:hnps_k512_...")
    p.add_argument("--created_after_ts", type=float, required=True,
                   help="Unix timestamp; only runs created after this are considered")
    p.add_argument("--poll_interval_sec", type=int, default=60)
    p.add_argument("--discover_timeout_sec", type=int, default=1800,
                   help="Abort if no matching run appears within this window")
    p.add_argument("--finalize_cmd", required=True, nargs="+",
                   help="Prefix command; run_id will be appended via --run_id")
    return p.parse_args()


def _discover_run(api: wandb.Api, path_prefix: str, variant_tag: str,
                  created_after_ts: float) -> str | None:
    runs = list(api.runs(path_prefix,
                         filters={"tags": {"$in": [variant_tag]}},
                         per_page=20, order="-created_at"))
    for r in runs:
        try:
            ts = r.created_at
        except Exception:
            continue
        import datetime as dt
        if isinstance(ts, str):
            try:
                ts_dt = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                ts_dt = None
        else:
            ts_dt = None
        if ts_dt is not None and ts_dt.timestamp() < created_after_ts:
            continue
        return r.id
    return None


def main() -> int:
    args = _parse()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be set in the environment")
    api = wandb.Api()
    path_prefix = f"{args.entity}/{args.project}"

    t0 = time.time()
    run_id = None
    print(f"[watcher] discovering run with tag={args.variant_tag} "
          f"created_after={args.created_after_ts}", flush=True)
    while run_id is None:
        try:
            run_id = _discover_run(api, path_prefix, args.variant_tag, args.created_after_ts)
        except Exception as exc:
            print(f"[watcher] discovery error: {exc}", flush=True)
        if run_id is not None:
            print(f"[watcher] found run_id={run_id}", flush=True)
            break
        if time.time() - t0 > args.discover_timeout_sec:
            print(f"[watcher] timeout waiting for run tag={args.variant_tag}", flush=True)
            return 3
        time.sleep(args.poll_interval_sec)

    cmd = list(args.finalize_cmd) + ["--run_id", run_id]
    print(f"[watcher] exec: {cmd}", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
