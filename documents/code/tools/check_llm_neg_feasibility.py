#!/usr/bin/env python3
"""
Phase 1 feasibility check for LLM-negative regeneration.

Because the cached shards
`train_s_zh_with_candidates_v4_llm_negative_test_gpu{0,1,2}.jsonl` carry
no `utter_id` field (confirmed empirically) and only total ~300 rows
(vs the 12499 rows in the upstream
`train_cleaned_with_retriever_results_varlen.jsonl`), the cache is
USELESS for a training-scale substitution.  This script's role is reduced
to producing a structured record of that finding + (optionally) timing a
100-row regen probe with Qwen3-30B-FP8 on a single GPU.  The orchestrator
reads the output markdown to choose between Path A (cache reuse), Path B
(regen in background), or Path C (abandon neg-source experiment in 8h).

No silent fallbacks: every required path is asserted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ======Configuration=====
UPSTREAM_JSONL = "/mnt/gemini/data1/jiaxuanluo/train_cleaned_with_retriever_results_varlen.jsonl"
CACHE_SHARDS = [
    "/mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_v4_llm_negative_test_gpu0.jsonl",
    "/mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_v4_llm_negative_test_gpu1.jsonl",
    "/mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates_v4_llm_negative_test_gpu2.jsonl",
]
# Below this cache-coverage fraction we cannot reuse the cache for training.
CACHE_COVERAGE_PATH_A = 0.95
# Below this extrapolated full-regen runtime we go Path B (regen in background).
PATH_B_MAX_HOURS = 3.0
# ======Configuration=====


def load_upstream_keys():
    """Return a set of identifying keys for upstream rows.

    Prefer utter_id; fall back to audios[0] (a stable absolute wav path).
    Raise LOUDLY if neither is present.
    """
    assert os.path.isfile(UPSTREAM_JSONL), f"upstream JSONL missing: {UPSTREAM_JSONL}"
    keys = set()
    n_rows = 0
    with open(UPSTREAM_JSONL) as f:
        for line in f:
            r = json.loads(line)
            k = r.get("utter_id") or (r.get("audios") or [None])[0]
            assert k, "upstream row has no utter_id nor audios[0]"
            keys.add(k)
            n_rows += 1
    return keys, n_rows


def load_cache_keys():
    """Return a set of identifying keys for cached LLM-neg rows.

    Cached shards were built without utter_id, so we fall back to audios[0].
    """
    keys = set()
    details = []
    for p in CACHE_SHARDS:
        if not os.path.isfile(p):
            details.append({"path": p, "rows": 0, "missing": True})
            continue
        n = 0
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                uid = r.get("utter_id")
                if uid:
                    keys.add(uid)
                else:
                    a = (r.get("audios") or [None])[0]
                    if a:
                        keys.add(a)
                n += 1
        details.append({"path": p, "rows": n, "has_utter_id": False if n and "utter_id" not in r else True})
    return keys, details


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--probe-json", default="",
                    help="Optional path to probe output. If missing, probe section is marked 'not run'.")
    args = ap.parse_args()

    upstream_keys, n_upstream = load_upstream_keys()
    cache_keys, cache_details = load_cache_keys()

    overlap = upstream_keys & cache_keys
    coverage = len(overlap) / max(1, len(upstream_keys))

    probe_report = None
    if args.probe_json and os.path.isfile(args.probe_json):
        with open(args.probe_json) as f:
            probe_report = json.load(f)

    if coverage >= CACHE_COVERAGE_PATH_A:
        decided = "A"
        reason = f"cache covers {coverage*100:.1f}% of upstream rows"
    elif probe_report is not None and probe_report.get("extrapolated_full_hours", 1e9) <= PATH_B_MAX_HOURS:
        decided = "B"
        reason = (
            f"cache coverage only {coverage*100:.1f}% but extrapolated full "
            f"regen {probe_report['extrapolated_full_hours']:.2f}h <= {PATH_B_MAX_HOURS}h"
        )
    else:
        decided = "C"
        reason = (
            f"cache coverage only {coverage*100:.1f}%; "
            + (f"probe extrapolated {probe_report['extrapolated_full_hours']:.2f}h > {PATH_B_MAX_HOURS}h"
               if probe_report is not None else "no probe timing available")
        )

    summary = {
        "upstream_rows": n_upstream,
        "cache_shards": cache_details,
        "cache_total_unique_keys": len(cache_keys),
        "overlap_with_upstream": len(overlap),
        "coverage_fraction": coverage,
        "coverage_path_a_threshold": CACHE_COVERAGE_PATH_A,
        "path_b_max_hours": PATH_B_MAX_HOURS,
        "probe_report": probe_report,
        "decided_path": decided,
        "decision_reason": reason,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write("# LLM-negative regen feasibility\n\n")
        f.write(f"- upstream rows: **{n_upstream}**\n")
        f.write(f"- cache total unique keys (utter_id OR audios[0]): **{len(cache_keys)}**\n")
        f.write(f"- overlap: **{len(overlap)}** ({coverage*100:.2f}%)\n")
        f.write(f"- coverage threshold for Path A: {CACHE_COVERAGE_PATH_A*100:.0f}%\n")
        f.write(f"- path B max hours: {PATH_B_MAX_HOURS}\n\n")

        f.write("## Cache shards\n")
        for d in cache_details:
            missing = " (MISSING)" if d.get("missing") else ""
            f.write(f"- rows={d.get('rows')} path={d['path']}{missing}\n")
        f.write("\n")

        if probe_report is not None:
            f.write("## Probe result (100-row regen timing)\n")
            for k, v in probe_report.items():
                f.write(f"- {k}: {v}\n")
        else:
            f.write("## Probe result\n- not run (insufficient GPU time or skipped)\n")

        f.write("\n## Decision\n")
        f.write(f"- **Path {decided}**: {reason}\n")
        if decided == "C":
            f.write("- Action: abandon LLM-neg substitution in 8h window; T2 falls back to rank=64 ablation only.\n")
        elif decided == "B":
            f.write("- Action: launch regen in background; T2 starts as rank=64; if regen finishes before T2 done, kick a second T2' with LLM-neg after T2.\n")
        else:
            f.write("- Action: build LLM-neg alt training JSONL from cache; T2 = r=32 with LLM-negs.\n")

    print(f"[phase1] decision=Path{decided} reason={reason}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
