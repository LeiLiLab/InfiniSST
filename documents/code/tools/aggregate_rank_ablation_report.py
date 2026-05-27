#!/usr/bin/env python3
"""
Build the final REPORT for the 8h audit + rank + (optional) neg-source run.

Collects:
  - audit summary (both d5 no-cap and d5_cap if available)
  - phase1 feasibility
  - per-model eval_results_by_paper.tsv rows for paper 2022.acl-long.110

and produces one human-readable REPORT.md plus a structured summary.json.

No silent fallbacks: if a model's tsv is missing, it is listed as
`(missing)` in the table with a warning in the preamble.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ======Configuration=====
# Columns we copy verbatim from eval_results_by_paper.tsv, keyed by the
# canonical header produced by offline_streamlaal_eval.py:
#   mode, lang_code, BLEU, StreamLAAL, StreamLAAL_CA, TERM_ACC,
#   TERM_CORRECT, TERM_TOTAL, TCR, TCR_ADOPTED, TCR_TOTAL,
#   TERM_FCR, FALSE_COPY, NEG_TOTAL, instances_log
REPORT_COLS = [
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
]
# ======Configuration=====


def read_tsv_last_row(tsv_path: str) -> Optional[List[str]]:
    if not os.path.isfile(tsv_path):
        return None
    with open(tsv_path) as f:
        rows = [r for r in f.read().strip().splitlines() if r]
    if len(rows) < 2:
        return None
    return rows[-1].split("\t")


def extract_paper110_row(combined_dir: str, tag: str) -> Dict:
    """Read a single-row eval_results_by_paper.tsv that already reflects
    paper-110-only results (either because eval was run with
    RUN_PAPERS_OVERRIDE=2022.acl-long.110 or because the baseline cache was
    post-processed by compute_paper110_metrics_from_cache.py).

    Fails loudly if the TSV is missing or malformed rather than silently
    degrading to defaults.
    """
    tsv = os.path.join(combined_dir, "eval_results_by_paper.tsv")
    out: Dict = {"tag": tag, "combined_dir": combined_dir, "tsv": tsv}
    if not os.path.isfile(tsv):
        out["status"] = "missing_tsv"
        return out

    with open(tsv) as f:
        rows = [r for r in f.read().strip().splitlines() if r]
    if len(rows) < 2:
        out["status"] = "empty_tsv"
        return out
    header = rows[0].split("\t")
    values = rows[-1].split("\t")
    pair = dict(zip(header, values))

    missing = [c for c in REPORT_COLS if c not in pair]
    if missing:
        out["status"] = f"missing_cols:{','.join(missing)}"
        return out

    out["status"] = "ok"
    for c in REPORT_COLS:
        out[c] = pair.get(c, "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-nocap-json", default="")
    ap.add_argument("--audit-cap-json", default="")
    ap.add_argument("--phase1-json", default="")
    # Model entries as "label:combined_dir".  Order decides table order.
    ap.add_argument("--model-entry", action="append", default=[],
                    help="label:combined_dir")
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    audit_nocap = json.load(open(args.audit_nocap_json)) if args.audit_nocap_json and os.path.isfile(args.audit_nocap_json) else None
    audit_cap = json.load(open(args.audit_cap_json)) if args.audit_cap_json and os.path.isfile(args.audit_cap_json) else None
    phase1 = json.load(open(args.phase1_json)) if args.phase1_json and os.path.isfile(args.phase1_json) else None

    model_rows: List[Dict] = []
    for entry in args.model_entry:
        assert ":" in entry, f"Bad --model-entry (missing ':'): {entry}"
        label, combined_dir = entry.split(":", 1)
        r = extract_paper110_row(combined_dir, label)
        model_rows.append(r)

    summary = {
        "audit_nocap": audit_nocap,
        "audit_cap": audit_cap,
        "phase1": phase1,
        "model_rows": model_rows,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write("# 8h audit + rank ablation (+ optional LLM-neg) report\n\n")
        f.write("Single-paper eval focus: `2022.acl-long.110`, lm=1, per-paper extracted glossary.\n\n")

        f.write("## Phase 0 — Data audit\n\n")
        for label, aud in [("d5 no-cap", audit_nocap), ("d5 cap", audit_cap)]:
            if aud is None:
                f.write(f"### {label}\n- (audit not available)\n\n")
                continue
            f.write(f"### {label}\n")
            f.write(f"- input: `{aud['input_jsonl']}`\n")
            f.write(f"- rows_total: {aud['rows_total']}\n")
            f.write(f"- rows_with_blocking_error: {aud['rows_with_blocking_error']} "
                    f"({aud['blocking_rate']*100:.2f}%)\n")
            f.write(f"- gated (blocking gate {aud['gate_threshold']*100:.2f}%): **{aud['gated']}**\n")
            ec = aud.get("error_categories_by_severity", {})
            for sev, lst in ec.items():
                if lst:
                    f.write(f"- {sev}: " + ", ".join([f"`{c}` x {n}" for c, n in lst]) + "\n")
            f.write("\n")

        f.write("## Phase 1 — LLM-neg feasibility\n\n")
        if phase1 is None:
            f.write("- (not run)\n\n")
        else:
            f.write(f"- upstream rows: {phase1.get('upstream_rows')}\n")
            f.write(f"- cache coverage: {phase1.get('coverage_fraction', 0)*100:.2f}%\n")
            f.write(f"- decided path: **Path {phase1.get('decided_path')}**\n")
            f.write(f"- reason: {phase1.get('decision_reason')}\n\n")

        f.write("## Paper-110 results (lm=1, per-paper extracted glossary)\n\n")
        f.write("| Model | status | BLEU | TERM_ACC | TERM n | TCR | TERM_FCR | StreamLAAL | StreamLAAL_CA |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in model_rows:
            if r.get("status") != "ok":
                f.write(f"| {r['tag']} | {r.get('status')} | — | — | — | — | — | — | — |\n")
                continue
            try:
                bleu_s = f"{float(r['BLEU']):.2f}"
            except ValueError:
                bleu_s = r.get("BLEU", "")
            try:
                term_acc_s = f"{float(r['TERM_ACC']) * 100:.2f}%"
            except ValueError:
                term_acc_s = r.get("TERM_ACC", "")
            try:
                slaal_s = f"{float(r['StreamLAAL']):.0f}"
                slaal_ca_s = f"{float(r['StreamLAAL_CA']):.0f}"
            except ValueError:
                slaal_s = r.get("StreamLAAL", "")
                slaal_ca_s = r.get("StreamLAAL_CA", "")
            term_n = f"{r.get('TERM_CORRECT','?')}/{r.get('TERM_TOTAL','?')}"
            f.write(
                f"| {r['tag']} | ok | {bleu_s} | {term_acc_s} | {term_n} | "
                f"{r.get('TCR','')} | {r.get('TERM_FCR','')} | "
                f"{slaal_s} | {slaal_ca_s} |\n"
            )

        f.write("\n## Model pointers\n\n")
        for r in model_rows:
            f.write(f"- {r['tag']}: {r['combined_dir']}\n")

        f.write("\n## Verdicts (to fill in after numbers land)\n\n")
        f.write("- Does rank matter (r=16 vs r=32 vs r=64)? ...\n")
        f.write("- Does neg-source matter (retriever top-K vs LLM-neg)? ...\n")
        f.write("- How does the new stack compare to the old SLM baseline? ...\n")

    print(f"[report] wrote {args.output_md}", flush=True)
    print(f"[report] wrote {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
