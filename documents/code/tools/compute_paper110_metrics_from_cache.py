#!/usr/bin/env python3
"""
Compute paper-2022.acl-long.110 BLEU + StreamLAAL + TERM_ACC + TCR + TERM_FCR
from an EXISTING cached per-paper-combined directory (which currently holds
5-paper aggregate metrics in eval_results_by_paper.tsv/log, but paper-110's
single instance is preserved line-by-line in per_paper_combined/instances.log).

We:
  1. Extract paper-110's single instance from the combined instances.log.
  2. Invoke offline_streamlaal_eval.py (mode=extracted_by_paper, but with
     only paper-110 selected) on that subset so the StreamLAAL/BLEU/TERM
     numbers are computed for paper-110 ONLY.
  3. Write an output eval_results_by_paper.tsv + .log in a new paper110_only
     subdirectory.

Assertions abound — no silent fallbacks per repo rules.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

# ======Configuration=====
PAPER_ID = "2022.acl-long.110"
PAPER_WAV_BASENAME = f"{PAPER_ID}.wav"
OFFLINE_EVAL_SCRIPT = (
    "/home/jiaxuanluo/InfiniSST/documents/code/offline_sst_eval/"
    "offline_streamlaal_eval.py"
)
GLOSSARY_MANIFEST = (
    "/home/jiaxuanluo/InfiniSST/documents/data/data_pre/"
    "extracted_glossary_by_paper_manifest.json"
)
ACL6060_DEV_YAML = "/mnt/data/siqiouyang/datasets/acl6060/dev.yaml"
ACL6060_DEV_REF = (
    "/mnt/data/siqiouyang/datasets/acl6060/dev/text/txt/"
    "ACL.6060.dev.en-xx.zh.txt"
)
DEFAULT_LATENCY_UNIT = "char"
# ======Configuration=====


def extract_paper110_line(combined_log: Path, out_log: Path) -> int:
    """Copy only the paper-110 line from combined instances.log into out_log.
    Returns the new local index (0) of that line.
    Fails loudly if paper-110 is not found."""
    assert combined_log.is_file(), f"Missing combined instances log: {combined_log}"
    matched = None
    for line in combined_log.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        src = obj.get("source", "")
        if isinstance(src, list):
            src = src[0] if src else ""
        if PAPER_WAV_BASENAME in str(src):
            assert matched is None, (
                f"Multiple paper-110 rows in {combined_log}; unexpected"
            )
            matched = obj
    assert matched is not None, (
        f"No paper-110 row found in {combined_log} (searched for "
        f"{PAPER_WAV_BASENAME!r})"
    )
    matched["index"] = 0
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text(json.dumps(matched, ensure_ascii=False) + "\n")
    return 0


def subset_yaml_and_ref_for_paper110(out_dir: Path) -> tuple[Path, Path]:
    """Write per-paper subsetted dev.yaml and ref.txt keeping only paper-110
    utterance entries (and their corresponding reference lines)."""
    import yaml
    out_dir.mkdir(parents=True, exist_ok=True)
    full_yaml = yaml.safe_load(Path(ACL6060_DEV_YAML).read_text())
    full_refs = Path(ACL6060_DEV_REF).read_text().splitlines()
    assert isinstance(full_yaml, list), f"Invalid dev.yaml: {ACL6060_DEV_YAML}"
    assert len(full_yaml) == len(full_refs), (
        f"Length mismatch dev.yaml={len(full_yaml)} vs ref={len(full_refs)}"
    )
    sub_yaml: list = []
    sub_refs: list[str] = []
    for entry, ref in zip(full_yaml, full_refs):
        if entry.get("wav") == PAPER_WAV_BASENAME:
            sub_yaml.append(entry)
            sub_refs.append(ref)
    assert sub_yaml, f"No {PAPER_WAV_BASENAME} entries found in {ACL6060_DEV_YAML}"
    yaml_out = out_dir / f"dev_paper110.yaml"
    ref_out = out_dir / f"dev_paper110.zh.txt"
    yaml_out.write_text(yaml.safe_dump(sub_yaml, allow_unicode=True))
    ref_out.write_text("\n".join(sub_refs) + "\n")
    print(f"[paper110] subsetted dev.yaml -> {yaml_out} ({len(sub_yaml)} lines)")
    return yaml_out, ref_out


def run_offline_eval(instances_log: Path, output_dir: Path,
                     latency_unit: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = output_dir / "eval_results_by_paper.tsv"
    out_log = output_dir / "eval_results_by_paper.log"
    sub_yaml, sub_ref = subset_yaml_and_ref_for_paper110(output_dir / "_subsetted")
    cmd = [
        sys.executable, OFFLINE_EVAL_SCRIPT,
        "--instances-log", str(instances_log),
        "--audio-yaml", str(sub_yaml),
        "--ref-file", str(sub_ref),
        "--mode", "extracted_by_paper",
        "--extracted-glossary-manifest", GLOSSARY_MANIFEST,
        "--lang-code", "zh",
        "--latency-unit", latency_unit,
        "--output-tsv", str(out_tsv),
        "--output-log", str(out_log),
        "--work-dir", str(output_dir / "_work"),
    ]
    print(f"[paper110] + {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    rc = subprocess.call(cmd)
    assert rc == 0, f"offline_streamlaal_eval.py failed rc={rc}"
    tsv_path = out_tsv
    log_path = out_log
    assert tsv_path.is_file(), f"Missing output tsv: {tsv_path}"
    rows = tsv_path.read_text().strip().splitlines()
    assert len(rows) >= 2, f"Empty tsv: {tsv_path}"
    header = rows[0].split("\t")
    values = rows[-1].split("\t")
    pair = dict(zip(header, values))
    return {"tsv": str(tsv_path), "log": str(log_path), "summary": pair}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-dir", required=True,
                    help="Existing per_paper_combined directory")
    ap.add_argument("--output-dir", required=True,
                    help="Where to write paper110-only artifacts")
    ap.add_argument("--latency-unit", default=DEFAULT_LATENCY_UNIT,
                    choices=["spm", "word", "char"])
    args = ap.parse_args()

    combined_dir = Path(args.combined_dir)
    out_dir = Path(args.output_dir)
    combined_log = combined_dir / "instances.log"
    paper110_log = out_dir / "instances.log"

    extract_paper110_line(combined_log, paper110_log)
    result = run_offline_eval(paper110_log, out_dir, args.latency_unit)

    summary = result["summary"]
    for k in ("BLEU", "StreamLAAL", "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL",
              "TCR", "TERM_FCR"):
        assert k in summary, f"Missing column {k} in tsv header"
    print(f"[paper110] OK  {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
