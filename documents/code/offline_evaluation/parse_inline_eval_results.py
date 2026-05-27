#!/usr/bin/env python3
"""Aggregate per-ckpt offline inline-eval logs into a single TSV.

Parses logs produced by inline_eval_retriever.sh. Each log contains one
`[EVAL_DEV] ...` and one `[EVAL_ACL6060] ...` line in the training-script
emission format (minimal-metrics: recall + per-tau sweep R / P_mic / P_mac
/ kept / noise per glossary size).

Extracted metrics per ckpt:
    - dev / acl: top1, recall@10
    - dev_gs10000 / acl_gs10000: r@10, sweep@0.80 R, sweep@0.80 P_mic, kept, noise
    - ood_gap_r10_gs10k     = acl_r10_gs10k - dev_r10_gs10k
    - ood_gap_sweep080_gs10k = acl_sweep080_R_gs10k - dev_sweep080_R_gs10k

Fail-loud: any log that is truncated (missing EVAL_DEV or EVAL_ACL6060) is
flagged in the 'status' column and its metric cells are left empty. We
never silently substitute NaN for a missing metric.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


# ---- regex helpers ---------------------------------------------------------

_GS10K_SWEEP_RE = re.compile(
    r"gs10000_sweep@0\.80:\s+R=(?P<R>[0-9.]+)\s+"
    r"P_mic=(?P<P_mic>[0-9.]+)\s+"
    r"P_mac=(?P<P_mac>[0-9.]+)\s+"
    r"kept=(?P<kept>[0-9.]+)\s+"
    r"noise=(?P<noise>[0-9.]+)"
)
_BASE_SWEEP_RE = re.compile(
    # Word boundary rejects 'gs1000_sweep' / 'gs10000_sweep' because '_' is
    # a word char; only the base-bank 'sweep@0.80' (preceded by space) has
    # a boundary here.
    r"\bsweep@0\.80:\s+R=(?P<R>[0-9.]+)\s+"
    r"P_mic=(?P<P_mic>[0-9.]+)\s+"
    r"P_mac=(?P<P_mac>[0-9.]+)\s+"
    r"kept=(?P<kept>[0-9.]+)\s+"
    r"noise=(?P<noise>[0-9.]+)"
)
_R10_GS10K_RE = re.compile(r"gs10000\(bank=\d+\):\s+r@10=(?P<r>[0-9.]+)")
_TOP1_RE = re.compile(r"\btop1=(?P<v>[0-9.]+)")
_R10_RE = re.compile(r"\brecall@10=(?P<v>[0-9.]+)")
_STEP_RE = re.compile(r"step=(\d+)\s+epoch=(\d+)")
_RESUME_RE = re.compile(r"\[RESUME\]\s+(\S+\.pt)\s+epoch=(\d+)\s+step=(\d+)")
_BEST_METRIC_RE = re.compile(r"Restored best_metric_value=([0-9.]+)")
_BEST_METRIC_SEC_RE = re.compile(r"Restored best_metric_secondary_value=([0-9.]+)")
_TID_RE = re.compile(r"\[TERM_ID_NORMALIZE\] mode=(\S+)")


def _parse_eval_line(line: str) -> Dict[str, Optional[float]]:
    """Extract metrics from a single EVAL_DEV / EVAL_ACL6060 log line."""
    result: Dict[str, Optional[float]] = {
        "top1": None,
        "r10": None,
        "r10_gs10k": None,
        "sweep080_R": None,
        "sweep080_P_mic": None,
        "sweep080_kept": None,
        "sweep080_noise": None,
        "sweep080_R_gs10k": None,
        "sweep080_P_mic_gs10k": None,
        "sweep080_kept_gs10k": None,
        "sweep080_noise_gs10k": None,
    }
    m = _TOP1_RE.search(line)
    if m:
        result["top1"] = float(m.group("v"))
    # Multiple recall@10 occurrences (primary + extra emit the same format);
    # take the FIRST match which corresponds to the primary (base) bank.
    m = _R10_RE.search(line)
    if m:
        result["r10"] = float(m.group("v"))
    m = _R10_GS10K_RE.search(line)
    if m:
        result["r10_gs10k"] = float(m.group("r"))
    m = _BASE_SWEEP_RE.search(line)
    if m:
        result["sweep080_R"] = float(m.group("R"))
        result["sweep080_P_mic"] = float(m.group("P_mic"))
        result["sweep080_kept"] = float(m.group("kept"))
        result["sweep080_noise"] = float(m.group("noise"))
    m = _GS10K_SWEEP_RE.search(line)
    if m:
        result["sweep080_R_gs10k"] = float(m.group("R"))
        result["sweep080_P_mic_gs10k"] = float(m.group("P_mic"))
        result["sweep080_kept_gs10k"] = float(m.group("kept"))
        result["sweep080_noise_gs10k"] = float(m.group("noise"))
    return result


def _parse_log(path: Path) -> Tuple[Dict[str, object], str]:
    """Parse one log file; return (flat-record, status).

    status == 'ok' when both EVAL_DEV and EVAL_ACL6060 were parsed
    successfully AND every numeric field is populated. Otherwise returns a
    descriptive failure string.
    """
    text = path.read_text(errors="replace")
    record: Dict[str, object] = {
        "tag": path.stem,
        "ckpt": None,
        "term_id_normalize": None,
        "step": None,
        "epoch": None,
        "best_metric_value": None,
        "best_metric_secondary_value": None,
    }
    # Per-ckpt metadata from the [RESUME] + [TERM_ID_NORMALIZE] lines.
    m = _RESUME_RE.search(text)
    if m:
        record["ckpt"] = m.group(1)
        record["epoch"] = int(m.group(2))
        record["step"] = int(m.group(3))
    m = _TID_RE.search(text)
    if m:
        record["term_id_normalize"] = m.group(1)
    m = _BEST_METRIC_RE.search(text)
    if m:
        record["best_metric_value"] = float(m.group(1))
    m = _BEST_METRIC_SEC_RE.search(text)
    if m:
        record["best_metric_secondary_value"] = float(m.group(1))

    dev_line = None
    acl_line = None
    for line in text.splitlines():
        if "[EVAL_DEV]" in line and "sweep@" in line:
            dev_line = line
        elif "[EVAL_ACL6060]" in line and "sweep@" in line:
            acl_line = line
    if dev_line is None or acl_line is None:
        return record, (
            f"missing_eval_line dev={'ok' if dev_line else 'missing'} "
            f"acl={'ok' if acl_line else 'missing'}"
        )

    dev = _parse_eval_line(dev_line)
    acl = _parse_eval_line(acl_line)
    for k, v in dev.items():
        record[f"dev_{k}"] = v
    for k, v in acl.items():
        record[f"acl_{k}"] = v

    # Derived OOD gaps: acl - dev. If either side is missing, leave None.
    def _gap(a_key: str, d_key: str) -> Optional[float]:
        a = record.get(a_key)
        d = record.get(d_key)
        if isinstance(a, float) and isinstance(d, float):
            return round(a - d, 4)
        return None

    record["ood_gap_r10"] = _gap("acl_r10", "dev_r10")
    record["ood_gap_r10_gs10k"] = _gap("acl_r10_gs10k", "dev_r10_gs10k")
    record["ood_gap_sweep080_R"] = _gap("acl_sweep080_R", "dev_sweep080_R")
    record["ood_gap_sweep080_R_gs10k"] = _gap(
        "acl_sweep080_R_gs10k", "dev_sweep080_R_gs10k"
    )

    missing = [
        k
        for k in (
            "dev_r10",
            "dev_r10_gs10k",
            "dev_sweep080_R_gs10k",
            "acl_r10",
            "acl_r10_gs10k",
            "acl_sweep080_R_gs10k",
        )
        if record.get(k) is None
    ]
    if missing:
        return record, "partial_parse missing=" + ",".join(missing)
    return record, "ok"


# ---- TSV output ------------------------------------------------------------

_TSV_COLUMNS = [
    "tag",
    "status",
    "term_id_normalize",
    "step",
    "epoch",
    "best_metric_value",  # training-time acl6060/recall@10_gs1000 at save
    "best_metric_secondary_value",  # training-time acl6060/recall@10_gs10000
    # Primary comparison fields (user's key metric):
    "dev_r10_gs10k",
    "acl_r10_gs10k",
    "ood_gap_r10_gs10k",
    "dev_sweep080_R_gs10k",
    "acl_sweep080_R_gs10k",
    "ood_gap_sweep080_R_gs10k",
    # Secondary context:
    "dev_r10",
    "acl_r10",
    "ood_gap_r10",
    "dev_sweep080_R",
    "acl_sweep080_R",
    "ood_gap_sweep080_R",
    "dev_top1",
    "acl_top1",
    "dev_sweep080_kept_gs10k",
    "acl_sweep080_kept_gs10k",
    "dev_sweep080_noise_gs10k",
    "acl_sweep080_noise_gs10k",
    "ckpt",
]


def _fmt_cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", required=True, type=Path)
    parser.add_argument("--out_tsv", required=True, type=Path)
    parser.add_argument(
        "--glob", default="*.log",
        help="Glob pattern under --log_dir (default '*.log').",
    )
    args = parser.parse_args()

    assert args.log_dir.is_dir(), f"--log_dir not a directory: {args.log_dir}"
    log_paths = sorted(args.log_dir.glob(args.glob))
    assert log_paths, (
        f"No log files match {args.log_dir}/{args.glob}"
    )

    rows = []
    for p in log_paths:
        record, status = _parse_log(p)
        record["status"] = status
        rows.append(record)
        status_tag = "OK " if status == "ok" else "!! "
        print(
            f"[PARSE] {status_tag}{p.name:50s}  "
            f"dev_r10_gs10k={_fmt_cell(record.get('dev_r10_gs10k'))}  "
            f"acl_r10_gs10k={_fmt_cell(record.get('acl_r10_gs10k'))}  "
            f"acl_sweep080_R_gs10k={_fmt_cell(record.get('acl_sweep080_R_gs10k'))}  "
            f"ood_gap_r10_gs10k={_fmt_cell(record.get('ood_gap_r10_gs10k'))}  "
            f"status={status}"
        )

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(_TSV_COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(_fmt_cell(r.get(c)) for c in _TSV_COLUMNS) + "\n")

    print(f"[DONE] wrote {len(rows)} rows -> {args.out_tsv}")


if __name__ == "__main__":
    main()
