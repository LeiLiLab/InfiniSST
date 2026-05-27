#!/usr/bin/env python3

"""
Validate the StreamLAAL/TERM summary TSV for the k1=10, k2 sweep, 2-glossary setup.

This validator checks:
- Expected coverage count (2 glossaries * 4 latency multipliers * 4 K2 values = 32).
- Each TSV row points to an existing output directory.
- TSV meta columns match what can be parsed from the output directory name.
- TSV metric columns match what can be parsed from that directory's post_eval.log.

All user-facing strings are in English.
"""

from __future__ import annotations

# ======Configuration=====
SUMMARY_TSV = "/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2/zh/k1_10_k2_sweep_glossary2_streamlaal_summary.tsv"

POST_EVAL_LOG_NAME = "post_eval.log"

EXPECTED_GLOSSARY_TAGS = (
    "glossary_acl6060",
    "extracted_glossary_with_translations",
)

EXPECTED_LATENCY_MULTIPLIERS = (1, 2, 3, 4)
EXPECTED_K2_VALUES = (5, 10, 15, 20)
EXPECTED_K1 = 10

FLOAT_TOLERANCE = 1e-9
FAIL_FAST = False
# ======Configuration=====

import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DIR_RE = re.compile(
    r"_g(?P<g>.+?)_cs(?P<cs>[0-9.]+)_hs(?P<hs>[0-9.]+)_lm(?P<lm>[0-9]+)_k2(?P<k2>[0-9]+)_k1(?P<k1>[0-9]+)$"
)


def _as_int(x: str) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _as_float(x: str) -> Optional[float]:
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _float_eq(a: Optional[float], b: Optional[float], tol: float) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


@dataclass(frozen=True)
class DirMeta:
    glossary_tag: str
    vllm_segment_sec: str
    hop_size: str
    latency_multiplier: int
    k2: int
    k1: int


@dataclass(frozen=True)
class PostEvalMetrics:
    bleu: float
    stream_laal: float
    stream_laal_ca: float
    term_acc: float
    term_correct: int
    term_total: int


def parse_meta_from_dir_name(dir_name: str) -> Optional[DirMeta]:
    m = DIR_RE.search(dir_name)
    if not m:
        return None
    gd = m.groupdict()
    lm = _as_int(gd["lm"])
    k2 = _as_int(gd["k2"])
    k1 = _as_int(gd["k1"])
    if lm is None or k2 is None or k1 is None:
        return None
    return DirMeta(
        glossary_tag=gd["g"],
        vllm_segment_sec=gd["cs"],
        hop_size=gd["hs"],
        latency_multiplier=lm,
        k2=k2,
        k1=k1,
    )


def parse_metrics_from_post_eval_log(path: Path) -> Optional[PostEvalMetrics]:
    if not path.is_file():
        return None
    txt = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Find the first line after the "BLEU\tStreamLAAL\tStreamLAAL_CA" header.
    bleu = stream_laal = stream_laal_ca = None
    for i, line in enumerate(txt):
        if line.strip() == "BLEU\tStreamLAAL\tStreamLAAL_CA" and i + 1 < len(txt):
            parts = txt[i + 1].split("\t")
            if len(parts) >= 3:
                bleu = _as_float(parts[0])
                stream_laal = _as_float(parts[1])
                stream_laal_ca = _as_float(parts[2])
            break

    term_acc = term_correct = term_total = None
    for line in txt:
        if line.startswith("TERM_ACC"):
            # Format: TERM_ACC\t0.8162\tCORRECT_TERMS\t1239\tTOTAL_TERMS\t1518
            parts = line.split("\t")
            if len(parts) >= 6:
                term_acc = _as_float(parts[1])
                term_correct = _as_int(parts[3])
                term_total = _as_int(parts[5])
            break

    if (
        bleu is None
        or stream_laal is None
        or stream_laal_ca is None
        or term_acc is None
        or term_correct is None
        or term_total is None
    ):
        return None

    return PostEvalMetrics(
        bleu=float(bleu),
        stream_laal=float(stream_laal),
        stream_laal_ca=float(stream_laal_ca),
        term_acc=float(term_acc),
        term_correct=int(term_correct),
        term_total=int(term_total),
    )


def _key(meta: DirMeta) -> Tuple[str, int, int]:
    return (meta.glossary_tag, meta.latency_multiplier, meta.k2)


def expected_keys() -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    for g in EXPECTED_GLOSSARY_TAGS:
        for lm in EXPECTED_LATENCY_MULTIPLIERS:
            for k2 in EXPECTED_K2_VALUES:
                out.append((g, lm, k2))
    return out


def main() -> int:
    summary_path = Path(SUMMARY_TSV)
    if not summary_path.is_file():
        print(f"[ERROR] Summary TSV not found: {summary_path}", file=sys.stderr)
        return 2

    with summary_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        print(f"[ERROR] Summary TSV is empty: {summary_path}", file=sys.stderr)
        return 3

    errors: List[str] = []
    seen_keys: Dict[Tuple[str, int, int], str] = {}

    for idx, row in enumerate(rows):
        out_path = (row.get("output_path") or "").strip()
        if not out_path:
            errors.append(f"row={idx}: missing output_path")
            if FAIL_FAST:
                break
            continue

        out_dir = Path(out_path)
        if not out_dir.is_dir():
            errors.append(f"row={idx}: output_path does not exist: {out_path}")
            if FAIL_FAST:
                break
            continue

        base = out_dir.name
        meta = parse_meta_from_dir_name(base)
        if meta is None:
            errors.append(f"row={idx}: cannot parse dir meta: {base}")
            if FAIL_FAST:
                break
            continue

        # Meta checks (TSV vs dirname)
        if row.get("glossary_tag", "") != meta.glossary_tag:
            errors.append(
                f"row={idx}: glossary_tag mismatch: tsv={row.get('glossary_tag')} dir={meta.glossary_tag} path={out_path}"
            )
        if row.get("vllm_segment_sec", "") != meta.vllm_segment_sec:
            errors.append(
                f"row={idx}: vllm_segment_sec mismatch: tsv={row.get('vllm_segment_sec')} dir={meta.vllm_segment_sec} path={out_path}"
            )
        if _as_int(row.get("latency_multiplier", "")) != meta.latency_multiplier:
            errors.append(
                f"row={idx}: latency_multiplier mismatch: tsv={row.get('latency_multiplier')} dir={meta.latency_multiplier} path={out_path}"
            )
        if _as_int(row.get("K2", "")) != meta.k2:
            errors.append(f"row={idx}: K2 mismatch: tsv={row.get('K2')} dir={meta.k2} path={out_path}")
        if _as_int(row.get("K1", "")) != meta.k1:
            errors.append(f"row={idx}: K1 mismatch: tsv={row.get('K1')} dir={meta.k1} path={out_path}")

        # Coverage key checks
        k = _key(meta)
        if k in seen_keys:
            errors.append(f"duplicate key {k}: {seen_keys[k]} AND {out_path}")
        else:
            seen_keys[k] = out_path

        # post_eval.log checks
        post_eval_path = out_dir / POST_EVAL_LOG_NAME
        m = parse_metrics_from_post_eval_log(post_eval_path)
        if m is None:
            errors.append(f"row={idx}: cannot parse post_eval.log: {post_eval_path}")
            if FAIL_FAST:
                break
            continue

        if not _float_eq(_as_float(row.get("BLEU", "")), m.bleu, FLOAT_TOLERANCE):
            errors.append(f"row={idx}: BLEU mismatch: tsv={row.get('BLEU')} log={m.bleu} path={out_path}")
        if not _float_eq(_as_float(row.get("StreamLAAL", "")), m.stream_laal, FLOAT_TOLERANCE):
            errors.append(
                f"row={idx}: StreamLAAL mismatch: tsv={row.get('StreamLAAL')} log={m.stream_laal} path={out_path}"
            )
        if not _float_eq(_as_float(row.get("StreamLAAL_CA", "")), m.stream_laal_ca, FLOAT_TOLERANCE):
            errors.append(
                f"row={idx}: StreamLAAL_CA mismatch: tsv={row.get('StreamLAAL_CA')} log={m.stream_laal_ca} path={out_path}"
            )
        if not _float_eq(_as_float(row.get("TERM_ACC", "")), m.term_acc, FLOAT_TOLERANCE):
            errors.append(f"row={idx}: TERM_ACC mismatch: tsv={row.get('TERM_ACC')} log={m.term_acc} path={out_path}")
        if _as_int(row.get("TERM_CORRECT", "")) != m.term_correct:
            errors.append(
                f"row={idx}: TERM_CORRECT mismatch: tsv={row.get('TERM_CORRECT')} log={m.term_correct} path={out_path}"
            )
        if _as_int(row.get("TERM_TOTAL", "")) != m.term_total:
            errors.append(
                f"row={idx}: TERM_TOTAL mismatch: tsv={row.get('TERM_TOTAL')} log={m.term_total} path={out_path}"
            )

        if FAIL_FAST and errors:
            break

    # Coverage summary
    exp = set(expected_keys())
    got = set(seen_keys.keys())
    missing = sorted(exp - got)
    extra = sorted(got - exp)

    print(f"[INFO] TSV rows: {len(rows)}")
    print(f"[INFO] Unique keys (glossary_tag, latency_multiplier, K2): {len(got)}")
    print(f"[INFO] Expected keys: {len(exp)}")
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
        for k in missing:
            print(f"  - {k}")
    if extra:
        print(f"[WARN] Extra keys: {len(extra)}")
        for k in extra:
            print(f"  - {k} -> {seen_keys.get(k,'')}")

    if errors:
        print(f"[ERROR] Found {len(errors)} issues:")
        for e in errors[:200]:
            print(f"  - {e}")
        if len(errors) > 200:
            print(f"[ERROR] ... truncated, total issues: {len(errors)}")
        return 1

    if missing or extra:
        print("[WARN] No row-level mismatches, but coverage is not perfect.")
        return 0

    print("[OK] Summary TSV matches per-run post_eval.log and dirname meta; coverage is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




