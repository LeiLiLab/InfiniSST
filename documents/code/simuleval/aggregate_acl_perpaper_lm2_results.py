#!/usr/bin/env python3
"""Aggregate ACL per-paper LM=1..4 SimulEval result TSVs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


GLOSSARY_TAGS = {
    "glossary_acl6060_gt_union_gs10000": "gs10k",
    "glossary_acl6060_gt_union_gs1000": "gs1k",
    "extracted_glossary": "raw",
}


def _to_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _to_int(row: Dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "") or 0))
    except ValueError:
        return 0


def _glossary_regime(path: Path) -> str:
    name = path.parent.name
    for marker, regime in GLOSSARY_TAGS.items():
        if marker in name:
            return regime
    return "unknown"


def _latency_multiplier(path: Path) -> int:
    match = re.search(r"_lm(\d+)_", path.parent.name)
    if not match:
        return 0
    return int(match.group(1))


def _read_result(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"Expected one data row in {path}, found {len(rows)}")
    row = dict(rows[0])
    row["_path"] = str(path)
    row["_regime"] = _glossary_regime(path)
    row["_lm"] = str(_latency_multiplier(path))
    return row


def aggregate(rows: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[int, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["_lm"]), row["_regime"])].append(row)

    out: List[Dict[str, Any]] = []
    for lm in (1, 2, 3, 4):
        for regime in ("raw", "gs1k", "gs10k"):
            items = grouped.get((lm, regime), [])
            if not items:
                continue

            n = len(items)
            term_correct = sum(_to_int(r, "TERM_CORRECT") for r in items)
            term_total = sum(_to_int(r, "TERM_TOTAL") for r in items)
            adoption_adopted = sum(_to_int(r, "TERM_ADOPTED") for r in items)
            adoption_total = sum(_to_int(r, "TERM_ADOPTION_TOTAL") for r in items)
            adoption_sentences = sum(_to_int(r, "TERM_ADOPTION_SENTENCES") for r in items)
            false_copy = sum(_to_int(r, "FALSE_COPY") for r in items)
            neg_total = sum(_to_int(r, "NEG_TOTAL") for r in items)
            false_copy_terms = sum(_to_int(r, "FALSE_COPY_TERMS") for r in items)
            source_false_copy = sum(_to_int(r, "SOURCE_FALSE_COPY") for r in items)
            source_neg_total = sum(_to_int(r, "SOURCE_NEG_TOTAL") for r in items)
            source_false_copy_terms = sum(_to_int(r, "SOURCE_FALSE_COPY_TERMS") for r in items)
            fcr_modes = sorted({r.get("TERM_FCR_MODE", "unknown") or "unknown" for r in items})
            adoption_weighted_sum = sum(
                _to_float(r, "TERM_ADOPTION") * _to_int(r, "TERM_ADOPTION_SENTENCES")
                for r in items
            )

            out.append(
                {
                    "lm": lm,
                    "regime": regime,
                    "papers": n,
                    "complete": "yes" if n == 5 else "no",
                    "BLEU_macro": sum(_to_float(r, "BLEU") for r in items) / n,
                    "StreamLAAL_macro": sum(_to_float(r, "StreamLAAL") for r in items) / n,
                    "TERM_ACC_micro": term_correct / term_total if term_total else 0.0,
                    "TERM_CORRECT": term_correct,
                    "TERM_TOTAL": term_total,
                    "TERM_ADOPTION_sentence_macro": (
                        adoption_weighted_sum / adoption_sentences if adoption_sentences else 0.0
                    ),
                    "TERM_ADOPTION_micro": (
                        adoption_adopted / adoption_total if adoption_total else 0.0
                    ),
                    "TERM_ADOPTED": adoption_adopted,
                    "TERM_ADOPTION_TOTAL": adoption_total,
                    "TERM_ADOPTION_SENTENCES": adoption_sentences,
                    "TERM_FCR_micro": false_copy / neg_total if neg_total else 0.0,
                    "TERM_FCR_MODE": ",".join(fcr_modes),
                    "FALSE_COPY": false_copy,
                    "NEG_TOTAL": neg_total,
                    "FALSE_COPY_TERMS": false_copy_terms,
                    "SOURCE_TERM_SENT_FCR_micro": (
                        source_false_copy / source_neg_total if source_neg_total else 0.0
                    ),
                    "SOURCE_FALSE_COPY": source_false_copy,
                    "SOURCE_NEG_TOTAL": source_neg_total,
                    "SOURCE_FALSE_COPY_TERMS": source_false_copy_terms,
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default="/mnt/gemini/data2/jiaxuanluo/acl_perpaper_lm1to4_raw1k10k_sorigin_finalrag_taurus",
    )
    ap.add_argument("--output-tsv", default="")
    args = ap.parse_args()

    base = Path(args.base_dir)
    rows = [_read_result(p) for p in sorted(base.rglob("eval_results.tsv"))]
    summary = aggregate(rows)

    fields = [
        "lm",
        "regime",
        "papers",
        "complete",
        "BLEU_macro",
        "StreamLAAL_macro",
        "TERM_ACC_micro",
        "TERM_CORRECT",
        "TERM_TOTAL",
        "TERM_ADOPTION_sentence_macro",
        "TERM_ADOPTION_micro",
        "TERM_ADOPTED",
        "TERM_ADOPTION_TOTAL",
        "TERM_ADOPTION_SENTENCES",
        "TERM_FCR_micro",
        "TERM_FCR_MODE",
        "FALSE_COPY",
        "NEG_TOTAL",
        "FALSE_COPY_TERMS",
        "SOURCE_TERM_SENT_FCR_micro",
        "SOURCE_FALSE_COPY",
        "SOURCE_NEG_TOTAL",
        "SOURCE_FALSE_COPY_TERMS",
    ]
    out_path = Path(args.output_tsv) if args.output_tsv else base / "summary_acl_perpaper_lm1to4.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote {out_path}")
    with out_path.open("r", encoding="utf-8") as f:
        print(f.read().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
