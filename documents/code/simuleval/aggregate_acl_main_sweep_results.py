#!/usr/bin/env python3
"""Aggregate ACL main-sweep one-setting eval_results.tsv files.

Expected layout:
  <base>/zh/d<density>_lm<LM>_k<K>_th<TAU>_g<GLOSSARY>_pp<PAPER>/eval_results.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PAPERS = (
    "2022.acl-long.110",
    "2022.acl-long.117",
    "2022.acl-long.268",
    "2022.acl-long.367",
    "2022.acl-long.590",
)


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


def _metadata(path: Path) -> Dict[str, str]:
    name = path.parent.name
    lm = re.search(r"_lm(\d+)_", name)
    tau = re.search(r"_th([^_]+)_", name)
    paper = re.search(r"_pp(2022\.acl-long\.\d+)$", name)
    if "gglossary_acl6060_gt_union_gs10000" in name:
        regime = "gs10k"
    elif "gglossary_acl6060_gt_union_gs1000" in name:
        regime = "gs1k"
    elif "gextracted_glossary__" in name:
        regime = "raw"
    else:
        regime = "unknown"
    return {
        "lm": lm.group(1) if lm else "",
        "tau": tau.group(1) if tau else "",
        "paper": paper.group(1) if paper else "",
        "regime": regime,
    }


def _read_result(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty TSV: {path}")
    row = dict(rows[-1])
    row.update({f"_{k}": v for k, v in _metadata(path).items()})
    row["_path"] = str(path)
    return row


def aggregate(rows: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        tau = row.get("_tau", "")
        regime = row.get("_regime", "unknown")
        try:
            lm = int(row.get("_lm", "") or 0)
        except ValueError:
            lm = 0
        grouped[(tau, lm, regime)].append(row)

    out: List[Dict[str, Any]] = []
    for tau in ("0.0", "0.75"):
        for lm in (1, 2, 3, 4):
            for regime in ("raw", "gs1k", "gs10k"):
                items = grouped.get((tau, lm, regime), [])
                if not items:
                    continue
                papers = sorted({r.get("_paper", "") for r in items if r.get("_paper")})
                n = len(items)
                term_correct = sum(_to_int(r, "TERM_CORRECT") for r in items)
                term_total = sum(_to_int(r, "TERM_TOTAL") for r in items)
                adoption_adopted = sum(_to_int(r, "TERM_ADOPTED") for r in items)
                adoption_total = sum(_to_int(r, "TERM_ADOPTION_TOTAL") for r in items)
                adoption_sentences = sum(_to_int(r, "TERM_ADOPTION_SENTENCES") for r in items)
                real_adopted = sum(_to_int(r, "REAL_TERM_ADOPTED") for r in items)
                real_total = sum(_to_int(r, "REAL_TERM_ADOPT_TOTAL") for r in items)
                real_sentences = sum(_to_int(r, "REAL_TERM_ADOPT_SENTENCES") for r in items)
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
                real_weighted_sum = sum(
                    _to_float(r, "REAL_TERM_ADOPT") * _to_int(r, "REAL_TERM_ADOPT_SENTENCES")
                    for r in items
                )
                out.append(
                    {
                        "tau": tau,
                        "lm": lm,
                        "regime": regime,
                        "papers": n,
                        "complete": "yes" if sorted(papers) == sorted(PAPERS) else "no",
                        "paper_ids": ",".join(papers),
                        "BLEU_macro": sum(_to_float(r, "BLEU") for r in items) / n,
                        "StreamLAAL_macro": sum(_to_float(r, "StreamLAAL") for r in items) / n,
                        "StreamLAAL_CA_macro": sum(_to_float(r, "StreamLAAL_CA") for r in items) / n,
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
                        "REAL_TERM_ADOPTION_sentence_macro": (
                            real_weighted_sum / real_sentences if real_sentences else 0.0
                        ),
                        "REAL_TERM_ADOPTION_micro": (
                            real_adopted / real_total if real_total else 0.0
                        ),
                        "REAL_TERM_ADOPTED": real_adopted,
                        "REAL_TERM_ADOPT_TOTAL": real_total,
                        "REAL_TERM_ADOPT_SENTENCES": real_sentences,
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
        default="/mnt/gemini/home/jiaxuanluo/acl_main_zh_v2_r32_srcgated_no_utterance",
    )
    ap.add_argument("--output-tsv", default="")
    args = ap.parse_args()

    base = Path(args.base_dir)
    rows = [_read_result(p) for p in sorted(base.rglob("eval_results.tsv"))]
    summary = aggregate(rows)

    fields = [
        "tau",
        "lm",
        "regime",
        "papers",
        "complete",
        "paper_ids",
        "BLEU_macro",
        "StreamLAAL_macro",
        "StreamLAAL_CA_macro",
        "TERM_ACC_micro",
        "TERM_CORRECT",
        "TERM_TOTAL",
        "TERM_ADOPTION_sentence_macro",
        "TERM_ADOPTION_micro",
        "TERM_ADOPTED",
        "TERM_ADOPTION_TOTAL",
        "TERM_ADOPTION_SENTENCES",
        "REAL_TERM_ADOPTION_sentence_macro",
        "REAL_TERM_ADOPTION_micro",
        "REAL_TERM_ADOPTED",
        "REAL_TERM_ADOPT_TOTAL",
        "REAL_TERM_ADOPT_SENTENCES",
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
    out_path = Path(args.output_tsv) if args.output_tsv else base / "summary_acl_main_zh_v2r32.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote {out_path}")
    print(out_path.read_text(encoding="utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
