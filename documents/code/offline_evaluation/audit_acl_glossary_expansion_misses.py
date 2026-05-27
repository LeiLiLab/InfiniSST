#!/usr/bin/env python3
"""Audit ACL term-level misses introduced by glossary expansion.

This diagnostic compares the ACL GT-only bank against an expanded bank
(GT terms + wiki terms).  It focuses on cases where the GT term was recalled
at tau in the GT-only bank but drops out of top-K after the expanded glossary
adds many near-neighbor terms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from audit_acl_boundary_samples import (  # noqa: E402
    ChunkMeta,
    _build_term_meta,
    _candidate_record,
    _flag_candidate,
    _format_candidate_line,
    _load_glossary_entries,
    _topk_indices_scores,
)
from threshold_sweep_maxsim import (  # noqa: E402
    Chunk,
    build_model,
    compute_sim,
    encode_audio_chunks,
    encode_terms,
)


def _log(msg: str) -> None:
    print(f"[EXPANSION_AUDIT] {msg}", flush=True)


@dataclass
class TermInstance:
    instance_id: str
    chunk_id: str
    utter_id: str
    chunk_idx: int
    audio_path: str
    chunk_src_text: str
    term: str
    term_key: str
    term_zh: str


def _load_acl_term_instances(
    jsonl_path: str,
    term_meta: Dict[str, Dict[str, str]],
) -> Tuple[List[Chunk], List[ChunkMeta], List[TermInstance], Dict[str, int]]:
    chunk_rows: Dict[str, Dict[str, object]] = {}
    instances: List[TermInstance] = []
    seen_instance = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            utter_id = str(obj.get("utter_id", ""))
            chunk_idx = int(obj.get("chunk_idx", 0))
            chunk_id = f"{utter_id}::{chunk_idx}"
            audio_path = str(obj.get("chunk_audio_path", ""))
            chunk_src_text = str(obj.get("chunk_src_text", ""))
            term_key = (obj.get("term_key", "") or obj.get("term", "") or "").strip().lower()

            chunk = chunk_rows.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "utter_id": utter_id,
                    "chunk_idx": chunk_idx,
                    "audio_path": audio_path,
                    "chunk_src_text": chunk_src_text,
                    "gt_terms": [],
                    "gt_terms_lower": [],
                    "gt_terms_zh": [],
                    "seen_gt": set(),
                    "has_term": False,
                },
            )

            if not term_key:
                continue

            display_term = term_meta.get(term_key, {}).get("term", obj.get("term", term_key))
            term_zh = term_meta.get(term_key, {}).get("zh", "")
            seen_gt = chunk["seen_gt"]
            if term_key not in seen_gt:
                seen_gt.add(term_key)
                chunk["has_term"] = True
                chunk["gt_terms"].append(display_term)
                chunk["gt_terms_lower"].append(term_key)
                chunk["gt_terms_zh"].append(term_zh)

            seen_instance[(chunk_id, term_key)] += 1
            instance_id = f"{chunk_id}::{term_key}::{seen_instance[(chunk_id, term_key)]}"
            instances.append(
                TermInstance(
                    instance_id=instance_id,
                    chunk_id=chunk_id,
                    utter_id=utter_id,
                    chunk_idx=chunk_idx,
                    audio_path=audio_path,
                    chunk_src_text=chunk_src_text,
                    term=str(display_term),
                    term_key=term_key,
                    term_zh=term_zh,
                )
            )

    ordered = sorted(chunk_rows.values(), key=lambda x: str(x["chunk_id"]))
    with_term: List[ChunkMeta] = []
    no_term: List[ChunkMeta] = []
    for row in ordered:
        meta = ChunkMeta(
            chunk_id=str(row["chunk_id"]),
            utter_id=str(row["utter_id"]),
            chunk_idx=int(row["chunk_idx"]),
            audio_path=str(row["audio_path"]),
            chunk_src_text=str(row["chunk_src_text"]),
            gt_terms=list(row["gt_terms"]),
            gt_terms_lower=list(row["gt_terms_lower"]),
            gt_terms_zh=list(row["gt_terms_zh"]),
            has_term=bool(row["has_term"]),
        )
        (with_term if meta.has_term else no_term).append(meta)

    metas = with_term + no_term
    chunks = [
        Chunk(
            chunk_id=meta.chunk_id,
            audio_path=meta.audio_path,
            gt_terms=set(meta.gt_terms_lower),
            has_term=meta.has_term,
        )
        for meta in metas
    ]
    chunk_to_row = {meta.chunk_id: idx for idx, meta in enumerate(metas)}
    return chunks, metas, instances, chunk_to_row


def _build_banks(
    instances: Sequence[TermInstance],
    wiki_terms: Sequence[str],
    gs_size: int,
    term_meta: Dict[str, Dict[str, str]],
) -> Tuple[List[str], List[str], Dict[str, int]]:
    base_bank = sorted({inst.term_key for inst in instances if inst.term_key})
    expanded = list(base_bank)
    already = set(expanded)
    for raw_term in wiki_terms:
        key = raw_term.strip().lower()
        if not key or key in already:
            continue
        expanded.append(key)
        already.add(key)
        if gs_size > 0 and len(expanded) >= gs_size:
            break
    for term in expanded:
        term_meta.setdefault(term, {"term": term, "zh": ""})
    return base_bank, expanded, {term: idx for idx, term in enumerate(expanded)}


@torch.no_grad()
def _compute_sim_numpy_batched(
    speech_embs: torch.Tensor,
    text_embs: torch.Tensor,
    _maxsim_score,
    batch_rows: int,
) -> np.ndarray:
    if batch_rows <= 0 or speech_embs.size(0) <= batch_rows:
        return compute_sim(speech_embs, text_embs, _maxsim_score).cpu().numpy()

    sims: List[np.ndarray] = []
    total = speech_embs.size(0)
    for start in range(0, total, batch_rows):
        end = min(start + batch_rows, total)
        chunk_sim = compute_sim(speech_embs[start:end], text_embs, _maxsim_score)
        sims.append(chunk_sim.cpu().numpy())
        _log(f"  sim rows {end}/{total}")
        del chunk_sim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(sims, axis=0)


def _rank_of(row: np.ndarray, term_idx: int) -> int:
    # One-indexed strict rank; ties at the same score share the earlier rank.
    return int(np.sum(row > row[term_idx]) + 1)


def _candidate_source(term_idx: int, base_size: int) -> str:
    return "acl_gt_bank" if term_idx < base_size else "wiki_extra"


def _candidate_record_for_target(
    rank: int,
    term_idx: int,
    score: float,
    expanded_bank: Sequence[str],
    term_meta: Dict[str, Dict[str, str]],
    target_idx: int,
    target_term: str,
    target_zh: str,
    target_score: float,
    base_size: int,
    small_gap: float,
) -> Dict[str, object]:
    term_key = expanded_bank[term_idx]
    entry = term_meta.get(term_key, {"term": term_key, "zh": ""})
    is_target = term_idx == target_idx
    if is_target:
        flags: List[str] = []
        auto_label = "target_gt_term"
    else:
        flags, auto_label = _flag_candidate(
            candidate_term=entry.get("term", term_key),
            candidate_zh=entry.get("zh", ""),
            score=float(score),
            gt_terms=[target_term],
            gt_zh_terms=[target_zh] if target_zh else [],
            best_gt_score=float(target_score),
            small_gap=small_gap,
        )
    return {
        "rank": rank,
        "term": entry.get("term", term_key),
        "term_key": term_key,
        "zh": entry.get("zh", ""),
        "score": round(float(score), 6),
        "source": _candidate_source(term_idx, base_size),
        "is_gt": is_target,
        "is_target": is_target,
        "is_acl_gt_bank": term_idx < base_size,
        "score_band": "0.7-0.8" if 0.7 <= float(score) < 0.8 else (
            ">=0.8" if float(score) >= 0.8 else "<0.7"
        ),
        "flags": flags,
        "auto_label": auto_label,
    }


def _hit_at_tau(row: np.ndarray, target_idx: int, topk: int, tau: float, bank_limit: int) -> bool:
    subrow = row[:bank_limit]
    k = min(topk, subrow.shape[0])
    top_idx, _ = _topk_indices_scores(subrow, k)
    return bool(target_idx in set(int(x) for x in top_idx) and row[target_idx] >= tau)


def _safe_float(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    top1_terms = Counter(str(row.get("top1_term") or "") for row in rows)
    target_terms = Counter(str(row.get("target_term") or "") for row in rows)
    top10_label_counts = Counter()
    top10_source_counts = Counter()
    wiki_07_08_top10 = 0
    wiki_ge_08_top10 = 0
    for row in rows:
        for cand in row.get("top30", [])[:10]:
            top10_label_counts[str(cand["auto_label"])] += 1
            top10_source_counts[str(cand["source"])] += 1
            if cand["source"] == "wiki_extra" and cand["score_band"] == "0.7-0.8":
                wiki_07_08_top10 += 1
            if cand["source"] == "wiki_extra" and cand["score_band"] == ">=0.8":
                wiki_ge_08_top10 += 1
    return {
        "rows": len(rows),
        "top1_terms": top1_terms.most_common(20),
        "target_terms": target_terms.most_common(20),
        "top10_auto_label_counts": dict(top10_label_counts),
        "top10_source_counts": dict(top10_source_counts),
        "wiki_0p70_0p80_in_top10": wiki_07_08_top10,
        "wiki_ge_0p80_in_top10": wiki_ge_08_top10,
    }


def _write_jsonl(path: str, rows: Sequence[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_tsv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    header = [
        "instance_id",
        "chunk_src_text",
        "target_term",
        "target_score",
        "base_rank",
        "expanded_rank",
        "base_hit_tau0p70",
        "expanded_hit_tau0p70",
        "base_hit_tau0p80",
        "expanded_hit_tau0p80",
        "top1_term",
        "top1_source",
        "top1_score",
        "top1_label",
        "wiki_0p70_0p80_above_target_top30",
        "wiki_ge_target_top30",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            values = [
                row["instance_id"],
                str(row["chunk_src_text"]).replace("\t", " "),
                row["target_term"],
                _safe_float(row["target_score"]),
                str(row["base_rank"]),
                str(row["expanded_rank"]),
                str(row["base_hit_tau0p70"]),
                str(row["expanded_hit_tau0p70"]),
                str(row["base_hit_tau0p80"]),
                str(row["expanded_hit_tau0p80"]),
                row["top1_term"],
                row["top1_source"],
                _safe_float(row["top1_score"]),
                row["top1_label"],
                str(row["wiki_0p70_0p80_above_target_top30"]),
                str(row["wiki_ge_target_top30"]),
            ]
            f.write("\t".join(values) + "\n")


def _write_report(
    path: str,
    args: argparse.Namespace,
    summary: Dict[str, object],
    regression_rows: Sequence[Dict[str, object]],
    miss_rows: Sequence[Dict[str, object]],
) -> None:
    lines: List[str] = []
    lines.append("# ACL Glossary Expansion Miss Audit")
    lines.append("")
    lines.append(f"- Model: `{args.model_path}`")
    lines.append(f"- ACL JSONL: `{args.acl_jsonl}`")
    lines.append(f"- Wiki glossary: `{args.wiki_glossary}`")
    lines.append(f"- Expanded glossary size: `{summary['expanded_bank_size']}`")
    lines.append(f"- Top-K dump: `{args.top_k}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in (
        "term_instances",
        "base_bank_size",
        "expanded_bank_size",
        "base_hit_tau0p70",
        "expanded_hit_tau0p70",
        "base_to_expanded_regressions_tau0p70",
        "base_hit_tau0p80",
        "expanded_hit_tau0p80",
        "base_to_expanded_regressions_tau0p80",
    ):
        lines.append(f"- `{key}`: `{summary[key]}`")
    lines.append("")
    lines.append("## Tau 0.70 Regression Aggregate")
    lines.append("")
    reg_summary = summary["regression_tau0p70"]
    lines.append(f"- Rows: `{reg_summary['rows']}`")
    lines.append(f"- Top-10 source counts: `{reg_summary['top10_source_counts']}`")
    lines.append(f"- Top-10 auto-label counts: `{reg_summary['top10_auto_label_counts']}`")
    lines.append(f"- Wiki `[0.70,0.80)` candidates in top-10: `{reg_summary['wiki_0p70_0p80_in_top10']}`")
    lines.append(f"- Wiki `>=0.80` candidates in top-10: `{reg_summary['wiki_ge_0p80_in_top10']}`")
    lines.append(f"- Most frequent target terms: `{reg_summary['target_terms'][:10]}`")
    lines.append(f"- Most frequent top-1 terms: `{reg_summary['top1_terms'][:10]}`")
    lines.append("")
    lines.append("## Representative Tau 0.70 Regressions")
    lines.append("")
    selected = sorted(
        regression_rows,
        key=lambda r: (
            -int(r["wiki_0p70_0p80_above_target_top30"]),
            -float(r["top1_score"]),
        ),
    )[: args.report_examples]
    for i, row in enumerate(selected, 1):
        lines.append(
            f"### {i}. `{row['instance_id']}` target=`{row['target_term']}` "
            f"score=`{row['target_score']:.6f}` rank `{row['base_rank']} -> {row['expanded_rank']}`"
        )
        lines.append("")
        lines.append(f"- Chunk text: `{row['chunk_src_text']}`")
        lines.append(
            f"- Wiki `[0.70,0.80)` above target in top-30: "
            f"`{row['wiki_0p70_0p80_above_target_top30']}`; "
            f"wiki above target in top-30: `{row['wiki_ge_target_top30']}`"
        )
        lines.append("- Expanded top-30:")
        for cand in row["top30"]:
            source = cand["source"]
            marker = " TARGET" if cand["is_target"] else ""
            lines.append(
                f"  - `{_format_candidate_line(cand)} "
                f"[source={source} band={cand['score_band']}{marker}]`"
            )
        lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for key, value in summary["paths"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--acl_jsonl", required=True)
    parser.add_argument("--wiki_glossary", required=True)
    parser.add_argument("--gt_glossary", default="")
    parser.add_argument("--candidate_glossary", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gs_size", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--eval_topk", type=int, default=10)
    parser.add_argument("--focus_tau", type=float, default=0.70)
    parser.add_argument("--compare_tau", type=float, default=0.80)
    parser.add_argument("--small_gap", type=float, default=0.03)
    parser.add_argument("--report_examples", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--target_dim", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2",
    )
    parser.add_argument("--pooling_type", type=str, default="transformer")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--use_maxsim", action="store_true", default=False)
    parser.add_argument("--maxsim_windows", type=str, default="2 3 4 5 6 7 8 10 12 16 20 24")
    parser.add_argument("--maxsim_stride", type=int, default=2)
    parser.add_argument("--text_lora_rank", type=int, default=128)
    parser.add_argument("--text_lora_alpha", type=int, default=256)
    parser.add_argument("--text_lora_target_modules", type=str, default="query key value dense")
    parser.add_argument("--text_pooling", type=str, default="cls")
    parser.add_argument("--sparse_weight", type=float, default=0.0)
    parser.add_argument("--sim_batch_rows", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    term_meta = _build_term_meta(args.wiki_glossary, args.candidate_glossary, args.gt_glossary)
    wiki_terms = [term for term, _ in _load_glossary_entries(args.wiki_glossary)]
    chunks, metas, instances, chunk_to_row = _load_acl_term_instances(args.acl_jsonl, term_meta)
    base_bank, expanded_bank, expanded_term_to_idx = _build_banks(
        instances, wiki_terms, args.gs_size, term_meta
    )
    base_size = len(base_bank)
    _log(
        f"ACL chunks={len(metas)} term_instances={len(instances)} "
        f"base_bank={base_size} expanded_bank={len(expanded_bank)}"
    )

    retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device = build_model(args)

    _log("Encoding audio chunks...")
    speech_embs = encode_audio_chunks(chunks, retriever, feat_ext, device)
    _log(f"speech_embs={tuple(speech_embs.shape)}")

    _log("Encoding expanded bank terms...")
    text_embs = encode_terms(expanded_bank, text_encoder, tokenizer, device)
    _log(f"text_embs={tuple(text_embs.shape)}")

    _log("Computing similarity matrix...")
    sim_np = _compute_sim_numpy_batched(
        speech_embs, text_embs, _maxsim_score, batch_rows=args.sim_batch_rows
    )
    _log(f"sim={tuple(sim_np.shape)}")

    focus_tag = f"tau{args.focus_tau:.2f}".replace(".", "p")
    compare_tag = f"tau{args.compare_tau:.2f}".replace(".", "p")
    miss_rows: List[Dict[str, object]] = []
    regression_rows: List[Dict[str, object]] = []
    all_rows: List[Dict[str, object]] = []
    metric_counts = Counter()

    for inst in instances:
        target_idx = expanded_term_to_idx.get(inst.term_key)
        if target_idx is None:
            continue
        row = sim_np[chunk_to_row[inst.chunk_id]]
        base_rank = _rank_of(row[:base_size], target_idx)
        expanded_rank = _rank_of(row, target_idx)
        target_score = float(row[target_idx])

        base_hit_focus = _hit_at_tau(row, target_idx, args.eval_topk, args.focus_tau, base_size)
        expanded_hit_focus = _hit_at_tau(
            row, target_idx, args.eval_topk, args.focus_tau, len(expanded_bank)
        )
        base_hit_compare = _hit_at_tau(row, target_idx, args.eval_topk, args.compare_tau, base_size)
        expanded_hit_compare = _hit_at_tau(
            row, target_idx, args.eval_topk, args.compare_tau, len(expanded_bank)
        )

        metric_counts[f"base_hit_{focus_tag}"] += int(base_hit_focus)
        metric_counts[f"expanded_hit_{focus_tag}"] += int(expanded_hit_focus)
        metric_counts[f"base_hit_{compare_tag}"] += int(base_hit_compare)
        metric_counts[f"expanded_hit_{compare_tag}"] += int(expanded_hit_compare)
        metric_counts[f"regression_{focus_tag}"] += int(base_hit_focus and not expanded_hit_focus)
        metric_counts[f"regression_{compare_tag}"] += int(base_hit_compare and not expanded_hit_compare)

        top_idx, top_scores = _topk_indices_scores(row, args.top_k)
        top30 = [
            _candidate_record_for_target(
                rank=rank,
                term_idx=int(term_idx),
                score=float(score),
                expanded_bank=expanded_bank,
                term_meta=term_meta,
                target_idx=target_idx,
                target_term=inst.term,
                target_zh=inst.term_zh,
                target_score=target_score,
                base_size=base_size,
                small_gap=args.small_gap,
            )
            for rank, (term_idx, score) in enumerate(zip(top_idx, top_scores), start=1)
        ]

        top1 = top30[0]
        wiki_07_08_above_target_top30 = sum(
            1
            for cand in top30
            if cand["source"] == "wiki_extra"
            and cand["score_band"] == "0.7-0.8"
            and float(cand["score"]) >= target_score
        )
        wiki_ge_target_top30 = sum(
            1
            for cand in top30
            if cand["source"] == "wiki_extra" and float(cand["score"]) >= target_score
        )

        out_row: Dict[str, object] = {
            "instance_id": inst.instance_id,
            "chunk_id": inst.chunk_id,
            "utter_id": inst.utter_id,
            "chunk_idx": inst.chunk_idx,
            "audio_path": inst.audio_path,
            "chunk_src_text": inst.chunk_src_text,
            "target_term": inst.term,
            "target_term_key": inst.term_key,
            "target_zh": inst.term_zh,
            "target_score": round(target_score, 6),
            "base_rank": base_rank,
            "expanded_rank": expanded_rank,
            "base_hit_tau0p70": base_hit_focus,
            "expanded_hit_tau0p70": expanded_hit_focus,
            "base_hit_tau0p80": base_hit_compare,
            "expanded_hit_tau0p80": expanded_hit_compare,
            "top1_term": top1["term"],
            "top1_term_key": top1["term_key"],
            "top1_source": top1["source"],
            "top1_score": top1["score"],
            "top1_label": top1["auto_label"],
            "top1_flags": top1["flags"],
            "wiki_0p70_0p80_above_target_top30": wiki_07_08_above_target_top30,
            "wiki_ge_target_top30": wiki_ge_target_top30,
            "top30": top30,
        }
        all_rows.append(out_row)
        if not expanded_hit_focus:
            miss_rows.append(out_row)
        if base_hit_focus and not expanded_hit_focus:
            regression_rows.append(out_row)

    prefix = f"acl6060_gs{args.gs_size}_{focus_tag}"
    all_path = os.path.join(args.output_dir, f"{prefix}_all_term_instances_top{args.top_k}.jsonl")
    miss_path = os.path.join(args.output_dir, f"{prefix}_misses_top{args.top_k}.jsonl")
    regression_path = os.path.join(args.output_dir, f"{prefix}_base_to_expanded_regressions_top{args.top_k}.jsonl")
    miss_tsv = os.path.join(args.output_dir, f"{prefix}_misses_top{args.top_k}.tsv")
    regression_tsv = os.path.join(args.output_dir, f"{prefix}_base_to_expanded_regressions_top{args.top_k}.tsv")
    summary_path = os.path.join(args.output_dir, "acl6060_glossary_expansion_miss_summary.json")
    report_path = os.path.join(args.output_dir, "acl6060_glossary_expansion_miss_report.md")

    _write_jsonl(all_path, all_rows)
    _write_jsonl(miss_path, miss_rows)
    _write_jsonl(regression_path, regression_rows)
    _write_tsv(miss_tsv, miss_rows)
    _write_tsv(regression_tsv, regression_rows)

    n = max(len(instances), 1)
    summary: Dict[str, object] = {
        "model_path": args.model_path,
        "acl_jsonl": args.acl_jsonl,
        "wiki_glossary": args.wiki_glossary,
        "output_dir": args.output_dir,
        "elapsed_seconds": round(time.time() - t0, 2),
        "term_instances": len(instances),
        "acl_chunks": len(metas),
        "base_bank_size": base_size,
        "expanded_bank_size": len(expanded_bank),
        "top_k_dump": args.top_k,
        "eval_topk": args.eval_topk,
        "base_hit_tau0p70": {
            "count": metric_counts[f"base_hit_{focus_tag}"],
            "rate": metric_counts[f"base_hit_{focus_tag}"] / n,
        },
        "expanded_hit_tau0p70": {
            "count": metric_counts[f"expanded_hit_{focus_tag}"],
            "rate": metric_counts[f"expanded_hit_{focus_tag}"] / n,
        },
        "base_to_expanded_regressions_tau0p70": {
            "count": metric_counts[f"regression_{focus_tag}"],
            "rate": metric_counts[f"regression_{focus_tag}"] / n,
        },
        "base_hit_tau0p80": {
            "count": metric_counts[f"base_hit_{compare_tag}"],
            "rate": metric_counts[f"base_hit_{compare_tag}"] / n,
        },
        "expanded_hit_tau0p80": {
            "count": metric_counts[f"expanded_hit_{compare_tag}"],
            "rate": metric_counts[f"expanded_hit_{compare_tag}"] / n,
        },
        "base_to_expanded_regressions_tau0p80": {
            "count": metric_counts[f"regression_{compare_tag}"],
            "rate": metric_counts[f"regression_{compare_tag}"] / n,
        },
        "misses_tau0p70": _summarize_rows(miss_rows),
        "regression_tau0p70": _summarize_rows(regression_rows),
        "paths": {
            "all_jsonl": all_path,
            "miss_jsonl": miss_path,
            "regression_jsonl": regression_path,
            "miss_tsv": miss_tsv,
            "regression_tsv": regression_tsv,
            "summary_json": summary_path,
            "report_md": report_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _write_report(report_path, args, summary, regression_rows, miss_rows)
    _log(f"Done in {time.time() - t0:.1f}s")
    _log(f"Summary: {summary_path}")
    _log(f"Report: {report_path}")
    _log(f"Regression rows: {regression_path}")


if __name__ == "__main__":
    main()
