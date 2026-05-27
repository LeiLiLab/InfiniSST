#!/usr/bin/env python3
"""Audit ACL6060 gs10000 boundary retrieval samples for one checkpoint.

This script reuses the MaxSim offline-eval stack to dump per-chunk top-10
retrieval results, then extracts a boundary subset around a tau band so we can
inspect whether high-scoring non-GT candidates are likely false negatives,
very similar terms, or clear noise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
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

from threshold_sweep_maxsim import (  # noqa: E402
    Chunk,
    build_model,
    compute_sim,
    encode_audio_chunks,
    encode_terms,
)

TOP_K = 10
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _log(msg: str) -> None:
    print(f"[BOUNDARY] {msg}", flush=True)


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


def _load_glossary_entries(path: str) -> List[Tuple[str, str]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items: List[Tuple[str, str]] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                term = value.get("term", key)
                zh = (
                    value.get("translation", "")
                    or value.get("zh", "")
                    or value.get("target_translations", {}).get("zh", "")
                )
            else:
                term = key
                zh = ""
            items.append((str(term), str(zh)))
    elif isinstance(raw, list):
        for value in raw:
            if isinstance(value, dict):
                term = value.get("term", "")
                zh = (
                    value.get("translation", "")
                    or value.get("zh", "")
                    or value.get("target_translations", {}).get("zh", "")
                )
            else:
                term = str(value)
                zh = ""
            items.append((str(term), str(zh)))
    else:
        raise ValueError(f"Unexpected glossary format in {path}: {type(raw)}")
    return items


def _build_term_meta(*glossary_paths: str) -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    for path in glossary_paths:
        for term, zh in _load_glossary_entries(path):
            key = term.strip().lower()
            if not key:
                continue
            entry = meta.setdefault(key, {"term": term, "zh": ""})
            if term and (not entry["term"] or entry["term"] == key):
                entry["term"] = term
            if zh and not entry["zh"]:
                entry["zh"] = zh
    return meta


def _normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower()))


def _compact_text(text: str) -> str:
    return "".join(TOKEN_RE.findall((text or "").lower()))


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _acronym(text: str) -> str:
    toks = [t for t in _tokenize(text) if t not in STOPWORDS]
    if not toks:
        return ""
    if len(toks) == 1:
        return toks[0]
    return "".join(tok[0] for tok in toks)


def _token_overlap(a: str, b: str) -> float:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for xa, xb in zip(a, b):
        if xa != xb:
            break
        n += 1
    return n


@dataclass
class ChunkMeta:
    chunk_id: str
    utter_id: str
    chunk_idx: int
    audio_path: str
    chunk_src_text: str
    gt_terms: List[str]
    gt_terms_lower: List[str]
    gt_terms_zh: List[str]
    has_term: bool


def load_acl_chunks_with_meta(
    jsonl_path: str,
    gt_term_meta: Dict[str, Dict[str, str]],
) -> Tuple[List[Chunk], List[ChunkMeta]]:
    groups: Dict[str, Dict[str, object]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            utter_id = str(obj.get("utter_id", ""))
            chunk_idx = int(obj.get("chunk_idx", 0))
            chunk_id = f"{utter_id}::{chunk_idx}"
            group = groups.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "utter_id": utter_id,
                    "chunk_idx": chunk_idx,
                    "audio_path": obj.get("chunk_audio_path", ""),
                    "chunk_src_text": obj.get("chunk_src_text", ""),
                    "gt_terms": [],
                    "gt_terms_lower": [],
                    "gt_terms_zh": [],
                    "seen_gt": set(),
                    "has_term": False,
                },
            )
            term = (obj.get("term_key", "") or obj.get("term", "") or "").strip()
            if not term:
                continue
            term_lower = term.lower()
            seen_gt = group["seen_gt"]
            if term_lower in seen_gt:
                continue
            seen_gt.add(term_lower)
            group["has_term"] = True
            display_term = gt_term_meta.get(term_lower, {}).get("term", term)
            gt_zh = gt_term_meta.get(term_lower, {}).get("zh", "")
            group["gt_terms"].append(display_term)
            group["gt_terms_lower"].append(term_lower)
            group["gt_terms_zh"].append(gt_zh)

    ordered = sorted(groups.values(), key=lambda x: x["chunk_id"])
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
    return chunks, metas


def build_bank(
    metas: Sequence[ChunkMeta],
    wiki_terms: Sequence[str],
    gs_size: int,
    term_meta: Dict[str, Dict[str, str]],
) -> List[str]:
    gt_terms = sorted({term for meta in metas for term in meta.gt_terms_lower})
    bank = list(gt_terms)
    already_in = set(bank)
    for wiki_term in wiki_terms:
        key = wiki_term.strip().lower()
        if not key or key in already_in:
            continue
        bank.append(key)
        already_in.add(key)
        if gs_size > 0 and len(bank) >= gs_size:
            break
    for term in bank:
        term_meta.setdefault(term, {"term": term, "zh": ""})
    return bank


def _topk_indices_scores(row: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = min(k, row.shape[0])
    idx = np.argpartition(-row, n - 1)[:n]
    scores = row[idx]
    order = np.argsort(-scores)
    return idx[order], scores[order]


def _flag_candidate(
    candidate_term: str,
    candidate_zh: str,
    score: float,
    gt_terms: Sequence[str],
    gt_zh_terms: Sequence[str],
    best_gt_score: Optional[float],
    small_gap: float,
) -> Tuple[List[str], str]:
    cand_norm = _normalize_text(candidate_term)
    cand_compact = _compact_text(candidate_term)
    cand_acronym = _acronym(candidate_term)
    cand_zh_norm = _normalize_text(candidate_zh)

    gt_norms = [_normalize_text(t) for t in gt_terms]
    gt_compacts = [_compact_text(t) for t in gt_terms]
    gt_acronyms = [_acronym(t) for t in gt_terms]
    gt_zh_norms = [_normalize_text(t) for t in gt_zh_terms if t]

    flags: List[str] = []
    max_token_overlap = 0.0
    has_token_containment = False
    has_long_prefix_overlap = False

    for gt_term, gt_norm, gt_compact, gt_acronym in zip(
        gt_terms, gt_norms, gt_compacts, gt_acronyms
    ):
        if cand_norm and cand_norm == gt_norm:
            flags.append("normalized_exact_match")
        if cand_compact and cand_compact == gt_compact:
            flags.append("compact_exact_match")
        if cand_acronym and gt_acronym and cand_acronym == gt_acronym:
            flags.append("acronym_overlap")
        max_token_overlap = max(max_token_overlap, _token_overlap(candidate_term, gt_term))
        cand_tokens = set(_tokenize(candidate_term))
        gt_tokens = set(_tokenize(gt_term))
        if cand_tokens and gt_tokens and cand_tokens != gt_tokens:
            if cand_tokens.issubset(gt_tokens) or gt_tokens.issubset(cand_tokens):
                has_token_containment = True
        prefix_len = _common_prefix_len(cand_compact, gt_compact)
        min_len = min(len(cand_compact), len(gt_compact))
        if min_len >= 6 and prefix_len >= 6 and prefix_len / max(min_len, 1) >= 0.7:
            has_long_prefix_overlap = True

    if has_token_containment:
        flags.append("token_containment")
    if has_long_prefix_overlap:
        flags.append("shared_long_prefix")
    if max_token_overlap >= 0.5:
        flags.append("high_token_overlap")
    elif max_token_overlap >= 0.25:
        flags.append("partial_token_overlap")
    if cand_zh_norm and cand_zh_norm in gt_zh_norms:
        flags.append("shared_zh_gloss")
    if best_gt_score is not None and (best_gt_score - score) <= small_gap:
        flags.append("small_gap_to_gt")

    # Dedupe while preserving order.
    flags = list(dict.fromkeys(flags))
    if any(flag in flags for flag in ("normalized_exact_match", "compact_exact_match", "shared_zh_gloss")):
        label = "likely_false_negative"
    elif any(flag in flags for flag in ("high_token_overlap", "token_containment", "shared_long_prefix", "acronym_overlap", "partial_token_overlap")):
        label = "very_similar_term"
    else:
        label = "clear_noise"
    return flags, label


def _candidate_record(
    rank: int,
    term_idx: int,
    score: float,
    bank: Sequence[str],
    term_meta: Dict[str, Dict[str, str]],
    gt_idx_set: set[int],
    gt_terms: Sequence[str],
    gt_zh_terms: Sequence[str],
    best_gt_score: Optional[float],
    small_gap: float,
) -> Dict[str, object]:
    term_key = bank[term_idx]
    entry = term_meta.get(term_key, {"term": term_key, "zh": ""})
    is_gt = term_idx in gt_idx_set
    flags: List[str] = []
    auto_label = "gt_term" if is_gt else "clear_noise"
    if not is_gt:
        flags, auto_label = _flag_candidate(
            candidate_term=entry.get("term", term_key),
            candidate_zh=entry.get("zh", ""),
            score=float(score),
            gt_terms=gt_terms,
            gt_zh_terms=gt_zh_terms,
            best_gt_score=best_gt_score,
            small_gap=small_gap,
        )
    return {
        "rank": rank,
        "term": entry.get("term", term_key),
        "term_key": term_key,
        "zh": entry.get("zh", ""),
        "score": round(float(score), 6),
        "is_gt": is_gt,
        "flags": flags,
        "auto_label": auto_label,
    }


def _top_non_gt(candidates: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    for cand in candidates:
        if not cand["is_gt"]:
            return cand
    return None


def _top_gt(candidates: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    for cand in candidates:
        if cand["is_gt"]:
            return cand
    return None


def _select_representatives(
    rows: Sequence[Dict[str, object]],
    limit_per_label: int,
) -> Dict[str, List[Dict[str, object]]]:
    grouped = {
        "likely_false_negative": [],
        "very_similar_term": [],
        "clear_noise": [],
    }
    ordered = sorted(
        rows,
        key=lambda r: (
            0 if "gt_missing_or_outranked" in r["subgroups"] else 1,
            -(r["top_non_gt_score"] or -1.0),
        ),
    )
    for row in ordered:
        label = row["boundary_label"]
        if label not in grouped:
            continue
        if len(grouped[label]) >= limit_per_label:
            continue
        grouped[label].append(row)
    return grouped


def _format_candidate_line(cand: Dict[str, object]) -> str:
    suffix: List[str] = []
    if cand["is_gt"]:
        suffix.append("GT")
    if cand["auto_label"] != "gt_term":
        suffix.append(str(cand["auto_label"]))
    if cand["flags"]:
        suffix.append(",".join(cand["flags"]))
    extra = f" [{' | '.join(suffix)}]" if suffix else ""
    zh = f" / {cand['zh']}" if cand["zh"] else ""
    return f"{cand['rank']:>2}. {cand['term']}{zh}  score={cand['score']:.6f}{extra}"


def write_markdown_report(
    path: str,
    model_path: str,
    acl_jsonl: str,
    output_dir: str,
    lower_tau: float,
    upper_tau: float,
    tau_ref: float,
    summary: Dict[str, object],
    grouped_examples: Dict[str, List[Dict[str, object]]],
) -> None:
    lines: List[str] = []
    lines.append("# ACL Boundary Audit for `tys70s0y`")
    lines.append("")
    lines.append(f"- Model: `{model_path}`")
    lines.append(f"- ACL JSONL: `{acl_jsonl}`")
    lines.append(f"- Output dir: `{output_dir}`")
    lines.append(f"- Boundary band: top non-GT score in `[{lower_tau:.2f}, {upper_tau:.2f}]`")
    lines.append(f"- Reference tau: `{tau_ref:.2f}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Total ACL chunks: `{summary['total_chunks']}`; with-term chunks: `{summary['with_term_chunks']}`."
    )
    lines.append(
        f"- Full dump rows: `{summary['full_dump_rows']}`; boundary rows: `{summary['boundary_rows']}`."
    )
    lines.append(
        f"- Boundary auto-label counts: likely false negative `{summary['label_counts'].get('likely_false_negative', 0)}`, "
        f"very similar `{summary['label_counts'].get('very_similar_term', 0)}`, "
        f"clear noise `{summary['label_counts'].get('clear_noise', 0)}`."
    )
    lines.append(
        f"- Boundary subgroup counts: gt-in-top10-near-tau `{summary['subgroup_counts'].get('gt_in_top10_near_tau', 0)}`, "
        f"gt-missing-or-outranked `{summary['subgroup_counts'].get('gt_missing_or_outranked', 0)}`, "
        f"other `{summary['subgroup_counts'].get('other_boundary', 0)}`."
    )
    lines.append("")

    title_map = {
        "likely_false_negative": "Likely False Negatives / Alias Collisions",
        "very_similar_term": "Very Similar Terms",
        "clear_noise": "Clear Noise",
    }
    for label in ("likely_false_negative", "very_similar_term", "clear_noise"):
        lines.append(f"## {title_map[label]}")
        lines.append("")
        examples = grouped_examples.get(label, [])
        if not examples:
            lines.append("No representative examples selected.")
            lines.append("")
            continue
        for idx, row in enumerate(examples, start=1):
            lines.append(
                f"### {idx}. `{row['chunk_id']}`  |  top_non_gt=`{row['top_non_gt_term']}` "
                f"({row['top_non_gt_score']:.6f})"
            )
            lines.append("")
            lines.append(f"- Subgroups: `{', '.join(row['subgroups'])}`")
            lines.append(f"- GT terms: `{', '.join(row['gt_terms']) or '(none)'}`")
            if row["gt_terms_zh"]:
                lines.append(f"- GT zh: `{', '.join([zh for zh in row['gt_terms_zh'] if zh])}`")
            if row["gt_best_score"] is not None:
                lines.append(
                    f"- Best GT score: `{row['gt_best_score']:.6f}`"
                    + (
                        f", GT top10 rank `{row['gt_top10_rank']}`"
                        if row["gt_top10_rank"] is not None
                        else ", GT not in top10"
                    )
                )
            lines.append(f"- Chunk text: `{row['chunk_src_text']}`")
            lines.append("- Top-10:")
            for cand in row["top10"]:
                lines.append(f"  - `{_format_candidate_line(cand)}`")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--acl_jsonl", required=True)
    parser.add_argument("--wiki_glossary", required=True)
    parser.add_argument("--gt_glossary", default="")
    parser.add_argument("--candidate_glossary", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gs_size", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lower_tau", type=float, default=0.75)
    parser.add_argument("--upper_tau", type=float, default=0.85)
    parser.add_argument("--tau_ref", type=float, default=0.80)
    parser.add_argument("--gt_near_margin", type=float, default=0.03)
    parser.add_argument("--small_gap", type=float, default=0.03)
    parser.add_argument("--report_examples_per_label", type=int, default=6)

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
    full_dump_path = os.path.join(args.output_dir, "acl6060_gs10000_top10_dump.jsonl")
    boundary_jsonl_path = os.path.join(args.output_dir, "acl6060_boundary_samples.jsonl")
    boundary_tsv_path = os.path.join(args.output_dir, "acl6060_boundary_samples.tsv")
    summary_path = os.path.join(args.output_dir, "acl6060_boundary_summary.json")
    report_path = os.path.join(args.output_dir, "acl6060_boundary_report.md")

    t0 = time.time()
    term_meta = _build_term_meta(args.wiki_glossary, args.candidate_glossary, args.gt_glossary)
    wiki_terms = [term for term, _ in _load_glossary_entries(args.wiki_glossary)]
    chunks, metas = load_acl_chunks_with_meta(args.acl_jsonl, term_meta)
    with_term_count = sum(1 for meta in metas if meta.has_term)
    _log(f"ACL chunks: total={len(metas)} with_term={with_term_count}")

    bank = build_bank(metas, wiki_terms, args.gs_size, term_meta)
    term_to_idx = {term: idx for idx, term in enumerate(bank)}
    _log(f"Bank size={len(bank)} (gs_size={args.gs_size})")

    retriever, text_encoder, tokenizer, feat_ext, _maxsim_score, device = build_model(args)

    _log("Encoding audio chunks...")
    speech_embs = encode_audio_chunks(chunks, retriever, feat_ext, device)
    _log(f"speech_embs={tuple(speech_embs.shape)}")

    _log("Encoding bank terms...")
    text_embs = encode_terms(bank, text_encoder, tokenizer, device)
    _log(f"text_embs={tuple(text_embs.shape)}")

    _log("Computing similarity matrix...")
    sim_np = _compute_sim_numpy_batched(
        speech_embs, text_embs, _maxsim_score, batch_rows=args.sim_batch_rows
    )
    _log(f"sim={tuple(sim_np.shape)}")

    gt_idx_per_chunk: List[List[int]] = []
    for meta in metas:
        gt_idx_per_chunk.append([term_to_idx[t] for t in meta.gt_terms_lower if t in term_to_idx])

    full_rows: List[Dict[str, object]] = []
    boundary_rows: List[Dict[str, object]] = []
    label_counts = {"likely_false_negative": 0, "very_similar_term": 0, "clear_noise": 0}
    subgroup_counts = {"gt_in_top10_near_tau": 0, "gt_missing_or_outranked": 0, "other_boundary": 0}

    with open(full_dump_path, "w", encoding="utf-8") as f_full, open(
        boundary_jsonl_path, "w", encoding="utf-8"
    ) as f_boundary:
        for idx, meta in enumerate(metas):
            row = sim_np[idx]
            top_idx, top_scores = _topk_indices_scores(row, TOP_K)
            gt_idx_set = set(gt_idx_per_chunk[idx])
            best_gt_score: Optional[float] = None
            if gt_idx_set:
                best_gt_score = float(np.max([row[g] for g in gt_idx_set]))
            top10 = [
                _candidate_record(
                    rank=rank,
                    term_idx=int(term_idx),
                    score=float(score),
                    bank=bank,
                    term_meta=term_meta,
                    gt_idx_set=gt_idx_set,
                    gt_terms=meta.gt_terms,
                    gt_zh_terms=meta.gt_terms_zh,
                    best_gt_score=best_gt_score,
                    small_gap=args.small_gap,
                )
                for rank, (term_idx, score) in enumerate(zip(top_idx, top_scores), start=1)
            ]

            top_non_gt = _top_non_gt(top10)
            top_gt = _top_gt(top10)
            gt_top10_rank = int(top_gt["rank"]) if top_gt else None
            gt_top10_score = float(top_gt["score"]) if top_gt else None

            base_row: Dict[str, object] = {
                "chunk_id": meta.chunk_id,
                "utter_id": meta.utter_id,
                "chunk_idx": meta.chunk_idx,
                "audio_path": meta.audio_path,
                "chunk_src_text": meta.chunk_src_text,
                "gt_terms": meta.gt_terms,
                "gt_terms_zh": [zh for zh in meta.gt_terms_zh if zh],
                "has_term": meta.has_term,
                "top10": top10,
                "gt_best_score": round(best_gt_score, 6) if best_gt_score is not None else None,
                "gt_in_top10": gt_top10_rank is not None,
                "gt_top10_rank": gt_top10_rank,
                "gt_top10_score": gt_top10_score,
                "top_non_gt_term": top_non_gt["term"] if top_non_gt else None,
                "top_non_gt_term_key": top_non_gt["term_key"] if top_non_gt else None,
                "top_non_gt_score": float(top_non_gt["score"]) if top_non_gt else None,
                "top_non_gt_label": top_non_gt["auto_label"] if top_non_gt else None,
                "top_non_gt_flags": top_non_gt["flags"] if top_non_gt else [],
            }
            full_rows.append(base_row)
            f_full.write(json.dumps(base_row, ensure_ascii=False) + "\n")

            if not meta.has_term or top_non_gt is None:
                continue
            top_non_gt_score = float(top_non_gt["score"])
            if not (args.lower_tau <= top_non_gt_score <= args.upper_tau):
                continue

            subgroups: List[str] = []
            if gt_top10_rank is not None and gt_top10_score is not None:
                if gt_top10_score <= args.tau_ref + args.gt_near_margin:
                    subgroups.append("gt_in_top10_near_tau")
            if gt_top10_rank is None or (best_gt_score is not None and top_non_gt_score >= best_gt_score):
                subgroups.append("gt_missing_or_outranked")
            if not subgroups:
                subgroups.append("other_boundary")

            boundary_row = dict(base_row)
            boundary_row["subgroups"] = subgroups
            boundary_row["boundary_label"] = top_non_gt["auto_label"]
            boundary_row["boundary_score_band"] = [args.lower_tau, args.upper_tau]
            boundary_row["tau_ref"] = args.tau_ref
            boundary_rows.append(boundary_row)
            f_boundary.write(json.dumps(boundary_row, ensure_ascii=False) + "\n")

            label_counts[top_non_gt["auto_label"]] += 1
            for subgroup in subgroups:
                subgroup_counts[subgroup] += 1

    # TSV summary for quick filtering in spreadsheets / shell.
    with open(boundary_tsv_path, "w", encoding="utf-8") as f:
        header = [
            "chunk_id",
            "chunk_src_text",
            "gt_terms",
            "gt_terms_zh",
            "gt_best_score",
            "gt_in_top10",
            "gt_top10_rank",
            "gt_top10_score",
            "top_non_gt_term",
            "top_non_gt_score",
            "top_non_gt_label",
            "top_non_gt_flags",
            "subgroups",
        ]
        f.write("\t".join(header) + "\n")
        for row in boundary_rows:
            values = [
                row["chunk_id"],
                row["chunk_src_text"].replace("\t", " "),
                "|".join(row["gt_terms"]),
                "|".join(row["gt_terms_zh"]),
                "" if row["gt_best_score"] is None else f"{row['gt_best_score']:.6f}",
                str(row["gt_in_top10"]),
                "" if row["gt_top10_rank"] is None else str(row["gt_top10_rank"]),
                "" if row["gt_top10_score"] is None else f"{row['gt_top10_score']:.6f}",
                row["top_non_gt_term"] or "",
                "" if row["top_non_gt_score"] is None else f"{row['top_non_gt_score']:.6f}",
                row["top_non_gt_label"] or "",
                "|".join(row["top_non_gt_flags"]),
                "|".join(row["subgroups"]),
            ]
            f.write("\t".join(values) + "\n")

    grouped_examples = _select_representatives(
        boundary_rows,
        limit_per_label=args.report_examples_per_label,
    )
    summary = {
        "model_path": args.model_path,
        "acl_jsonl": args.acl_jsonl,
        "output_dir": args.output_dir,
        "elapsed_seconds": round(time.time() - t0, 2),
        "total_chunks": len(metas),
        "with_term_chunks": with_term_count,
        "bank_size": len(bank),
        "full_dump_rows": len(full_rows),
        "boundary_rows": len(boundary_rows),
        "label_counts": label_counts,
        "subgroup_counts": subgroup_counts,
        "paths": {
            "full_dump_jsonl": full_dump_path,
            "boundary_jsonl": boundary_jsonl_path,
            "boundary_tsv": boundary_tsv_path,
            "summary_json": summary_path,
            "report_md": report_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_markdown_report(
        path=report_path,
        model_path=args.model_path,
        acl_jsonl=args.acl_jsonl,
        output_dir=args.output_dir,
        lower_tau=args.lower_tau,
        upper_tau=args.upper_tau,
        tau_ref=args.tau_ref,
        summary=summary,
        grouped_examples=grouped_examples,
    )

    _log(f"Done in {time.time() - t0:.1f}s")
    _log(f"Full dump: {full_dump_path}")
    _log(f"Boundary subset: {boundary_jsonl_path}")
    _log(f"Boundary TSV: {boundary_tsv_path}")
    _log(f"Summary: {summary_path}")
    _log(f"Report: {report_path}")


if __name__ == "__main__":
    main()
