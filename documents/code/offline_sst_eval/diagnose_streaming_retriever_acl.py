#!/usr/bin/env python3
"""Diagnose ACL6060 streaming RAG failures at sentence and term-map level.

The online SimulEval run produces one full-paper prediction plus a runtime
JSONL containing each vLLM call's final term_map.  This script aligns those
term_maps back to ACL sentence intervals, then separates:

  * retriever recall on source-side terms,
  * retriever precision / no-term noise after tau filtering,
  * SLM adoption when a retrieved term was actually shown in term_map.

Sentence-level ``no_term_avg_retrieved`` reflects deduped keys overlapping
each SimulEval sentence (what the LLM pipeline ultimately saw).  For an
apples-to-apples comparison with offline ACL dev ``noterm_noise@topK_tau*``
metrics, see ``vllm_no_term_acl_chunk_*``: per vLLM RAG call, restricted to
calls whose encoded audio window aligns with an ACL JSONL no-term chunk, with
the same top-K and tau counting rule as training eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_DIR = Path(
    "/mnt/gemini/data2/jiaxuanluo/acl_perpaper_lm1to4_raw1k10k_sner_tcmrag_tau0p75_aries"
)
DEFAULT_ACL_DEV_JSONL = Path(
    "/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/"
    "acl6060_dev_dataset.jsonl"
)
# Must match documents/code/data_pre/acl/prepare_acl6060_dev_dataset.py
ACL_DEV_CHUNK_SEC = 1.92
ACL_DEV_STRIDE_SEC = 0.96
TARGET_LANG = "zh"


@dataclass(frozen=True)
class GlossaryTerm:
    key: str
    term: str
    translation: str


@dataclass
class RuntimeCall:
    kind: str
    trigger: str
    segment_idx: int
    start_sec: float
    end_sec: float
    references: List[Dict[str, Any]]
    encoded_start_sec: float
    encoded_end_sec: float
    lookback_sec: float


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _source_contains(source_text: str, term: str) -> bool:
    source_norm = _normalise_space(source_text).casefold()
    term_norm = _normalise_space(term).casefold()
    if not source_norm or not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        return re.search(pattern, source_norm) is not None
    return term_norm in source_norm


def _text_contains(text: str, needle: str) -> bool:
    text_norm = _normalise_space(text)
    needle_norm = _normalise_space(needle)
    return bool(needle_norm) and needle_norm in text_norm


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_glossary(path: Path) -> List[GlossaryTerm]:
    data = _load_json(path)
    entries: Iterable[Any]
    if isinstance(data, dict):
        entries = data.values()
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unsupported glossary format: {path}")

    out: List[GlossaryTerm] = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        term = _normalise_space(entry.get("term") or entry.get("source") or "")
        translations = entry.get("target_translations")
        translation = ""
        if isinstance(translations, dict):
            translation = _normalise_space(translations.get(TARGET_LANG) or "")
        if not translation:
            translation = _normalise_space(
                entry.get("translation")
                or entry.get("target_translation")
                or entry.get(TARGET_LANG)
                or ""
            )
        if not term or not translation:
            continue
        key = (entry.get("key") or term).strip().casefold()
        dedup_key = (key, translation)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.append(GlossaryTerm(key=key, term=term, translation=translation))
    return out


def _infer_glossary_path(eval_dir: Path, paper_id: str) -> Path:
    name = eval_dir.name
    if "gglossary_acl6060_gt_union_gs10000" in name:
        return REPO_ROOT / "retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"
    if "gglossary_acl6060_gt_union_gs1000" in name:
        return REPO_ROOT / "retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs1000.json"
    if "gextracted_glossary__" in name:
        return (
            REPO_ROOT
            / "documents/data/data_pre/extracted_glossaries_by_paper"
            / f"extracted_glossary__{paper_id}.json"
        )
    raise ValueError(f"Could not infer glossary from eval dir name: {name}")


def _parse_eval_dir_metadata(eval_dir: Path) -> Dict[str, str]:
    name = eval_dir.name
    lm_m = re.search(r"_lm([^_]+)_", name)
    paper_m = re.search(r"_pp(2022\.acl-long\.\d+)$", name)
    if "gglossary_acl6060_gt_union_gs10000" in name:
        regime = "gs10k"
    elif "gglossary_acl6060_gt_union_gs1000" in name:
        regime = "gs1k"
    elif "gextracted_glossary__" in name:
        regime = "raw"
    else:
        regime = "unknown"
    return {
        "lm": lm_m.group(1) if lm_m else "",
        "paper": paper_m.group(1) if paper_m else "",
        "regime": regime,
    }


def _read_eval_result(eval_dir: Path) -> Dict[str, str]:
    path = eval_dir / "eval_results.tsv"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty eval_results.tsv: {path}")
    return rows[-1]


def _candidate_eval_dirs(base_dir: Path) -> List[Path]:
    root = base_dir / "zh" if (base_dir / "zh").is_dir() else base_dir
    return sorted(p.parent for p in root.glob("**/eval_results.tsv"))


def _select_eval_dir(
    base_dir: Path,
    eval_dir: Optional[Path],
    paper: str,
    lm: str,
    regime: str,
    metric: str,
) -> Tuple[Path, Dict[str, str], Dict[str, str]]:
    if eval_dir:
        meta = _parse_eval_dir_metadata(eval_dir)
        return eval_dir, meta, _read_eval_result(eval_dir)

    candidates: List[Tuple[float, Path, Dict[str, str], Dict[str, str]]] = []
    for d in _candidate_eval_dirs(base_dir):
        meta = _parse_eval_dir_metadata(d)
        if paper and meta["paper"] != paper:
            continue
        if lm and meta["lm"] != lm:
            continue
        if regime and meta["regime"] != regime:
            continue
        row = _read_eval_result(d)
        if metric not in row:
            raise KeyError(f"Metric {metric!r} not found in {d / 'eval_results.tsv'}")
        try:
            value = float(row[metric])
        except (TypeError, ValueError):
            continue
        candidates.append((value, d, meta, row))
    if not candidates:
        raise FileNotFoundError(
            f"No eval dirs under {base_dir} for paper={paper or '*'} lm={lm or '*'} "
            f"regime={regime or '*'}"
        )
    candidates.sort(key=lambda x: x[0])
    _value, d, meta, row = candidates[0]
    return d, meta, row


def _paper_inputs_dir(base_dir: Path, eval_dir: Path) -> Path:
    root = base_dir / "zh" if (base_dir / "zh").is_dir() else eval_dir.parent
    return root / "__paper_inputs__" / "lists"


def _load_audio_intervals(audio_yaml: Path) -> List[Tuple[float, float]]:
    entries = yaml.safe_load(audio_yaml.read_text(encoding="utf-8"))
    intervals: List[Tuple[float, float]] = []
    cursor = 0.0
    for item in entries:
        if not isinstance(item, dict):
            continue
        duration = float(item.get("duration") or 0.0)
        start = float(item.get("offset", cursor))
        end = start + duration
        intervals.append((start, end))
        cursor = end
    return intervals


def _find_runtime_jsonl(eval_dir: Path) -> Path:
    candidates = sorted(eval_dir.glob("runtime_omni_vllm_maxsim_rag_*.jsonl"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one runtime_omni_vllm_maxsim_rag_*.jsonl in {eval_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _load_runtime_calls(runtime_path: Path, segment_sec: float) -> Tuple[List[RuntimeCall], List[RuntimeCall]]:
    final_calls: List[RuntimeCall] = []
    window_calls: List[RuntimeCall] = []
    for rec in _iter_jsonl(runtime_path):
        rec_type = rec.get("type")
        if rec_type not in {"rag", "rag_window"}:
            continue
        end_sec = float(rec.get("rag_audio_duration") or 0.0)
        start_sec = max(0.0, end_sec - segment_sec)
        current_start_sec = float(rec.get("current_start_sec", start_sec))
        current_end_sec = float(rec.get("current_end_sec", end_sec))
        lookback_sec = float(rec.get("lookback_sec", 0.0))
        encoded_start_sec = max(0.0, current_start_sec - lookback_sec)
        call = RuntimeCall(
            kind=str(rec_type),
            trigger=str(rec.get("trigger") or ("vllm_final" if rec_type == "rag" else "")),
            segment_idx=int(rec.get("segment_idx", -1)),
            start_sec=current_start_sec,
            end_sec=current_end_sec,
            references=list(rec.get("references") or []),
            encoded_start_sec=encoded_start_sec,
            encoded_end_sec=current_end_sec,
            lookback_sec=lookback_sec,
        )
        if rec_type == "rag":
            final_calls.append(call)
        else:
            window_calls.append(call)
    if not final_calls:
        # Newer logs may have only rag_window; use vLLM-final windows as final maps.
        final_calls = [c for c in window_calls if c.trigger == "vllm_final"]
    return final_calls, window_calls


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and a_end > b_start


def _best_refs_for_sentence(
    calls: Sequence[RuntimeCall],
    sent_start: float,
    sent_end: float,
) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for call in calls:
        if not _overlaps(call.start_sec, call.end_sec, sent_start, sent_end):
            continue
        for ref in call.references:
            term = _normalise_space(ref.get("term") or "")
            key = _normalise_space(ref.get("key") or term).casefold()
            if not key:
                continue
            score = float(ref.get("score") or 0.0)
            current = by_key.get(key)
            if current is None or score > float(current.get("score") or 0.0):
                by_key[key] = {
                    "key": key,
                    "term": term,
                    "translation": _normalise_space(ref.get("translation") or ""),
                    "score": score,
                }
    return by_key


def _nearby_calls(
    calls: Sequence[RuntimeCall],
    window_calls: Sequence[RuntimeCall],
    sent_start: float,
    sent_end: float,
    limit: int,
) -> List[Dict[str, Any]]:
    mid = (sent_start + sent_end) / 2.0
    overlapping = [
        c for c in calls
        if _overlaps(c.start_sec, c.end_sec, sent_start, sent_end)
    ]
    if overlapping:
        ranked = sorted(overlapping, key=lambda c: (c.start_sec, c.end_sec, c.segment_idx))
    else:
        ranked = sorted(calls, key=lambda c: abs(c.end_sec - mid))[:limit]
    out: List[Dict[str, Any]] = []
    timeline_by_segment = {
        w.segment_idx: w for w in window_calls if w.trigger == "vllm_timeline"
    }
    for call in ranked:
        timeline = timeline_by_segment.get(call.segment_idx)
        current_start = timeline.start_sec if timeline else call.start_sec
        current_end = timeline.end_sec if timeline else call.end_sec
        encoded_start = timeline.encoded_start_sec if timeline else call.encoded_start_sec
        encoded_end = timeline.encoded_end_sec if timeline else call.encoded_end_sec
        lookback = timeline.lookback_sec if timeline else call.lookback_sec
        previous = [
            w for w in window_calls
            if w.trigger != "vllm_timeline"
            if w.end_sec <= call.end_sec and not (w.kind == "rag_window" and w.trigger == "vllm_final" and w.end_sec == call.end_sec)
        ][-2:]
        out.append({
            "segment_idx": call.segment_idx,
            "current_sec": [round(current_start, 2), round(current_end, 2)],
            "encoded_sec": [round(encoded_start, 2), round(encoded_end, 2)],
            "lookback_sec": round(lookback, 2),
            "overlaps_sentence": _overlaps(current_start, current_end, sent_start, sent_end),
            "final_term_map_count": len(call.references),
            "final_term_map": _refs_summary(call.references, len(call.references)),
            "previous_stride_windows_available": bool(previous),
            "previous_stride_windows": [
                {
                    "trigger": p.trigger,
                    "window_sec": [round(p.start_sec, 2), round(p.end_sec, 2)],
                    "term_map_count": len(p.references),
                    "term_map": _refs_summary(p.references, len(p.references)),
                }
                for p in previous
            ],
        })
    return out


def _refs_summary(refs: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    out = []
    for ref in refs[:limit]:
        out.append({
            "term": _normalise_space(ref.get("term") or ""),
            "translation": _normalise_space(ref.get("translation") or ""),
            "score": round(float(ref.get("score") or 0.0), 4),
        })
    return out


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _runtime_overhead_metrics(runtime_path: Path) -> Dict[str, Any]:
    """Summarize vLLM-blocking RAG overhead only.

    Intermediate stride retrieves are counted separately and excluded from
    `rag_blocking_*` unless the runtime explicitly marked them as blocking.
    """
    blocking: List[float] = []
    nonblocking: List[float] = []
    for rec in _iter_jsonl(runtime_path):
        if rec.get("type") not in {"rag", "rag_window"}:
            continue
        sec_raw = rec.get("rag_blocking_sec")
        if sec_raw is None:
            sec_raw = rec.get("rag_sec")
        try:
            sec = float(sec_raw)
        except (TypeError, ValueError):
            continue
        if rec.get("type") == "rag":
            blocking.append(sec)
            continue
        if bool(rec.get("blocking_for_vllm", False)):
            blocking.append(sec)
        else:
            nonblocking.append(sec)

    return {
        "rag_blocking_call_count": len(blocking),
        "rag_blocking_total_sec": sum(blocking),
        "rag_blocking_avg_sec": _safe_div(sum(blocking), len(blocking)),
        "rag_blocking_p50_sec": _percentile(blocking, 0.50),
        "rag_blocking_p95_sec": _percentile(blocking, 0.95),
        "rag_nonblocking_call_count": len(nonblocking),
        "rag_nonblocking_total_sec": sum(nonblocking),
        "rag_nonblocking_avg_sec": _safe_div(sum(nonblocking), len(nonblocking)),
    }


def _parse_rag_tau_from_eval_dir(eval_dir: Path) -> Optional[float]:
    m = re.search(r"_th([\d.]+)_", eval_dir.name)
    if not m:
        return None
    return float(m.group(1))


def _parse_rag_topk_from_eval_dir(eval_dir: Path) -> Optional[int]:
    m = re.search(r"_k(\d+)_", eval_dir.name)
    if not m:
        return None
    return int(m.group(1))


def _jsonl_row_has_acl_term(row: Dict[str, Any]) -> bool:
    key = str(row.get("term_key", "") or row.get("term", "") or "").strip()
    return bool(key)


def _overlap_interval_len(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)


def _window_aligns_acl_chunk(
    w0: float,
    w1: float,
    c0: float,
    c1: float,
    min_frac: float,
) -> bool:
    """True if overlap is at least min_frac * min(duration(w), duration(c))."""
    d_w = max(0.0, w1 - w0)
    d_c = max(0.0, c1 - c0)
    if d_w <= 0.0 or d_c <= 0.0:
        return False
    ov = _overlap_interval_len(w0, w1, c0, c1)
    return ov >= min_frac * min(d_w, d_c)


def _load_acl_no_term_chunk_intervals(
    jsonl_path: Path,
    paper_id: str,
    audio_end_sec: float,
    chunk_sec: float,
    stride_sec: float,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """Intervals [start, end) for ACL dev chunks with no gold term (JSONL rows).

    Matches ``prepare_acl6060_dev_dataset.save_jsonl``: no-term chunks are rows
    with empty ``term`` / ``term_key``; with-term chunks emit one row per term.
    """
    by_chunk: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    rows_paper = 0
    for row in _iter_jsonl(jsonl_path):
        if str(row.get("utter_id", "")).strip() != paper_id:
            continue
        rows_paper += 1
        try:
            ci = int(row.get("chunk_idx", -1))
        except (TypeError, ValueError):
            continue
        if ci < 0:
            continue
        by_chunk[ci].append(row)

    intervals: List[Tuple[float, float]] = []
    no_term_chunks = 0
    with_term_chunks = 0
    for ci in sorted(by_chunk.keys()):
        rows = by_chunk[ci]
        if any(_jsonl_row_has_acl_term(r) for r in rows):
            with_term_chunks += 1
            continue
        no_term_chunks += 1
        c0 = ci * stride_sec
        c1 = min(c0 + chunk_sec, audio_end_sec)
        if c0 < audio_end_sec and c1 > c0:
            intervals.append((c0, c1))

    meta = {
        "acl_dev_jsonl_rows_for_paper": rows_paper,
        "acl_dev_distinct_chunks": len(by_chunk),
        "acl_dev_no_term_chunks": no_term_chunks,
        "acl_dev_with_term_chunks": with_term_chunks,
    }
    return intervals, meta


def _kept_refs_topk_tau(
    references: Sequence[Dict[str, Any]],
    tau: float,
    topk: int,
) -> int:
    """Count refs with score>=tau among top-``topk`` by score (offline eval style).

    Mirrors ``_compute_noterm_noise`` in ``qwen3_glossary_neg_train.py``: per
    query, take top-K scores, count how many pass the threshold.
    """
    if not references or topk <= 0:
        return 0
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ref in references:
        try:
            s = float(ref.get("score") or 0.0)
        except (TypeError, ValueError):
            s = 0.0
        scored.append((s, ref))
    scored.sort(key=lambda x: x[0], reverse=True)
    k = min(topk, len(scored))
    return sum(1 for s, _ in scored[:k] if s >= tau)


def _vllm_acl_no_term_chunk_metrics(
    final_calls: Sequence[RuntimeCall],
    no_term_intervals: Sequence[Tuple[float, float]],
    tau: float,
    topk: int,
    overlap_min_frac: float,
) -> Dict[str, Any]:
    """Per vLLM (``rag``) call: restrict to calls aligned to ACL no-term chunks."""
    if not no_term_intervals:
        return {
            "vllm_no_term_acl_chunk_calls": 0,
            "vllm_no_term_acl_chunk_avg_kept_topk_tau": 0.0,
            "vllm_no_term_acl_chunk_p50_kept": 0.0,
            "vllm_no_term_acl_chunk_calls_with_kept_gt0": 0,
            "vllm_no_term_acl_chunk_noise_call_rate": 0.0,
            "vllm_no_term_acl_chunk_tau": tau,
            "vllm_no_term_acl_chunk_topk": topk,
            "vllm_no_term_acl_chunk_overlap_min_frac": overlap_min_frac,
        }

    kept_list: List[float] = []
    calls_matched = 0
    with_kept = 0
    seen_call: Set[Tuple[int, float, float]] = set()
    for call in final_calls:
        w0 = float(call.encoded_start_sec)
        w1 = float(call.encoded_end_sec)
        dedup_key = (call.segment_idx, round(w0, 4), round(w1, 4))
        if dedup_key in seen_call:
            continue
        seen_call.add(dedup_key)
        aligned = any(
            _window_aligns_acl_chunk(w0, w1, c0, c1, overlap_min_frac)
            for c0, c1 in no_term_intervals
        )
        if not aligned:
            continue
        calls_matched += 1
        k = _kept_refs_topk_tau(call.references, tau, topk)
        kept_list.append(float(k))
        if k > 0:
            with_kept += 1

    return {
        "vllm_no_term_acl_chunk_calls": calls_matched,
        "vllm_no_term_acl_chunk_avg_kept_topk_tau": _safe_div(sum(kept_list), len(kept_list)),
        "vllm_no_term_acl_chunk_p50_kept": _percentile(kept_list, 0.50),
        "vllm_no_term_acl_chunk_calls_with_kept_gt0": with_kept,
        "vllm_no_term_acl_chunk_noise_call_rate": _safe_div(with_kept, calls_matched),
        "vllm_no_term_acl_chunk_tau": tau,
        "vllm_no_term_acl_chunk_topk": topk,
        "vllm_no_term_acl_chunk_overlap_min_frac": overlap_min_frac,
    }


def diagnose(
    base_dir: Path,
    eval_dir: Path,
    meta: Dict[str, str],
    eval_row: Dict[str, str],
    sample_limit: int,
    enable_acl_chunk_metrics: bool = True,
    acl_dev_jsonl: Optional[Path] = None,
    acl_chunk_sec: float = ACL_DEV_CHUNK_SEC,
    acl_stride_sec: float = ACL_DEV_STRIDE_SEC,
    no_term_overlap_frac: float = 0.5,
    rag_tau: Optional[float] = None,
    rag_topk: Optional[int] = None,
) -> Dict[str, Any]:
    paper_id = meta["paper"]
    lm = float(meta["lm"])
    segment_sec = 0.96 * lm

    inputs_dir = _paper_inputs_dir(base_dir, eval_dir)
    source_path = inputs_dir / f"dev.source_text.en__{paper_id}.txt"
    ref_path = inputs_dir / f"dev.ref.zh__{paper_id}.txt"
    audio_yaml = inputs_dir / f"audio__{paper_id}.yaml"
    term_adoption_path = eval_dir / "term_adoption.json"
    runtime_path = _find_runtime_jsonl(eval_dir)
    glossary_path = _infer_glossary_path(eval_dir, paper_id)

    sources = _read_lines(source_path)
    refs = _read_lines(ref_path)
    intervals = _load_audio_intervals(audio_yaml)
    adoption = _load_json(term_adoption_path) if term_adoption_path.is_file() else {}
    hyp_by_idx = {
        int(row.get("index", idx)): str(row.get("hypothesis") or "")
        for idx, row in enumerate(adoption.get("sentences") or [])
        if isinstance(row, dict)
    }
    glossary_terms = _load_glossary(glossary_path)
    terms_by_key = {t.key: t for t in glossary_terms}
    final_calls, window_calls = _load_runtime_calls(runtime_path, segment_sec)
    overhead = _runtime_overhead_metrics(runtime_path)

    if not enable_acl_chunk_metrics:
        acl_path: Optional[Path] = None
    elif acl_dev_jsonl is not None:
        acl_path = acl_dev_jsonl
    else:
        acl_path = DEFAULT_ACL_DEV_JSONL

    tau_used: float
    tau_source: str
    if rag_tau is not None:
        tau_used = float(rag_tau)
        tau_source = "cli"
    else:
        parsed_tau = _parse_rag_tau_from_eval_dir(eval_dir)
        if parsed_tau is not None:
            tau_used = parsed_tau
            tau_source = "eval_dir"
        else:
            tau_used = 0.0
            tau_source = "default_0_missing_th_in_name"

    topk_used = rag_topk if rag_topk is not None else (_parse_rag_topk_from_eval_dir(eval_dir) or 10)

    audio_end_sec = float(intervals[-1][1]) if intervals else 0.0
    acl_no_term_chunk_diag: Dict[str, Any] = {
        "chunk_sec": acl_chunk_sec,
        "stride_sec": acl_stride_sec,
        "encoded_window_note": (
            "Each vLLM RAG call uses [encoded_start_sec, encoded_end_sec); aligned to an ACL "
            "no-term chunk when overlap >= overlap_min_frac * min(d_encoded, d_chunk)."
        ),
    }
    no_term_ivals: List[Tuple[float, float]] = []
    if not enable_acl_chunk_metrics:
        acl_no_term_chunk_diag["skipped_reason"] = "disabled_enable_acl_chunk_metrics_false"
    elif acl_path is None:
        acl_no_term_chunk_diag["skipped_reason"] = "internal_no_acl_path"
    elif not acl_path.is_file():
        acl_no_term_chunk_diag["acl_dev_jsonl"] = str(acl_path)
        acl_no_term_chunk_diag["skipped_reason"] = f"acl_dev_jsonl_not_found:{acl_path}"
    else:
        acl_no_term_chunk_diag["acl_dev_jsonl"] = str(acl_path)
        no_term_ivals, chunk_counts = _load_acl_no_term_chunk_intervals(
            acl_path,
            paper_id,
            audio_end_sec=audio_end_sec,
            chunk_sec=acl_chunk_sec,
            stride_sec=acl_stride_sec,
        )
        acl_no_term_chunk_diag.update(chunk_counts)
        acl_no_term_chunk_diag["no_term_intervals_count"] = len(no_term_ivals)

    acl_no_term_chunk_diag["rag_tau_used"] = tau_used
    acl_no_term_chunk_diag["rag_tau_source"] = tau_source
    acl_no_term_chunk_diag["rag_topk_used"] = topk_used
    acl_no_term_chunk_diag["overlap_min_frac"] = no_term_overlap_frac

    sentence_rows: List[Dict[str, Any]] = []
    totals = {
        "source_terms": 0,
        "source_ref_terms": 0,
        "retrieved_terms": 0,
        "retrieved_source_terms": 0,
        "retrieved_source_ref_terms": 0,
        "retrieved_adopted_source_terms": 0,
        "source_adopted_terms": 0,
        "no_term_sentences": 0,
        "no_term_retrieved_terms": 0,
        "no_term_with_noise": 0,
        "with_term_sentences": 0,
        "with_retrieval_sentences": 0,
    }

    for idx, source in enumerate(sources):
        ref_text = refs[idx] if idx < len(refs) else ""
        hyp_text = hyp_by_idx.get(idx, "")
        sent_start, sent_end = intervals[idx] if idx < len(intervals) else (idx * segment_sec, (idx + 1) * segment_sec)

        source_terms: Dict[str, GlossaryTerm] = {}
        source_ref_terms: Dict[str, GlossaryTerm] = {}
        for term in glossary_terms:
            if _source_contains(source, term.term):
                source_terms[term.key] = term
                if _text_contains(ref_text, term.translation):
                    source_ref_terms[term.key] = term

        retrieved = _best_refs_for_sentence(final_calls, sent_start, sent_end)
        retrieved_keys = set(retrieved)
        source_keys = set(source_terms)
        source_ref_keys = set(source_ref_terms)
        hit_source = retrieved_keys & source_keys
        hit_source_ref = retrieved_keys & source_ref_keys
        false_positive_keys = retrieved_keys - source_keys
        missed_source_keys = source_keys - retrieved_keys

        adopted_source_keys = {
            k for k, term in source_terms.items()
            if _text_contains(hyp_text, term.translation)
        }
        adopted_retrieved_source_keys = {
            k for k in hit_source
            if k in terms_by_key and _text_contains(hyp_text, terms_by_key[k].translation)
        }

        totals["source_terms"] += len(source_terms)
        totals["source_ref_terms"] += len(source_ref_terms)
        totals["retrieved_terms"] += len(retrieved)
        totals["retrieved_source_terms"] += len(hit_source)
        totals["retrieved_source_ref_terms"] += len(hit_source_ref)
        totals["retrieved_adopted_source_terms"] += len(adopted_retrieved_source_keys)
        totals["source_adopted_terms"] += len(adopted_source_keys)
        if source_terms:
            totals["with_term_sentences"] += 1
        else:
            totals["no_term_sentences"] += 1
            totals["no_term_retrieved_terms"] += len(retrieved)
            if retrieved:
                totals["no_term_with_noise"] += 1
        if retrieved:
            totals["with_retrieval_sentences"] += 1

        sentence_rows.append({
            "index": idx,
            "time_sec": [round(sent_start, 3), round(sent_end, 3)],
            "source": source,
            "reference": ref_text,
            "hypothesis": hyp_text,
            "source_terms": [
                {"term": t.term, "translation": t.translation}
                for t in sorted(source_terms.values(), key=lambda x: x.term.casefold())
            ],
            "source_ref_terms": [
                {"term": t.term, "translation": t.translation}
                for t in sorted(source_ref_terms.values(), key=lambda x: x.term.casefold())
            ],
            "retrieved_terms": sorted(retrieved.values(), key=lambda x: -float(x.get("score") or 0.0)),
            "retrieved_source_terms": sorted(hit_source),
            "retrieved_source_term_details": sorted(
                [retrieved[k] for k in hit_source],
                key=lambda x: -float(x.get("score") or 0.0),
            ),
            "missed_source_terms": [
                {"term": source_terms[k].term, "translation": source_terms[k].translation}
                for k in sorted(missed_source_keys)
            ],
            "false_positive_terms": sorted(
                [retrieved[k] for k in false_positive_keys],
                key=lambda x: -float(x.get("score") or 0.0),
            ),
            "adopted_source_terms": sorted(adopted_source_keys),
            "adopted_retrieved_source_terms": sorted(adopted_retrieved_source_keys),
            "term_recall": _safe_div(len(hit_source), len(source_terms)),
            "term_precision": _safe_div(len(hit_source), len(retrieved)),
            "adoption_given_retrieved": _safe_div(len(adopted_retrieved_source_keys), len(hit_source)),
            "nearby_calls": _nearby_calls(final_calls, window_calls, sent_start, sent_end, limit=4),
        })

    with_terms = [r for r in sentence_rows if r["source_terms"]]
    with_retrieval = [r for r in sentence_rows if r["retrieved_terms"]]
    metrics = {
        "term_recall_source_micro": _safe_div(totals["retrieved_source_terms"], totals["source_terms"]),
        "term_precision_source_micro": _safe_div(totals["retrieved_source_terms"], totals["retrieved_terms"]),
        "term_recall_source_ref_micro": _safe_div(totals["retrieved_source_ref_terms"], totals["source_ref_terms"]),
        "term_recall_source_macro": _safe_div(sum(float(r["term_recall"]) for r in with_terms), len(with_terms)),
        "term_precision_macro_retrieved": _safe_div(sum(float(r["term_precision"]) for r in with_retrieval), len(with_retrieval)),
        "adoption_given_retrieved_source_micro": _safe_div(
            totals["retrieved_adopted_source_terms"], totals["retrieved_source_terms"]
        ),
        "adoption_source_micro": _safe_div(totals["source_adopted_terms"], totals["source_terms"]),
        "no_term_avg_retrieved": _safe_div(totals["no_term_retrieved_terms"], totals["no_term_sentences"]),
        "no_term_noise_sentence_rate": _safe_div(totals["no_term_with_noise"], totals["no_term_sentences"]),
        "avg_retrieved_per_sentence": _safe_div(totals["retrieved_terms"], len(sentence_rows)),
        "sentences": len(sentence_rows),
        **totals,
    }

    if enable_acl_chunk_metrics and acl_path is not None and acl_path.is_file():
        metrics.update(
            _vllm_acl_no_term_chunk_metrics(
                final_calls,
                no_term_ivals,
                tau_used,
                topk_used,
                no_term_overlap_frac,
            )
        )
    else:
        metrics.update({
            "vllm_no_term_acl_chunk_calls": None,
            "vllm_no_term_acl_chunk_avg_kept_topk_tau": None,
            "vllm_no_term_acl_chunk_p50_kept": None,
            "vllm_no_term_acl_chunk_calls_with_kept_gt0": None,
            "vllm_no_term_acl_chunk_noise_call_rate": None,
            "vllm_no_term_acl_chunk_tau": tau_used,
            "vllm_no_term_acl_chunk_topk": topk_used,
            "vllm_no_term_acl_chunk_overlap_min_frac": no_term_overlap_frac,
        })

    bad_with_terms = sorted(
        with_terms,
        key=lambda r: (
            float(r["term_recall"]),
            float(r["adoption_given_retrieved"]) if r["retrieved_source_terms"] else 0.0,
            -len(r["false_positive_terms"]),
        ),
    )
    noisy_no_terms = sorted(
        [r for r in sentence_rows if not r["source_terms"] and r["retrieved_terms"]],
        key=lambda r: (-len(r["retrieved_terms"]), r["index"]),
    )
    samples = {
        "worst_with_term_sentences": bad_with_terms[:sample_limit],
        "no_term_noise_sentences": noisy_no_terms[:sample_limit],
    }

    return {
        "eval_dir": str(eval_dir),
        "runtime_jsonl": str(runtime_path),
        "glossary_path": str(glossary_path),
        "paper": paper_id,
        "lm": meta["lm"],
        "regime": meta["regime"],
        "segment_sec": segment_sec,
        "eval_results": eval_row,
        "existing_term_adoption": {
            k: adoption.get(k)
            for k in (
                "term_adoption",
                "term_adoption_micro",
                "adopted",
                "total",
                "term_fcr",
                "false_copy_sentences",
                "no_gold_sentences",
            )
            if k in adoption
        },
        "runtime_log_has_stride_windows": any(c.kind == "rag_window" for c in window_calls),
        "rag_overhead": overhead,
        "metrics": metrics,
        "acl_no_term_chunk_alignment": acl_no_term_chunk_diag,
        "samples": samples,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    m = report["metrics"]
    lines = [
        f"# Streaming Retriever Diagnosis: {report['paper']} lm={report['lm']} {report['regime']}",
        "",
        f"- Eval dir: `{report['eval_dir']}`",
        f"- Runtime log: `{report['runtime_jsonl']}`",
        f"- Runtime has independent stride windows: `{report['runtime_log_has_stride_windows']}`",
        "",
        "## RAG Overhead",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, val in (report.get("rag_overhead") or {}).items():
        if isinstance(val, float):
            lines.append(f"| `{key}` | {val:.4f} |")
        else:
            lines.append(f"| `{key}` | {val} |")
    lines.extend([
        "",
        "## Metrics",
        "",
        "Sentence-level `no_term_avg_retrieved` is not comparable to offline ACL "
        "`noterm_noise@topK_tau*` (per no-term chunk). Use `vllm_no_term_acl_chunk_*` below.",
        "",
        "| metric | value |",
        "|---|---:|",
    ])
    for key in [
        "term_recall_source_micro",
        "term_precision_source_micro",
        "term_recall_source_ref_micro",
        "adoption_given_retrieved_source_micro",
        "adoption_source_micro",
        "no_term_avg_retrieved",
        "no_term_noise_sentence_rate",
        "vllm_no_term_acl_chunk_calls",
        "vllm_no_term_acl_chunk_avg_kept_topk_tau",
        "vllm_no_term_acl_chunk_p50_kept",
        "vllm_no_term_acl_chunk_noise_call_rate",
        "vllm_no_term_acl_chunk_tau",
        "vllm_no_term_acl_chunk_topk",
        "vllm_no_term_acl_chunk_overlap_min_frac",
        "avg_retrieved_per_sentence",
        "source_terms",
        "retrieved_terms",
        "retrieved_source_terms",
        "no_term_sentences",
    ]:
        val = m.get(key)
        if val is None:
            lines.append(f"| `{key}` | N/A |")
        elif isinstance(val, float):
            lines.append(f"| `{key}` | {val:.4f} |")
        else:
            lines.append(f"| `{key}` | {val} |")

    align = report.get("acl_no_term_chunk_alignment") or {}
    if align:
        lines.extend(["", "## ACL no-term chunk alignment (diagnostic)", ""])
        lines.append(f"- JSONL: `{align.get('acl_dev_jsonl', '')}`")
        if "skipped_reason" in align:
            lines.append(f"- Skipped / partial: `{align['skipped_reason']}`")
        for k in (
            "acl_dev_distinct_chunks",
            "acl_dev_no_term_chunks",
            "no_term_intervals_count",
            "rag_tau_used",
            "rag_tau_source",
            "rag_topk_used",
            "overlap_min_frac",
        ):
            if k in align:
                lines.append(f"- `{k}`: {align[k]}")

    def add_samples(title: str, samples: Sequence[Dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}", ""])
        for row in samples:
            lines.extend([
                f"### Sentence {row['index']} [{row['time_sec'][0]}, {row['time_sec'][1]}]",
                "",
                f"- Source: {row['source']}",
                f"- Reference: {row['reference']}",
                f"- Hypothesis: {row['hypothesis']}",
                f"- Recall / precision / adoption_given_retrieved: "
                f"{row['term_recall']:.3f} / {row['term_precision']:.3f} / "
                f"{row['adoption_given_retrieved']:.3f}",
                f"- Source terms: {json.dumps(row['source_terms'], ensure_ascii=False)}",
                f"- Retrieved source terms: "
                f"{json.dumps(_refs_summary(row.get('retrieved_source_term_details', []), 12), ensure_ascii=False)}",
                f"- Missed source terms: {json.dumps(row['missed_source_terms'], ensure_ascii=False)}",
                f"- Retrieved terms (top 12 / total {len(row['retrieved_terms'])}): "
                f"{json.dumps(_refs_summary(row['retrieved_terms'], 12), ensure_ascii=False)}",
                f"- False positives (top 12 / total {len(row['false_positive_terms'])}): "
                f"{json.dumps(_refs_summary(row['false_positive_terms'], 12), ensure_ascii=False)}",
                "",
                "Overlapping vLLM-pre retriever calls:",
            ])
            for call in row["nearby_calls"]:
                lines.append(
                    f"- seg={call['segment_idx']} "
                    f"current={call.get('current_sec', call.get('window_sec'))} "
                    f"encoded={call.get('encoded_sec', call.get('window_sec'))} "
                    f"lookback={call.get('lookback_sec', 0.0)} "
                    f"overlap={call.get('overlaps_sentence', True)} "
                    f"final_count={call.get('final_term_map_count', len(call['final_term_map']))} "
                    f"final={json.dumps(call['final_term_map'], ensure_ascii=False)}"
                )
                for prev in call["previous_stride_windows"]:
                    lines.append(
                        f"  - prev {prev['trigger']} win={prev['window_sec']} "
                        f"count={prev.get('term_map_count', len(prev['term_map']))} "
                        f"{json.dumps(prev['term_map'], ensure_ascii=False)}"
                    )
            lines.append("")

    add_samples("Worst With-Term Samples", report["samples"]["worst_with_term_sentences"])
    add_samples("No-Term Noise Samples", report["samples"]["no_term_noise_sentences"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--eval-dir", default="")
    parser.add_argument("--paper", default="", help="Paper id to diagnose; empty selects worst.")
    parser.add_argument("--lm", default="1", help="Latency multiplier filter; empty means all.")
    parser.add_argument("--regime", default="gs10k", choices=["", "raw", "gs1k", "gs10k"])
    parser.add_argument("--worst-metric", default="TERM_ACC")
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument(
        "--acl-dev-jsonl",
        default="",
        help=(
            "ACL6060 dev JSONL (utter_id + chunk_idx). Empty uses built-in default path or "
            "ACL6060_DEV_JSONL if set."
        ),
    )
    parser.add_argument(
        "--skip-acl-chunk-metrics",
        action="store_true",
        help="Skip ACL JSONL alignment and vllm_no_term_acl_chunk_* metrics.",
    )
    parser.add_argument(
        "--no-term-overlap-frac",
        type=float,
        default=0.5,
        help="Min overlap / min(d_encoded, d_chunk) to align a vLLM call to an ACL no-term chunk.",
    )
    parser.add_argument(
        "--rag-tau",
        type=float,
        default=None,
        help="RAG score threshold (default: parse _thX.X_ from eval dirname).",
    )
    parser.add_argument(
        "--rag-topk",
        type=int,
        default=None,
        help="Top-K for kept count (default: parse _kN_ from eval dirname, else 10).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    eval_dir_arg = Path(args.eval_dir) if args.eval_dir else None
    eval_dir, meta, eval_row = _select_eval_dir(
        base_dir=base_dir,
        eval_dir=eval_dir_arg,
        paper=args.paper,
        lm=args.lm,
        regime=args.regime,
        metric=args.worst_metric,
    )
    env_acl = os.environ.get("ACL6060_DEV_JSONL", "").strip()
    acl_cli = args.acl_dev_jsonl.strip()
    if args.skip_acl_chunk_metrics:
        enable_acl_chunk = False
        acl_jsonl_path: Optional[Path] = None
    elif acl_cli:
        enable_acl_chunk = True
        acl_jsonl_path = Path(acl_cli)
    elif env_acl:
        enable_acl_chunk = True
        acl_jsonl_path = Path(env_acl)
    else:
        enable_acl_chunk = True
        acl_jsonl_path = None

    report = diagnose(
        base_dir=base_dir,
        eval_dir=eval_dir,
        meta=meta,
        eval_row=eval_row,
        sample_limit=args.sample_limit,
        enable_acl_chunk_metrics=enable_acl_chunk,
        acl_dev_jsonl=acl_jsonl_path,
        no_term_overlap_frac=args.no_term_overlap_frac,
        rag_tau=args.rag_tau,
        rag_topk=args.rag_topk,
    )

    out_json = Path(args.output_json) if args.output_json else eval_dir / "streaming_retriever_diagnosis.json"
    out_md = Path(args.output_md) if args.output_md else eval_dir / "streaming_retriever_diagnosis.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, out_md)

    metrics = report["metrics"]
    print(f"selected_eval_dir\t{eval_dir}")
    print(f"paper\t{report['paper']}\tlm\t{report['lm']}\tregime\t{report['regime']}")
    print(
        "retriever\t"
        f"recall_source_micro={metrics['term_recall_source_micro']:.6f}\t"
        f"precision_source_micro={metrics['term_precision_source_micro']:.6f}\t"
        f"no_term_avg={metrics['no_term_avg_retrieved']:.6f}\t"
        f"no_term_noise_rate={metrics['no_term_noise_sentence_rate']:.6f}\t"
        f"adopt_given_retrieved={metrics['adoption_given_retrieved_source_micro']:.6f}"
    )
    nt_avg = metrics.get("vllm_no_term_acl_chunk_avg_kept_topk_tau")
    if nt_avg is not None:
        print(
            "vllm_acl_no_term_chunk\t"
            f"n_calls={metrics.get('vllm_no_term_acl_chunk_calls')}\t"
            f"avg_kept_topk_tau={float(nt_avg):.6f}\t"
            f"noise_call_rate={float(metrics.get('vllm_no_term_acl_chunk_noise_call_rate', 0.0)):.6f}\t"
            f"tau={metrics.get('vllm_no_term_acl_chunk_tau')}\t"
            f"topk={metrics.get('vllm_no_term_acl_chunk_topk')}"
        )
    else:
        print("vllm_acl_no_term_chunk\tN/A (ACL JSONL disabled or missing)")
    print(f"wrote_json\t{out_json}")
    print(f"wrote_md\t{out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
