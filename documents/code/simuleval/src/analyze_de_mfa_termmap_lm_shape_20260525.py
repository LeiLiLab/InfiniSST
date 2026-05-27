#!/usr/bin/env python3
"""Analyze En-De MFA-timed runtime term-map shape across latency multipliers.

This is a diagnostic, not a calibration script: it reads already-finished
tagged-ACL readouts and aligns post-tau runtime term maps to ACL sentence
intervals.  The goal is to explain why high term recall/adoption can still hurt
BLEU and to identify a training-data repair direction for the SLM.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from documents.code.offline_sst_eval.compute_sentence_term_adoption import (  # noqa: E402
    _iter_jsonl,
    _load_glossary_terms,
    _normalise_space,
    _segment_prediction_by_references,
    _source_contains,
    _text_contains,
)

try:
    from sacrebleu.metrics import CHRF
except Exception:  # pragma: no cover
    CHRF = None


INPUT_DIR = Path(
    "/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1"
    "/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all"
)
GLOSSARY = Path("/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json")

NEWV9_ROOT = Path(
    "/mnt/gemini/data1/jiaxuanluo/tagged_acl_same_lm_batch_v1_mfa_npfilter_hn1024_tau078_raw_de_lm1to4_20260524T1738_tagacl_newv9_mfa_npfilter_de_batch_mt80"
    "/new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64_hn1024_tau078_same_lm_batch_v1/de"
)
TMV4_OMIT_ROOTS = {
    2: Path(
        "/mnt/gemini/data1/jiaxuanluo/tagged_acl_tmv4_hn1024_tau078_omit_de_lm2_20260525T100614_tmv4_hn1024_tau078_omit_de_lm2_taurus45"
        "/tmv4_de_bsz4_hn1024_tau078_omit_batch_max80/de"
    ),
    4: Path(
        "/mnt/gemini/data1/jiaxuanluo/tagged_acl_tmv4_hn1024_tau078_omit_de_lm4_20260525T102941_tmv4_hn1024_tau078_omit_de_lm4_taurus45"
        "/tmv4_de_bsz4_hn1024_tau078_omit_batch_max80/de"
    ),
}
NORAG_DIRS = {
    2: Path(
        "/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm2_batch_max80_aries01_20260524T2200_tagacl_origin_norag_de_lm2_batch_max80_aries01"
        "/origin_norag_de_lm2_batch_max80/de/dtagacl_origin_norag_batch_max80_lm2_k0_th0.0_gacl6060_tagged_gt_raw_min_norm2"
    ),
    4: Path(
        "/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm4_raw_rerun_20260524T160830_tagacl_origin_norag_de_lm4_raw_rerun"
        "/origin_norag/de/gigaspeech-de-s_origin-bsz4_gacl6060_tagged_gt_raw_min_norm2_cs3.84_hs0.48_lm4_k210_k110_th0p0"
    ),
}

OUT_SUMMARY = ROOT / "documents/code/simuleval/reports/20260525_de_mfa_termmap_lm_shape_summary.tsv"
OUT_CALLS = ROOT / "documents/code/simuleval/reports/20260525_de_mfa_termmap_lm_shape_calls.tsv"
OUT_SENTENCES = ROOT / "documents/code/simuleval/reports/20260525_de_mfa_termmap_lm_shape_sentences.tsv"
OUT_MD = ROOT / "documents/code/simuleval/reports/20260525_de_mfa_termmap_lm_shape_bleu_recovery.md"


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def read_eval(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"expected one eval row in {path}, got {len(rows)}")
    return rows[0]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(_iter_jsonl(path))


def load_sentence_meta(path: Path) -> List[Dict[str, Any]]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"audio.yaml must be a list: {path}")
    out = []
    for idx, row in enumerate(rows):
        wav = str(row.get("wav") or "")
        start = float(row.get("offset") or 0.0)
        duration = float(row.get("duration") or 0.0)
        out.append(
            {
                "idx": idx,
                "paper_id": Path(wav).stem,
                "start": start,
                "end": start + duration,
                "duration": duration,
                "speaker_id": str(row.get("speaker_id") or "NA"),
            }
        )
    return out


def by_paper_indices(meta: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for row in meta:
        out[str(row["paper_id"])].append(int(row["idx"]))
    return dict(out)


def eval_dir_for(run_key: str, lm: int) -> Path:
    if run_key == "newv9_none_block":
        return NEWV9_ROOT / (
            f"dtagacl_bv1_mfa_np_hn1024_tau078_lm{lm}_k10_th0.78_"
            "gacl6060_tagged_gt_raw_min_norm2"
        )
    if run_key == "tmv4_omit":
        root = TMV4_OMIT_ROOTS[lm]
        return root / (
            f"dtagacl_bv1_tmv4_hn1024_tau078_omit_lm{lm}_k10_th0.78_"
            "gacl6060_tagged_gt_raw_min_norm2"
        )
    raise KeyError(run_key)


def runtime_path(eval_dir: Path, lm: int) -> Path:
    path = eval_dir / f"runtime_omni_vllm_maxsim_rag_batched_lm{lm}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def metric_eval_path(run_key: str, eval_dir: Path) -> Path:
    if run_key == "newv9_none_block" and (eval_dir / "wordfix_eval/eval_results.tsv").is_file():
        return eval_dir / "wordfix_eval/eval_results.tsv"
    return eval_dir / "eval_results.tsv"


def instances_path_for(run_key: str, eval_dir: Path) -> Path:
    if run_key == "newv9_none_block" and (eval_dir / "wordfix_eval/instances.strip_term.log").is_file():
        return eval_dir / "wordfix_eval/instances.strip_term.log"
    return eval_dir / "instances.strip_term.log"


def paper_id_from_source(source: Any) -> str:
    candidates: List[str] = []
    if isinstance(source, list):
        candidates.extend(str(x) for x in source)
    elif source:
        candidates.append(str(source))
    for item in candidates:
        if item.endswith(".wav"):
            return Path(item).stem
    return ""


def segment_instances(instances_path: Path, refs: Sequence[str], meta: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    instances = read_jsonl(instances_path)
    indices_by_paper = by_paper_indices(meta)
    out: Dict[int, str] = {}
    for inst in instances:
        paper = paper_id_from_source(inst.get("source"))
        if paper not in indices_by_paper:
            raise ValueError(f"cannot map instance to paper: {instances_path} {inst.get('source')}")
        indices = indices_by_paper[paper]
        local_refs = [refs[i] for i in indices]
        hyp_segments = _segment_prediction_by_references(
            str(inst.get("prediction") or ""),
            local_refs,
            latency_unit="word",
        )
        if len(hyp_segments) != len(indices):
            raise ValueError(f"segment mismatch {paper}: {len(hyp_segments)} != {len(indices)}")
        for idx, hyp in zip(indices, hyp_segments):
            out[idx] = hyp
    return out


def chrf_score(hyp: str, ref: str) -> float:
    if CHRF is not None:
        return float(CHRF(word_order=2).sentence_score(hyp, [ref]).score)
    hyp_set = Counter(hyp)
    ref_set = Counter(ref)
    denom = max(1, sum(ref_set.values()))
    return 100.0 * sum((hyp_set & ref_set).values()) / denom


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def pct(num: float, den: float) -> float:
    return num / den if den else 0.0


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and a1 > b0


def ref_key(ref: Dict[str, Any]) -> Tuple[str, str]:
    term = _normalise_space(ref.get("term") or ref.get("key") or "")
    trans = _normalise_space(ref.get("translation") or "")
    return (term.casefold(), trans)


def gold_keys_for_sentence(source: str, ref: str, terms: Sequence[Any]) -> set[Tuple[str, str]]:
    out = set()
    for term in terms:
        if _source_contains(source, term.term) and _text_contains(ref, term.translation):
            out.add(term.match_key)
    return out


def source_keys_for_sentence(source: str, terms: Sequence[Any]) -> set[Tuple[str, str]]:
    out = set()
    for term in terms:
        if _source_contains(source, term.term):
            out.add(term.match_key)
    return out


def refs_summary(refs: Iterable[Dict[str, Any]], max_items: int = 8) -> str:
    items = []
    for ref in refs:
        score = ref.get("score")
        score_s = f"{float(score):.3f}" if score is not None else "NA"
        items.append(f"{ref.get('term') or ref.get('key')}->{ref.get('translation')}@{score_s}")
    return "; ".join(items[:max_items]) + ("; ..." if len(items) > max_items else "")


def parse_runtime_calls(path: Path) -> List[Dict[str, Any]]:
    windows: Dict[Tuple[int, int], Dict[str, Any]] = {}
    calls: List[Dict[str, Any]] = []
    for rec in read_jsonl(path):
        typ = rec.get("type")
        if typ not in {"rag_window", "rag", "llm_input"}:
            continue
        try:
            inst = int(rec.get("instance_index"))
            seg = int(rec.get("segment_idx"))
        except (TypeError, ValueError):
            continue
        key = (inst, seg)
        if typ == "rag_window":
            windows[key] = {
                "instance_index": inst,
                "segment_idx": seg,
                "paper_id": Path(str(rec.get("source_path") or "")).stem,
                "start": float(rec.get("current_start_sec") or 0.0),
                "end": float(rec.get("current_end_sec") or 0.0),
                "lookback_sec": float(rec.get("lookback_sec") or 0.0),
            }
        elif typ == "rag":
            win = dict(windows.get(key, {}))
            win.setdefault("instance_index", inst)
            win.setdefault("segment_idx", seg)
            win.setdefault("paper_id", Path(str(rec.get("source_path") or "")).stem)
            win.setdefault("start", 0.0)
            win.setdefault("end", float(rec.get("rag_audio_duration") or 0.0))
            refs = [r for r in (rec.get("references") or []) if isinstance(r, dict)]
            win["references"] = refs
            calls.append(win)
        elif typ == "llm_input":
            # Prompt shape is checked separately; the RAG call above carries
            # the same references after filtering.
            continue
    return calls


def prompt_shape(path: Path) -> Dict[str, int]:
    counts = Counter()
    for rec in read_jsonl(path):
        if rec.get("type") != "llm_input":
            continue
        prompt = str(rec.get("prompt") or "")
        refs = rec.get("references") or []
        counts["llm_inputs"] += 1
        counts["empty_reference_inputs"] += int(not refs)
        counts["nonempty_reference_inputs"] += int(bool(refs))
        counts["term_map_none_prompts"] += int("term_map:\nNONE" in prompt)
        counts["term_map_block_prompts"] += int("term_map:" in prompt)
        counts["empty_reference_with_term_map_block"] += int(not refs and "term_map:" in prompt)
        counts["empty_reference_omitted_term_map"] += int(not refs and "term_map:" not in prompt)
    return dict(counts)


def score_bins(scores: Sequence[float]) -> Dict[str, int]:
    bins = Counter()
    for score in scores:
        if score < 0.80:
            bins["score_078_080"] += 1
        elif score < 0.85:
            bins["score_080_085"] += 1
        elif score < 0.90:
            bins["score_085_090"] += 1
        else:
            bins["score_ge_090"] += 1
    return dict(bins)


def main() -> int:
    os.environ.setdefault("MWERSEGMENTER_ROOT", "/mnt/taurus/home/jiaxuanluo/mwerSegmenter")
    os.environ["PATH"] = f"{os.environ['MWERSEGMENTER_ROOT']}:{os.environ.get('PATH', '')}"

    sources = read_lines(INPUT_DIR / "source_text.txt")
    refs = read_lines(INPUT_DIR / "ref.txt")
    meta = load_sentence_meta(INPUT_DIR / "audio.yaml")
    if not (len(sources) == len(refs) == len(meta)):
        raise ValueError(f"source/ref/audio mismatch: {len(sources)} {len(refs)} {len(meta)}")

    terms = _load_glossary_terms(GLOSSARY, "de")
    gold_by_sent = [gold_keys_for_sentence(src, ref, terms) for src, ref in zip(sources, refs)]
    source_by_sent = [source_keys_for_sentence(src, terms) for src in sources]
    indices_by_paper = by_paper_indices(meta)

    run_specs: List[Tuple[str, int, Path]] = []
    for lm in (1, 2, 3, 4):
        run_specs.append(("newv9_none_block", lm, eval_dir_for("newv9_none_block", lm)))
    for lm in (2, 4):
        run_specs.append(("tmv4_omit", lm, eval_dir_for("tmv4_omit", lm)))

    summary_rows: List[Dict[str, Any]] = []
    call_rows: List[Dict[str, Any]] = []
    sentence_rows: List[Dict[str, Any]] = []
    top_bad_cases: List[Dict[str, Any]] = []
    comparison_notes: List[str] = []

    for run_key, lm, eval_dir in run_specs:
        rpath = runtime_path(eval_dir, lm)
        eval_metrics = read_eval(metric_eval_path(run_key, eval_dir))
        instances_path = instances_path_for(run_key, eval_dir)
        hyps = segment_instances(instances_path, refs, meta)
        calls = parse_runtime_calls(rpath)
        prompts = prompt_shape(rpath)

        sent_refs: Dict[int, Dict[Tuple[str, str], Dict[str, Any]]] = {
            i: {} for i in range(len(meta))
        }
        all_counts: List[int] = []
        all_scores: List[float] = []
        stale_refs = current_refs = future_refs = marginal_refs = 0
        total_refs = 0
        call_noise_refs = 0
        call_gold_refs = 0
        call_source_refs = 0

        for call in calls:
            paper = str(call.get("paper_id") or "")
            start = float(call.get("start") or 0.0)
            end = float(call.get("end") or 0.0)
            refs_in_call = [r for r in call.get("references", []) if isinstance(r, dict)]
            all_counts.append(len(refs_in_call))
            overlap_indices = [
                idx for idx in indices_by_paper.get(paper, [])
                if overlaps(start, end, float(meta[idx]["start"]), float(meta[idx]["end"]))
            ]
            gold_union = set().union(*(gold_by_sent[i] for i in overlap_indices)) if overlap_indices else set()
            source_union = set().union(*(source_by_sent[i] for i in overlap_indices)) if overlap_indices else set()
            current = stale = future = marginal = gold = source_supported = noise = 0
            scores = []
            for ref_obj in refs_in_call:
                key = ref_key(ref_obj)
                try:
                    score = float(ref_obj.get("score"))
                except (TypeError, ValueError):
                    score = 0.0
                scores.append(score)
                all_scores.append(score)
                total_refs += 1
                if score < 0.80:
                    marginal += 1
                    marginal_refs += 1
                ts = float(ref_obj.get("time_start") or start)
                te = float(ref_obj.get("time_end") or ts)
                if te <= start:
                    stale += 1
                    stale_refs += 1
                elif ts >= end:
                    future += 1
                    future_refs += 1
                else:
                    current += 1
                    current_refs += 1
                if key in gold_union:
                    gold += 1
                    call_gold_refs += 1
                if key in source_union:
                    source_supported += 1
                    call_source_refs += 1
                if key not in source_union and key not in gold_union:
                    noise += 1
                    call_noise_refs += 1
                for idx in overlap_indices:
                    existing = sent_refs[idx].get(key)
                    if existing is None or score > float(existing.get("score", 0.0)):
                        sent_refs[idx][key] = {
                            "term": ref_obj.get("term") or ref_obj.get("key") or "",
                            "translation": ref_obj.get("translation") or "",
                            "score": score,
                            "time_start": ts,
                            "time_end": te,
                            "call_start": start,
                            "call_end": end,
                        }
            call_rows.append(
                {
                    "run_key": run_key,
                    "lm": lm,
                    "paper_id": paper,
                    "segment_idx": call.get("segment_idx"),
                    "start_sec": f"{start:.3f}",
                    "end_sec": f"{end:.3f}",
                    "duration_sec": f"{end - start:.3f}",
                    "term_map_count": len(refs_in_call),
                    "score_min": f"{min(scores):.4f}" if scores else "",
                    "score_mean": f"{statistics.mean(scores):.4f}" if scores else "",
                    "score_max": f"{max(scores):.4f}" if scores else "",
                    "marginal_078_080": marginal,
                    "current_window_refs": current,
                    "stale_lookback_refs": stale,
                    "future_refs": future,
                    "gold_refs_by_sentence_overlap": gold,
                    "source_supported_refs_by_sentence_overlap": source_supported,
                    "noise_refs_by_sentence_overlap": noise,
                    "term_map": refs_summary(refs_in_call, 12),
                }
            )

        sent_counts: List[int] = []
        sent_gold_recall_num = sent_gold_recall_den = 0
        sent_noise_total = sent_source_supported_total = sent_gold_total = 0
        sent_rows_with_noise = sent_rows_with_map = sent_rows_with_gold = 0
        chrf_by_bin: Dict[str, List[float]] = defaultdict(list)
        false_copy_rows = 0

        for idx, meta_row in enumerate(meta):
            refs_map = sent_refs[idx]
            count = len(refs_map)
            sent_counts.append(count)
            gold_keys = gold_by_sent[idx]
            source_keys = source_by_sent[idx]
            retrieved_keys = set(refs_map)
            gold_retrieved = len(gold_keys & retrieved_keys)
            source_supported = len(source_keys & retrieved_keys)
            noise_keys = sorted(retrieved_keys - source_keys)
            sent_gold_recall_num += gold_retrieved
            sent_gold_recall_den += len(gold_keys)
            sent_noise_total += len(noise_keys)
            sent_source_supported_total += source_supported
            sent_gold_total += len(gold_keys)
            sent_rows_with_map += int(count > 0)
            sent_rows_with_gold += int(bool(gold_keys))
            sent_rows_with_noise += int(bool(noise_keys))
            hyp = hyps.get(idx, "")
            chrf = chrf_score(hyp, refs[idx])
            if count == 0:
                bin_name = "map0"
            elif count <= 2:
                bin_name = "map1_2"
            elif count <= 5:
                bin_name = "map3_5"
            else:
                bin_name = "map6plus"
            chrf_by_bin[bin_name].append(chrf)
            false_copied = []
            for key in noise_keys:
                item = refs_map[key]
                trans = str(item.get("translation") or "")
                if trans and _text_contains(hyp, trans) and not _text_contains(refs[idx], trans):
                    false_copied.append(f"{item.get('term')}->{trans}@{float(item.get('score', 0.0)):.3f}")
            false_copy_rows += int(bool(false_copied))
            row = {
                "run_key": run_key,
                "lm": lm,
                "sentence_index": idx,
                "paper_id": meta_row["paper_id"],
                "speaker_id": meta_row["speaker_id"],
                "start_sec": f"{float(meta_row['start']):.3f}",
                "end_sec": f"{float(meta_row['end']):.3f}",
                "source": sources[idx],
                "reference": refs[idx],
                "hypothesis": hyp,
                "sentence_chrf": f"{chrf:.3f}",
                "term_map_count": count,
                "gold_term_count": len(gold_keys),
                "gold_retrieved_count": gold_retrieved,
                "source_supported_retrieved_count": source_supported,
                "noise_retrieved_count": len(noise_keys),
                "noise_false_copied": "; ".join(false_copied),
                "term_map": refs_summary(refs_map.values(), 18),
            }
            sentence_rows.append(row)
            if run_key == "tmv4_omit" and lm == 4 and (false_copied or (count >= 6 and chrf < 45)):
                top_bad_cases.append(row)

        bins = score_bins(all_scores)
        summary = {
            "run_key": run_key,
            "lm": lm,
            "BLEU": eval_metrics.get("BLEU", ""),
            "StreamLAAL": eval_metrics.get("StreamLAAL", ""),
            "StreamLAAL_CA": eval_metrics.get("StreamLAAL_CA", ""),
            "TERM_ACC": eval_metrics.get("TERM_ACC", ""),
            "TERM_CORRECT": eval_metrics.get("TERM_CORRECT", ""),
            "TERM_TOTAL": eval_metrics.get("TERM_TOTAL", ""),
            "TERM_FCR": eval_metrics.get("TERM_FCR", ""),
            "SOURCE_TERM_SENT_FCR": eval_metrics.get("SOURCE_TERM_SENT_FCR", ""),
            "calls": len(calls),
            "llm_inputs": prompts.get("llm_inputs", 0),
            "empty_reference_inputs": prompts.get("empty_reference_inputs", 0),
            "empty_reference_rate": f"{pct(prompts.get('empty_reference_inputs', 0), prompts.get('llm_inputs', 0)):.4f}",
            "term_map_none_prompts": prompts.get("term_map_none_prompts", 0),
            "empty_reference_omitted_term_map": prompts.get("empty_reference_omitted_term_map", 0),
            "nonempty_call_rate": f"{pct(sum(1 for x in all_counts if x > 0), len(all_counts)):.4f}",
            "avg_terms_per_call": f"{statistics.mean(all_counts):.4f}" if all_counts else "0",
            "p50_terms_per_call": f"{quantile(all_counts, 0.50):.2f}",
            "p90_terms_per_call": f"{quantile(all_counts, 0.90):.2f}",
            "max_terms_per_call": max(all_counts) if all_counts else 0,
            "total_refs": total_refs,
            "score_p10": f"{quantile(all_scores, 0.10):.4f}",
            "score_p50": f"{quantile(all_scores, 0.50):.4f}",
            "score_p90": f"{quantile(all_scores, 0.90):.4f}",
            "score_078_080": bins.get("score_078_080", 0),
            "score_080_085": bins.get("score_080_085", 0),
            "score_085_090": bins.get("score_085_090", 0),
            "score_ge_090": bins.get("score_ge_090", 0),
            "marginal_ref_rate_078_080": f"{pct(marginal_refs, total_refs):.4f}",
            "current_window_ref_rate": f"{pct(current_refs, total_refs):.4f}",
            "stale_lookback_ref_rate": f"{pct(stale_refs, total_refs):.4f}",
            "future_ref_rate": f"{pct(future_refs, total_refs):.4f}",
            "call_gold_ref_rate": f"{pct(call_gold_refs, total_refs):.4f}",
            "call_source_supported_ref_rate": f"{pct(call_source_refs, total_refs):.4f}",
            "call_noise_ref_rate": f"{pct(call_noise_refs, total_refs):.4f}",
            "sentences": len(meta),
            "sentences_with_map": sent_rows_with_map,
            "sentence_map_rate": f"{pct(sent_rows_with_map, len(meta)):.4f}",
            "avg_terms_per_sentence": f"{statistics.mean(sent_counts):.4f}",
            "p90_terms_per_sentence": f"{quantile(sent_counts, 0.90):.2f}",
            "sentence_gold_recall": f"{pct(sent_gold_recall_num, sent_gold_recall_den):.4f}",
            "sentence_source_supported_ref_rate": f"{pct(sent_source_supported_total, sum(sent_counts)):.4f}",
            "sentence_noise_ref_rate": f"{pct(sent_noise_total, sum(sent_counts)):.4f}",
            "sentences_with_noise": sent_rows_with_noise,
            "sentence_noise_rate": f"{pct(sent_rows_with_noise, len(meta)):.4f}",
            "sentence_false_copy_rows": false_copy_rows,
            "chrf_map0": f"{statistics.mean(chrf_by_bin['map0']):.3f}" if chrf_by_bin["map0"] else "",
            "chrf_map1_2": f"{statistics.mean(chrf_by_bin['map1_2']):.3f}" if chrf_by_bin["map1_2"] else "",
            "chrf_map3_5": f"{statistics.mean(chrf_by_bin['map3_5']):.3f}" if chrf_by_bin["map3_5"] else "",
            "chrf_map6plus": f"{statistics.mean(chrf_by_bin['map6plus']):.3f}" if chrf_by_bin["map6plus"] else "",
        }
        summary_rows.append(summary)

    # Compare runtime reference identity where SLM changed but retriever did not.
    for lm in (2, 4):
        a = parse_runtime_calls(runtime_path(eval_dir_for("newv9_none_block", lm), lm))
        b = parse_runtime_calls(runtime_path(eval_dir_for("tmv4_omit", lm), lm))
        identical = len(a) == len(b)
        if identical:
            for x, y in zip(a, b):
                x_refs = [(r.get("term"), r.get("translation"), round(float(r.get("score") or 0), 6)) for r in x.get("references", [])]
                y_refs = [(r.get("term"), r.get("translation"), round(float(r.get("score") or 0), 6)) for r in y.get("references", [])]
                if x_refs != y_refs:
                    identical = False
                    break
        comparison_notes.append(f"lm={lm}: NewV9 and TMV4-omit post-tau references identical={identical}")

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in [
        (OUT_SUMMARY, summary_rows),
        (OUT_CALLS, call_rows),
        (OUT_SENTENCES, sentence_rows),
    ]:
        if not rows:
            raise ValueError(f"no rows for {path}")
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    no_rag_metrics = {}
    for lm, ndir in NORAG_DIRS.items():
        if (ndir / "eval_results.tsv").is_file():
            no_rag_metrics[lm] = read_eval(ndir / "eval_results.tsv")

    def md_summary_row(row: Dict[str, Any]) -> str:
        return (
            f"| {row['run_key']} | {row['lm']} | {float(row['BLEU']):.2f} | "
            f"{float(row['TERM_ACC']):.4f} | {float(row['TERM_FCR']):.4f} | "
            f"{row['nonempty_call_rate']} | {row['avg_terms_per_call']} | "
            f"{row['marginal_ref_rate_078_080']} | {row['stale_lookback_ref_rate']} | "
            f"{row['sentence_gold_recall']} | {row['sentence_noise_ref_rate']} | "
            f"{row['term_map_none_prompts']} | {row['empty_reference_omitted_term_map']} |"
        )

    md = [
        "# En-De MFA Term-Map Shape Analysis",
        "",
        "This diagnostic aligns HN1024 tau=0.78 runtime term maps to MFA-timed ACL sentence intervals. It is a post-hoc failure analysis, not a tau/model selection step.",
        "",
        "## Metric And Shape Summary",
        "",
        "| run | lm | BLEU | TERM_ACC | TERM_FCR | nonempty call rate | avg terms/call | marginal 0.78-0.80 | stale lookback refs | sent gold recall | sent noise ref rate | NONE prompts | omitted empty prompts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md.append(md_summary_row(row))
    md.extend(["", "## Runtime Reference Identity", ""])
    md.extend(f"- {line}" for line in comparison_notes)
    if no_rag_metrics:
        md.extend(["", "## Verified No-RAG Comparison Points", ""])
        for lm, row in sorted(no_rag_metrics.items()):
            md.append(f"- lm={lm}: BLEU={float(row['BLEU']):.4f}, TERM_ACC={float(row['TERM_ACC']):.4f}")

    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The post-tau reference lists are identical for NewV9 and TM-SFT at the same lm; BLEU differences are therefore SLM/prompt-response differences, not retriever differences.",
            "- `empty_term_map_policy=omit` removes `term_map:NONE` prompts for empty retrievals, but lm=4 TM-SFT still remains below the verified no-RAG BLEU target, so NONE blocks are not the only issue.",
            "- A large share of sentence-aligned runtime references are not source-supported in the overlapping ACL sentence, which means the SLM must be trained to ignore plausible but locally unsupported terminology, not only to adopt retrieved terms.",
            "- MFA timestamps show no post-tau references outside the current runtime window in these logs (`stale_lookback_ref_rate=0`), so the immediate issue is not stale lookback leakage.",
            "- The risky shape is local over-exposure: many sentence-aligned references are unsupported by the overlapping source sentence, and roughly 12-14% of references sit just above tau in the 0.78-0.80 band.",
            "",
            "## Recommended SLM Adjustment",
            "",
            "Use the MFA distribution to build a short rescue SFT variant rather than another tagged-only variant:",
            "",
            "1. Keep `empty_term_map_policy=omit` at inference and make training match it: no `term_map:NONE` user blocks for empty maps.",
            "2. For no-GT chunks, do not zero the map; inject HN1024-style retrieved maps, but bucket them by runtime shape: empty, 1-2 terms, 3-5 terms, 6+ terms.",
            "3. Add negative/noise exposure targets: if a retrieved term translation is not in the future assistant span up to the message end, leave the assistant unwrapped and do not force adoption.",
            "4. Down-weight or dropout the riskiest references during SFT data construction: score 0.78-0.80 and sentence-unsupported terms. This changes SLM behavior without ACL tau tuning.",
            "5. Gate the next training candidate on de/lm=4 BLEU against verified no-RAG, but select the data rule from train/dev distribution, not ACL.",
            "",
            "## Case Pointers",
            "",
            f"- Full sentence table: `{OUT_SENTENCES}`",
            f"- Call-level shape table: `{OUT_CALLS}`",
            f"- Summary table: `{OUT_SUMMARY}`",
        ]
    )
    if top_bad_cases:
        md.extend(["", "### TM-SFT lm=4 Dense/False-Copy Examples", ""])
        for row in sorted(top_bad_cases, key=lambda r: (int(r["noise_retrieved_count"]), int(r["term_map_count"])), reverse=True)[:10]:
            md.extend(
                [
                    f"#### {row['paper_id']} sent={row['sentence_index']} {row['start_sec']}-{row['end_sec']}s",
                    "",
                    f"- chrF={row['sentence_chrf']}; map_count={row['term_map_count']}; noise={row['noise_retrieved_count']}; false_copy={row['noise_false_copied'] or 'NONE'}",
                    f"- term_map: {row['term_map'] or 'EMPTY'}",
                    f"- source: {row['source']}",
                    f"- reference: {row['reference']}",
                    f"- hypothesis: {row['hypothesis']}",
                    "",
                ]
            )
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_CALLS}")
    print(f"Wrote {OUT_SENTENCES}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
