#!/usr/bin/env python3
"""Sentence-level DE lm=2 RASST case analysis.

This script compares the current NewV9 RASST run against the older TM-SFT
+ HN1024 run and the verified no-RAG baseline.  It uses the ACL sentence
offsets in audio.yaml to align runtime term maps to source sentences.
"""

from __future__ import annotations

import csv
import json
import os
import re
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
    _runtime_records_to_sentence_term_map_keys,
    _segment_prediction_by_references,
    _source_contains,
    _split_runtime_records_by_instance,
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

NEWV9_DIR = Path(
    "/mnt/gemini/data1/jiaxuanluo/tagged_acl_same_lm_batch_v1_mfa_npfilter_hn1024_tau078_raw_de_lm1to4_20260524T1738_tagacl_newv9_mfa_npfilter_de_batch_mt80"
    "/new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64_hn1024_tau078_same_lm_batch_v1/de"
    "/dtagacl_bv1_mfa_np_hn1024_tau078_lm2_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2"
)
TMV4_DIR = Path(
    "/mnt/gemini/data1/jiaxuanluo/tagged_acl_de_lm2_tmv4_hn1024_batch_20260524T2135_tagacl_de_lm2_tmv4_hn1024_batch"
    "/tmv4_de_bsz4_hn1024_tau078_batch_lm2/de"
    "/dtagacl_bv1_tmv4_hn1024_hn1024_tau078_lm2_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2"
)
NORAG_DIR = Path(
    "/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm2_batch_max80_aries01_20260524T2200_tagacl_origin_norag_de_lm2_batch_max80_aries01"
    "/origin_norag_de_lm2_batch_max80/de"
    "/dtagacl_origin_norag_batch_max80_lm2_k0_th0.0_gacl6060_tagged_gt_raw_min_norm2"
)

OUT_TSV = ROOT / "documents/code/simuleval/reports/20260525_de_lm2_rasst_bleu_case_report.tsv"
OUT_MD = ROOT / "documents/code/simuleval/reports/20260525_de_lm2_rasst_bleu_case_report.md"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(_iter_jsonl(path))


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def paper_id_from_source(source: Any) -> str:
    values: List[str] = []
    if isinstance(source, list):
        values = [str(x) for x in source]
    elif source:
        values = [str(source)]
    for item in values:
        if item.endswith(".wav"):
            return Path(item).stem
    return ""


def load_sentence_meta(path: Path) -> List[Dict[str, Any]]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"audio.yaml must be a list: {path}")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        wav = str(row.get("wav") or "")
        start = float(row.get("offset") or 0.0)
        dur = float(row.get("duration") or 0.0)
        out.append(
            {
                "global_idx": idx,
                "paper_id": Path(wav).stem,
                "start": start,
                "end": start + dur,
                "duration": dur,
                "speaker_id": str(row.get("speaker_id") or "NA"),
            }
        )
    return out


def by_paper_indices(meta: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = defaultdict(list)
    for row in meta:
        out[str(row["paper_id"])].append(int(row["global_idx"]))
    return dict(out)


def segment_instances(
    instances_path: Path,
    refs: Sequence[str],
    meta: Sequence[Dict[str, Any]],
) -> Dict[int, str]:
    instances = read_jsonl(instances_path)
    indices_by_paper = by_paper_indices(meta)
    segmented: Dict[int, str] = {}
    for inst in instances:
        paper = paper_id_from_source(inst.get("source"))
        if paper not in indices_by_paper:
            raise ValueError(f"Cannot map instance source to paper: {instances_path} {inst.get('source')}")
        indices = indices_by_paper[paper]
        local_refs = [refs[i] for i in indices]
        hyp_segments = _segment_prediction_by_references(
            str(inst.get("prediction") or ""),
            local_refs,
            latency_unit="word",
        )
        if len(hyp_segments) != len(indices):
            raise ValueError(f"Segment count mismatch for {paper}: {len(hyp_segments)} != {len(indices)}")
        for idx, hyp in zip(indices, hyp_segments):
            segmented[idx] = hyp
    return segmented


def load_sentence_term_map(runtime_path: Path, meta: Sequence[Dict[str, Any]]) -> Dict[int, List[Tuple[str, str]]]:
    groups = _split_runtime_records_by_instance(runtime_path)
    indices_by_paper = by_paper_indices(meta)
    source_order = ["2022.acl-long.268", "2022.acl-long.367", "2022.acl-long.590", "2022.acl-long.110", "2022.acl-long.117"]
    out: Dict[int, List[Tuple[str, str]]] = {int(row["global_idx"]): [] for row in meta}
    if len(groups) != len(source_order):
        raise ValueError(f"Runtime groups {len(groups)} != expected papers {len(source_order)}")
    for group, paper in zip(groups, source_order):
        indices = indices_by_paper[paper]
        intervals = [(float(meta[i]["start"]), float(meta[i]["end"])) for i in indices]
        local_keys = _runtime_records_to_sentence_term_map_keys(group, intervals)
        if len(local_keys) != len(indices):
            raise ValueError(f"Term-map key mismatch for {paper}: {len(local_keys)} != {len(indices)}")
        for idx, keys in zip(indices, local_keys):
            out[idx] = sorted((term, trans) for term, trans in keys if trans)
    return out


def count_runtime_tags_by_sentence(runtime_path: Path, meta: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    indices_by_paper = by_paper_indices(meta)
    intervals_by_paper = {
        paper: [(i, float(meta[i]["start"]), float(meta[i]["end"])) for i in indices]
        for paper, indices in indices_by_paper.items()
    }
    tags: Dict[int, int] = defaultdict(int)
    windows: Dict[Tuple[int, int], Tuple[float, float, str]] = {}
    for rec in read_jsonl(runtime_path):
        typ = rec.get("type")
        try:
            inst = int(rec.get("instance_index"))
            seg = int(rec.get("segment_idx"))
        except (TypeError, ValueError):
            continue
        if typ == "rag_window":
            windows[(inst, seg)] = (
                float(rec.get("current_start_sec") or 0.0),
                float(rec.get("current_end_sec") or 0.0),
                Path(str(rec.get("source_path") or "")).stem,
            )
        elif typ == "llm_output":
            count = str(rec.get("text") or "").count("<term>")
            if not count:
                continue
            start, end, paper = windows.get((inst, seg), (None, None, ""))
            if start is None or not paper:
                continue
            for idx, sent_start, sent_end in intervals_by_paper.get(paper, []):
                if start < sent_end and end > sent_start:
                    tags[idx] += count
    return tags


def simple_words(text: str) -> List[str]:
    return re.findall(r"\w+", _normalise_space(text), flags=re.UNICODE)


def chrf_score(hyp: str, ref: str) -> float:
    if CHRF is not None:
        return float(CHRF(word_order=2).sentence_score(hyp, [ref]).score)
    # Fallback: rough character overlap score.
    hyp_chars = Counter(hyp)
    ref_chars = Counter(ref)
    overlap = sum((hyp_chars & ref_chars).values())
    denom = max(1, sum(ref_chars.values()))
    return 100.0 * overlap / denom


def load_eval_metrics(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows[0] if rows else {}


def term_summary_for_sentence(source: str, ref: str, hyp: str, terms: Iterable[Any]) -> Tuple[List[str], List[str], List[str]]:
    gold = []
    adopted = []
    missed = []
    for term in terms:
        if _source_contains(source, term.term) and _text_contains(ref, term.translation):
            item = f"{term.term}->{term.translation}"
            gold.append(item)
            if _text_contains(hyp, term.translation):
                adopted.append(item)
            else:
                missed.append(item)
    return gold, adopted, missed


def main() -> int:
    os.environ.setdefault("MWERSEGMENTER_ROOT", "/mnt/taurus/home/jiaxuanluo/mwerSegmenter")
    os.environ["PATH"] = f"{os.environ['MWERSEGMENTER_ROOT']}:{os.environ.get('PATH', '')}"

    sources = read_lines(INPUT_DIR / "source_text.txt")
    refs = read_lines(INPUT_DIR / "ref.txt")
    meta = load_sentence_meta(INPUT_DIR / "audio.yaml")
    if not (len(sources) == len(refs) == len(meta)):
        raise ValueError(f"source/ref/audio size mismatch: {len(sources)} {len(refs)} {len(meta)}")

    terms = _load_glossary_terms(GLOSSARY, "de")
    hyps = {
        "newv9": segment_instances(NEWV9_DIR / "wordfix_eval/instances.strip_term.log", refs, meta),
        "tmv4": segment_instances(TMV4_DIR / "instances.strip_term.log", refs, meta),
        "norag": segment_instances(NORAG_DIR / "instances.strip_term.log", refs, meta),
    }
    term_map = load_sentence_term_map(NEWV9_DIR / "runtime_omni_vllm_maxsim_rag_batched_lm2.jsonl", meta)
    tag_counts = count_runtime_tags_by_sentence(NEWV9_DIR / "runtime_omni_vllm_maxsim_rag_batched_lm2.jsonl", meta)

    runtime_new = [r for r in read_jsonl(NEWV9_DIR / "runtime_omni_vllm_maxsim_rag_batched_lm2.jsonl") if r.get("type") == "llm_input"]
    runtime_tm = [r for r in read_jsonl(TMV4_DIR / "runtime_omni_vllm_maxsim_rag_batched_lm2.jsonl") if r.get("type") == "llm_input"]
    refs_identical = all(
        [(x.get("term"), x.get("translation"), round(float(x.get("score", 0)), 6)) for x in a.get("references", [])]
        == [(x.get("term"), x.get("translation"), round(float(x.get("score", 0)), 6)) for x in b.get("references", [])]
        for a, b in zip(runtime_new, runtime_tm)
    )

    rows: List[Dict[str, Any]] = []
    for idx, (source, ref) in enumerate(zip(sources, refs)):
        n = hyps["newv9"][idx]
        t = hyps["tmv4"][idx]
        b = hyps["norag"][idx]
        gold, n_adopted, n_missed = term_summary_for_sentence(source, ref, n, terms)
        _, t_adopted, t_missed = term_summary_for_sentence(source, ref, t, terms)
        map_items = term_map.get(idx, [])
        false_copies = []
        for term, trans in map_items:
            if not _source_contains(source, term) and not _text_contains(ref, trans) and _text_contains(n, trans):
                false_copies.append(f"{term}->{trans}")
        row = {
            "sentence_index": idx,
            "paper_id": meta[idx]["paper_id"],
            "speaker_id": meta[idx]["speaker_id"],
            "start_sec": f"{meta[idx]['start']:.3f}",
            "end_sec": f"{meta[idx]['end']:.3f}",
            "duration_sec": f"{meta[idx]['duration']:.3f}",
            "source": source,
            "reference": ref,
            "newv9_hyp": n,
            "tmv4_hyp": t,
            "norag_hyp": b,
            "newv9_chrf": f"{chrf_score(n, ref):.3f}",
            "tmv4_chrf": f"{chrf_score(t, ref):.3f}",
            "norag_chrf": f"{chrf_score(b, ref):.3f}",
            "tmv4_minus_newv9_chrf": f"{chrf_score(t, ref) - chrf_score(n, ref):.3f}",
            "norag_minus_newv9_chrf": f"{chrf_score(b, ref) - chrf_score(n, ref):.3f}",
            "newv9_words": len(simple_words(n)),
            "tmv4_words": len(simple_words(t)),
            "norag_words": len(simple_words(b)),
            "ref_words": len(simple_words(ref)),
            "term_map_count": len(map_items),
            "term_map": "; ".join(f"{term}->{trans}" for term, trans in map_items),
            "gold_terms": "; ".join(gold),
            "newv9_adopted": "; ".join(n_adopted),
            "newv9_missed": "; ".join(n_missed),
            "tmv4_adopted": "; ".join(t_adopted),
            "tmv4_missed": "; ".join(t_missed),
            "newv9_false_copy_from_term_map": "; ".join(false_copies),
            "newv9_term_tag_chunks_overlapping_sentence": tag_counts.get(idx, 0),
        }
        rows.append(row)

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    raw_new_text = (NEWV9_DIR / "instances.log").read_text(encoding="utf-8", errors="replace")
    raw_tm_text = (TMV4_DIR / "instances.log").read_text(encoding="utf-8", errors="replace")
    raw_norag_text = (NORAG_DIR / "instances.log").read_text(encoding="utf-8", errors="replace")
    metrics = {
        "newv9": load_eval_metrics(NEWV9_DIR / "wordfix_eval/eval_results.tsv"),
        "tmv4": load_eval_metrics(TMV4_DIR / "eval_results.tsv"),
        "norag": load_eval_metrics(NORAG_DIR / "eval_results.tsv"),
    }

    def top_cases(predicate, n: int = 8) -> List[Dict[str, Any]]:
        cand = [r for r in rows if predicate(r)]
        cand.sort(key=lambda r: float(r["tmv4_minus_newv9_chrf"]), reverse=True)
        return cand[:n]

    def md_case(row: Dict[str, Any]) -> str:
        return (
            f"### {row['paper_id']} sent={row['sentence_index']} "
            f"{row['start_sec']}-{row['end_sec']}s\n\n"
            f"- chrF NewV9/TM-SFT/no-RAG: {row['newv9_chrf']} / {row['tmv4_chrf']} / {row['norag_chrf']} "
            f"(TM-New={row['tmv4_minus_newv9_chrf']})\n"
            f"- term_map: {row['term_map'] or 'EMPTY'}\n"
            f"- gold terms: {row['gold_terms'] or 'NONE'}\n"
            f"- NewV9 false copy from term_map: {row['newv9_false_copy_from_term_map'] or 'NONE'}\n"
            f"- NewV9 overlapping `<term>` chunks: {row['newv9_term_tag_chunks_overlapping_sentence']}\n"
            f"- source: {row['source']}\n"
            f"- reference: {row['reference']}\n"
            f"- NewV9: {row['newv9_hyp']}\n"
            f"- TM-SFT+HN1024: {row['tmv4_hyp']}\n"
            f"- no-RAG: {row['norag_hyp']}\n"
        )

    summary_lines = [
        "# DE lm=2 RASST BLEU Case Analysis",
        "",
        "## Global evidence",
        "",
        f"- NewV9 RASST: BLEU={metrics['newv9'].get('BLEU')} TERM_ACC={metrics['newv9'].get('TERM_ACC')} REAL_TERM_ADOPT={metrics['newv9'].get('REAL_TERM_ADOPT')} TERM_FCR={metrics['newv9'].get('TERM_FCR')}",
        f"- TM-SFT + HN1024: BLEU={metrics['tmv4'].get('BLEU')} TERM_ACC={metrics['tmv4'].get('TERM_ACC')} REAL_TERM_ADOPT=0.853383 TERM_FCR=0.147139",
        f"- no-RAG verified: BLEU={metrics['norag'].get('BLEU')} TERM_ACC={metrics['norag'].get('TERM_ACC')}",
        f"- Runtime term maps are identical between NewV9 and TM-SFT + HN1024: {refs_identical}.",
        f"- NewV9 raw output tag counts: <term>={raw_new_text.count('<term>')} </term>={raw_new_text.count('</term>')}; TM-SFT={raw_tm_text.count('<term>')}; no-RAG={raw_norag_text.count('<term>')}.",
        f"- Sentence rows written to `{OUT_TSV}`.",
        "",
        "## Distribution",
        "",
    ]
    total = len(rows)
    nonempty = sum(1 for r in rows if int(r["term_map_count"]) > 0)
    worse_tm = sum(1 for r in rows if float(r["tmv4_minus_newv9_chrf"]) > 5)
    worse_tm_map = sum(1 for r in rows if int(r["term_map_count"]) > 0 and float(r["tmv4_minus_newv9_chrf"]) > 5)
    false_copy_rows = sum(1 for r in rows if r["newv9_false_copy_from_term_map"])
    tag_rows = sum(1 for r in rows if int(r["newv9_term_tag_chunks_overlapping_sentence"]) > 0)
    summary_lines.extend(
        [
            f"- Sentences: {total}; with aligned term_map: {nonempty}; NewV9 worse than TM-SFT by chrF>5: {worse_tm}; among term_map sentences: {worse_tm_map}.",
            f"- NewV9 false-copied at least one exposed non-gold term in {false_copy_rows} sentence rows.",
            f"- NewV9 emitted `<term>` in chunks overlapping {tag_rows} sentence rows.",
            "",
            "## Top NewV9-worse Cases With Term Maps",
            "",
        ]
    )
    for row in top_cases(lambda r: int(r["term_map_count"]) > 0 and float(r["tmv4_minus_newv9_chrf"]) > 5):
        summary_lines.append(md_case(row))
    summary_lines.extend(["", "## Top NewV9 False-Copy Cases", ""])
    for row in top_cases(lambda r: bool(r["newv9_false_copy_from_term_map"])):
        summary_lines.append(md_case(row))
    OUT_MD.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote {OUT_TSV}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
