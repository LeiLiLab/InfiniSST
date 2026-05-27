#!/usr/bin/env python3
"""Diagnose TERM_ACC by analyzing per-term retrieval and adoption patterns.

For each paper, loads:
  - per-paper glossary  (GT terms with zh translations)
  - runtime JSONL       (per-segment RAG term_map and LLM output)
  - instances.log       (full model prediction)
  - reference file      (ground-truth translation)

Outputs per-term statistics:
  - provided:        whether the term was ever in any segment's term_map
  - segment_count:   how many segments included this term
  - max_score:       highest retrieval score across all segments
  - avg_score:       average retrieval score
  - adopted:         whether the zh translation appears in the full prediction
  - relevant:        whether the zh translation appears in the reference

Then compares adopted vs not-adopted groups to check for cross-chunk signal weakness.
"""

from __future__ import annotations

# ======Configuration=====
EVAL_BASE = "/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed/zh"
GLOSSARY_DIR = "/home/jiaxuanluo/InfiniSST/documents/data/data_pre/extracted_glossaries_by_paper"
PAPER_INPUTS_DIR = "/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed/zh/__paper_inputs__/lists"

PAPERS = ["2022.acl-long.110", "2022.acl-long.367", "2022.acl-long.590"]
CONFIG_PREFIX = "dold_slm_lm1_k10_gextracted_glossary_"
TARGET_LANG = "zh"
# ======Configuration=====

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import statistics


@dataclass
class TermStats:
    term_en: str
    term_zh: str
    relevant: bool = False
    provided: bool = False
    adopted: bool = False
    segment_indices: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    @property
    def segment_count(self) -> int:
        return len(set(self.segment_indices))

    @property
    def max_score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def avg_score(self) -> float:
        return statistics.mean(self.scores) if self.scores else 0.0


def load_glossary(paper_id: str) -> Dict[str, Dict]:
    """Load per-paper glossary, return {term_en_lower: entry}."""
    path = Path(GLOSSARY_DIR) / f"extracted_glossary__{paper_id}.json"
    assert path.is_file(), f"Glossary not found: {path}"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for key, entry in data.items():
        term_en = entry.get("term", key).strip()
        zh = entry.get("target_translations", {}).get(TARGET_LANG, "")
        if term_en and zh:
            result[term_en.lower()] = {"term": term_en, "zh": zh}
    return result


def find_runtime_jsonl(eval_dir: Path) -> Path:
    """Find the runtime JSONL file in an eval directory."""
    candidates = list(eval_dir.glob("runtime_omni_vllm_maxsim_rag_*.jsonl"))
    assert len(candidates) == 1, (
        f"Expected exactly 1 runtime JSONL in {eval_dir}, found {len(candidates)}: {candidates}"
    )
    return candidates[0]


def load_runtime_jsonl(jsonl_path: Path) -> List[dict]:
    """Load all records from a runtime JSONL."""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_instances_prediction(eval_dir: Path) -> str:
    """Load the full model prediction from instances.log."""
    instances_path = eval_dir / "instances.log"
    assert instances_path.is_file(), f"instances.log not found: {instances_path}"
    with open(instances_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) >= 1, f"instances.log is empty: {instances_path}"
    entry = json.loads(lines[0])
    return entry.get("prediction", "")


def load_reference(paper_id: str) -> str:
    """Load the reference translation for a paper."""
    ref_path = Path(PAPER_INPUTS_DIR) / f"dev.target.{TARGET_LANG}__{paper_id}.txt"
    assert ref_path.is_file(), f"Reference not found: {ref_path}"
    with open(ref_path, encoding="utf-8") as f:
        return f.read().strip()


def analyze_paper(paper_id: str) -> Tuple[str, List[TermStats]]:
    """Analyze term retrieval and adoption for one paper."""
    glossary = load_glossary(paper_id)

    eval_dir_name = f"{CONFIG_PREFIX}_{paper_id}_pp{paper_id}"
    eval_dir = Path(EVAL_BASE) / eval_dir_name
    assert eval_dir.is_dir(), f"Eval directory not found: {eval_dir}"

    runtime_jsonl = find_runtime_jsonl(eval_dir)
    records = load_runtime_jsonl(runtime_jsonl)
    prediction = load_instances_prediction(eval_dir)
    reference = load_reference(paper_id)

    term_stats: Dict[str, TermStats] = {}
    for key, entry in glossary.items():
        ts = TermStats(term_en=entry["term"], term_zh=entry["zh"])
        ts.relevant = entry["zh"] in reference
        ts.adopted = entry["zh"] in prediction
        term_stats[key] = ts

    for rec in records:
        if rec.get("type") not in ("rag", "llm_input"):
            continue
        seg_idx = rec.get("segment_idx", -1)
        refs = rec.get("references", [])
        for ref in refs:
            term_en = (ref.get("term") or "").strip()
            score = ref.get("score", 0.0)
            key = term_en.lower()
            if key in term_stats:
                term_stats[key].provided = True
                term_stats[key].segment_indices.append(seg_idx)
                term_stats[key].scores.append(score)

    return paper_id, list(term_stats.values())


def print_report(paper_id: str, stats_list: List[TermStats]) -> None:
    """Print a detailed report for one paper."""
    relevant = [s for s in stats_list if s.relevant]
    provided_relevant = [s for s in relevant if s.provided]
    adopted_relevant = [s for s in relevant if s.adopted]
    provided_and_adopted = [s for s in relevant if s.provided and s.adopted]
    provided_not_adopted = [s for s in relevant if s.provided and not s.adopted]
    not_provided = [s for s in relevant if not s.provided]

    print(f"\n{'='*80}")
    print(f"Paper: {paper_id}")
    print(f"{'='*80}")
    print(f"  Total glossary terms:          {len(stats_list)}")
    print(f"  Relevant (zh in reference):    {len(relevant)}")
    print(f"  Provided (in any term_map):    {len(provided_relevant)} / {len(relevant)} relevant")
    print(f"  Adopted  (zh in prediction):   {len(adopted_relevant)} / {len(relevant)} relevant")
    print(f"  Provided AND adopted:          {len(provided_and_adopted)}")
    print(f"  Provided but NOT adopted:      {len(provided_not_adopted)}")
    print(f"  NOT provided (retriever miss): {len(not_provided)}")

    if provided_and_adopted:
        seg_counts = [s.segment_count for s in provided_and_adopted]
        max_scores = [s.max_score for s in provided_and_adopted]
        avg_scores = [s.avg_score for s in provided_and_adopted]
        print(f"\n  --- Provided AND Adopted (n={len(provided_and_adopted)}) ---")
        print(f"    segment_count:  mean={statistics.mean(seg_counts):.1f}  median={statistics.median(seg_counts):.1f}  min={min(seg_counts)}  max={max(seg_counts)}")
        print(f"    max_score:      mean={statistics.mean(max_scores):.4f}  median={statistics.median(max_scores):.4f}")
        print(f"    avg_score:      mean={statistics.mean(avg_scores):.4f}  median={statistics.median(avg_scores):.4f}")

    if provided_not_adopted:
        seg_counts = [s.segment_count for s in provided_not_adopted]
        max_scores = [s.max_score for s in provided_not_adopted]
        avg_scores = [s.avg_score for s in provided_not_adopted]
        print(f"\n  --- Provided but NOT Adopted (n={len(provided_not_adopted)}) ---")
        print(f"    segment_count:  mean={statistics.mean(seg_counts):.1f}  median={statistics.median(seg_counts):.1f}  min={min(seg_counts)}  max={max(seg_counts)}")
        print(f"    max_score:      mean={statistics.mean(max_scores):.4f}  median={statistics.median(max_scores):.4f}")
        print(f"    avg_score:      mean={statistics.mean(avg_scores):.4f}  median={statistics.median(avg_scores):.4f}")

        print(f"\n    Per-term detail (provided but NOT adopted):")
        for s in sorted(provided_not_adopted, key=lambda x: x.max_score):
            print(f"      {s.term_en:40s} -> {s.term_zh:15s}  seg_count={s.segment_count:3d}  max_score={s.max_score:.4f}  avg_score={s.avg_score:.4f}")

    if not_provided:
        print(f"\n  --- NOT Provided / Retriever Miss (n={len(not_provided)}) ---")
        for s in not_provided:
            print(f"      {s.term_en:40s} -> {s.term_zh:15s}  (NEVER in term_map)")

    # Cross-chunk signal analysis: compare segment density between adopted and not-adopted
    if provided_and_adopted and provided_not_adopted:
        adopted_mean_seg = statistics.mean([s.segment_count for s in provided_and_adopted])
        not_adopted_mean_seg = statistics.mean([s.segment_count for s in provided_not_adopted])
        adopted_mean_max_score = statistics.mean([s.max_score for s in provided_and_adopted])
        not_adopted_mean_max_score = statistics.mean([s.max_score for s in provided_not_adopted])
        print(f"\n  --- Cross-Chunk Signal Comparison ---")
        print(f"    Adopted group:      avg_seg_count={adopted_mean_seg:.1f}  avg_max_score={adopted_mean_max_score:.4f}")
        print(f"    Not-adopted group:  avg_seg_count={not_adopted_mean_seg:.1f}  avg_max_score={not_adopted_mean_max_score:.4f}")
        if not_adopted_mean_seg < adopted_mean_seg * 0.7:
            print(f"    >> Not-adopted terms appear in FEWER segments (cross-chunk signal weakness likely)")
        if not_adopted_mean_max_score < adopted_mean_max_score * 0.85:
            print(f"    >> Not-adopted terms have LOWER max retrieval scores")


def main():
    for paper_id in PAPERS:
        try:
            paper_id_str, stats = analyze_paper(paper_id)
            print_report(paper_id_str, stats)
        except Exception as e:
            print(f"\nERROR processing {paper_id}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
