#!/usr/bin/env python3
"""
Build ACL eval glossaries for multiple sizes using ACL-dev GT term set.

Rule:
- GT terms are derived from acl6060_dev_dataset.jsonl (term_key/term).
- For every glossary size, include ALL GT terms first.
- Fill remaining slots with wiki terms (excluding GT overlap).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set


# ======Configuration=====
ACL_DEV_JSONL = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
)
ACL_GT_TRANSLATION_JSON = "/mnt/gemini/data2/jiaxuanluo/tcr_fcr_eval/acl_combined_glossary.json"
WIKI_ENRICHED_JSON = (
    "/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/"
    "wiki_glossary_nlp_ai_cs_enriched.json"
)
OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/tcr_fcr_eval"
GLOSSARY_SIZES = [100, 1000, 10000]
ACL_GT_TERMS_OUTPUT = "/mnt/gemini/data2/jiaxuanluo/tcr_fcr_eval/acl_gt_terms_from_dev95.json"
# ======Configuration=====


def load_acl_gt_terms(acl_jsonl_path: str) -> Set[str]:
    gt_terms: Set[str] = set()
    with open(acl_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            term = (sample.get("term_key") or sample.get("term") or "").strip().lower()
            if term:
                gt_terms.add(term)
    return gt_terms


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_terms = sorted(load_acl_gt_terms(ACL_DEV_JSONL))
    assert len(gt_terms) > 0, "No GT terms found in ACL dev JSONL."

    acl_translation_map: Dict[str, Dict] = json.load(open(ACL_GT_TRANSLATION_JSON, "r", encoding="utf-8"))
    wiki_entries: List[Dict] = json.load(open(WIKI_ENRICHED_JSON, "r", encoding="utf-8"))

    gt_glossary_entries: List[Dict] = []
    gt_translation_95: Dict[str, Dict[str, str]] = {}
    for term in gt_terms:
        assert term in acl_translation_map, f"Missing ACL translation for GT term: {term}"
        zh = acl_translation_map[term].get("zh", "").strip()
        assert zh, f"Empty zh translation for GT term: {term}"
        gt_translation_95[term] = {
            "term": acl_translation_map[term]["term"],
            "zh": zh,
        }
        gt_glossary_entries.append(
            {
                "term": acl_translation_map[term]["term"],
                "target_translations": {"zh": zh},
                "source": "acl_gt",
            }
        )

    gt_term_set = set(gt_terms)
    wiki_fill_entries: List[Dict] = []
    for item in wiki_entries:
        term = item.get("term", "").strip()
        if not term:
            continue
        key = term.lower()
        if key in gt_term_set:
            continue
        zh = (
            item.get("target_translations", {}).get("zh", "").strip()
            or term
        )
        wiki_fill_entries.append(
            {
                "term": term,
                "target_translations": {"zh": zh},
                "source": "wiki_fill",
            }
        )

    print(f"[INFO] ACL GT unique terms: {len(gt_glossary_entries)}")
    print(f"[INFO] Wiki filler candidates: {len(wiki_fill_entries)}")

    with open(ACL_GT_TERMS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(gt_translation_95, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved ACL GT (95) translation map -> {ACL_GT_TERMS_OUTPUT}")

    for gs in GLOSSARY_SIZES:
        assert gs >= len(gt_glossary_entries), (
            f"Glossary size gs={gs} is smaller than GT term count={len(gt_glossary_entries)}. "
            "Increase gs or revise policy."
        )
        need_fill = gs - len(gt_glossary_entries)
        assert need_fill <= len(wiki_fill_entries), (
            f"Not enough wiki terms to fill gs={gs}: need {need_fill}, have {len(wiki_fill_entries)}."
        )

        merged = gt_glossary_entries + wiki_fill_entries[:need_fill]
        assert len(merged) == gs, f"Merged glossary size mismatch for gs={gs}: got {len(merged)}"

        out_path = output_dir / f"acl_glossary_gs{gs}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        n_gt = sum(1 for x in merged if x["source"] == "acl_gt")
        n_wiki = sum(1 for x in merged if x["source"] == "wiki_fill")
        print(f"[INFO] gs={gs}: total={len(merged)} gt={n_gt} wiki={n_wiki} -> {out_path}")


if __name__ == "__main__":
    main()
