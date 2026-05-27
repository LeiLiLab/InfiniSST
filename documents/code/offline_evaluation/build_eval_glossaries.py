#!/usr/bin/env python3
"""
Build eval glossaries for DEV and ACL with sizes {100, 1000, 10000}.

Policy:
- Always include ALL GT terms first.
- If requested gs > GT count, fill with wiki terms.
- If requested gs <= GT count, keep all GT terms (do not truncate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


# ======Configuration=====
DEV_RETRIEVER_RESULTS = "/mnt/gemini/data2/jiaxuanluo/tcr_fcr_eval/dev_retriever_results.jsonl"
ACL_DEV_JSONL = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
)
ACL_GT_TRANSLATION_JSON = "/mnt/gemini/data2/jiaxuanluo/tcr_fcr_eval/acl_combined_glossary.json"
WIKI_ENRICHED_JSON = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/"
    "wiki_glossary_nlp_ai_cs_enriched.json"
)
OUTPUT_DIR = "/mnt/gemini/data2/jiaxuanluo/tcr_fcr_eval"
GLOSSARY_SIZES = [100, 1000, 10000]
# ======Configuration=====


def _load_wiki_fillers() -> List[Tuple[str, str]]:
    wiki = json.load(open(WIKI_ENRICHED_JSON, "r", encoding="utf-8"))
    fillers: List[Tuple[str, str]] = []
    for item in wiki:
        term = item.get("term", "").strip()
        if not term:
            continue
        zh = item.get("target_translations", {}).get("zh", "").strip() or term
        fillers.append((term, zh))
    return fillers


def _build_glossary_entries(
    gt_map: Dict[str, Dict[str, str]],
    wiki_fillers: List[Tuple[str, str]],
    gs: int,
) -> List[Dict]:
    gt_entries = [
        {
            "term": v["term"],
            "target_translations": {"zh": v["zh"]},
            "source": "gt",
        }
        for _, v in sorted(gt_map.items(), key=lambda kv: kv[0])
    ]
    gt_keys = set(gt_map.keys())

    wiki_entries = []
    for term, zh in wiki_fillers:
        if term.lower() in gt_keys:
            continue
        wiki_entries.append(
            {
                "term": term,
                "target_translations": {"zh": zh},
                "source": "wiki_fill",
            }
        )

    if gs <= len(gt_entries):
        # User-requested policy: never truncate GT terms.
        return gt_entries

    need = gs - len(gt_entries)
    assert need <= len(wiki_entries), (
        f"Not enough wiki fillers for gs={gs}: need={need}, have={len(wiki_entries)}"
    )
    return gt_entries + wiki_entries[:need]


def _build_dev_gt_map() -> Dict[str, Dict[str, str]]:
    gt_map: Dict[str, Dict[str, str]] = {}
    with open(DEV_RETRIEVER_RESULTS, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for t in row.get("gt_terms", []):
                term = t["term"].strip()
                zh = t["zh"].strip()
                if not term or not zh:
                    continue
                gt_map[term.lower()] = {"term": term, "zh": zh}
    assert len(gt_map) > 0, "No DEV GT terms found."
    return gt_map


def _build_acl_gt_map() -> Dict[str, Dict[str, str]]:
    acl_terms = set()
    with open(ACL_DEV_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            term = (sample.get("term_key") or sample.get("term") or "").strip().lower()
            if term:
                acl_terms.add(term)

    source_map = json.load(open(ACL_GT_TRANSLATION_JSON, "r", encoding="utf-8"))
    gt_map: Dict[str, Dict[str, str]] = {}
    for term in sorted(acl_terms):
        assert term in source_map, f"Missing ACL GT translation: {term}"
        zh = source_map[term].get("zh", "").strip()
        assert zh, f"Empty zh for ACL GT term: {term}"
        gt_map[term] = {"term": source_map[term]["term"], "zh": zh}
    assert len(gt_map) > 0, "No ACL GT terms found."
    return gt_map


def _save_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    wiki_fillers = _load_wiki_fillers()
    dev_gt_map = _build_dev_gt_map()
    acl_gt_map = _build_acl_gt_map()

    _save_json(out_dir / "acl_gt_terms_from_dev95.json", acl_gt_map)

    print(f"[INFO] DEV GT unique terms: {len(dev_gt_map)}")
    print(f"[INFO] ACL GT unique terms: {len(acl_gt_map)}")
    print(f"[INFO] Wiki filler candidates: {len(wiki_fillers)}")

    for gs in GLOSSARY_SIZES:
        dev_entries = _build_glossary_entries(dev_gt_map, wiki_fillers, gs)
        acl_entries = _build_glossary_entries(acl_gt_map, wiki_fillers, gs)

        dev_path = out_dir / f"dev_glossary_gs{gs}.json"
        acl_path = out_dir / f"acl_glossary_gs{gs}.json"
        _save_json(dev_path, dev_entries)
        _save_json(acl_path, acl_entries)

        dev_gt = sum(1 for x in dev_entries if x["source"] == "gt")
        dev_wiki = sum(1 for x in dev_entries if x["source"] == "wiki_fill")
        acl_gt = sum(1 for x in acl_entries if x["source"] == "gt")
        acl_wiki = sum(1 for x in acl_entries if x["source"] == "wiki_fill")

        print(
            f"[INFO] gs={gs} DEV total={len(dev_entries)} gt={dev_gt} wiki={dev_wiki} -> {dev_path}"
        )
        print(
            f"[INFO] gs={gs} ACL total={len(acl_entries)} gt={acl_gt} wiki={acl_wiki} -> {acl_path}"
        )


if __name__ == "__main__":
    main()
