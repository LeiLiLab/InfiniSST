#!/usr/bin/env python3

# ======Configuration=====
MANIFEST_PATH = "/home/jiaxuanluo/InfiniSST/documents/data/data_pre/extracted_glossary_by_paper_manifest.json"
OUTPUT_PATH = "/home/jiaxuanluo/InfiniSST/documents/data/data_pre/merged_paper_glossary.json"

# Normalization
LOWERCASE_KEYS = True
COLLAPSE_WHITESPACE = True

# Fields
SOURCE_PAPERS_FIELD = "source_papers"
PRIMARY_SOURCE_PAPER_FIELD = "source_paper"
# =======================

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_WS_RE = re.compile(r"\s+")


def _normalize_term(term: str) -> str:
    t = term.strip()
    if COLLAPSE_WHITESPACE:
        t = _WS_RE.sub(" ", t)
    if LOWERCASE_KEYS:
        t = t.lower()
    return t


def _as_list_unique(xs: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _prefer_non_empty(a: str, b: str) -> str:
    if a and a.strip():
        return a
    return b


def _prefer_longer(a: str, b: str) -> str:
    a2 = a or ""
    b2 = b or ""
    return a2 if len(a2) >= len(b2) else b2


def _merge_target_translations(
    old: Dict[str, str],
    new: Dict[str, str],
) -> Dict[str, str]:
    merged = dict(old)
    for lang, val in new.items():
        if not val:
            continue
        merged[lang] = _prefer_non_empty(merged.get(lang, ""), val)
    return merged


@dataclass
class MergeStats:
    papers: int = 0
    raw_terms: int = 0
    merged_terms: int = 0
    key_collisions: int = 0
    translation_conflicts: int = 0


def merge_glossary_entry(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
    incoming_source_paper: str,
    stats: MergeStats,
) -> Dict[str, Any]:
    out = dict(existing)

    # Core identity
    out["term"] = _prefer_non_empty(str(out.get("term", "")), str(incoming.get("term", "")))

    # Merge boolean-like fields
    out["confused"] = bool(out.get("confused", False)) or bool(incoming.get("confused", False))
    out["is_acronym"] = bool(out.get("is_acronym", False)) or bool(incoming.get("is_acronym", False))

    # Prefer informative strings
    out["classification_reason"] = _prefer_non_empty(
        str(out.get("classification_reason", "")),
        str(incoming.get("classification_reason", "")),
    )
    out["short_description"] = _prefer_longer(
        str(out.get("short_description", "")),
        str(incoming.get("short_description", "")),
    )
    out["full_form"] = _prefer_longer(str(out.get("full_form", "")), str(incoming.get("full_form", "")))
    out["url"] = _prefer_non_empty(str(out.get("url", "")), str(incoming.get("url", "")))
    out["target_translation_source"] = _prefer_non_empty(
        str(out.get("target_translation_source", "")),
        str(incoming.get("target_translation_source", "")),
    )

    # Target translations
    old_tt = out.get("target_translations") or {}
    new_tt = incoming.get("target_translations") or {}
    if isinstance(old_tt, dict) and isinstance(new_tt, dict):
        # Track conflicts (different non-empty values for same language)
        for lang, val in new_tt.items():
            if not val:
                continue
            prev = old_tt.get(lang, "")
            if prev and prev != val:
                stats.translation_conflicts += 1
        out["target_translations"] = _merge_target_translations(old_tt, new_tt)

    # Aggregate sources
    existing_sources = out.get(SOURCE_PAPERS_FIELD) or []
    if not isinstance(existing_sources, list):
        existing_sources = [str(existing_sources)]
    incoming_sources = []
    if incoming_source_paper:
        incoming_sources.append(incoming_source_paper)
    incoming_sp = str(incoming.get(PRIMARY_SOURCE_PAPER_FIELD, "")).strip()
    if incoming_sp:
        incoming_sources.append(incoming_sp)

    out[SOURCE_PAPERS_FIELD] = _as_list_unique([*existing_sources, *incoming_sources])
    if out[SOURCE_PAPERS_FIELD]:
        out[PRIMARY_SOURCE_PAPER_FIELD] = out[SOURCE_PAPERS_FIELD][0]

    return out


def main() -> None:
    manifest = json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8", errors="replace"))
    papers = manifest.get("papers") or {}

    merged: Dict[str, Dict[str, Any]] = {}
    stats = MergeStats(papers=len(papers))

    for paper_id, info in papers.items():
        glossary_path = str((info or {}).get("glossary_path", "")).strip()
        pdf_name = str((info or {}).get("pdf_name", "")).strip()
        if not glossary_path:
            continue

        paper_tag = pdf_name or f"{paper_id}.pdf"
        gobj = json.loads(Path(glossary_path).read_text(encoding="utf-8", errors="replace"))
        if not isinstance(gobj, dict):
            raise ValueError(f"Glossary must be a JSON object: {glossary_path}")

        for k, v in gobj.items():
            stats.raw_terms += 1
            if not isinstance(v, dict):
                continue

            raw_term = str(v.get("term") or k)
            norm_key = _normalize_term(raw_term)
            if not norm_key:
                continue

            incoming = dict(v)
            # Ensure term field is present and consistent
            incoming["term"] = raw_term
            # Carry a per-paper hint (kept only via aggregation)
            incoming[PRIMARY_SOURCE_PAPER_FIELD] = _prefer_non_empty(str(incoming.get(PRIMARY_SOURCE_PAPER_FIELD, "")), paper_tag)

            if norm_key in merged:
                stats.key_collisions += 1
                merged[norm_key] = merge_glossary_entry(merged[norm_key], incoming, paper_tag, stats)
            else:
                out = dict(incoming)
                out[SOURCE_PAPERS_FIELD] = _as_list_unique([paper_tag])
                out[PRIMARY_SOURCE_PAPER_FIELD] = paper_tag
                merged[norm_key] = out

    stats.merged_terms = len(merged)

    payload = {
        "meta": {
            "source": "merged_paper_glossary",
            "manifest_path": MANIFEST_PATH,
            "papers": sorted(list(papers.keys())),
            "stats": {
                "papers": stats.papers,
                "raw_terms": stats.raw_terms,
                "merged_terms": stats.merged_terms,
                "key_collisions": stats.key_collisions,
                "translation_conflicts": stats.translation_conflicts,
            },
        },
        "terms": merged,
    }

    out_path = Path(OUTPUT_PATH)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[INFO] Wrote merged glossary.")
    print(f"[INFO] output_path={OUTPUT_PATH}")
    print(f"[INFO] papers={stats.papers} raw_terms={stats.raw_terms} merged_terms={stats.merged_terms}")
    print(f"[INFO] key_collisions={stats.key_collisions} translation_conflicts={stats.translation_conflicts}")


if __name__ == "__main__":
    main()

