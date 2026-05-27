#!/usr/bin/env python3

"""
Compute Term Copy Rate (TCR) from SimulEval agent runtime JSONL log.

TCR measures: of the GT terms that (1) appear in the reference translation
(i.e. are relevant to the talk) AND (2) were provided in the term_map,
how many did the model actually adopt in its output?

  TCR = |relevant ∩ provided ∩ adopted| / |relevant ∩ provided|

This requires:
  - runtime JSONL:  to know which terms were provided in term_maps
  - reference text:  to know which terms are relevant to the talk
  - model output:    to check adoption (from runtime JSONL llm_output records)

Input:  runtime JSONL + reference target file (Chinese translations)
Output: TSV line:  TCR <float> TCR_ADOPTED <int> TCR_TOTAL <int>
"""

from __future__ import annotations

# ======Configuration=====
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

TARGET_LANG_DEFAULT = "zh"
# ======Configuration=====


def _load_glossary_terms(path: Path, target_lang: str) -> Dict[str, str]:
    """Load glossary, return {zh_translation: english_term}."""
    with open(path, "r", encoding="utf-8") as f:
        gdata = json.load(f)

    result: Dict[str, str] = {}
    if isinstance(gdata, dict):
        for k, v in gdata.items():
            if not isinstance(v, dict):
                continue
            term = v.get("term", k)
            zh = v.get("target_translations", {}).get(target_lang, "")
            if zh:
                result[zh] = term
    elif isinstance(gdata, list):
        for entry in gdata:
            if not isinstance(entry, dict):
                continue
            term = entry.get("term", "")
            zh = entry.get("target_translations", {}).get(target_lang, "")
            if zh:
                result[zh] = term
    else:
        assert False, f"Unexpected glossary format: {type(gdata)}"
    return result


def compute_tcr(
    log_path: Path,
    ref_text: str,
    glossary_zh_to_term: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """
    Compute TCR from runtime log + reference text.

    Steps:
      1. From runtime JSONL: collect provided translations (from term_maps)
         and full model output.
      2. If glossary given: filter provided terms to GT terms only.
      3. Relevant = provided terms whose translation appears in ref_text.
      4. Adopted = relevant terms whose translation appears in model output.
      5. TCR = adopted / relevant.
    """
    provided_translations: Dict[str, str] = {}  # zh_translation -> term
    output_parts: List[str] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = rec.get("type", "")
            if rtype == "llm_input":
                for ref in rec.get("references") or []:
                    term = (ref.get("term") or "").strip()
                    translation = (ref.get("translation") or "").strip()
                    if not translation:
                        continue
                    if translation not in provided_translations:
                        provided_translations[translation] = term
            elif rtype == "llm_output":
                text = rec.get("text", "")
                if text:
                    output_parts.append(text)

    assert provided_translations, f"No terms found in term_maps in {log_path}"

    # Filter to GT terms if glossary provided
    if glossary_zh_to_term is not None:
        gt_zh_set = set(glossary_zh_to_term.keys())
        provided_translations = {
            zh: term for zh, term in provided_translations.items()
            if zh in gt_zh_set
        }

    full_output = "".join(output_parts)

    relevant = {zh: term for zh, term in provided_translations.items() if zh in ref_text}
    adopted = {zh for zh in relevant if zh in full_output}

    total = len(relevant)
    adopted_count = len(adopted)

    return {
        "tcr": adopted_count / total if total > 0 else 0.0,
        "adopted": adopted_count,
        "total": total,
        "provided_unique": len(provided_translations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute TCR from agent runtime JSONL + reference text."
    )
    parser.add_argument(
        "--runtime-log", type=str, required=True,
        help="Path to runtime JSONL file (may contain multiple talks).",
    )
    parser.add_argument(
        "--ref-file", type=str, required=True,
        help="Path to reference target text file (one sentence per line).",
    )
    parser.add_argument(
        "--target-lang", type=str, default=TARGET_LANG_DEFAULT,
        help="Target language code for glossary translation lookup.",
    )
    parser.add_argument(
        "--glossary-path", type=str, default="",
        help="Optional glossary JSON to filter to GT terms only.",
    )
    args = parser.parse_args()

    log_path = Path(args.runtime_log)
    assert log_path.is_file(), f"Runtime log not found: {log_path}"

    ref_path = Path(args.ref_file)
    assert ref_path.is_file(), f"Reference file not found: {ref_path}"
    ref_text = ref_path.read_text(encoding="utf-8")

    glossary_zh_to_term: Optional[Dict[str, str]] = None
    if args.glossary_path:
        glossary_zh_to_term = _load_glossary_terms(
            Path(args.glossary_path), args.target_lang,
        )

    result = compute_tcr(log_path, ref_text, glossary_zh_to_term)
    print(
        "TCR\t{tcr:.4f}\tTCR_ADOPTED\t{adopted:d}\tTCR_TOTAL\t{total:d}".format(
            tcr=result["tcr"],
            adopted=int(result["adopted"]),
            total=int(result["total"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
