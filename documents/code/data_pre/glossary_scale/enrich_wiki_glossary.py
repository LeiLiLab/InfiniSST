#!/usr/bin/env python3
"""
Enrich wiki glossary JSON files with:
  1. short_desc – from Wikipedia action API page descriptions
  2. target_translations – {zh, ja, de}, preferably from Wikipedia langlinks
     and optionally from Gemini as a fallback

Pipeline phases:
  Phase 1 (--phase wiki_desc):  Fetch short_desc from Wikipedia.  No API key needed.
  Phase 2 (--phase wiki_langlinks): Fetch zh/ja/de Wikipedia page titles. No API key.
  Phase 3 (--phase translate): Batch-translate missing terms+desc via Gemini.

Intermediate state is saved after each batch so the job is resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ======Configuration=====
INPUT_PATH = str(
    Path(__file__).parent / "wiki_glossary_nlp_ai_cs.json"
)
OUTPUT_PATH = str(
    Path(__file__).parent / "wiki_glossary_nlp_ai_cs_enriched.json"
)
PRESET_PATHS = {
    "nlp_ai_cs": (
        Path(__file__).parent / "wiki_glossary_nlp_ai_cs.json",
        Path(__file__).parent / "wiki_glossary_nlp_ai_cs_enriched.json",
    ),
    "medicine": (
        Path(__file__).parent / "wiki_glossary_medicine.json",
        Path(__file__).parent / "wiki_glossary_medicine_enriched.json",
    ),
}
DOMAIN_TRANSLATION_CONTEXT = {
    "nlp_ai_cs": "computer science and NLP terminology",
    "medicine": "medical, pharmacological, oncology, and clinical terminology",
}

WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary"
WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "InfiniSST-GlossaryEnrich/1.0 (research; jiaxuanluo@example.com)",
}
WIKI_BATCH_SIZE = 50
WIKI_LANGLINKS_BATCH_SIZE = 50
WIKI_DELAY_SEC = 0.02
TARGET_LANGS = ("zh", "ja", "de")

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_BATCH_SIZE = 80
GEMINI_DELAY_SEC = 0.5
GEMINI_MAX_RETRIES = 3
SAVE_EVERY = 200
# ======Configuration=====


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: Wikipedia short descriptions (batch via action API)
# ---------------------------------------------------------------------------

def fetch_short_desc_batch(titles: List[str]) -> Dict[str, str]:
    """Fetch short descriptions for a batch of titles via MediaWiki action API."""
    result: Dict[str, str] = {}
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "description",
        "format": "json",
        "formatversion": "2",
    }
    try:
        resp = requests.get(
            WIKI_ACTION_API, params=params, headers=WIKI_HEADERS, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        for page in data.get("query", {}).get("pages", []):
            title = page.get("title", "")
            desc = page.get("description", "")
            if title and desc:
                result[title.lower()] = desc
    except Exception as exc:
        _log(f"  WARN batch fetch failed: {exc}")
    return result


def phase_wiki_desc(glossary: List[Dict], output_path: str) -> List[Dict]:
    """Fetch Wikipedia short descriptions for all terms."""
    _log("Phase 1: Fetching Wikipedia short descriptions")

    need_desc = [i for i, item in enumerate(glossary) if not item.get("short_desc")]
    _log(f"  {len(need_desc)} terms need short_desc (of {len(glossary)} total)")

    done = 0
    for batch_start in range(0, len(need_desc), WIKI_BATCH_SIZE):
        batch_idx = need_desc[batch_start:batch_start + WIKI_BATCH_SIZE]
        batch_terms = [glossary[i]["term"] for i in batch_idx]

        descs = fetch_short_desc_batch(batch_terms)

        for i in batch_idx:
            key = glossary[i]["term"].lower()
            if key in descs:
                glossary[i]["short_desc"] = descs[key]

        done += len(batch_idx)
        time.sleep(WIKI_DELAY_SEC)

        if done % SAVE_EVERY == 0 or done >= len(need_desc):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)
            has_desc = sum(1 for x in glossary if x.get("short_desc"))
            _log(f"  {done}/{len(need_desc)} fetched, {has_desc} with desc. Saved.")

    has_desc = sum(1 for x in glossary if x.get("short_desc"))
    _log(f"Phase 1 done: {has_desc}/{len(glossary)} terms have short_desc")
    return glossary


# ---------------------------------------------------------------------------
# Phase 2: Wikipedia language links
# ---------------------------------------------------------------------------

def _has_all_target_translations(item: Dict) -> bool:
    translations = item.get("target_translations") or {}
    return all(translations.get(lang) for lang in TARGET_LANGS)


def _resolve_alias(alias_map: Dict[str, str], key: str) -> str:
    seen = set()
    while key in alias_map and key not in seen:
        seen.add(key)
        key = alias_map[key]
    return key


def fetch_langlinks_batch(titles: List[str]) -> Dict[str, Dict[str, str]]:
    """Fetch zh/ja/de page-title langlinks for a batch of English titles."""
    result: Dict[str, Dict[str, str]] = {title.lower(): {} for title in titles}
    for lang in TARGET_LANGS:
        params = {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "langlinks",
            "lllang": lang,
            "lllimit": "max",
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
        }
        try:
            resp = requests.get(
                WIKI_ACTION_API, params=params, headers=WIKI_HEADERS, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            _log(f"  WARN langlinks batch failed lang={lang}: {exc}")
            continue

        query = data.get("query", {})
        alias_map: Dict[str, str] = {}
        for norm in query.get("normalized", []) or []:
            src = str(norm.get("from") or "").lower()
            dst = str(norm.get("to") or "").lower()
            if src and dst:
                alias_map[src] = dst
        for redir in query.get("redirects", []) or []:
            src = str(redir.get("from") or "").lower()
            dst = str(redir.get("to") or "").lower()
            if src and dst:
                alias_map[src] = dst

        canonical: Dict[str, str] = {}
        for page in query.get("pages", []) or []:
            title = str(page.get("title") or "").lower()
            if not title:
                continue
            for link in page.get("langlinks", []) or []:
                if link.get("lang") == lang and link.get("title"):
                    canonical[title] = str(link["title"]).strip()
                    break

        for title in titles:
            key = title.lower()
            resolved = _resolve_alias(alias_map, key)
            if resolved in canonical:
                result[key][lang] = canonical[resolved]
    return result


def phase_wiki_langlinks(glossary: List[Dict], output_path: str) -> List[Dict]:
    """Fetch Wikipedia zh/ja/de page-title translations for all terms."""
    _log("Phase 2: Fetching Wikipedia langlinks for zh/ja/de")

    need_links = [
        i for i, item in enumerate(glossary) if not _has_all_target_translations(item)
    ]
    _log(f"  {len(need_links)} terms need langlinks (of {len(glossary)} total)")

    done = 0
    for batch_start in range(0, len(need_links), WIKI_LANGLINKS_BATCH_SIZE):
        batch_idx = need_links[batch_start:batch_start + WIKI_LANGLINKS_BATCH_SIZE]
        batch_terms = [glossary[i]["term"] for i in batch_idx]

        langlinks = fetch_langlinks_batch(batch_terms)

        for i in batch_idx:
            key = glossary[i]["term"].lower()
            found = langlinks.get(key) or {}
            if not found:
                continue
            current = dict(glossary[i].get("target_translations") or {})
            current.update({lang: title for lang, title in found.items() if title})
            if current:
                glossary[i]["target_translations"] = current

        done += len(batch_idx)
        time.sleep(WIKI_DELAY_SEC)

        if done % SAVE_EVERY == 0 or done >= len(need_links):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)
            full = sum(1 for x in glossary if _has_all_target_translations(x))
            any_lang = sum(1 for x in glossary if x.get("target_translations"))
            _log(
                f"  {done}/{len(need_links)} fetched, "
                f"{full} with all langs, {any_lang} with any lang. Saved."
            )

    full = sum(1 for x in glossary if _has_all_target_translations(x))
    any_lang = sum(1 for x in glossary if x.get("target_translations"))
    _log(
        f"Phase 2 done: {full}/{len(glossary)} terms have all target langlinks; "
        f"{any_lang} have at least one"
    )
    return glossary


# ---------------------------------------------------------------------------
# Phase 3: Gemini translation fallback
# ---------------------------------------------------------------------------

def build_translation_prompt(items: List[Dict], domain: str = "nlp_ai_cs") -> str:
    """Build a prompt for batch translation."""
    lines = []
    for idx, item in enumerate(items):
        term = item["term"]
        desc = item.get("short_desc", "")
        if desc:
            lines.append(f"{idx}|{term}|{desc}")
        else:
            lines.append(f"{idx}|{term}|")

    domain_context = DOMAIN_TRANSLATION_CONTEXT.get(
        domain, DOMAIN_TRANSLATION_CONTEXT["nlp_ai_cs"]
    )
    prompt = (
        f"You are a professional translator specializing in {domain_context}.\n"
        "Below is a list of English technical terms with optional short descriptions.\n"
        "For each line, provide translations of the TERM (not the description) into Chinese (zh), Japanese (ja), and German (de).\n"
        "The description is provided only as context to help you understand the term's meaning.\n\n"
        "Rules:\n"
        "- If the term is a proper noun, abbreviation, brand name, person name, or widely used in its original English form "
        "(e.g. BERT, GPT, TensorFlow, Alan Turing), keep the original English term as the translation.\n"
        "- If the term is difficult to translate or has no established translation in the target language, "
        "use the original English term as-is.\n"
        "- Output EXACTLY one line per input, in the format: idx|zh_translation|ja_translation|de_translation\n"
        "- Do NOT add explanations, headers, or extra text.\n\n"
        "Input:\n"
    )
    prompt += "\n".join(lines)
    return prompt


def parse_gemini_response(text: str, count: int) -> List[Optional[Dict[str, str]]]:
    """Parse Gemini response into list of {zh, ja, de} dicts."""
    results: List[Optional[Dict[str, str]]] = [None] * count
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0].strip())
        except ValueError:
            continue
        if 0 <= idx < count:
            results[idx] = {
                "zh": parts[1].strip(),
                "ja": parts[2].strip(),
                "de": parts[3].strip(),
            }
    return results


def translate_batch_gemini(
    items: List[Dict], api_key: str, domain: str = "nlp_ai_cs"
) -> List[Optional[Dict[str, str]]]:
    """Call Gemini API to translate a batch of terms."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = build_translation_prompt(items, domain=domain)

    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            text = response.text
            return parse_gemini_response(text, len(items))
        except Exception as exc:
            _log(f"  Gemini attempt {attempt+1} failed: {exc}")
            if attempt < GEMINI_MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))

    return [None] * len(items)


def phase_translate(
    glossary: List[Dict],
    output_path: str,
    api_key: str,
    domain: str = "nlp_ai_cs",
    allow_empty_batches: bool = False,
) -> List[Dict]:
    """Fill missing target translations using Gemini, preserving wiki langlinks."""
    _log("Phase 3: Filling missing target translations via Gemini")

    need_trans = [
        i for i, item in enumerate(glossary)
        if not _has_all_target_translations(item)
    ]
    _log(f"  {len(need_trans)} terms need translation (of {len(glossary)} total)")

    done = 0
    for batch_start in range(0, len(need_trans), GEMINI_BATCH_SIZE):
        batch_idx = need_trans[batch_start:batch_start + GEMINI_BATCH_SIZE]
        batch_items = [glossary[i] for i in batch_idx]

        translations = translate_batch_gemini(batch_items, api_key, domain=domain)
        if not any(translations) and not allow_empty_batches:
            raise RuntimeError(
                "Gemini returned no translations for a full batch; aborting so "
                "the job does not silently burn through all batches. Check API "
                "quota/key or rerun with --allow_empty_translation_batches."
            )

        for i, trans in zip(batch_idx, translations):
            if trans:
                current = dict(glossary[i].get("target_translations") or {})
                for lang in TARGET_LANGS:
                    if not current.get(lang) and trans.get(lang):
                        current[lang] = trans[lang]
                glossary[i]["target_translations"] = current

        done += len(batch_idx)
        time.sleep(GEMINI_DELAY_SEC)

        if done % SAVE_EVERY == 0 or done >= len(need_trans):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)
            has_trans = sum(
                1 for x in glossary if x.get("target_translations", {}).get("zh")
            )
            full = sum(1 for x in glossary if _has_all_target_translations(x))
            _log(
                f"  {done}/{len(need_trans)} translated, "
                f"{full} with all langs, {has_trans} with zh. Saved."
            )

    has_trans = sum(
        1 for x in glossary if x.get("target_translations", {}).get("zh")
    )
    full = sum(1 for x in glossary if _has_all_target_translations(x))
    _log(
        f"Phase 3 done: {full}/{len(glossary)} terms have all target translations; "
        f"{has_trans} have zh"
    )
    return glossary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global GEMINI_MODEL
    parser = argparse.ArgumentParser(description="Enrich wiki glossary with desc + translations")
    parser.add_argument("--preset", choices=sorted(PRESET_PATHS), default="nlp_ai_cs")
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--phase",
        choices=["wiki_desc", "wiki_langlinks", "wiki_all", "translate", "both"],
        default="wiki_all",
    )
    parser.add_argument("--gemini_api_key", type=str, default="",
                        help="Gemini API key (or set GEMINI_API_KEY / GOOGLE_API_KEY)")
    parser.add_argument("--gemini_model", type=str, default=GEMINI_MODEL)
    parser.add_argument(
        "--allow_empty_translation_batches",
        action="store_true",
        default=False,
        help="Continue even if a Gemini batch returns no parsed translations.",
    )
    args = parser.parse_args()
    GEMINI_MODEL = args.gemini_model

    preset_input, preset_output = PRESET_PATHS[args.preset]
    if not args.input:
        args.input = str(preset_input)
    if not args.output:
        args.output = str(preset_output)

    api_key = (
        args.gemini_api_key
        or os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("GOOGLE_API_KEY", "")
    )

    if os.path.exists(args.output):
        _log(f"Resuming from existing output: {args.output}")
        with open(args.output, encoding="utf-8") as f:
            glossary = json.load(f)
    else:
        _log(f"Loading input: {args.input}")
        with open(args.input, encoding="utf-8") as f:
            glossary = json.load(f)

    _log(f"Glossary: {len(glossary)} terms")

    if args.phase in ("wiki_desc", "wiki_all", "both"):
        glossary = phase_wiki_desc(glossary, args.output)

    if args.phase in ("wiki_langlinks", "wiki_all", "both"):
        glossary = phase_wiki_langlinks(glossary, args.output)

    if args.phase in ("translate", "both"):
        assert api_key, (
            "Gemini API key required for translation fallback. "
            "Set --gemini_api_key, GEMINI_API_KEY, or GOOGLE_API_KEY."
        )
        glossary = phase_translate(
            glossary,
            args.output,
            api_key,
            domain=args.preset,
            allow_empty_batches=args.allow_empty_translation_batches,
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

    has_desc = sum(1 for x in glossary if x.get("short_desc"))
    has_zh = sum(1 for x in glossary if x.get("target_translations", {}).get("zh"))
    _log(f"Final: {len(glossary)} terms, {has_desc} with desc, {has_zh} with zh")
    _log(f"Output: {args.output}")


if __name__ == "__main__":
    main()
