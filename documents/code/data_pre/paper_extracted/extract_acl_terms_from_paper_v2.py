#!/usr/bin/env python3
"""
Extract technical glossary terms from ACL papers using LLM API.
Version 2: PDF to text (no upload), with per-chunk extraction and integrated translations.

All user-facing strings are in English.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ======Configuration=====
# Directories
PAPERS_DIR = Path(__file__).parent / "papers"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_GLOSSARIES_DIR = OUTPUT_DIR / "extracted_glossaries_by_paper"
OUTPUT_LISTS_DIR = OUTPUT_DIR / "extracted_glossary_lists_by_paper"
OUTPUT_MANIFEST = OUTPUT_DIR / "extracted_glossary_by_paper_manifest.json"

# LLM provider (this version supports both google.genai and google.generativeai)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
if LLM_PROVIDER != "gemini":
    print("This version only supports Gemini API.")
    print("Set GEMINI_API_KEY environment variable.")
    exit(1)

# Try both google.genai (newer SDK) and google.generativeai (older SDK)
try:
    from google import genai
    from google.genai import types

    GEMINI_SDK = "genai"
except Exception:
    try:
        import google.generativeai as genai  # type: ignore

        GEMINI_SDK = "generativeai"
    except Exception as e:
        raise RuntimeError("google-genai or google-generativeai is required. Install one and set GEMINI_API_KEY.") from e

if not os.environ.get("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set")
    exit(1)

if GEMINI_SDK == "genai":
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
else:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])  # type: ignore
    client = None  # type: ignore

MODEL_NAME = os.environ.get("MODEL_NAME", "models/gemini-2.5-flash")
print(f"Using Gemini model: {MODEL_NAME} (SDK: {GEMINI_SDK})")

# LLM params
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))

# PDF->text processing
PDF_TEXT_MIN_CHARS = int(os.environ.get("PDF_TEXT_MIN_CHARS", "1000"))
PDF_TEXT_MAX_CHARS = int(os.environ.get("PDF_TEXT_MAX_CHARS", "250000"))

# Chunking (split paper text into smaller pieces to avoid LLM truncation)
TEXT_CHUNK_CHARS = int(os.environ.get("TEXT_CHUNK_CHARS", "25000"))
MAX_TEXT_CHUNKS = int(os.environ.get("MAX_TEXT_CHUNKS", "6"))
CHUNK_SEPARATOR = "\n\n===NEXT_CHUNK===\n\n"
USE_PER_CHUNK_LLM = int(os.environ.get("USE_PER_CHUNK_LLM", "1"))
# ======Configuration=====

PAPER_TEXT_TOKEN = "__PAPER_TEXT__"

EXTRACTION_PROMPT_TEMPLATE = """Extract technical glossary terms from the following ACL paper text and provide translations.

Focus on:
- Domain-specific terms (NLP / speech / ML)
- Terms likely mentioned in oral presentations
- Substantive technical concepts (not common words)
- Acronyms

For each term, provide:
1. The term itself (in English)
2. Translations in Chinese (zh), German (de), and Japanese (ja)

Return ONLY a valid JSON array (no markdown, no commentary). Format:
[
  {
    "term": "BERT",
    "target_translations": {
      "zh": "BERT",
      "de": "BERT",
      "ja": "BERT"
    }
  },
  {
    "term": "attention mechanism",
    "target_translations": {
      "zh": "注意力机制",
      "de": "Aufmerksamkeitsmechanismus",
      "ja": "注意機構"
    }
  }
]

Constraints:
- Extract as many relevant technical terms as you can.
- Do not include author names, affiliations, or generic phrases.
- If the same term appears multiple times, list it only once.
- For acronyms, provide the acronym itself (not the full form).

PAPER_TEXT:
__PAPER_TEXT__

Return ONLY the JSON array, nothing else."""


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _extract_json_array(text: str) -> str:
    """
    Best-effort extraction of the first JSON array from a mixed response.
    """
    t = _strip_code_fences(text)
    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1].strip()
    m = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", t)
    if m:
        return m.group(0).strip()
    return t


def _try_parse_terms(text: str) -> List[Dict]:
    candidate = _extract_json_array(text)
    parsed = json.loads(candidate)
    if not isinstance(parsed, list):
        raise ValueError("Parsed JSON is not a list")
    return parsed


def _safe_get_response_text(resp: Any) -> str:
    """
    Extract text from response in a defensive way for both google.genai and google.generativeai SDKs.
    """
    chunks = []
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            cand = resp.candidates[0]
            if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts") and cand.content.parts:
                for p in cand.content.parts:
                    if hasattr(p, "text") and p.text:
                        chunks.append(p.text)
        elif hasattr(resp, "text"):
            return resp.text.strip()
    except Exception:
        pass

    if chunks:
        return "".join(chunks).strip()

    # Fallback
    try:
        return str(resp).strip()
    except Exception:
        return ""


def _repair_json_with_llm(client: Any, model_name: str, broken_json_text: str) -> str:
    """
    Ask the model to repair/complete the JSON. This is cheaper than re-reading the PDF.
    """
    repair_prompt = (
        "Fix and complete the JSON array below. "
        "Return ONLY a valid JSON array. "
        'Do not add any markdown, do not add any extra keys beyond "term" and "target_translations".\n\n'
        "BROKEN_JSON:\n"
        f"{broken_json_text}\n"
    )

    if GEMINI_SDK == "genai":
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=repair_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=min(MAX_OUTPUT_TOKENS, 8192),
                    response_mime_type="application/json",
                ),
            )
        except Exception:
            resp = client.models.generate_content(
                model=model_name,
                contents=repair_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=min(MAX_OUTPUT_TOKENS, 8192),
                ),
            )
    else:
        model = genai.GenerativeModel(model_name)  # type: ignore
        try:
            resp = model.generate_content(
                repair_prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": min(MAX_OUTPUT_TOKENS, 8192),
                    "response_mime_type": "application/json",
                },
            )
        except TypeError:
            resp = model.generate_content(
                repair_prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": min(MAX_OUTPUT_TOKENS, 8192),
                },
            )

    return _safe_get_response_text(resp)


def _normalize_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_pdf_text_pdfminer(pdf_path: Path) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:
        return None
    try:
        return str(extract_text(str(pdf_path)))
    except Exception:
        return None


def _extract_pdf_text_pypdf(pdf_path: Path) -> Optional[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return None
    try:
        reader = PdfReader(str(pdf_path))
        parts: List[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                parts.append(t)
        return "\n\n".join(parts)
    except Exception:
        return None


def _pdf_to_text(pdf_path: Path) -> str:
    txt = _extract_pdf_text_pdfminer(pdf_path)
    if not txt:
        txt = _extract_pdf_text_pypdf(pdf_path)
    if not txt:
        raise RuntimeError("Failed to extract text from PDF (pdfminer and pypdf both failed).")
    txt = _normalize_text(txt)
    if len(txt) < PDF_TEXT_MIN_CHARS:
        raise RuntimeError(f"Extracted text too short: {len(txt)} chars")
    if len(txt) > PDF_TEXT_MAX_CHARS:
        txt = txt[:PDF_TEXT_MAX_CHARS]
    return txt


def _split_text_into_chunks(text: str, chunk_chars: int, max_chunks: int) -> List[str]:
    if chunk_chars <= 0:
        return [text]
    if max_chunks <= 0:
        return [text]
    chunks: List[str] = []
    for i in range(0, len(text), chunk_chars):
        chunks.append(text[i : i + chunk_chars])
    if len(chunks) <= max_chunks:
        return chunks
    idxs = [int(round(j * (len(chunks) - 1) / float(max_chunks - 1))) for j in range(max_chunks)]
    out = [chunks[i] for i in idxs]
    return out


def extract_terms_from_paper_text(client: Any, paper_text: str) -> List[Dict]:
    """Ask Gemini to extract terms from already-extracted paper text."""
    response_text = ""
    try:
        chunks = _split_text_into_chunks(paper_text, TEXT_CHUNK_CHARS, MAX_TEXT_CHUNKS)
        print(f"  Split into {len(chunks)} chunks (chunk_chars={TEXT_CHUNK_CHARS}, max_chunks={MAX_TEXT_CHUNKS})")

        if not USE_PER_CHUNK_LLM:
            packed = CHUNK_SEPARATOR.join([f"[CHUNK {i+1}/{len(chunks)}]\n{c}" for i, c in enumerate(chunks)])
            return _extract_terms_single_call(client, packed, chunk_idx=0)

        all_terms: List[Dict] = []
        for i, c in enumerate(chunks):
            packed = f"[CHUNK {i+1}/{len(chunks)}]\n{c}"
            terms = _extract_terms_single_call(client, packed, chunk_idx=i + 1)
            all_terms.extend(terms)
        print(f"  Total terms from all chunks (before dedup): {len(all_terms)}")
        return all_terms
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse JSON: {e}")
        print(f"  Response: {response_text[:500]}")
        return []
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return []


def _extract_terms_single_call(client: Any, packed_text: str, chunk_idx: int = 0) -> List[Dict]:
    prompt = EXTRACTION_PROMPT_TEMPLATE.replace(PAPER_TEXT_TOKEN, packed_text)
    print(f"  [Chunk {chunk_idx}] packed_text_len={len(packed_text)} chars, calling LLM...")

    if GEMINI_SDK == "genai":
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                ),
            )
        except Exception:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
    else:
        model = genai.GenerativeModel(MODEL_NAME)  # type: ignore
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                },
            )
        except TypeError:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
            )

    response_text = _safe_get_response_text(resp)
    print(f"  [Chunk {chunk_idx}] response_len={len(response_text)} chars")
    print(f"  [Chunk {chunk_idx}] response_head(200): {response_text[:200]}")
    print(f"  [Chunk {chunk_idx}] response_tail(200): {response_text[-200:]}")

    try:
        terms = _try_parse_terms(response_text)
        print(f"  [Chunk {chunk_idx}] ✓ Parsed {len(terms)} terms")
        return terms
    except Exception as parse_err:
        print(f"  [Chunk {chunk_idx}] JSON parse failed, attempting repair: {type(parse_err).__name__}: {parse_err}")
        repaired = _repair_json_with_llm(client, MODEL_NAME, response_text)
        print(f"  [Chunk {chunk_idx}] repaired_len={len(repaired)} chars, repaired_head(200): {repaired[:200]}")
        try:
            terms = _try_parse_terms(repaired)
            print(f"  [Chunk {chunk_idx}] ✓ Parsed {len(terms)} terms after repair")
            return terms
        except Exception as parse_err_2:
            print(f"  [Chunk {chunk_idx}] ✗ Repair parse failed: {type(parse_err_2).__name__}: {parse_err_2}")
            print(f"  [Chunk {chunk_idx}] repaired_body(800): {_strip_code_fences(repaired)[:800]}")
            return []


def _paper_terms_to_glossary(terms: List[Dict[str, Any]], pdf_name: str) -> Dict[str, Dict[str, Any]]:
    seen_lower: set[str] = set()
    out: Dict[str, Dict[str, Any]] = {}
    for obj in terms:
        term = str(obj.get("term", "")).strip()
        if not term:
            continue
        key_lower = term.lower()
        if key_lower in seen_lower:
            continue
        seen_lower.add(key_lower)

        target_translations = obj.get("target_translations", {})
        if not isinstance(target_translations, dict):
            target_translations = {}

        # Use lowercased key for dict (consistent with existing glossary format)
        out[key_lower] = {
            "term": term,
            "classification_reason": "llm_extracted",
            "confused": False,
            "short_description": "",
            "full_form": "",
            "is_acronym": False,
            "source_paper": pdf_name,
            "target_translations": target_translations,
            "target_translation_source": "llm translation" if target_translations else "",
            "url": "",
        }
    return out


def extract_terms_from_paper(client: Any, pdf_path: Path) -> Dict[str, Dict[str, Any]]:
    """Extract terms from a single PDF paper (returns glossary dict)."""
    print(f"  Extracting text from PDF: {pdf_path.name}")
    paper_text = _pdf_to_text(pdf_path)
    print(f"  Extracted {len(paper_text)} chars from {pdf_path.name}")

    terms = extract_terms_from_paper_text(client, paper_text)
    print(f"  ✓ Extracted terms (paper): {len(terms)}")

    glossary = _paper_terms_to_glossary(terms, pdf_path.name)
    return glossary


def main():
    """Main function to process all papers and extract glossary (per-paper output)."""
    print("=" * 80)
    print("ACL Paper Glossary Extraction (v2 - PDF to text, no upload)")
    print("=" * 80)
    print(f"\nPapers directory: {PAPERS_DIR}")

    if not PAPERS_DIR.exists():
        print(f"Error: Papers directory not found: {PAPERS_DIR}")
        return

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files\n")

    if not pdf_files:
        print("No PDF files found!")
        return

    OUTPUT_GLOSSARIES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LISTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "papers": {},
        "total_papers": len(pdf_files),
    }

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        print("-" * 80)

        glossary = extract_terms_from_paper(client, pdf_path)

        paper_id = pdf_path.stem  # e.g. "2022.acl-long.110"
        glossary_out_path = OUTPUT_GLOSSARIES_DIR / f"extracted_glossary__{paper_id}.json"
        glossary_list_out_path = OUTPUT_LISTS_DIR / f"extracted_glossary_list__{paper_id}.json"

        with glossary_out_path.open("w", encoding="utf-8") as f:
            json.dump(glossary, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Paper glossary saved: {glossary_out_path}")

        glossary_list = sorted(glossary.keys(), key=lambda s: s.lower())
        with glossary_list_out_path.open("w", encoding="utf-8") as f:
            json.dump(glossary_list, f, indent=2, ensure_ascii=False)

        manifest["papers"][paper_id] = {
            "pdf_name": pdf_path.name,
            "glossary_path": str(glossary_out_path),
            "glossary_list_path": str(glossary_list_out_path),
            "term_count": len(glossary),
        }

        if i < len(pdf_files):
            time.sleep(2)

    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    for paper_id, info in manifest["papers"].items():
        print(f"  {paper_id}: {info['term_count']} terms")
    print(f"\n✓ Manifest saved to: {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
