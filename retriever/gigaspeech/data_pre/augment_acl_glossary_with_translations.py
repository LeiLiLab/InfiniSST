#!/usr/bin/env python3
"""
Augment extracted ACL glossary with target translations (zh/de/ja).

Input: extracted_glossary.json (dict: original_term -> info)
Output: enriched glossary (dict: lowercased_key -> info) with:
  - target_translations: {"zh": "...", "de": "...", "ja": "..."}
  - target_translation_source: "gold translation" | "llm translation" | "mixed"

    Preference:
  - Use the provided technical context (description/full form) from the paper.
  - Ask LLM to provide accurate technical translations for academic NLP/ML.

Requirements:
  - GEMINI_API_KEY env var
  - google-genai package
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


try:
    from google import genai
    from google.genai import types
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "google-genai is required. Install it (pip install google-genai) and set GEMINI_API_KEY."
    ) from e


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str:
    """
    Best-effort extraction of the first JSON object from a mixed response.
    """
    t = _strip_code_fences(text)
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1].strip()
    m = re.search(r"\{\s*\"[\s\S]*?\}\s*", t)
    if m:
        return m.group(0).strip()
    return t


def _repair_json_with_llm(client: genai.Client, model_name: str, broken_json_text: str) -> str:
    prompt = (
        "Fix the JSON object below. Return ONLY a valid JSON object. "
        "Do not add markdown. Do not add any keys beyond:\n"
        "- check_presence\n"
        "- target_translations\n"
        "- target_translation_source\n\n"
        "BROKEN_JSON:\n"
        f"{broken_json_text}\n"
    )
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )
    except Exception:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2048,
            ),
        )
    return _safe_get_response_text(resp)


def _try_parse_translation_object(text: str) -> Dict[str, Any]:
    candidate = _extract_json_object(text)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object")
    return parsed

def _safe_get_response_text(resp: Any) -> str:
    """
    Extract text from response in a defensive way for the new google-genai SDK.
    Handles truncated responses (MAX_TOKENS) and different part types (thought vs text).
    """
    chunks = []
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            cand = resp.candidates[0]
            if hasattr(cand, "content") and cand.content and hasattr(cand.content, "parts") and cand.content.parts:
                for p in cand.content.parts:
                    # Prefer .text if available
                    if hasattr(p, "text") and p.text:
                        chunks.append(p.text)
                    # For newer models that might return thought parts, we usually skip them 
                    # for the final translation, but if there's no text, we might want to know.
                    # Here we strictly look for text chunks.
    except Exception:
        pass

    if chunks:
        return "".join(chunks).strip()

    # Fallback to standard .text if our manual extraction failed
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
    except Exception:
        pass

    return ""


def _candidate_finish_reason(resp: Any) -> str:
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            cand = resp.candidates[0]
            if hasattr(cand, "finish_reason"):
                return str(cand.finish_reason)
    except Exception:
        pass
    return "UNKNOWN"


def _call_llm_translation(
    client: genai.Client,
    model_name: str,
    contents: str | List[Any],
    gen_cfg_dict: Dict[str, Any],
    retries: int = 2,
    sleep_seconds: float = 0.5,
) -> str:
    """
    Call Gemini and return response text using the new google-genai SDK.
    """
    last_err: Optional[Exception] = None
    cfg_dict = dict(gen_cfg_dict)
    
    for attempt in range(retries + 1):
        try:
            # Convert dict config to GenerateContentConfig
            config = types.GenerateContentConfig(**cfg_dict)
            
            resp = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            txt = _safe_get_response_text(resp)
            if txt:
                return txt
            
            # No text parts; treat as retryable
            fr = _candidate_finish_reason(resp)
            raise ValueError(f"Empty response text (finish_reason={fr})")
        except Exception as e:
            last_err = e
            # Retry with more conservative settings
            if attempt < retries:
                cfg_dict.pop("response_mime_type", None)
                cfg_dict["temperature"] = 0.0
                time.sleep(sleep_seconds * (attempt + 1))
                continue
            raise
    assert last_err is not None
    raise last_err


def _fallback_prompt(term: str, want_langs: List[str], is_acronym: bool, context: str = "") -> str:
    acronym_clause = "true" if is_acronym else "false"
    context_clause = f"Context: {context}\n" if context else ""
    lang_template = ", ".join([f'"{l}": "..."' for l in want_langs])
    return (
        "You are a technical terminology translator for academic NLP/ML.\n\n"
        f'Term: "{term}"\n'
        f"IsAcronym: {acronym_clause}\n"
        f"{context_clause}\n"
        "Translate the term into the target languages. If IsAcronym is true, keep it unchanged if appropriate.\n"
        "Return ONLY JSON, no markdown.\n\n"
        "{\n"
        '  "check_presence": true,\n'
        f'  "target_translations": {{{lang_template}}},\n'
        '  "target_translation_source": "llm translation"\n'
        "}\n"
    )


def _is_probably_acronym(term: str, is_acronym_field: Optional[bool]) -> bool:
    if is_acronym_field is True:
        return True
    t = term.strip()
    if not t:
        return False
    if len(t) <= 10 and t.isupper():
        return True
    # Mixed-case acronym-like
    if any(c.isupper() for c in t) and not t.islower() and len(t) <= 12 and " " not in t:
        return True
    return False


def _make_prompt(
    term: str,
    is_acronym: bool,
    short_description: str,
    full_form: str,
    want_langs: List[str],
) -> str:
    desc = (short_description or "").strip()
    ff = (full_form or "").strip()
    
    context_parts = []
    if ff:
        context_parts.append(f'Full Form: "{ff}"')
    if desc:
        context_parts.append(f'Description: "{desc}"')
    
    context_clause = "\n".join(context_parts)
    acronym_clause = "true" if is_acronym else "false"

    lang_template = ", ".join([f'"{l}": "..."' for l in want_langs])

    return (
        "You are a technical terminology translator for academic NLP/ML papers.\n\n"
        f'Term: "{term}"\n'
        f"IsAcronym: {acronym_clause}\n"
        f"{context_clause}\n\n"
        "Translate the term into the requested target languages using the provided technical context.\n"
        "Rules:\n"
        "- If IsAcronym is true, keep the acronym unchanged in the translation if appropriate for the target language.\n"
        "- Ensure translations are accurate within the context of NLP/ML research.\n"
        "- Return ONLY JSON, no markdown.\n\n"
        "Return a JSON object with exactly these keys:\n"
        "{\n"
        '  "check_presence": true,\n'
        f'  "target_translations": {{{lang_template}}},\n'
        '  "target_translation_source": "llm translation"\n'
        "}\n"
    )


def _normalize_key(term: str) -> str:
    return (term or "").strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment extracted_glossary.json with zh/de/ja translations via Gemini using paper context.")
    ap.add_argument(
        "--input_glossary",
        default="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary.json",
        help="Path to extracted_glossary.json",
    )
    ap.add_argument(
        "--output_glossary",
        default="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json",
        help="Output JSON path",
    )
    ap.add_argument(
        "--resume_from",
        default="",
        help="If set to an existing JSON output, already translated keys will be skipped and merged.",
    )
    ap.add_argument(
        "--zh", action="store_true", default=True, help="Enable Chinese translation"
    )
    ap.add_argument(
        "--de", action="store_true", default=True, help="Enable German translation"
    )
    ap.add_argument(
        "--ja", action="store_true", default=True, help="Enable Japanese translation"
    )
    ap.add_argument(
        "--papers_dir",
        default="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/papers",
        help="Directory containing PDF papers for context.",
    )
    ap.add_argument("--max_terms", type=int, default=0, help="If >0, limit to first N terms (for debugging)")
    ap.add_argument("--sleep_seconds", type=float, default=float(os.environ.get("SLEEP_SECONDS", "0.2")))
    ap.add_argument(
        "--print_entries",
        action="store_true",
        help="Print each processed entry's key + target_translations to stdout (debug/inspection).",
    )
    ap.add_argument(
        "--print_gemini_output",
        action="store_true",
        help="Print Gemini raw outputs (primary + fallback) as JSONL to stdout (debug/inspection).",
    )
    ap.add_argument(
        "--print_gemini_output_max_chars",
        type=int,
        default=int(os.environ.get("PRINT_GEMINI_OUTPUT_MAX_CHARS", "8000")),
        help="Max chars to print for Gemini raw output (truncate if longer).",
    )

    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Error: GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model_name = os.environ.get("MODEL_NAME", "gemini-2.0-flash")
    print(f"Using Gemini model: {model_name}")

    print("Loading glossary...")
    with open(args.input_glossary, "r", encoding="utf-8") as f:
        raw_glossary: Dict[str, Dict[str, Any]] = json.load(f)

    want_langs = []
    if args.zh:
        want_langs.append("zh")
    if args.de:
        want_langs.append("de")
    if args.ja:
        want_langs.append("ja")
    
    if not want_langs:
        raise SystemExit("Error: No target languages enabled.")

    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from existing output: {args.resume_from}")
        with open(args.resume_from, "r", encoding="utf-8") as f:
            existing = json.load(f)

    out: Dict[str, Dict[str, Any]] = dict(existing)

    # Stable iteration order
    items = list(raw_glossary.items())
    items.sort(key=lambda kv: (kv[0] or "").lower())
    if args.max_terms and args.max_terms > 0:
        items = items[: args.max_terms]

    total = len(items)
    print(f"Processing {total} terms...")

    uploaded_files: Dict[str, Any] = {}

    try:
        for idx, (orig_key, info) in enumerate(items, 1):
            term = (info.get("term") or orig_key or "").strip()
            if not term:
                continue
            k = _normalize_key(term)
            if k in out and out[k].get("target_translations") and all(
                (out[k]["target_translations"].get(l) for l in want_langs)
            ):
                if idx % 10 == 0:
                    print(f"Progress: {idx}/{total} (skipping {term!r})")
                continue

            is_acronym = _is_probably_acronym(term, info.get("is_acronym"))
            short_description = info.get("short_description", "")
            full_form = info.get("full_form", "")
            source_paper = info.get("source_paper", "").split(",")[0].strip()

            # 实时打印当前处理的术语
            print(f"[{idx}/{total}] Processing: {term!r} | paper: {source_paper or 'N/A'}")

            prompt = _make_prompt(
                term=term,
                is_acronym=is_acronym,
                short_description=short_description,
                full_form=full_form,
                want_langs=want_langs,
            )

            # Prepare content for LLM
            contents: List[Any] = [prompt]
            if source_paper and args.papers_dir:
                paper_path = os.path.join(args.papers_dir, source_paper)
                if os.path.exists(paper_path):
                    if source_paper not in uploaded_files:
                        try:
                            print(f"  Uploading context paper: {source_paper}")
                            uploaded = client.files.upload(file=paper_path)
                            print(f"  Waiting for paper to be ACTIVE: {source_paper}")
                            while uploaded.state == "PROCESSING":
                                time.sleep(2)
                                uploaded = client.files.get(name=uploaded.name)
                            
                            if uploaded.state == "FAILED":
                                print(f"  Warning: Paper processing failed for {source_paper}")
                                del uploaded_files[source_paper]
                            else:
                                uploaded_files[source_paper] = uploaded
                                print(f"  Paper ready: {source_paper}")
                        except Exception as upload_err:
                            print(f"  Warning: Failed to upload {source_paper}: {upload_err}")
                    
                    if source_paper in uploaded_files:
                        contents.append(uploaded_files[source_paper])

            # Generate
            gen_cfg: Dict[str, Any] = {"temperature": 0.1, "max_output_tokens": 8192}
            try:
                gen_cfg["response_mime_type"] = "application/json"
            except Exception:
                pass

            try:
                resp_text = _call_llm_translation(
                    client=client,
                    model_name=model_name,
                    contents=contents,
                    gen_cfg_dict=gen_cfg,
                    retries=int(os.environ.get("LLM_RETRIES", "2")),
                    sleep_seconds=float(os.environ.get("LLM_RETRY_SLEEP", "0.6")),
                )
                if args.print_gemini_output:
                    s = resp_text
                    truncated = False
                    if args.print_gemini_output_max_chars > 0 and len(s) > args.print_gemini_output_max_chars:
                        s = s[: args.print_gemini_output_max_chars]
                        truncated = True
                    print(
                        json.dumps(
                            {
                                "phase": "primary",
                                "key": k,
                                "term": term,
                                "truncated": truncated,
                                "text": s,
                            },
                            ensure_ascii=False,
                        )
                    )
                try:
                    parsed = _try_parse_translation_object(resp_text)
                except Exception as parse_err:
                    repaired = _repair_json_with_llm(client, model_name, resp_text)
                    parsed = _try_parse_translation_object(repaired)
            except Exception as e:
                # Try a minimal prompt as a last resort (often helps with preview/strict models)
                try:
                    fallback_context = f"{full_form} {short_description}".strip()
                    fp = _fallback_prompt(term=term, want_langs=want_langs, is_acronym=is_acronym, context=fallback_context)
                    
                    # For fallback, we also try to use the paper if it was successfully uploaded
                    fallback_contents: List[Any] = [fp]
                    if source_paper in uploaded_files:
                        fallback_contents.append(uploaded_files[source_paper])
                    else:
                        fallback_contents = fp  # type: ignore

                    resp_text_2 = _call_llm_translation(
                        client=client,
                        model_name=model_name,
                        contents=fallback_contents,
                        gen_cfg_dict={"temperature": 0.0, "max_output_tokens": 1024},
                        retries=1,
                        sleep_seconds=float(os.environ.get("LLM_RETRY_SLEEP", "0.6")),
                    )
                    if args.print_gemini_output:
                        s2 = resp_text_2
                        truncated2 = False
                        if args.print_gemini_output_max_chars > 0 and len(s2) > args.print_gemini_output_max_chars:
                            s2 = s2[: args.print_gemini_output_max_chars]
                            truncated2 = True
                        print(
                            json.dumps(
                                {
                                    "phase": "fallback",
                                    "key": k,
                                    "term": term,
                                    "truncated": truncated2,
                                    "text": s2,
                                },
                                ensure_ascii=False,
                            )
                        )
                    try:
                        parsed = _try_parse_translation_object(resp_text_2)
                    except Exception:
                        repaired = _repair_json_with_llm(client, model_name, resp_text_2)
                        parsed = _try_parse_translation_object(repaired)
                except Exception as e2:
                    print(f"ERROR: LLM call failed for term={term!r}: {type(e).__name__}: {e}")
                    print(f"ERROR: Fallback LLM call failed for term={term!r}: {type(e2).__name__}: {e2}")
                    parsed = {
                        "check_presence": True,
                        "target_translations": {},
                        "target_translation_source": "llm translation",
                    }

            target_translations: Dict[str, str] = {}
            src_norm = "llm translation"

            tt = parsed.get("target_translations") or {}
            if isinstance(tt, dict):
                for l in want_langs:
                    v = tt.get(l, "")
                    if isinstance(v, str) and v.strip():
                        target_translations[l] = v.strip()

            # If model failed to return some languages, fill conservatively
            for l in want_langs:
                if l not in target_translations:
                    target_translations[l] = term if is_acronym else ""

            # Merge into info
            new_info = dict(info)
            new_info["term"] = term
            new_info.setdefault("target_translations", {})
            merged_tt = dict(new_info.get("target_translations") or {})
            merged_tt.update(target_translations)
            new_info["target_translations"] = merged_tt
            new_info["target_translation_source"] = src_norm

            out[k] = new_info
            if args.print_entries:
                print(
                    json.dumps(
                        {
                            "key": k,
                            "term": term,
                            "target_translations": new_info.get("target_translations", {}),
                            "source": new_info.get("target_translation_source", ""),
                        },
                        ensure_ascii=False,
                    )
                )

            # Periodic checkpoint
            if idx % 20 == 0 or idx == total:
                os.makedirs(os.path.dirname(args.output_glossary), exist_ok=True)
                with open(args.output_glossary, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    finally:
        for paper, file_obj in uploaded_files.items():
            try:
                print(f"Cleaning up context paper: {paper}")
                client.files.delete(name=file_obj.name)
            except Exception as e:
                print(f"Warning: Failed to delete {paper}: {e}")

    print(f"Done. Wrote: {args.output_glossary} | terms={len(out)}")


if __name__ == "__main__":
    main()


