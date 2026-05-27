#!/usr/bin/env python3
"""
Generate TTS audio for glossary terms using OpenAI TTS API.

Produces 16 kHz mono WAV files compatible with the offline evaluation pipeline.
Generates TTS only for terms that do NOT already have TTS audio.

Usage:
    # Dry run to see what would be generated and cost estimate
    python generate_tts_for_glossary.py --glossary_size 1000 --dry_run

    # Smoke test: generate first 10 terms
    OPENAI_API_KEY=sk-xxx python generate_tts_for_glossary.py --limit 10

    # Generate for 1000-term glossary expansion
    OPENAI_API_KEY=sk-xxx python generate_tts_for_glossary.py --glossary_size 1000

    # Generate all (up to 10k)
    OPENAI_API_KEY=sk-xxx python generate_tts_for_glossary.py --glossary_size 10000
"""

from __future__ import annotations

# ======Configuration=====
WIKI_GLOSSARY_PATH = (
    "/home/jiaxuanluo/InfiniSST/documents/code/data_pre/"
    "glossary_scale/wiki_glossary_nlp_ai_cs.json"
)
EXISTING_TTS_JSONL = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_dev_offline_eval/acl6060_dev_dataset_with_tts.jsonl"
)
EXISTING_TTS_ROOT = "/mnt/gemini/data/siqiouyang/acl_terms"

OUTPUT_TTS_DIR = "/mnt/gemini/data2/jiaxuanluo/wiki_glossary_tts"
OUTPUT_MAPPING_JSONL = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_dev_offline_eval/wiki_glossary_tts_mapping.jsonl"
)
COMBINED_TTS_MAPPING_JSONL = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_dev_offline_eval/acl6060_combined_tts_mapping.jsonl"
)

OPENAI_TTS_MODEL = "tts-1"
OPENAI_TTS_VOICE = "alloy"
OPENAI_TTS_RESPONSE_FORMAT = "wav"
TARGET_SAMPLE_RATE = 16000
OPENAI_TTS_COST_PER_MILLION_CHARS = 15.0

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0
DEFAULT_CONCURRENT_REQUESTS = 8
# ======Configuration=====

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import soundfile as sf


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def _load_wiki_glossary(path: str) -> List[str]:
    assert os.path.isfile(path), f"Wiki glossary not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    terms: List[str] = []
    seen: Set[str] = set()
    for e in entries:
        term = e["term"].strip()
        key = term.lower()
        assert key, f"Empty term: {e!r}"
        if key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def _load_existing_tts_terms(jsonl_path: str) -> Dict[str, str]:
    """Returns dict: lowercase_term -> tts_audio_path."""
    mapping: Dict[str, str] = {}
    if not os.path.isfile(jsonl_path):
        return mapping
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("term", "").strip().lower()
            p = obj.get("tts_audio_path", "").strip()
            if t and p:
                mapping[t] = p
    return mapping


def _generate_single_tts(
    client,
    term: str,
    output_path: str,
) -> Tuple[bool, str, float]:
    """Generate TTS for a single term. Returns (success, error_msg, cost_chars)."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=term,
                response_format=OPENAI_TTS_RESPONSE_FORMAT,
            )
            wav_bytes = response.content

            audio, sr = sf.read(io.BytesIO(wav_bytes))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            if sr != TARGET_SAMPLE_RATE:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(TARGET_SAMPLE_RATE, sr)
                audio = resample_poly(audio, TARGET_SAMPLE_RATE // g, sr // g).astype(np.float32)

            sf.write(output_path, audio, TARGET_SAMPLE_RATE)
            return True, "", float(len(term))

        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            else:
                return False, str(exc), 0.0

    return False, "max retries exceeded", 0.0


def _safe_filename(term: str, idx: int) -> str:
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in term)
    safe = safe.strip().replace(" ", "_")[:80]
    return f"{idx:05d}_{safe}.wav"


def _build_combined_mapping(existing_jsonl: str, wiki_mapping_jsonl: str, output_path: str) -> int:
    """Merge existing GT TTS mapping + newly generated wiki TTS mapping."""
    all_mappings: Dict[str, str] = {}

    if os.path.isfile(existing_jsonl):
        with open(existing_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = obj.get("term", "").strip().lower()
                p = obj.get("tts_audio_path", "").strip()
                if t and p:
                    all_mappings[t] = p

    if os.path.isfile(wiki_mapping_jsonl):
        with open(wiki_mapping_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = obj.get("term", "").strip().lower()
                p = obj.get("tts_audio_path", "").strip()
                if t and p:
                    all_mappings[t] = p

    with open(output_path, "w", encoding="utf-8") as f:
        for term_key, path in sorted(all_mappings.items()):
            f.write(json.dumps({"term": term_key, "tts_audio_path": path},
                               ensure_ascii=False) + "\n")
    return len(all_mappings)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate TTS audio for wiki glossary terms via OpenAI API"
    )
    parser.add_argument("--glossary_size", type=int, default=1000,
                        help="Target glossary size (includes GT terms)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of NEW terms to generate (0=no limit)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_TTS_DIR)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan and cost estimate without calling API")
    parser.add_argument("--concurrent", type=int, default=DEFAULT_CONCURRENT_REQUESTS)
    parser.add_argument("--api_key", type=str, default="",
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--skip_combined", action="store_true",
                        help="Skip building the combined mapping file")
    args = parser.parse_args()

    _log(f"glossary_size={args.glossary_size}, limit={args.limit}, dry_run={args.dry_run}")

    wiki_terms = _load_wiki_glossary(WIKI_GLOSSARY_PATH)
    _log(f"Wiki glossary loaded: {len(wiki_terms)} terms")

    existing_tts = _load_existing_tts_terms(EXISTING_TTS_JSONL)
    _log(f"Existing TTS terms (GT): {len(existing_tts)}")

    already_generated: Dict[str, str] = {}
    mapping_path = Path(OUTPUT_MAPPING_JSONL)
    if mapping_path.exists():
        already_generated = _load_existing_tts_terms(str(mapping_path))
        valid_count = sum(1 for p in already_generated.values() if os.path.isfile(p))
        _log(f"Already generated (previous runs): {len(already_generated)} "
             f"(valid files: {valid_count})")

    terms_to_generate: List[Tuple[int, str]] = []
    for idx, term in enumerate(wiki_terms):
        if idx >= args.glossary_size:
            break
        key = term.lower()
        if key in existing_tts:
            continue
        if key in already_generated and os.path.isfile(already_generated[key]):
            continue
        terms_to_generate.append((idx, term))

    if args.limit > 0:
        terms_to_generate = terms_to_generate[:args.limit]

    total_chars = sum(len(t) for _, t in terms_to_generate)
    estimated_cost = total_chars * OPENAI_TTS_COST_PER_MILLION_CHARS / 1_000_000

    _log("=" * 60)
    _log(f"PLAN SUMMARY")
    _log(f"  Target glossary size:    {args.glossary_size}")
    _log(f"  Existing GT TTS terms:   {len(existing_tts)}")
    _log(f"  Already generated:       {len(already_generated)}")
    _log(f"  Terms to generate:       {len(terms_to_generate)}")
    _log(f"  Total characters:        {total_chars}")
    _log(f"  Estimated cost (tts-1):  ${estimated_cost:.4f}")
    _log("=" * 60)

    if args.dry_run:
        _log("DRY RUN — no API calls made. Sample terms:")
        for idx, term in terms_to_generate[:20]:
            print(f"  [{idx:05d}] {term} ({len(term)} chars)")
        if len(terms_to_generate) > 20:
            print(f"  ... and {len(terms_to_generate) - 20} more")
        return 0

    if not terms_to_generate:
        _log("Nothing to generate.")
        if not args.skip_combined:
            count = _build_combined_mapping(
                EXISTING_TTS_JSONL, str(mapping_path), COMBINED_TTS_MAPPING_JSONL
            )
            _log(f"Combined TTS mapping updated: {count} terms -> {COMBINED_TTS_MAPPING_JSONL}")
        return 0

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    assert api_key, (
        "OpenAI API key is required. Set OPENAI_API_KEY env var or pass --api_key."
    )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Starting TTS generation with {args.concurrent} concurrent workers ...")
    t_start = time.time()
    success_count = 0
    fail_count = 0
    total_cost_chars = 0.0
    new_mappings: List[Dict] = []

    def _task(idx_term):
        idx, term = idx_term
        fname = _safe_filename(term, idx)
        out_path = str(output_dir / fname)
        ok, err, chars = _generate_single_tts(client, term, out_path)
        return idx, term, out_path, ok, err, chars

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {executor.submit(_task, it): it for it in terms_to_generate}
        for i, future in enumerate(as_completed(futures)):
            idx, term, out_path, ok, err, chars = future.result()
            if ok:
                success_count += 1
                total_cost_chars += chars
                new_mappings.append({
                    "term": term.lower(),
                    "tts_audio_path": out_path,
                })
            else:
                fail_count += 1
                _warn(f"Failed [{idx:05d}] {term!r}: {err}")

            if (i + 1) % 50 == 0 or (i + 1) == len(terms_to_generate):
                elapsed = time.time() - t_start
                _log(
                    f"Progress: {i+1}/{len(terms_to_generate)} "
                    f"(ok={success_count}, fail={fail_count}, "
                    f"elapsed={elapsed:.1f}s)"
                )

    elapsed = time.time() - t_start
    actual_cost = total_cost_chars * OPENAI_TTS_COST_PER_MILLION_CHARS / 1_000_000

    _log("=" * 60)
    _log(f"DONE in {elapsed:.1f}s")
    _log(f"  Success: {success_count}")
    _log(f"  Failed:  {fail_count}")
    _log(f"  Total chars sent: {total_cost_chars:.0f}")
    _log(f"  Actual cost:      ${actual_cost:.4f}")
    _log("=" * 60)

    if new_mappings:
        with mapping_path.open("a", encoding="utf-8") as f:
            for m in new_mappings:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        _log(f"Appended {len(new_mappings)} entries to {mapping_path}")

    if not args.skip_combined:
        count = _build_combined_mapping(
            EXISTING_TTS_JSONL, str(mapping_path), COMBINED_TTS_MAPPING_JSONL
        )
        _log(f"Combined TTS mapping: {count} terms -> {COMBINED_TTS_MAPPING_JSONL}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
