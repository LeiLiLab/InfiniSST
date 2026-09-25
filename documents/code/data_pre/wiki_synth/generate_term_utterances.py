#!/usr/bin/env python3
"""
Generate term-anchored micro-utterance texts for TTS synthesis using Gemini API.

For each term, asks Gemini to generate multiple short, natural utterances.
Each utterance must:
  1. Contain the term exactly once (verbatim, case-insensitive match accepted)
  2. Be short enough for ~1.92s speech (target 5-15 words)
  3. Sound natural, like a snippet from a lecture, tutorial, or discussion

Requires GEMINI_API_KEY in the process environment.
Uses async concurrency to parallelize API calls for speed.
Supports resume: if the output file already exists, completed terms are skipped.

Output: a JSONL file with one row per utterance:
  {"term": ..., "utterance": ..., "variant_idx": ...}

Usage:
    python generate_term_utterances.py --terms wiki_synth_terms.json
    python generate_term_utterances.py --terms wiki_synth_terms.json --variants_per_term 6
    python generate_term_utterances.py --terms wiki_synth_terms.json --concurrency 15
    python generate_term_utterances.py --terms wiki_synth_terms.json --smoke_test 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from typing import Dict, List, Set, Tuple

from google import genai
from google.genai import types

# ======Configuration=====
GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_VARIANTS_PER_TERM = 6
BATCH_SIZE = 40
CONCURRENCY = 15
MAX_RETRIES = 5
RETRY_BASE_WAIT_SEC = 3
TEMPERATURE = 1.0
MAX_OUTPUT_TOKENS = 32768
PROGRESS_INTERVAL = 10
# ======Configuration=====

SYSTEM_INSTRUCTION = """\
You are a dataset builder for speech recognition research.
Your task: given a list of terms with their short descriptions, generate short
English utterances (5-15 words each) that a speaker might say in a lecture,
tutorial, or technical discussion. Use the description as context to produce
accurate, meaningful sentences — but do NOT just copy the description.
Each utterance MUST contain the given term EXACTLY as written (same spelling,
but case may differ). Vary sentence structure and term position (beginning,
middle, end). Do NOT repeat the same pattern across terms.
Output valid JSON only — no markdown fences, no explanation.\
"""


def build_prompt(term_entries: List[Dict[str, str]], variants: int) -> str:
    """Build prompt with term + short_description context."""
    term_info = [
        {"term": e["term"], "description": e.get("short_description", "")}
        for e in term_entries
    ]
    info_json = json.dumps(term_info, ensure_ascii=False)
    return (
        f"Generate {variants} short utterances (5-15 words) for EACH of the "
        f"following terms. Use the description as context to make the utterance "
        f"accurate and meaningful. Each utterance must contain the term verbatim.\n\n"
        f"Terms with descriptions:\n{info_json}\n\n"
        f"Return a JSON object mapping each term to a list of {variants} "
        f"utterance strings. Example format:\n"
        f'{{"Seismology": ["Seismology helps us understand earthquake patterns.", '
        f'"The field of seismology has advanced rapidly.", ...]}}\n\n'
        f"IMPORTANT: output raw JSON only, no markdown code fences."
    )


def parse_gemini_response(raw_text: str) -> Dict[str, List[str]]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    return data


def validate_utterance(term: str, utterance: str) -> bool:
    return term.lower() in utterance.lower()


async def generate_batch_async(
    client: genai.Client,
    batch_entries: List[Dict[str, str]],
    variants: int,
    semaphore: asyncio.Semaphore,
    batch_idx: int,
) -> Tuple[int, Dict[str, List[str]]]:
    """Call Gemini for a batch of terms with retry, guarded by semaphore."""
    prompt = build_prompt(batch_entries, variants)

    async with semaphore:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                )
                result = parse_gemini_response(response.text)
                return batch_idx, result
            except Exception as exc:
                wait = RETRY_BASE_WAIT_SEC * (2 ** attempt)
                print(
                    f"  [Batch {batch_idx} RETRY {attempt + 1}/{MAX_RETRIES}] "
                    f"{type(exc).__name__}: {exc} — waiting {wait}s",
                    flush=True,
                )
                if attempt == MAX_RETRIES:
                    print(
                        f"  [Batch {batch_idx} FATAL] giving up after "
                        f"{MAX_RETRIES} retries",
                        flush=True,
                    )
                    return batch_idx, {}
                await asyncio.sleep(wait)

    return batch_idx, {}


async def run_generation(
    terms_data: List[Dict],
    done_terms: Set[str],
    output_path: str,
    variants: int,
    batch_size: int,
    concurrency: int,
) -> None:
    remaining = [e for e in terms_data if e["term"] not in done_terms]
    print(
        f"Total terms: {len(terms_data)}, "
        f"already done: {len(done_terms)}, "
        f"remaining: {len(remaining)}",
        flush=True,
    )
    if not remaining:
        print("Nothing to do — all terms already processed.", flush=True)
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    semaphore = asyncio.Semaphore(concurrency)

    batches: List[Tuple[int, List[Dict[str, str]]]] = []
    for i in range(0, len(remaining), batch_size):
        batch_entries = remaining[i: i + batch_size]
        batches.append((i // batch_size, batch_entries))

    total_batches = len(batches)
    print(
        f"Launching {total_batches} batches with concurrency={concurrency} ...",
        flush=True,
    )

    total_written = 0
    total_skipped = 0
    completed_batches = 0
    start_time = time.time()

    with open(output_path, "a", encoding="utf-8") as fout:

        async def process_and_write(batch_idx: int, batch_entries: List[Dict[str, str]]):
            nonlocal total_written, total_skipped, completed_batches

            idx, result = await generate_batch_async(
                client, batch_entries, variants, semaphore, batch_idx
            )

            written = 0
            skipped = 0
            rows = []
            batch_terms = [e["term"] for e in batch_entries]
            for term in batch_terms:
                utterances = result.get(term, [])
                if not utterances:
                    for key in result:
                        if key.lower() == term.lower():
                            utterances = result[key]
                            break
                if not utterances:
                    skipped += 1
                    continue

                for vidx, utt in enumerate(utterances):
                    if not validate_utterance(term, utt):
                        skipped += 1
                        continue
                    rows.append(
                        json.dumps(
                            {"term": term, "utterance": utt, "variant_idx": vidx},
                            ensure_ascii=False,
                        )
                    )
                    written += 1

            if rows:
                fout.write("\n".join(rows) + "\n")
                fout.flush()

            total_written += written
            total_skipped += skipped
            completed_batches += 1

            if completed_batches % PROGRESS_INTERVAL == 0 or completed_batches == total_batches:
                elapsed = time.time() - start_time
                rate = completed_batches / elapsed if elapsed > 0 else 0
                eta = (total_batches - completed_batches) / rate if rate > 0 else 0
                print(
                    f"  [Progress] {completed_batches}/{total_batches} batches "
                    f"({total_written} utterances) "
                    f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s",
                    flush=True,
                )

        tasks = [
            process_and_write(batch_idx, batch_entries)
            for batch_idx, batch_entries in batches
        ]
        await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}", flush=True)
    print(
        f"Done. Wrote {total_written} utterances, "
        f"skipped {total_skipped} invalid. "
        f"Time: {elapsed:.1f}s",
        flush=True,
    )
    print(f"Output: {output_path}", flush=True)
    print(f"{'=' * 60}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate term-anchored utterances via Gemini API (async)"
    )
    parser.add_argument(
        "--terms", type=str, required=True,
        help="Path to wiki_synth_terms.json",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Output JSONL path (default: <terms_dir>/wiki_synth_utterances.jsonl)",
    )
    parser.add_argument(
        "--variants_per_term", type=int, default=DEFAULT_VARIANTS_PER_TERM,
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY,
    )
    parser.add_argument(
        "--smoke_test", type=int, default=0,
        help="If >0, only process the first N terms",
    )
    args = parser.parse_args()

    assert os.path.isfile(args.terms), f"Terms file not found: {args.terms}"
    with open(args.terms, "r", encoding="utf-8") as f:
        terms_data = json.load(f)
    assert isinstance(terms_data, list) and len(terms_data) > 0

    if args.smoke_test > 0:
        terms_data = terms_data[: args.smoke_test]
        print(f"[SMOKE TEST] Processing only {len(terms_data)} terms", flush=True)

    output_path = args.output
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(args.terms), "wiki_synth_utterances.jsonl"
        )

    done_terms: Set[str] = set()
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done_terms.add(row["term"])
        print(
            f"[RESUME] Found {len(done_terms)} already-completed terms",
            flush=True,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    asyncio.run(
        run_generation(
            terms_data, done_terms, output_path,
            args.variants_per_term, args.batch_size, args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
