#!/usr/bin/env python3
"""
Build a TTS term bank (npy + symlinked wav directory) from term_train_dataset_final.jsonl.

The output format matches what StreamingQwen3RAGRetrieverV4._initialize_tts_term_bank expects:
  - A .npy file: 1-D string array where index i corresponds to wav file "{i+1}.wav"
    (duplicate term_keys allowed — each entry maps to a distinct audio sample)
  - A wav directory: symlinks named "1.wav", "2.wav", ... pointing to original audio chunks

Usage:
    python build_tts_bank_from_term_train.py \
        --term-train-jsonl /path/to/term_train_dataset_final.jsonl \
        --glossary-json    /path/to/glossary_for_zh_rate1.0_k20.json \
        --output-npy       /path/to/output_terms.npy \
        --output-wav-dir   /path/to/output_wav_dir \
        --max-prototypes-per-term 8
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ============================== Configuration ==============================
DEFAULT_MAX_PROTOTYPES_PER_TERM = 8
DEFAULT_SEED = 42
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
# ===========================================================================

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TTS bank (npy + wav symlinks) from term_train_dataset JSONL."
    )
    parser.add_argument(
        "--term-train-jsonl", required=True,
        help="Path to term_train_dataset_final_with_tts.jsonl (must have tts_audio_path field)",
    )
    parser.add_argument(
        "--glossary-json", required=True,
        help="Path to the training glossary JSON (e.g. glossary_for_zh_rate1.0_k20.json). "
             "Only terms present in this glossary will be included in the TTS bank.",
    )
    parser.add_argument("--output-npy", required=True, help="Output .npy file path")
    parser.add_argument("--output-wav-dir", required=True, help="Output directory for wav symlinks")
    parser.add_argument(
        "--max-prototypes-per-term", type=int, default=DEFAULT_MAX_PROTOTYPES_PER_TERM,
        help=f"Max audio samples to keep per term (default: {DEFAULT_MAX_PROTOTYPES_PER_TERM})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for sampling when a term has more audios than max (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--skip-exist-check", action="store_true",
        help="Skip checking whether source wav files exist (faster but riskier)",
    )
    return parser.parse_args()


def load_glossary_keys(glossary_path: str) -> set:
    logger.info("Loading glossary from %s ...", glossary_path)
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
    keys = {k.strip().lower() for k in glossary.keys()}
    logger.info("Glossary contains %d unique term keys.", len(keys))
    return keys


def collect_term_audios(jsonl_path: str, glossary_keys: set, skip_exist_check: bool) -> dict:
    """Return {term_key: [audio_path, ...]} for terms present in glossary."""
    logger.info("Scanning %s for matching terms ...", jsonl_path)
    term_audios: dict = defaultdict(list)
    total_rows = 0
    skipped_not_in_glossary = 0
    skipped_missing_audio = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total_rows += 1
            obj = json.loads(line)
            term_key = obj.get("term_key", "").strip().lower()
            audio_path = obj.get("tts_audio_path", "").strip()

            if not term_key or term_key not in glossary_keys:
                skipped_not_in_glossary += 1
                continue
            if not audio_path:
                skipped_missing_audio += 1
                continue
            if not skip_exist_check and not os.path.isfile(audio_path):
                skipped_missing_audio += 1
                continue

            term_audios[term_key].append(audio_path)

            if total_rows % 500_000 == 0:
                logger.info("  ... scanned %d rows, matched %d terms so far", total_rows, len(term_audios))

    logger.info(
        "Scan complete: %d rows, %d matched terms, %d skipped (not in glossary), %d skipped (no audio)",
        total_rows, len(term_audios), skipped_not_in_glossary, skipped_missing_audio,
    )
    return dict(term_audios)


def sample_and_build(
    term_audios: dict,
    max_prototypes: int,
    seed: int,
) -> list:
    """Return list of (term_key, audio_path) pairs, capped per term."""
    rng = random.Random(seed)
    pairs = []
    for term_key in sorted(term_audios.keys()):
        paths = term_audios[term_key]
        if len(paths) > max_prototypes:
            paths = rng.sample(paths, max_prototypes)
        for p in paths:
            pairs.append((term_key, p))
    return pairs


def write_npy_and_symlinks(
    pairs: list,
    output_npy: str,
    output_wav_dir: str,
) -> None:
    os.makedirs(os.path.dirname(output_npy) or ".", exist_ok=True)
    os.makedirs(output_wav_dir, exist_ok=True)

    term_strings = []
    created = 0
    for idx, (term_key, audio_path) in enumerate(pairs):
        term_strings.append(term_key)
        link_name = os.path.join(output_wav_dir, f"{idx + 1}.wav")
        abs_audio = os.path.abspath(audio_path)
        if os.path.islink(link_name) or os.path.exists(link_name):
            os.remove(link_name)
        os.symlink(abs_audio, link_name)
        created += 1

    arr = np.array(term_strings, dtype=str)
    np.save(output_npy, arr)
    logger.info("Wrote npy: %s  (shape=%s, dtype=%s)", output_npy, arr.shape, arr.dtype)
    logger.info("Created %d symlinks in %s", created, output_wav_dir)


def main() -> None:
    args = parse_args()

    glossary_keys = load_glossary_keys(args.glossary_json)

    term_audios = collect_term_audios(
        args.term_train_jsonl, glossary_keys, args.skip_exist_check
    )

    pairs = sample_and_build(term_audios, args.max_prototypes_per_term, args.seed)
    unique_terms = len(set(tk for tk, _ in pairs))
    logger.info(
        "Final TTS bank: %d unique terms, %d total (term, audio) pairs "
        "(max %d prototypes/term)",
        unique_terms, len(pairs), args.max_prototypes_per_term,
    )

    write_npy_and_symlinks(pairs, args.output_npy, args.output_wav_dir)

    coverage = unique_terms / max(1, len(glossary_keys)) * 100
    logger.info(
        "Glossary coverage: %d / %d = %.2f%%",
        unique_terms, len(glossary_keys), coverage,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
