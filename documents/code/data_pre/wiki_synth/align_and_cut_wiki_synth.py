#!/usr/bin/env python3
"""
MFA alignment + 1.92s chunk cutting for wiki_synth TTS data.

Reads JSONL with fields:
  {term, utterance, variant_idx, clean_audio_path, [noisy_audio_path]}

MFA alignment runs on the clean audio.  The 1.92s chunk is cut from clean,
then WHAM noise is mixed directly onto the clean chunk to create the noisy
version.  This guarantees **identical speech timing** between clean and noisy.

Output: TWO training rows per entry (one clean, one noisy), each with format:
  {"term": ..., "term_key": ..., "chunk_src_text": ...,
   "utter_id": ..., "chunk_idx": 0, "chunk_audio_path": ..., "audio_type": "clean"|"noisy"}

Supports sharding for parallel SLURM array jobs.

Usage:
    # Smoke test (10 entries)
    python align_and_cut_wiki_synth.py \\
        --data INPUT.jsonl --work-dir /mnt/data/.../work \\
        --output-audio-dir /mnt/data/.../chunks \\
        --output-jsonl OUT.jsonl --noise-dir /mnt/data/siqiouyang/datasets/wham_wav \\
        --smoke-test 10

    # Full sharded run (shard 0 of 20)
    python align_and_cut_wiki_synth.py \\
        --data INPUT.jsonl --shard-id 0 --num-shards 20 \\
        --work-dir /mnt/data/.../work \\
        --output-audio-dir /mnt/data/.../chunks \\
        --output-jsonl OUT_shard_00.jsonl \\
        --noise-dir /mnt/data/siqiouyang/datasets/wham_wav
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# ======Configuration=====
SAMPLE_RATE = 16000
CHUNK_SEC = 1.92
CHUNK_SAMPLES = int(CHUNK_SEC * SAMPLE_RATE)  # 30720

MIN_CONTEXT_PAD_SEC = 0.05  # minimum context on each side of the term

MFA_CONDA_ENV = "mfa"
MFA_CONDA_PREFIX = ""  # if set, overrides MFA_CONDA_ENV with --prefix
MFA_NUM_JOBS = 32
MFA_ACOUSTIC_MODEL = "english_mfa"
MFA_DICTIONARY = "english_mfa"

SNR_LOW_DB = 5
SNR_HIGH_DB = 25

WORD_NORM_RE = re.compile(r"[^\w'-]")
RANDOM_SEED = 42
# ======Configuration=====


@dataclass
class WordInterval:
    word: str
    start: float
    end: float


def normalize_word(w: str) -> str:
    return WORD_NORM_RE.sub("", w.lower().strip())


def load_data(
    jsonl_path: str,
    shard_id: int = 0,
    num_shards: int = 1,
    smoke_test: int = 0,
) -> List[Tuple[int, dict]]:
    """Load JSONL and return shard slice as (global_idx, entry) pairs."""
    all_entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            all_entries.append((idx, json.loads(line)))

    total = len(all_entries)
    shard_size = (total + num_shards - 1) // num_shards
    start = shard_id * shard_size
    end = min(start + shard_size, total)
    shard_entries = all_entries[start:end]

    if smoke_test > 0:
        shard_entries = shard_entries[:smoke_test]

    print(
        f"[INFO] Total entries: {total}, shard {shard_id}/{num_shards}: "
        f"[{start}:{end}] -> {len(shard_entries)} entries"
        + (f" (smoke_test={smoke_test})" if smoke_test > 0 else ""),
        flush=True,
    )
    return shard_entries


def _resolve_audio_path(entry: dict) -> str:
    """Get the audio path for MFA alignment (prefer clean_audio_path)."""
    if "clean_audio_path" in entry:
        return entry["clean_audio_path"]
    return entry["tts_audio_path"]


def prepare_mfa_input(
    entries: List[Tuple[int, dict]],
    mfa_input_dir: str,
    remap_src=None,
) -> Dict[str, Tuple[int, dict]]:
    """Create flat MFA input directory: symlink WAVs + write .lab transcripts.

    Uses clean_audio_path for MFA alignment (better quality alignment).
    Returns mapping: utt_id -> (global_idx, entry).
    remap_src: optional callable to remap source audio paths for cross-machine access.
    """
    os.makedirs(mfa_input_dir, exist_ok=True)

    utt_map: Dict[str, Tuple[int, dict]] = {}
    skipped = 0
    for global_idx, entry in entries:
        utt_id = f"utt_{global_idx:07d}"
        wav_src = _resolve_audio_path(entry)
        if remap_src is not None:
            wav_src = remap_src(wav_src)

        if not os.path.isfile(wav_src):
            print(f"[WARN] WAV not found, skipping: {wav_src}", flush=True)
            skipped += 1
            continue

        wav_link = os.path.join(mfa_input_dir, f"{utt_id}.wav")
        lab_path = os.path.join(mfa_input_dir, f"{utt_id}.lab")

        if os.path.islink(wav_link) or os.path.exists(wav_link):
            os.remove(wav_link)
        os.symlink(os.path.abspath(wav_src), wav_link)

        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(entry["utterance"])

        utt_map[utt_id] = (global_idx, entry)

    assert len(utt_map) > 0, f"No valid entries after prepare (skipped={skipped})"
    print(
        f"[INFO] Prepared MFA input: {len(utt_map)} utterances "
        f"(skipped {skipped}) in {mfa_input_dir}",
        flush=True,
    )
    return utt_map


def run_mfa(
    mfa_input_dir: str,
    mfa_output_dir: str,
    mfa_cache_dir: str,
    num_jobs: int,
    conda_prefix: str = "",
    mfa_model_dir: str = "",
) -> None:
    """Run MFA forced alignment on the prepared corpus.

    When conda_prefix is set, invokes MFA binary directly instead of
    using 'conda run' (avoids cross-machine conda activation issues).

    When mfa_model_dir is set, dictionary and acoustic model are resolved
    to absolute file paths under that directory, bypassing MFA's built-in
    pretrained-name lookup which depends on ~/Documents/MFA existing.
    """
    os.makedirs(mfa_output_dir, exist_ok=True)
    os.makedirs(mfa_cache_dir, exist_ok=True)

    if mfa_model_dir:
        dict_path = os.path.join(mfa_model_dir, "pretrained_models", "dictionary", f"{MFA_DICTIONARY}.dict")
        acoustic_path = os.path.join(mfa_model_dir, "pretrained_models", "acoustic", f"{MFA_ACOUSTIC_MODEL}.zip")
        assert os.path.isfile(dict_path), f"MFA dictionary not found: {dict_path}"
        assert os.path.isfile(acoustic_path), f"MFA acoustic model not found: {acoustic_path}"
        dictionary = dict_path
        acoustic_model = acoustic_path
    else:
        dictionary = MFA_DICTIONARY
        acoustic_model = MFA_ACOUSTIC_MODEL

    mfa_args = [
        "align",
        "--clean", "--final_clean",
        "--single_speaker",
        "--num_jobs", str(num_jobs),
        "--temporary_directory", mfa_cache_dir,
        "--overwrite",
        "--output_format", "short_textgrid",
        mfa_input_dir,
        dictionary,
        acoustic_model,
        mfa_output_dir,
    ]

    env = None
    if conda_prefix:
        mfa_bin = os.path.join(conda_prefix, "bin", "mfa")
        assert os.path.isfile(mfa_bin), f"MFA binary not found: {mfa_bin}"
        cmd = [mfa_bin] + mfa_args
        env = os.environ.copy()
        env["PATH"] = os.path.join(conda_prefix, "bin") + ":" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = os.path.join(conda_prefix, "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    else:
        cmd = ["conda", "run", "-n", MFA_CONDA_ENV, "mfa"] + mfa_args

    print(f"[INFO] Running MFA: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        print(f"[WARN] MFA stderr:\n{proc.stderr}", flush=True)
        raise RuntimeError(
            f"MFA failed with exit code {proc.returncode}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    print("[INFO] MFA alignment completed.", flush=True)


def parse_short_textgrid(tg_path: str) -> List[WordInterval]:
    """Parse MFA short TextGrid format to extract word intervals."""
    with open(tg_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]

    assert len(lines) >= 7, f"TextGrid too short: {tg_path}"

    idx = 0
    while idx < len(lines) and lines[idx] != "<exists>":
        idx += 1
    assert idx < len(lines), f"No <exists> tag in {tg_path}"
    idx += 1

    num_tiers = int(lines[idx])
    idx += 1

    words: List[WordInterval] = []

    for _ in range(num_tiers):
        _tier_class = lines[idx].strip('"')
        idx += 1
        tier_name = lines[idx].strip('"')
        idx += 1
        _tier_xmin = float(lines[idx])
        idx += 1
        _tier_xmax = float(lines[idx])
        idx += 1
        num_intervals = int(lines[idx])
        idx += 1

        for _ in range(num_intervals):
            xmin = float(lines[idx])
            idx += 1
            xmax = float(lines[idx])
            idx += 1
            text = lines[idx].strip('"')
            idx += 1

            if tier_name == "words" and text.strip():
                words.append(WordInterval(word=text, start=xmin, end=xmax))

    return words


def find_textgrid(mfa_output_dir: str, utt_id: str) -> Optional[str]:
    """Locate the TextGrid output for a given utterance ID."""
    candidates = [
        os.path.join(mfa_output_dir, f"{utt_id}.TextGrid"),
        os.path.join(mfa_output_dir, utt_id, f"{utt_id}.TextGrid"),
        os.path.join(mfa_output_dir, "unknown", f"{utt_id}.TextGrid"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def find_term_span(
    term: str,
    word_intervals: List[WordInterval],
) -> Optional[Tuple[float, float]]:
    """Find time span of a multi-word term using greedy normalized matching."""
    term_words = [normalize_word(w) for w in term.split() if normalize_word(w)]
    if not term_words:
        return None

    mfa_words_norm = [normalize_word(wi.word) for wi in word_intervals]

    for start_idx in range(len(mfa_words_norm) - len(term_words) + 1):
        if all(
            mfa_words_norm[start_idx + j] == tw
            for j, tw in enumerate(term_words)
        ):
            t_start = word_intervals[start_idx].start
            t_end = word_intervals[start_idx + len(term_words) - 1].end
            return (t_start, t_end)

    return None


PUNCT_STRIP_RE = re.compile(r"[^\w'-]")


def find_term_span_by_position(
    term: str,
    utterance: str,
    word_intervals: List[WordInterval],
) -> Optional[Tuple[float, float]]:
    """Fallback: locate term span by word-index correspondence between utterance and MFA.

    When MFA maps OOV words to <unk>, exact matching fails. This approach finds
    the term's position in the original utterance and maps those word indices
    to MFA intervals (which maintain 1:1 correspondence with input words).
    """
    utt_words = [PUNCT_STRIP_RE.sub("", w) for w in utterance.split()]
    utt_words = [w for w in utt_words if w]
    term_words = [PUNCT_STRIP_RE.sub("", w) for w in term.split()]
    term_words = [w for w in term_words if w]

    if not term_words or not utt_words:
        return None

    term_start_idx = None
    for i in range(len(utt_words) - len(term_words) + 1):
        if all(
            utt_words[i + j].lower() == tw.lower()
            for j, tw in enumerate(term_words)
        ):
            term_start_idx = i
            break

    if term_start_idx is None:
        return None

    term_end_idx = term_start_idx + len(term_words) - 1

    if len(word_intervals) != len(utt_words):
        return None

    return (word_intervals[term_start_idx].start, word_intervals[term_end_idx].end)


def get_chunk_text(
    word_intervals: List[WordInterval],
    chunk_start: float,
    chunk_end: float,
    utterance: str = "",
) -> str:
    """Get text of words whose midpoints fall within the chunk time window.

    When MFA word count matches utterance word count, uses original utterance
    words (preserving casing, replacing <unk>) instead of MFA's lowercased output.
    """
    utt_words: List[str] = []
    if utterance:
        utt_words = [PUNCT_STRIP_RE.sub("", w) for w in utterance.split()]
        utt_words = [w for w in utt_words if w]

    use_original = len(word_intervals) == len(utt_words) and len(utt_words) > 0

    words = []
    for i, wi in enumerate(word_intervals):
        mid = (wi.start + wi.end) / 2
        if chunk_start <= mid < chunk_end:
            if use_original:
                words.append(utt_words[i])
            else:
                words.append(wi.word)
    return " ".join(words)


def load_noise_paths(noise_dir: str) -> list[str]:
    """Load all WHAM noise WAV paths from directory."""
    paths = sorted([
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith(".wav")
    ])
    assert len(paths) > 0, f"No .wav files found in {noise_dir}"
    return paths


def add_wham_noise(
    clean_chunk: np.ndarray,
    noise_paths: list[str],
    rng: random.Random,
) -> np.ndarray:
    """Mix a random WHAM noise clip into clean_chunk at a random SNR."""
    noise_file = rng.choice(noise_paths)
    snr_db = rng.uniform(SNR_LOW_DB, SNR_HIGH_DB)

    noise, noise_sr = sf.read(noise_file, dtype="float32")
    if noise.ndim > 1:
        noise = noise.mean(axis=1)
    assert noise_sr == SAMPLE_RATE, (
        f"Noise SR {noise_sr} != expected {SAMPLE_RATE}"
    )

    n_clean = len(clean_chunk)
    if len(noise) < n_clean:
        repeats = (n_clean // len(noise)) + 1
        noise = np.tile(noise, repeats)[:n_clean]
    else:
        max_start = len(noise) - n_clean
        start = rng.randint(0, max_start)
        noise = noise[start: start + n_clean]

    clean_power = np.mean(clean_chunk ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    snr_linear = 10 ** (snr_db / 10.0)
    noise_gain = np.sqrt(clean_power / (noise_power * snr_linear))

    mixed = clean_chunk + noise_gain * noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def cut_chunk_around_term(
    wav_path: str,
    term_start: float,
    term_end: float,
    rng: random.Random,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Cut a 1.92s chunk centered on the term with random jitter.

    Returns (chunk_audio, chunk_start_sec, chunk_end_sec) or None.
    """
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {sr}Hz: {wav_path}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    total_duration = len(audio) / SAMPLE_RATE

    term_dur = term_end - term_start
    if term_dur >= CHUNK_SEC:
        center = (term_start + term_end) / 2
        chunk_start = max(0.0, center - CHUNK_SEC / 2)
    else:
        available_pad = CHUNK_SEC - term_dur
        min_pre = min(MIN_CONTEXT_PAD_SEC, available_pad)
        max_pre = max(min_pre, available_pad - MIN_CONTEXT_PAD_SEC)
        pre_pad = rng.uniform(min_pre, max_pre)
        chunk_start = max(0.0, term_start - pre_pad)

    if chunk_start + CHUNK_SEC > total_duration:
        chunk_start = max(0.0, total_duration - CHUNK_SEC)

    chunk_end = chunk_start + CHUNK_SEC

    start_sample = int(chunk_start * SAMPLE_RATE)
    end_sample = start_sample + CHUNK_SAMPLES

    chunk_audio = audio[start_sample:end_sample]
    if len(chunk_audio) < CHUNK_SAMPLES:
        chunk_audio = np.pad(
            chunk_audio,
            (0, CHUNK_SAMPLES - len(chunk_audio)),
            mode="constant",
        )

    return chunk_audio, chunk_start, chunk_end


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MFA alignment + 1.92s chunk cutting for wiki_synth TTS data"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to wiki_synth_utterances_1M_with_tts.jsonl",
    )
    parser.add_argument(
        "--work-dir", type=str, required=True,
        help="MFA working directory (use fast local storage, e.g. /mnt/data/...)",
    )
    parser.add_argument(
        "--output-audio-dir", type=str, required=True,
        help="Output directory for 1.92s chunk WAV files",
    )
    parser.add_argument(
        "--output-jsonl", type=str, required=True,
        help="Output training JSONL path",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--global-idx-offset", type=int, default=0,
        help="Add this offset to JSONL row indices before generating utter_id/chunk names. "
             "Useful when processing a suffix split separately but preserving full-dataset ids.",
    )
    parser.add_argument("--mfa-num-jobs", type=int, default=MFA_NUM_JOBS)
    parser.add_argument(
        "--mfa-conda-prefix", type=str, default="",
        help="Full path to MFA conda env (overrides 'conda run -n mfa' with '--prefix')",
    )
    parser.add_argument(
        "--skip-mfa", action="store_true",
        help="Skip MFA step (reuse existing TextGrids)",
    )
    parser.add_argument(
        "--remap-src", nargs=2, metavar=("FROM", "TO"), default=None,
        help="Remap source audio path prefix for reading (e.g. /mnt/data/ /mnt/taurus/data/)",
    )
    parser.add_argument(
        "--remap-output", nargs=2, metavar=("FROM", "TO"), default=None,
        help="Remap output chunk_audio_path prefix in JSONL (e.g. /mnt/data/ /mnt/aries/data/)",
    )
    parser.add_argument(
        "--noise-dir", type=str, default="",
        help="Directory containing WHAM noise WAV files. Required for noisy chunk generation.",
    )
    parser.add_argument(
        "--mfa-model-dir", type=str, default="",
        help="Absolute path to MFA root (e.g. /mnt/taurus/home/.../Documents/MFA). "
             "When set, dictionary and acoustic model are resolved to absolute paths "
             "so pretrained-name lookup (which depends on ~/Documents/MFA) is bypassed.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--smoke-test", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed + args.shard_id)

    noise_paths: list[str] = []
    if args.noise_dir:
        noise_paths = load_noise_paths(args.noise_dir)
        print(f"[INFO] Loaded {len(noise_paths)} WHAM noise clips from {args.noise_dir}",
              flush=True)

    shard_tag = f"shard_{args.shard_id:02d}"
    mfa_input_dir = os.path.join(args.work_dir, shard_tag, "mfa_input")
    mfa_output_dir = os.path.join(args.work_dir, shard_tag, "mfa_output")
    mfa_cache_dir = os.path.join(args.work_dir, shard_tag, "mfa_cache")
    output_audio_shard = os.path.join(args.output_audio_dir, shard_tag)

    os.makedirs(output_audio_shard, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    # Step 1: Load data (shard slice)
    entries = load_data(args.data, args.shard_id, args.num_shards, args.smoke_test)
    if args.global_idx_offset:
        entries = [(global_idx + args.global_idx_offset, entry) for global_idx, entry in entries]
    assert len(entries) > 0, "No entries to process in this shard"

    # Build path remap function for source audio
    def remap_src(path: str) -> str:
        if args.remap_src and path.startswith(args.remap_src[0]):
            return args.remap_src[1] + path[len(args.remap_src[0]):]
        return path

    def remap_output(path: str) -> str:
        if args.remap_output and path.startswith(args.remap_output[0]):
            return args.remap_output[1] + path[len(args.remap_output[0]):]
        return path

    # Step 2: Prepare MFA input (symlinks + .lab files)
    utt_map = prepare_mfa_input(entries, mfa_input_dir, remap_src)

    # Step 3: Run MFA forced alignment
    if not args.skip_mfa:
        run_mfa(mfa_input_dir, mfa_output_dir, mfa_cache_dir,
                args.mfa_num_jobs, args.mfa_conda_prefix,
                args.mfa_model_dir)
    else:
        print("[INFO] Skipping MFA (--skip-mfa). Using existing TextGrids.", flush=True)

    # Step 4: Parse TextGrids → cut clean chunk → add WHAM noise → save both
    success = 0
    no_textgrid = 0
    no_term_span = 0
    fallback_used = 0

    results = []
    for utt_id, (global_idx, entry) in utt_map.items():
        tg_path = find_textgrid(mfa_output_dir, utt_id)
        if tg_path is None:
            no_textgrid += 1
            continue

        try:
            word_intervals = parse_short_textgrid(tg_path)
        except Exception as exc:
            print(f"[WARN] Failed to parse TextGrid for {utt_id}: {exc}", flush=True)
            no_textgrid += 1
            continue

        if not word_intervals:
            no_textgrid += 1
            continue

        term_span = find_term_span(entry["term"], word_intervals)
        if term_span is None:
            term_span = find_term_span_by_position(
                entry["term"], entry["utterance"], word_intervals
            )
            if term_span is not None:
                fallback_used += 1
        if term_span is None:
            no_term_span += 1
            continue

        term_start, term_end = term_span

        clean_path = entry.get("clean_audio_path", entry.get("tts_audio_path"))
        clean_wav = remap_src(clean_path)
        if not os.path.isfile(clean_wav):
            print(f"[WARN] clean WAV missing: {clean_wav}", flush=True)
            continue

        cut_result = cut_chunk_around_term(clean_wav, term_start, term_end, rng)
        if cut_result is None:
            continue

        clean_chunk, chunk_start, chunk_end = cut_result

        chunk_src_text = get_chunk_text(
            word_intervals, chunk_start, chunk_end, entry["utterance"]
        )
        mfa_term_start_in_chunk = max(0.0, term_start - chunk_start)
        mfa_term_end_in_chunk = min(CHUNK_SEC, term_end - chunk_start)
        mfa_term_duration = max(0.0, term_end - term_start)

        clean_filename = f"{utt_id}_clean.wav"
        clean_chunk_path = os.path.join(output_audio_shard, clean_filename)
        sf.write(clean_chunk_path, clean_chunk, SAMPLE_RATE, subtype="PCM_16")

        results.append({
            "term": entry["term"],
            "term_key": entry["term"].lower(),
            "chunk_src_text": chunk_src_text,
            "utter_id": f"wiki_synth_{global_idx:07d}",
            "chunk_idx": 0,
            "chunk_audio_path": remap_output(os.path.abspath(clean_chunk_path)),
            "audio_type": "clean",
            "mfa_term_start_in_chunk": mfa_term_start_in_chunk,
            "mfa_term_end_in_chunk": mfa_term_end_in_chunk,
            "mfa_term_duration": mfa_term_duration,
        })

        # --- Noisy version: reuse EXACT same sample offsets as clean ---
        start_sample = int(chunk_start * SAMPLE_RATE)
        noisy_chunk = None

        noisy_src = entry.get("noisy_audio_path", "")
        if noisy_src:
            noisy_wav = remap_src(noisy_src)
            if os.path.isfile(noisy_wav):
                noisy_audio, n_sr = sf.read(noisy_wav, dtype="float32")
                assert n_sr == SAMPLE_RATE, f"Noisy SR {n_sr} != {SAMPLE_RATE}"
                if noisy_audio.ndim > 1:
                    noisy_audio = noisy_audio.mean(axis=1)
                noisy_chunk = noisy_audio[start_sample:start_sample + CHUNK_SAMPLES]
                if len(noisy_chunk) < CHUNK_SAMPLES:
                    noisy_chunk = np.pad(
                        noisy_chunk,
                        (0, CHUNK_SAMPLES - len(noisy_chunk)),
                        mode="constant",
                    )

        if noisy_chunk is None and noise_paths:
            noisy_chunk = add_wham_noise(clean_chunk, noise_paths, rng)

        if noisy_chunk is not None:
            noisy_filename = f"{utt_id}_noisy.wav"
            noisy_chunk_path = os.path.join(output_audio_shard, noisy_filename)
            sf.write(noisy_chunk_path, noisy_chunk, SAMPLE_RATE, subtype="PCM_16")

            results.append({
                "term": entry["term"],
                "term_key": entry["term"].lower(),
                "chunk_src_text": chunk_src_text,
                "utter_id": f"wiki_synth_{global_idx:07d}",
                "chunk_idx": 0,
                "chunk_audio_path": remap_output(os.path.abspath(noisy_chunk_path)),
                "audio_type": "noisy",
                "mfa_term_start_in_chunk": mfa_term_start_in_chunk,
                "mfa_term_end_in_chunk": mfa_term_end_in_chunk,
                "mfa_term_duration": mfa_term_duration,
            })

        success += 1

    # Write output JSONL
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    clean_rows = sum(1 for r in results if r.get("audio_type") == "clean")
    noisy_rows = sum(1 for r in results if r.get("audio_type") == "noisy")
    print(f"\n{'=' * 60}", flush=True)
    print(f"[RESULT] Shard {args.shard_id}/{args.num_shards}", flush=True)
    print(f"  Input entries:    {len(entries)}", flush=True)
    print(f"  MFA prepared:     {len(utt_map)}", flush=True)
    print(f"  Terms aligned:    {success}", flush=True)
    print(f"  Training rows:    {len(results)} (clean={clean_rows}, noisy={noisy_rows})", flush=True)
    print(f"  Fallback used:    {fallback_used}", flush=True)
    print(f"  No TextGrid:      {no_textgrid}", flush=True)
    print(f"  No term span:     {no_term_span}", flush=True)
    print(f"  Output JSONL:     {args.output_jsonl}", flush=True)
    print(f"  Audio chunks dir: {output_audio_shard}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
