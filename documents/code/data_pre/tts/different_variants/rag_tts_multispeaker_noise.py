"""TTS generation with multi-speaker voice cloning and noise augmentation.

Based on rag_tts_wiki_synth.py, with two key additions:
  1. Multi-speaker: randomly select a VCTK speaker prompt per utterance
     for CosyVoice zero-shot voice cloning (instead of a single fixed voice).
  2. Noise augmentation: randomly mix in WHAM! noise at a random SNR
     to simulate realistic acoustic conditions.

Usage:
    python rag_tts_multispeaker_noise.py \
        --data wiki_synth_utterances.jsonl \
        --output-dir /mnt/data/jiaxuanluo/wiki_synth_utterances_tts_augmented \
        --speaker-dir /mnt/data/siqiouyang/datasets/vctk_speaker_prompts \
        --noise-dir /mnt/data/siqiouyang/datasets/wham_wav \
        --shard-id 0 --num-shards 8 --batch-size 4
"""

import os
import sys
import json
import random
import argparse
import concurrent.futures

COSYVOICE_ROOT = os.environ.get("COSYVOICE_ROOT", "/mnt/gemini/home/jiaxuanluo/CosyVoice")
sys.path.append(COSYVOICE_ROOT)
sys.path.append(os.path.join(COSYVOICE_ROOT, "third_party/Matcha-TTS"))

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

# ======Configuration=====
SPEAKER_DIR = "/mnt/data/siqiouyang/datasets/vctk_speaker_prompts"
NOISE_DIR = "/mnt/data/siqiouyang/datasets/wham_wav"

SNR_LOW_DB = 5
SNR_HIGH_DB = 25

PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"

SPEAKER_INDEX_FILENAME = "speaker_index.json"

DEFAULT_REF_TEXT = "希望你以后能够做的比我还好呦。"
DEFAULT_REF_AUDIO = "/mnt/gemini/home/jiaxuanluo/CosyVoice/asset/zero_shot_prompt.wav"
# ======Configuration=====


def load_data(data_path: str, no_dedup: bool = False) -> list[dict]:
    """Load JSONL data file.

    By default, keeps only the first utterance per term (backward-compatible).
    With no_dedup=True, keeps ALL rows (for multi-variant per term).
    """
    data = []
    if no_dedup:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    seen_terms: set[str] = set()
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            term = entry["term"]
            if term not in seen_terms:
                seen_terms.add(term)
                data.append(entry)
    return data


def load_speaker_prompts(speaker_dir: str) -> list[dict]:
    """Load speaker prompts with per-speaker ref_text from speaker_index.json.

    Each returned dict has keys: speaker_id, wav_path, text, gender, accent.
    """
    index_path = os.path.join(speaker_dir, SPEAKER_INDEX_FILENAME)
    assert os.path.isfile(index_path), (
        f"Speaker index not found: {index_path}. "
        f"Expected a JSON file with speaker_id, wav_path, text fields."
    )

    with open(index_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    speakers = []
    for entry in entries:
        local_wav = os.path.join(speaker_dir, f"{entry['speaker_id']}.wav")
        assert os.path.isfile(local_wav), f"Speaker WAV not found: {local_wav}"
        speakers.append({
            "speaker_id": entry["speaker_id"],
            "wav_path": local_wav,
            "text": entry["text"],
            "gender": entry.get("gender", "Unknown"),
            "accent": entry.get("accent", "Unknown"),
        })

    assert len(speakers) > 0, f"No speakers found in {index_path}"
    print(f"Loaded {len(speakers)} speaker prompts from {index_path}")
    return speakers


def load_noise_paths(noise_dir: str) -> list[str]:
    """Load all noise WAV paths from directory."""
    paths = sorted([
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith(".wav")
    ])
    assert len(paths) > 0, f"No .wav files found in {noise_dir}"
    print(f"Loaded {len(paths)} noise clips from {noise_dir}")
    return paths


def mix_noise(clean_audio: np.ndarray, noise_path: str,
              snr_db: float, target_sr: int) -> np.ndarray:
    """Mix a noise clip into clean audio at the specified SNR (dB)."""
    noise, noise_sr = sf.read(noise_path, dtype="float32")

    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    if noise_sr != target_sr:
        noise_tensor = torch.from_numpy(noise).unsqueeze(0)
        noise_tensor = torchaudio.functional.resample(noise_tensor, noise_sr, target_sr)
        noise = noise_tensor.squeeze(0).numpy()

    n_clean = len(clean_audio)
    if len(noise) < n_clean:
        repeats = (n_clean // len(noise)) + 1
        noise = np.tile(noise, repeats)[:n_clean]
    else:
        max_start = len(noise) - n_clean
        start = random.randint(0, max_start)
        noise = noise[start : start + n_clean]

    clean_power = np.mean(clean_audio ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    snr_linear = 10 ** (snr_db / 10.0)
    noise_gain = np.sqrt(clean_power / (noise_power * snr_linear))

    mixed = clean_audio + noise_gain * noise

    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed


def _generate_one(cosyvoice, utterance: str, ref_text: str, ref_audio: str,
                  text_frontend: bool, clean_audio_path: str, noisy_audio_path: str,
                  sampling_rate: int, resampler,
                  noise_paths: list[str] | None,
                  snr_low: float, snr_high: float, rng_seed: int):
    """Generate clean + noisy TTS variants for one utterance.

    Runs TTS inference once, saves the clean version, then mixes noise and
    saves the noisy version.  Returns (clean_ok, noisy_ok).
    """
    try:
        rng = random.Random(rng_seed)

        for _, output in enumerate(cosyvoice.inference_zero_shot(
            utterance,
            PROMPT_PREFIX + ref_text,
            ref_audio,
            stream=False,
            text_frontend=text_frontend,
        )):
            audio_tensor = output["tts_speech"]
            if resampler is not None:
                audio_tensor = resampler(audio_tensor)
            clean_audio = audio_tensor.squeeze(0).cpu().numpy()

            os.makedirs(os.path.dirname(clean_audio_path), exist_ok=True)
            sf.write(clean_audio_path, clean_audio, samplerate=sampling_rate, format="WAV")

            noisy_ok = False
            if noise_paths is not None:
                noise_path = rng.choice(noise_paths)
                snr_db = rng.uniform(snr_low, snr_high)
                noisy_audio = mix_noise(clean_audio, noise_path, snr_db, sampling_rate)
                os.makedirs(os.path.dirname(noisy_audio_path), exist_ok=True)
                sf.write(noisy_audio_path, noisy_audio, samplerate=sampling_rate, format="WAV")
                noisy_ok = True

            return True, noisy_ok

        return False, False
    except Exception as e:
        print(f"Error generating TTS for '{utterance[:50]}': {e}")
        return False, False


def main(args):
    data = load_data(args.data, no_dedup=args.no_dedup)
    total = len(data)
    dedup_tag = "all" if args.no_dedup else "unique-term"
    print(f"Loaded {total} {dedup_tag} utterances from {args.data}")

    shard_indices = list(range(args.shard_id, total, args.num_shards))
    print(f"Shard {args.shard_id}/{args.num_shards}: processing {len(shard_indices)} utterances")

    # --- Speaker prompts (with per-speaker ref_text) ---
    speaker_prompts = None
    if args.speaker_dir:
        speaker_prompts = load_speaker_prompts(args.speaker_dir)
        accents = set(s["accent"] for s in speaker_prompts)
        genders = {s["gender"] for s in speaker_prompts}
        print(f"  Accents: {sorted(accents)}")
        print(f"  Genders: {sorted(genders)}")

    # --- Noise clips ---
    noise_paths = None
    if args.noise_dir:
        noise_paths = load_noise_paths(args.noise_dir)

    # --- CosyVoice model ---
    print(f"Loading CosyVoice model from {args.model_dir}...")
    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_trt=args.load_trt,
        load_vllm=True,
        fp16=False,
    )
    print(f"Model loaded. Sample rate: {cosyvoice.sample_rate}")

    os.makedirs(args.output_dir, exist_ok=True)

    resampler = None
    if cosyvoice.sample_rate != args.sampling_rate:
        print(f"Initializing resampler: {cosyvoice.sample_rate} Hz -> {args.sampling_rate} Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=cosyvoice.sample_rate,
            new_freq=args.sampling_rate,
        )

    # --- Partition into skip vs todo ---
    results = []
    todo_items = []
    skipped = 0

    clean_subdir = "clean"
    noisy_subdir = "noisy"

    for line_idx in shard_indices:
        entry = data[line_idx]
        utterance = entry["utterance"]
        if not utterance or not utterance.strip():
            continue

        chunk_dir = f"{line_idx // 10000:04d}"
        clean_path = os.path.join(args.output_dir, clean_subdir, chunk_dir, f"{line_idx}.wav")
        noisy_path = os.path.join(args.output_dir, noisy_subdir, chunk_dir, f"{line_idx}.wav")

        if os.path.exists(clean_path) and (noise_paths is None or os.path.exists(noisy_path)):
            skipped += 1
            entry_with_tts = entry.copy()
            entry_with_tts["clean_audio_path"] = clean_path
            if noise_paths is not None:
                entry_with_tts["noisy_audio_path"] = noisy_path
            results.append(entry_with_tts)
            continue

        # Deterministic per-utterance: pick speaker
        idx_rng = random.Random(line_idx)
        if speaker_prompts is not None:
            speaker = idx_rng.choice(speaker_prompts)
            ref_audio = speaker["wav_path"]
            ref_text = speaker["text"]
        else:
            ref_audio = args.ref_audio
            ref_text = args.ref_text

        todo_items.append((line_idx, entry, clean_path, noisy_path, ref_text, ref_audio))

    print(f"Skipped {skipped} already completed, {len(todo_items)} remaining")

    # --- Generate (clean + noisy for each utterance) ---
    successful, failed = 0, 0

    if args.batch_size <= 1:
        for line_idx, entry, clean_path, noisy_path, ref_text, ref_audio in tqdm(
            todo_items, desc=f"Shard {args.shard_id} TTS"
        ):
            clean_ok, noisy_ok = _generate_one(
                cosyvoice, entry["utterance"], ref_text, ref_audio,
                args.text_frontend, clean_path, noisy_path,
                args.sampling_rate, resampler, noise_paths,
                args.snr_low, args.snr_high, rng_seed=line_idx,
            )
            if clean_ok:
                successful += 1
                entry_with_tts = entry.copy()
                entry_with_tts["clean_audio_path"] = clean_path
                if noisy_ok:
                    entry_with_tts["noisy_audio_path"] = noisy_path
                results.append(entry_with_tts)
            else:
                failed += 1
    else:
        print(f"Using concurrent batch_size={args.batch_size} for vLLM batching")
        pbar = tqdm(total=len(todo_items),
                    desc=f"Shard {args.shard_id} TTS (bs={args.batch_size})")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            pending: dict[concurrent.futures.Future, tuple] = {}
            todo_iter = iter(todo_items)

            for item in todo_iter:
                line_idx, entry, clean_path, noisy_path, ref_text, ref_audio = item
                future = executor.submit(
                    _generate_one,
                    cosyvoice, entry["utterance"], ref_text, ref_audio,
                    args.text_frontend, clean_path, noisy_path,
                    args.sampling_rate, resampler, noise_paths,
                    args.snr_low, args.snr_high, rng_seed=line_idx,
                )
                pending[future] = item
                if len(pending) >= args.batch_size:
                    break

            while pending:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    line_idx, entry, clean_path, noisy_path, _, _ = pending.pop(future)
                    clean_ok, noisy_ok = future.result()
                    if clean_ok:
                        successful += 1
                        entry_with_tts = entry.copy()
                        entry_with_tts["clean_audio_path"] = clean_path
                        if noisy_ok:
                            entry_with_tts["noisy_audio_path"] = noisy_path
                        results.append(entry_with_tts)
                    else:
                        failed += 1
                    pbar.update(1)

                    next_item = next(todo_iter, None)
                    if next_item is not None:
                        nl, ne, cp, np_, rt, ra = next_item
                        new_future = executor.submit(
                            _generate_one,
                            cosyvoice, ne["utterance"], rt, ra,
                            args.text_frontend, cp, np_,
                            args.sampling_rate, resampler, noise_paths,
                            args.snr_low, args.snr_high, rng_seed=nl,
                        )
                        pending[new_future] = next_item

        pbar.close()

    # --- Save shard output ---
    output_prefix = args.output_jsonl_prefix or "wiki_synth_utterances_1M_all_with_tts"
    output_jsonl = os.path.join(
        os.path.dirname(args.data),
        f"{output_prefix}_shard{args.shard_id}.jsonl",
    )
    print(f"\nSaving {len(results)} results to {output_jsonl}...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        f"Done! Shard {args.shard_id}: {successful} successful, {failed} failed, "
        f"{skipped} skipped out of {len(shard_indices)} utterances."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-speaker + noise-augmented TTS for wiki synthetic utterances"
    )

    # --- Data ---
    parser.add_argument("--data", type=str, required=True,
                        help="Path to JSONL file containing utterances")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save generated audio files")

    # --- Sharding ---
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)

    # --- Model ---
    parser.add_argument("--model-dir", type=str,
                        default="/mnt/gemini/home/jiaxuanluo/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--load-trt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--text-frontend", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Concurrent vLLM requests per GPU (1 = sequential)")

    # --- Multi-speaker ---
    parser.add_argument("--speaker-dir", type=str, default=SPEAKER_DIR,
                        help="Directory of speaker prompt WAVs for zero-shot cloning. "
                             "Set to empty string to use a single fixed speaker.")
    parser.add_argument("--ref-audio", type=str, default=DEFAULT_REF_AUDIO,
                        help="Fallback reference audio (used when --speaker-dir is empty)")
    parser.add_argument("--ref-text", type=str, default=DEFAULT_REF_TEXT,
                        help="Reference text for zero-shot prompt. When using VCTK speakers "
                             "without transcripts, this is an approximate placeholder.")

    # --- Noise augmentation ---
    parser.add_argument("--noise-dir", type=str, default=NOISE_DIR,
                        help="Directory of noise WAVs (WHAM!). "
                             "Set to empty string to disable noise augmentation.")
    parser.add_argument("--snr-low", type=float, default=SNR_LOW_DB,
                        help="Minimum SNR in dB for noise mixing")
    parser.add_argument("--snr-high", type=float, default=SNR_HIGH_DB,
                        help="Maximum SNR in dB for noise mixing")

    # --- Multi-variant ---
    parser.add_argument("--no_dedup", action="store_true",
                        help="Keep ALL rows from the input JSONL (no per-term dedup). "
                             "Required when input has multiple variant_idx per term.")
    parser.add_argument("--output_jsonl_prefix", type=str, default="",
                        help="Prefix for the per-shard output JSONL filename. "
                             "Default: wiki_synth_utterances_1M_all_with_tts")

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.speaker_dir and not os.path.isdir(args.speaker_dir):
        parser.error(f"--speaker-dir not found: {args.speaker_dir}")
    if args.noise_dir and not os.path.isdir(args.noise_dir):
        parser.error(f"--noise-dir not found: {args.noise_dir}")

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        set_all_random_seed(args.seed)
    main(args)
