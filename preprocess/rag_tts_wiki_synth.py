"""TTS generation using CosyVoice for wiki synthetic utterances.

Generates TTS audio for utterances from wiki_synth_utterances_1M_all.jsonl
using CosyVoice zero-shot inference with random VCTK speaker prompts.
Supports sharding for SLURM array parallelism.

python preprocess/rag_tts_wiki_synth.py \
    --data /data/group_data/li_lab/siqiouya/datasets/gigaspeech/wiki_synth_utterances_1M_all.jsonl \
    --output-dir /data/group_data/li_lab/siqiouya/datasets/gigaspeech/wiki_synth_utterances_tts \
    --shard-id 0 --num-shards 24
"""

import os
import sys
import json
import glob
import random
import argparse

# Add CosyVoice to path
sys.path.append('/home/siqiouya/code/CosyVoice')
sys.path.append('/home/siqiouya/code/CosyVoice/third_party/Matcha-TTS')

import numpy as np
import soundfile as sf
import torchaudio
from tqdm import tqdm

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed


def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix clean audio with noise at a given SNR level.

    Args:
        clean: Clean audio array (1D)
        noise: Noise audio array (1D), must be >= len(clean)
        snr_db: Target signal-to-noise ratio in dB

    Returns:
        Noisy audio array (1D), same length as clean
    """
    # Trim noise to match clean length (random offset)
    if len(noise) > len(clean):
        max_offset = len(noise) - len(clean)
        offset = random.randint(0, max_offset)
        noise = noise[offset:offset + len(clean)]

    # Compute scaling factor for noise
    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(clean_power / (noise_power * snr_linear))

    return clean + scale * noise


def load_data(data_path: str) -> list[dict]:
    """Load JSONL data file, keeping only the first utterance per term.

    Args:
        data_path: Path to the JSONL file

    Returns:
        List of entries (one per unique term)
    """
    data = []
    seen_terms = set()
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            term = entry["term"]
            if term not in seen_terms:
                seen_terms.add(term)
                data.append(entry)
    return data


def main(args):
    """Run TTS generation using CosyVoice.

    Args:
        args: Parsed CLI args from parse_args().
    """

    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    total = len(data)
    print(f"Loaded {total} utterances")

    # Compute shard indices (strided)
    shard_indices = list(range(args.shard_id, total, args.num_shards))
    print(f"Shard {args.shard_id}/{args.num_shards}: processing {len(shard_indices)} utterances")

    # Load VCTK speaker prompts
    print(f"Loading speaker prompts from {args.speaker_index}...")
    with open(args.speaker_index, "r") as f:
        speaker_prompts = json.load(f)
    print(f"Loaded {len(speaker_prompts)} speaker prompts")

    # Load noise files
    noise_files = sorted(glob.glob(os.path.join(args.noise_dir, "*.wav")))
    print(f"Loaded {len(noise_files)} noise files from {args.noise_dir}")

    # Initialize CosyVoice model
    print(f"Loading CosyVoice model from {args.model_dir}...")
    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_trt=True,
        load_vllm=True,
        fp16=False
    )
    print(f"Model loaded. Sample rate: {cosyvoice.sample_rate}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create resampler if needed
    resampler = None
    if cosyvoice.sample_rate != args.sampling_rate:
        print(f"Initializing resampler: {cosyvoice.sample_rate} Hz -> {args.sampling_rate} Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=cosyvoice.sample_rate,
            new_freq=args.sampling_rate
        )

    # Process each utterance in this shard
    results = []
    successful = 0
    failed = 0

    for line_idx in tqdm(shard_indices, desc=f"Shard {args.shard_id} TTS"):
        entry = data[line_idx]
        utterance = entry["utterance"]

        # Skip empty utterances
        if not utterance or not utterance.strip():
            print(f"Skipping empty utterance at line {line_idx}")
            failed += 1
            continue

        # Chunked subdirectory: line_idx // 10000, zero-padded to 4 digits
        chunk_dir = f"{line_idx // 10000:04d}"
        subdir = os.path.join(args.output_dir, chunk_dir)
        os.makedirs(subdir, exist_ok=True)

        tts_audio_path = os.path.join(subdir, f"{line_idx}.wav")

        # Skip if file already exists (resumability)
        if os.path.exists(tts_audio_path):
            successful += 1
            entry_with_tts = entry.copy()
            entry_with_tts["tts_audio_path"] = tts_audio_path
            noisy_path = os.path.join(subdir, f"{line_idx}_noisy.wav")
            if os.path.exists(noisy_path):
                entry_with_tts["tts_noisy_audio_path"] = noisy_path
            results.append(entry_with_tts)
            continue

        # Set random seed for reproducibility
        if args.seed is not None:
            set_all_random_seed(args.seed + line_idx)

        # Select random speaker prompt (deterministic per line_idx)
        rng = random.Random(args.seed + line_idx)
        speaker = rng.choice(speaker_prompts)
        ref_audio = speaker["wav_path"]
        ref_text = speaker["text"]

        # Generate TTS audio using CosyVoice zero-shot
        try:
            audio_generated = False
            for i, output in enumerate(cosyvoice.inference_zero_shot(
                utterance,
                'You are a helpful assistant.<|endofprompt|>' + ref_text,
                ref_audio,
                stream=False,
                text_frontend=args.text_frontend
            )):
                audio_tensor = output['tts_speech']  # Shape: [1, time]

                if resampler is not None:
                    audio_tensor = resampler(audio_tensor)

                audio_numpy = audio_tensor.squeeze(0).cpu().numpy()
                sf.write(tts_audio_path, audio_numpy, samplerate=args.sampling_rate, format="WAV")
                audio_generated = True

                # Add noise augmentation
                noisy_audio_path = os.path.join(subdir, f"{line_idx}_noisy.wav")
                if not os.path.exists(noisy_audio_path):
                    # Pick random SNR from mixture of ranges: 5-10, 10-15, 15-20 dB
                    snr_db = rng.uniform(args.snr_min, args.snr_max)
                    # Try up to 5 random noise files to find one longer than TTS
                    for _ in range(5):
                        noise_path = rng.choice(noise_files)
                        noise_data, noise_sr = sf.read(noise_path)
                        if noise_sr != args.sampling_rate:
                            continue
                        if len(noise_data) >= len(audio_numpy):
                            noisy = add_noise(audio_numpy, noise_data, snr_db)
                            sf.write(noisy_audio_path, noisy, samplerate=args.sampling_rate, format="WAV")
                            break

                break  # Only take the first output

            if not audio_generated:
                print(f"Warning: No audio generated for line {line_idx}: '{utterance[:50]}'")
                failed += 1
                continue

            successful += 1
            entry_with_tts = entry.copy()
            entry_with_tts["tts_audio_path"] = tts_audio_path
            if os.path.exists(os.path.join(subdir, f"{line_idx}_noisy.wav")):
                entry_with_tts["tts_noisy_audio_path"] = os.path.join(subdir, f"{line_idx}_noisy.wav")
            results.append(entry_with_tts)

        except Exception as e:
            print(f"Error generating TTS for line {line_idx} ('{utterance[:50]}'): {e}")
            failed += 1
            continue

    # Save output JSONL for this shard
    output_jsonl = os.path.join(
        os.path.dirname(args.data),
        f"wiki_synth_utterances_1M_all_with_tts_shard{args.shard_id}.jsonl"
    )
    print(f"\nSaving {len(results)} results to {output_jsonl}...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Done! Shard {args.shard_id}: {successful} successful, {failed} failed out of {len(shard_indices)} utterances.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TTS for wiki synthetic utterances using CosyVoice")

    parser.add_argument(
        "--data",
        type=str,
        default="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/wiki_synth_utterances_1M_all.jsonl",
        help="Path to the JSONL file containing utterances",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/group_data/li_lab/siqiouya/datasets/gigaspeech/wiki_synth_utterances_tts",
        help="Directory to save generated audio files",
    )

    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard ID for parallel processing (default: 0)",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards (default: 1)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/user_data/siqiouya/ckpts/pretrained/tts/Fun-CosyVoice3-0.5B",
        help="Path to CosyVoice model directory",
    )

    parser.add_argument(
        "--speaker-index",
        type=str,
        default="/data/group_data/li_lab/siqiouya/datasets/vctk_speaker_prompts/speaker_index.json",
        help="Path to VCTK speaker index JSON (list of {wav_path, text, speaker_id})",
    )

    parser.add_argument(
        "--noise-dir",
        type=str,
        default="/data/group_data/li_lab/siqiouya/datasets/wham_wav",
        help="Directory containing noise wav files",
    )

    parser.add_argument(
        "--snr-min",
        type=float,
        default=5.0,
        help="Minimum SNR in dB for noise augmentation (default: 5.0)",
    )

    parser.add_argument(
        "--snr-max",
        type=float,
        default=20.0,
        help="Maximum SNR in dB for noise augmentation (default: 20.0)",
    )

    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate for audio saving (default: 16000)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--text-frontend",
        action="store_true",
        default=True,
        help="Enable text frontend normalization (default: True)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
