"""TTS generation using CosyVoice for voice cloning.

Generates TTS audio for terms using CosyVoice zero-shot inference
with random reference audio from the dataset.
"""

import os
import sys
import json
import random
import argparse
from collections import Counter

# Add CosyVoice to path
sys.path.append('/home/siqiouya/code/CosyVoice')
sys.path.append('/home/siqiouya/code/CosyVoice/third_party/Matcha-TTS')

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed


def parse_utter_id(utter_id: str) -> tuple[str, str]:
    """Parse utter_id to extract recording ID and segment number.
    
    Args:
        utter_id: String like "POD0000001195_7"
        
    Returns:
        Tuple of (recording_id, segment_num), e.g., ("POD0000001195", "7")
    """
    parts = utter_id.rsplit("_", 1)
    return parts[0], parts[1]


def load_data(data_path: str) -> list[dict]:
    """Load JSONL data file.
    
    Args:
        data_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing data rows
    """
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main(args):
    """Run TTS generation using CosyVoice.

    Args:
        args: Parsed CLI args from parse_args().
    """
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    print(f"Loaded {len(data)} samples")
    
    # Initialize CosyVoice model
    print(f"Loading CosyVoice model from {args.model_dir}...")
    print("Model settings: load_trt=True, load_vllm=True, fp16=False")
    cosyvoice = AutoModel(
        model_dir=args.model_dir,
        load_trt=True,
        load_vllm=True,
        fp16=False
    )
    print(f"Model loaded. Sample rate: {cosyvoice.sample_rate}")

    # Base TTS output directory
    base_output_dir = "/data/group_data/li_lab/siqiouya/datasets/gigaspeech/term_train_tts"
    
    # Create resampler if needed
    resampler = None
    if cosyvoice.sample_rate != args.sampling_rate:
        print(f"Initializing resampler: {cosyvoice.sample_rate} Hz -> {args.sampling_rate} Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=cosyvoice.sample_rate,
            new_freq=args.sampling_rate
        )
    
    # Process each data sample
    results = []
    cnt = Counter()
    for idx, sample in enumerate(tqdm(data, desc="Generating TTS")):
        term = sample["term"]
        utter_id = sample["utter_id"]
        chunk_idx = sample["chunk_idx"]
        
        # Select a random chunk_audio_path from the data as reference
        ref_sample = random.choice(data)
        # ref_audio = ref_sample["chunk_audio_path"]
        # ref_text = ref_sample["chunk_src_text"]
        ref_audio = "/home/siqiouya/code/Expressive-S2S/data/stresstest/ground_truth/audio_0.wav"
        ref_text = "I didn't say he stole the money."
        
        # Parse utter_id to get directory structure
        recording_id, segment_num = parse_utter_id(utter_id)
        
        # Create output path
        output_subdir = os.path.join(base_output_dir, recording_id, segment_num)
        os.makedirs(output_subdir, exist_ok=True)
        key = "{}_{}_{}".format(recording_id, segment_num, chunk_idx)
        tts_audio_path = os.path.join(output_subdir, f"chunk_{chunk_idx}_{cnt[key]}.wav")
        cnt[key] += 1

        if cnt[key] > 1:
            print(tts_audio_path)
        
        # Set random seed for reproducibility
        if args.seed is not None:
            set_all_random_seed(args.seed + idx)

        if not os.path.exists(tts_audio_path):        
            # Generate TTS audio using CosyVoice zero-shot
            try:
                # inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False)
                audio_generated = False
                for i, output in enumerate(cosyvoice.inference_zero_shot(
                    term,           # tts_text: the term to synthesize
                    'You are a helpful assistant.<|endofprompt|>' + ref_text,       # prompt_text: reference text
                    ref_audio,      # prompt_wav: reference audio path
                    stream=False,
                    text_frontend=args.text_frontend
                )):
                    # Get the audio tensor
                    audio_tensor = output['tts_speech']  # Shape: [1, time]
                    
                    # Resample if necessary
                    if resampler is not None:
                        audio_tensor = resampler(audio_tensor)
                    
                    # Convert to numpy
                    audio_numpy = audio_tensor.squeeze(0).cpu().numpy()
                    
                    # Save audio file
                    sf.write(tts_audio_path, audio_numpy, samplerate=args.sampling_rate, format="WAV")
                    audio_generated = True
                    break  # Only take the first output
                
                if not audio_generated:
                    print(f"Warning: No audio generated for sample {idx+1}")
                    continue
                    
            except Exception as e:
                print(f"Error generating TTS for sample {idx+1} (term: '{term}'): {e}")
                continue
        
        # Add tts_audio_path to the sample
        sample_with_tts = sample.copy()
        sample_with_tts["tts_audio_path"] = tts_audio_path
        results.append(sample_with_tts)
    
    # Save updated data to new JSONL file
    output_data_path = args.data.replace(".jsonl", "_with_tts.jsonl")
    print(f"\nSaving updated data to {output_data_path}...")
    with open(output_data_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Done! Processed {len(results)} samples.")


def parse_args():
    """Parse CLI arguments for TTS inference with CosyVoice.

    Returns:
        argparse.Namespace with CLI options.
    """
    parser = argparse.ArgumentParser(description="Generate TTS using CosyVoice for voice cloning")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the data jsonl",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/user_data/siqiouya/ckpts/pretrained/tts/Fun-CosyVoice3-0.5B",
        help="Path to CosyVoice model directory (default: /data/user_data/siqiouya/ckpts/pretrained/tts/Fun-CosyVoice3-0.5B)",
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
        help="Random seed for reproducibility (default: None)",
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
