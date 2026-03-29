"""TTS generation using CosyVoice for evaluation terms.

Generates TTS audio for terms from acl_terms.npy using CosyVoice zero-shot inference
with a fixed reference audio.

python preprocess/rag_tts_for_eval.py \
    --terms-path /data/group_data/li_lab/siqiouya/datasets/acl_6060/acl_terms.npy \
    --output-dir /data/group_data/li_lab/siqiouya/datasets/acl_6060/acl_terms \
    --overwrite
"""

import os
import sys
import json
import argparse

# Add CosyVoice to path
sys.path.append('/home/siqiouya/code/CosyVoice')
sys.path.append('/home/siqiouya/code/CosyVoice/third_party/Matcha-TTS')

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed


def load_terms(terms_path: str) -> list[str]:
    """Load terms from .npy file.
    
    Args:
        terms_path: Path to the .npy file containing terms
        
    Returns:
        List of terms
    """
    terms = np.load(terms_path, allow_pickle=True)
    # Convert to list if it's an array
    if isinstance(terms, np.ndarray):
        terms = terms.tolist()
    return terms


def main(args):
    """Run TTS generation using CosyVoice.

    Args:
        args: Parsed CLI args from parse_args().
    """
    
    # Load terms
    print(f"Loading terms from {args.terms_path}...")
    terms = load_terms(args.terms_path)
    print(f"Loaded {len(terms)} terms")
    
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Create resampler if needed
    resampler = None
    if cosyvoice.sample_rate != args.sampling_rate:
        print(f"Initializing resampler: {cosyvoice.sample_rate} Hz -> {args.sampling_rate} Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=cosyvoice.sample_rate,
            new_freq=args.sampling_rate
        )
    
    # Process each term
    successful = 0
    failed = 0
    term_to_path = {}  # Mapping from term to audio path
    
    for idx, term in enumerate(tqdm(terms, desc="Generating TTS")):
        # Skip empty terms
        if not term or not term.strip():
            print(f"Skipping empty term at index {idx}")
            failed += 1
            continue
            
        # Create output filename with numeric index (1-indexed)
        audio_filename = f"{idx + 1}.wav"
        tts_audio_path = os.path.join(args.output_dir, audio_filename)
        
        # Skip if file already exists
        if os.path.exists(tts_audio_path) and not args.overwrite:
            print(f"Skipping existing file: {tts_audio_path}")
            successful += 1
            term_to_path[term] = tts_audio_path
            continue
        
        # Set random seed for reproducibility
        if args.seed is not None:
            set_all_random_seed(args.seed + idx)
        
        # Generate TTS audio using CosyVoice zero-shot
        try:
            # inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False)
            audio_generated = False
            for i, output in enumerate(cosyvoice.inference_zero_shot(
                term,           # tts_text: the term to synthesize
                'You are a helpful assistant.<|endofprompt|>' + args.ref_text,  # prompt_text: reference text
                args.ref_audio,      # prompt_wav: reference audio path
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
                successful += 1
                # Add to mapping
                term_to_path[term] = tts_audio_path
                break  # Only take the first output
            
            if not audio_generated:
                print(f"Warning: No audio generated for term: '{term}'")
                failed += 1
                continue
                
        except Exception as e:
            print(f"Error generating TTS for term '{term}': {e}")
            failed += 1
            continue
    
    # Save term-to-path mapping
    mapping_path = os.path.join(args.output_dir, "term_to_path.json")
    print(f"\nSaving term-to-path mapping to {mapping_path}...")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(term_to_path, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone! Successfully processed {successful}/{len(terms)} terms.")
    if failed > 0:
        print(f"Failed to process {failed} terms.")
    print(f"Term-to-path mapping saved with {len(term_to_path)} entries.")


def parse_args():
    """Parse CLI arguments for TTS inference with CosyVoice.

    Returns:
        argparse.Namespace with CLI options.
    """
    parser = argparse.ArgumentParser(description="Generate TTS for evaluation terms using CosyVoice")

    parser.add_argument(
        "--terms-path",
        type=str,
        default="/data/group_data/li_lab/siqiouya/datasets/acl_6060/acl_terms.npy",
        help="Path to the .npy file containing terms (default: /data/group_data/li_lab/siqiouya/datasets/acl_6060/acl_terms.npy)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/group_data/li_lab/siqiouya/datasets/acl_6060/acl_terms",
        help="Directory to save generated audio files (default: /data/group_data/li_lab/siqiouya/datasets/acl_6060/acl_terms)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/user_data/siqiouya/ckpts/pretrained/tts/Fun-CosyVoice3-0.5B",
        help="Path to CosyVoice model directory (default: /data/user_data/siqiouya/ckpts/pretrained/tts/Fun-CosyVoice3-0.5B)",
    )

    parser.add_argument(
        "--ref-audio",
        type=str,
        default="/home/siqiouya/code/Expressive-S2S/data/stresstest/ground_truth/audio_0.wav",
        help="Path to reference audio file for voice cloning (default: /home/siqiouya/code/Expressive-S2S/data/stresstest/ground_truth/audio_0.wav)",
    )

    parser.add_argument(
        "--ref-text",
        type=str,
        default="I didn't say he stole the money.",
        help="Reference text matching the reference audio (default: 'I didn't say he stole the money.')",
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

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing audio files (default: False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
