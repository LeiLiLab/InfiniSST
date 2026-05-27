#!/usr/bin/env python3
"""
Pre-compute TTS prototype embeddings for the TTS bank and save as a .npz cache.

This avoids each shard re-encoding 170K+ wav files at runtime.
Output .npz contains:
  - "proto_term_keys": 1-D string array, term key per prototype   (shape: [N_protos])
  - "proto_embs":      2-D float32 array, L2-normalized embeddings (shape: [N_protos, D])

Usage:
    CUDA_VISIBLE_DEVICES=0 python precompute_tts_embeddings.py \
        --terms-npy      /path/to/terms.npy \
        --wav-dir        /path/to/wav/ \
        --model-path     /path/to/rag_model.pt \
        --glossary-json  /path/to/glossary.json \
        --output-npz     /path/to/tts_bank_cache.npz \
        --target-lang-code zh
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import librosa
import faiss

# ============================== Configuration ==============================
AUDIO_TARGET_LEN = 30720
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_PROTOTYPES_PER_TERM = 8
DEFAULT_SAMPLE_RATE = 16000
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
# ===========================================================================

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-compute TTS prototype embeddings.")
    p.add_argument("--terms-npy", required=True, help="Path to terms.npy (1-D string array)")
    p.add_argument("--wav-dir", required=True, help="Directory with {idx+1}.wav symlinks")
    p.add_argument("--model-path", required=True, help="Path to RAG model checkpoint (.pt)")
    p.add_argument("--glossary-json", required=True, help="Training glossary JSON")
    p.add_argument("--output-npz", required=True, help="Output .npz cache path")
    p.add_argument("--target-lang-code", default="zh")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-prototypes-per-term", type=int, default=DEFAULT_MAX_PROTOTYPES_PER_TERM)
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    p.add_argument(
        "--base-model-name", default="Atotti/Qwen3-Omni-AudioTransformer",
        help="HuggingFace model name for the audio encoder backbone",
    )
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--text-lora-r", type=int, default=16)
    return p.parse_args()


def load_glossary_keys(path: str) -> set:
    with open(path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
    return {k.strip().lower() for k in glossary.keys()}


def build_model(args):
    """Build the exact same model as StreamingQwen3RAGRetrieverV4 uses."""
    from transformers import WhisperFeatureExtractor
    from agents.streaming_qwen3_rag_retriever_v4 import Qwen3OmniRetriever

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Building Qwen3OmniRetriever (lora_r=%d, lora_alpha=%d) ...", args.lora_r, args.lora_alpha)

    model = Qwen3OmniRetriever(
        model_id=args.base_model_name,
        target_dim=1024,
        use_lora=True,
        lora_rank=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    logger.info("Loading tuned weights from %s ...", args.model_path)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_state_dict = checkpoint["model_state_dict"]
    else:
        raw_state_dict = checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}
    load_info = model.load_state_dict(state_dict, strict=False)
    logger.info("Model load info: missing=%d unexpected=%d", len(load_info.missing_keys), len(load_info.unexpected_keys))
    model = model.to(device).eval()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model_name)
    return model, feature_extractor, device


def collect_term_wav_paths(
    terms_npy_path: str,
    wav_dir: str,
    glossary_keys: set,
    max_per_term: int,
) -> Dict[str, List[str]]:
    terms_array = np.load(terms_npy_path, allow_pickle=True)
    term_to_paths: Dict[str, List[str]] = {}
    for idx in range(len(terms_array)):
        term_key = str(terms_array[idx]).strip().lower()
        if not term_key or term_key not in glossary_keys:
            continue
        wav_path = os.path.join(wav_dir, f"{idx + 1}.wav")
        if not os.path.isfile(wav_path):
            continue
        paths = term_to_paths.setdefault(term_key, [])
        if len(paths) < max_per_term:
            paths.append(wav_path)
    return term_to_paths


def encode_all_prototypes(
    term_to_paths: Dict[str, List[str]],
    model,
    feature_extractor,
    device,
    batch_size: int,
    sample_rate: int,
) -> tuple:
    """Encode all prototypes and return per-prototype embeddings with term owners.

    Returns:
        prototype_term_keys: 1-D string array of term keys (one per prototype)
        prototype_embs: 2-D float32 array of L2-normalized embeddings (one per prototype)
    """
    prototype_paths = []
    prototype_owner = []
    for term_key in sorted(term_to_paths.keys()):
        for p in term_to_paths[term_key]:
            prototype_paths.append(p)
            prototype_owner.append(term_key)

    total = len(prototype_paths)
    logger.info("Encoding %d prototypes across %d terms ...", total, len(term_to_paths))

    all_embs = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        audios = []
        for p in prototype_paths[start:end]:
            wav, _ = librosa.load(p, sr=sample_rate, mono=True)
            wav = np.asarray(wav, dtype=np.float32).flatten()
            max_val = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
            if max_val > 0:
                wav = wav / max_val
            if len(wav) < AUDIO_TARGET_LEN:
                wav = np.pad(wav, (0, AUDIO_TARGET_LEN - len(wav)), mode="constant")
            elif len(wav) > AUDIO_TARGET_LEN:
                wav = wav[:AUDIO_TARGET_LEN]
            audios.append(wav)

        inputs = feature_extractor(audios, sampling_rate=sample_rate, return_tensors="pt", padding=False)
        features = inputs.input_features
        bs, channels, mel_len = features.shape
        input_features = features.transpose(0, 1).reshape(channels, -1).to(device).to(torch.bfloat16)
        feature_lens = torch.full((bs,), mel_len, dtype=torch.long, device=device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb = model(input_features, feature_lens)
            emb = emb.detach().cpu().float().numpy()
        faiss.normalize_L2(emb)
        all_embs.append(emb.astype(np.float32, copy=False))

        done = min(end, total)
        if done % (batch_size * 50) < batch_size or done == total:
            logger.info("  encoded %d / %d prototypes (%.1f%%)", done, total, done / total * 100)

    all_embs = np.concatenate(all_embs, axis=0)
    prototype_term_keys = np.array(prototype_owner, dtype=str)

    return prototype_term_keys, all_embs


def main():
    args = parse_args()

    glossary_keys = load_glossary_keys(args.glossary_json)
    logger.info("Glossary: %d term keys", len(glossary_keys))

    term_to_paths = collect_term_wav_paths(
        args.terms_npy, args.wav_dir, glossary_keys, args.max_prototypes_per_term,
    )
    logger.info("Matched %d terms with wav files", len(term_to_paths))

    model, feature_extractor, device = build_model(args)

    proto_term_keys, proto_embs = encode_all_prototypes(
        term_to_paths, model, feature_extractor, device, args.batch_size, args.sample_rate,
    )

    unique_terms = len(set(proto_term_keys.tolist()))
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez(
        args.output_npz,
        proto_term_keys=proto_term_keys,
        proto_embs=proto_embs,
    )
    logger.info(
        "Saved cache: %s  (unique_terms=%d, protos=%d, emb_shape=%s)",
        args.output_npz, unique_terms, proto_embs.shape[0], proto_embs.shape,
    )


if __name__ == "__main__":
    main()
