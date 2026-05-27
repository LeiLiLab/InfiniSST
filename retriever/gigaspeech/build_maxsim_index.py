#!/usr/bin/env python3
"""
Build MaxSim text embedding index for inference.

Pre-encodes all glossary terms into a text embedding tensor using the
BgeM3TextEncoder from the MaxSim retriever. The output .pt file contains:
  - text_embs: [N, D] L2-normalized text embeddings
  - term_list: list of dicts with keys {key, term, target_translations}

This replaces the FAISS .pkl index used by the old sliding-window retriever.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

# ======Configuration=====
TEXT_MODEL_ID = "BAAI/bge-m3"
TEXT_LORA_RANK = 128
TEXT_LORA_ALPHA = 256
TEXT_POOLING = "cls"
SPARSE_WEIGHT = 0.7
TEXT_LORA_TARGET_MODULES = "query key value dense".split()
TEXT_ENCODE_BATCH = 256
# ======Configuration=====

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def build_text_encoder(device: torch.device, lora_rank: int, lora_alpha: int):
    """Build and return the BgeM3TextEncoder."""
    train_dir = _REPO_ROOT / "documents" / "code" / "train" / "term_train"
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    from qwen3_glossary_neg_train import BgeM3TextEncoder
    from transformers import AutoTokenizer

    text_encoder = BgeM3TextEncoder(
        model_id=TEXT_MODEL_ID,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=TEXT_LORA_TARGET_MODULES,
        full_finetune=False,
        sparse_weight=SPARSE_WEIGHT,
        text_pooling=TEXT_POOLING,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    return text_encoder, tokenizer


def load_text_checkpoint(
    text_encoder, model_path: str, device: torch.device
) -> None:
    """Load text encoder weights from a MaxSim checkpoint."""
    ckpt = torch.load(model_path, map_location=device)

    def _strip(sd):
        return {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }

    if "text_model_state_dict" in ckpt:
        text_encoder.load_state_dict(
            _strip(ckpt["text_model_state_dict"]), strict=False
        )
        _log("Loaded text_model_state_dict from checkpoint.")
    else:
        _log("WARNING: No text_model_state_dict found in checkpoint. Using base weights only.")

    text_encoder.eval()


def load_glossary(glossary_path: str) -> List[Dict]:
    """Load glossary JSON and return list of term entries.

    Supports two formats:
      - dict: {key: {term, target_translations, ...}}  (glossary_acl6060, per-paper)
      - list: [{term, target_translations, source, ...}, ...]  (acl_glossary_gs1000/gs10000)
    """
    with open(glossary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    term_list = []
    if isinstance(data, dict):
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            term = entry.get("term", key)
            target_translations = entry.get("target_translations", {})
            term_list.append({
                "key": key,
                "term": term,
                "target_translations": target_translations,
            })
    elif isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            term = entry.get("term", "")
            assert term, f"List-format glossary entry missing 'term': {entry}"
            target_translations = entry.get("target_translations", {})
            term_list.append({
                "key": term.lower(),
                "term": term,
                "target_translations": target_translations,
            })
    else:
        raise ValueError(
            f"Unsupported glossary format in {glossary_path}: "
            f"expected dict or list, got {type(data).__name__}"
        )

    assert len(term_list) > 0, f"No terms loaded from {glossary_path}"
    _log(f"Loaded {len(term_list)} terms from {glossary_path}")
    return term_list


@torch.no_grad()
def encode_terms(
    term_list: List[Dict],
    text_encoder,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Encode all terms into [N, D] text embeddings."""
    texts = [entry["term"] for entry in term_list]
    all_embs = []

    for start in range(0, len(texts), TEXT_ENCODE_BATCH):
        batch = texts[start : start + TEXT_ENCODE_BATCH]
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = text_encoder(tok.input_ids, tok.attention_mask)
        all_embs.append(embs.float())
        done = start + len(batch)
        if (start // TEXT_ENCODE_BATCH) % 10 == 0:
            _log(f"  encoded {done}/{len(texts)} text terms")

    return torch.cat(all_embs, dim=0)  # [N, D]


def main():
    parser = argparse.ArgumentParser(description="Build MaxSim text embedding index")
    parser.add_argument("--model-path", required=True, help="MaxSim retriever checkpoint (.pt)")
    parser.add_argument("--glossary-path", required=True, help="Glossary JSON file")
    parser.add_argument("--output-path", required=True, help="Output .pt index file")
    parser.add_argument("--device", default="cuda:0", help="Device for encoding")
    parser.add_argument("--text-lora-rank", type=int, default=TEXT_LORA_RANK)
    parser.add_argument("--text-lora-alpha", type=int, default=TEXT_LORA_ALPHA)
    args = parser.parse_args()

    assert Path(args.model_path).is_file(), f"Model not found: {args.model_path}"
    assert Path(args.glossary_path).is_file(), f"Glossary not found: {args.glossary_path}"

    device = torch.device(args.device)

    _log(f"Building MaxSim text index: glossary={args.glossary_path}")
    _log(f"Model: {args.model_path}")
    _log(f"Device: {device}, text_lora_rank={args.text_lora_rank}")

    text_encoder, tokenizer = build_text_encoder(
        device, args.text_lora_rank, args.text_lora_alpha
    )
    load_text_checkpoint(text_encoder, args.model_path, device)

    term_list = load_glossary(args.glossary_path)
    text_embs = encode_terms(term_list, text_encoder, tokenizer, device)

    _log(f"Text embeddings shape: {list(text_embs.shape)}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"text_embs": text_embs.cpu(), "term_list": term_list},
        str(output_path),
    )
    _log(f"Saved index to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
