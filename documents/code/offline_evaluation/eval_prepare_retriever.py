#!/usr/bin/env python3
"""
Prepare retriever results for TCR/FCR evaluation.

Handles two dataset formats:
  - dev:  JSONL with {messages, audios} (multi-turn conversations)
  - acl:  JSONL with {term, chunk_audio_path, utter_id, chunk_idx, ...}

Output: unified eval JSONL, one line per chunk:
  {
    "chunk_id": str,
    "audio_path": str,
    "gt_terms": [{"term": "...", "zh": "..."}],   # known GT terms
    "gt_translation": str,                          # GT translation (dev only)
    "retriever_top10": [{"term": "...", "zh": "...", "score": float}],
  }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ======Configuration=====
AUDIO_MODEL_ID = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_MODEL_ID = "BAAI/bge-m3"
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SAMPLES = 30720  # 1.92s * 16kHz

TEXT_ENCODE_BATCH = 256
AUDIO_ENCODE_BATCH = 32
TOP_K = 10

LORA_RANK = 128
LORA_ALPHA = 256
POOLING_TYPE = "transformer"
TEMPERATURE = 0.03
USE_MAXSIM = True
MAXSIM_WINDOWS = [6, 10, 16, 24]
MAXSIM_STRIDE = 2
TARGET_DIM = 1024

TEXT_LORA_RANK = 128
TEXT_LORA_ALPHA = 256
TEXT_POOLING = "cls"
SPARSE_WEIGHT = 0.7

LORA_TARGET_MODULES = "q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2".split()
TEXT_LORA_TARGET_MODULES = "query key value dense".split()

MODEL_PATH = "/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"
# ======Configuration=====


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def build_model(device: torch.device):
    sys.path.insert(0, str(_REPO_ROOT / "documents" / "code" / "train" / "term_train"))
    from qwen3_glossary_neg_train import (
        BgeM3TextEncoder,
        Qwen3OmniRetriever,
        _maxsim_score,
    )

    retriever = Qwen3OmniRetriever(
        model_id=AUDIO_MODEL_ID,
        target_dim=TARGET_DIM,
        use_lora=True,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_target_modules=LORA_TARGET_MODULES,
        temperature=TEMPERATURE,
        learn_temp=False,
        pooling_type=POOLING_TYPE,
        use_maxsim=USE_MAXSIM,
        maxsim_windows=MAXSIM_WINDOWS,
        maxsim_stride=MAXSIM_STRIDE,
    ).to(device)

    text_encoder = BgeM3TextEncoder(
        model_id=TEXT_MODEL_ID,
        lora_rank=TEXT_LORA_RANK,
        lora_alpha=TEXT_LORA_ALPHA,
        target_modules=TEXT_LORA_TARGET_MODULES,
        full_finetune=False,
        sparse_weight=SPARSE_WEIGHT,
        text_pooling=TEXT_POOLING,
    ).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)

    def _strip(sd):
        return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    retriever.load_state_dict(_strip(ckpt.get("model_state_dict", {})), strict=False)
    if "text_model_state_dict" in ckpt:
        text_encoder.load_state_dict(_strip(ckpt["text_model_state_dict"]), strict=False)

    retriever.eval()
    text_encoder.eval()
    return retriever, text_encoder, _maxsim_score


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path)
    assert sr == EXPECTED_SAMPLE_RATE, f"Unexpected SR {sr} for {path}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    mx = float(np.max(np.abs(audio))) if audio.size else 0.0
    if mx > 0:
        audio = audio / mx
    if audio.shape[0] < EXPECTED_CHUNK_SAMPLES:
        audio = np.pad(audio, (0, EXPECTED_CHUNK_SAMPLES - audio.shape[0]))
    elif audio.shape[0] > EXPECTED_CHUNK_SAMPLES:
        audio = audio[:EXPECTED_CHUNK_SAMPLES]
    return audio


@torch.no_grad()
def encode_glossary(terms, text_encoder, tokenizer, device):
    all_embs = []
    for start in range(0, len(terms), TEXT_ENCODE_BATCH):
        batch = terms[start:start + TEXT_ENCODE_BATCH]
        tok = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = text_encoder(tok.input_ids, tok.attention_mask)
        all_embs.append(embs.float())
    return F.normalize(torch.cat(all_embs, dim=0), p=2, dim=-1)


@torch.no_grad()
def encode_audio_batch(audio_arrays, retriever, feat_ext, device):
    inputs = feat_ext(audio_arrays, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False)
    features = inputs.input_features
    B, C, T_mel = features.shape
    input_features = features.transpose(0, 1).reshape(C, -1).to(device).to(torch.bfloat16)
    feature_lens = torch.full((B,), T_mel, dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        embs = retriever(input_features, feature_lens)
    return embs.float()


def retrieve_topk(speech_emb, text_embs, _maxsim_score, k=TOP_K):
    if speech_emb.ndim == 3:
        sim = _maxsim_score(speech_emb, text_embs)
    else:
        sim = speech_emb @ text_embs.T
    sim_np = sim.cpu().numpy().squeeze(0)
    n = min(k, sim_np.shape[0])
    top_idx = np.argpartition(-sim_np, n)[:n]
    top_sco = sim_np[top_idx]
    order = np.argsort(-top_sco)
    return top_idx[order], top_sco[order]


# ---------------------------------------------------------------------------
# Dataset parsers
# ---------------------------------------------------------------------------

def parse_dev_dataset(jsonl_path: str) -> List[Dict]:
    """Parse dev JSONL into per-chunk records."""
    chunks = []
    with open(jsonl_path) as f:
        for conv_idx, line in enumerate(f):
            d = json.loads(line.strip())
            audios = d.get("audios", [])
            messages = d.get("messages", [])

            user_msgs = [m for m in messages if m["role"] == "user"]
            asst_msgs = [m for m in messages if m["role"] == "assistant"]

            for ci, (umsg, amsg) in enumerate(zip(user_msgs, asst_msgs)):
                audio_path = audios[ci] if ci < len(audios) else ""
                gt_translation = amsg["content"]
                user_content = umsg["content"]

                existing_terms = []
                if "term_map:" in user_content:
                    for tline in user_content.split("\n"):
                        tline = tline.strip()
                        if "=" in tline and not tline.startswith("term_map"):
                            parts = tline.split("=", 1)
                            en_term = parts[0].strip()
                            zh_term = parts[1].strip() if len(parts) > 1 else ""
                            if en_term and zh_term:
                                existing_terms.append({"term": en_term, "zh": zh_term})

                gt_terms = []
                for t in existing_terms:
                    if t["zh"] in gt_translation:
                        gt_terms.append(t)

                chunks.append({
                    "chunk_id": f"dev_{conv_idx}::{ci}",
                    "audio_path": audio_path,
                    "gt_terms": gt_terms,
                    "gt_translation": gt_translation,
                })
    return chunks


def parse_acl_dataset(jsonl_path: str, glossary_zh_map: Dict[str, str],
                      gt_zh_map: Optional[Dict[str, str]] = None) -> List[Dict]:
    """Parse ACL JSONL into per-chunk records, merging multi-term lines.
    
    gt_zh_map: separate EN→ZH mapping for GT terms (from per-paper glossary).
               Falls back to glossary_zh_map if not provided.
    """
    effective_zh = gt_zh_map if gt_zh_map else glossary_zh_map
    chunk_groups: Dict[str, Dict] = {}
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line.strip())
            cid = f"{d['utter_id']}::{d['chunk_idx']}"
            if cid not in chunk_groups:
                chunk_groups[cid] = {
                    "chunk_id": cid,
                    "audio_path": d["chunk_audio_path"],
                    "gt_terms": [],
                    "gt_translation": "",
                }
            term = (d.get("term") or "").strip()
            if term:
                zh = effective_zh.get(term.lower(), "")
                if zh:
                    chunk_groups[cid]["gt_terms"].append({"term": term, "zh": zh})

    chunks = sorted(chunk_groups.values(), key=lambda x: x["chunk_id"])
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["dev", "acl"], required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--glossary_json", required=True,
                        help="For dev: 44K GT glossary; for ACL: wiki glossary")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--glossary_size", type=int, default=0,
                        help="ACL only: limit glossary to N terms (0=all)")
    parser.add_argument("--acl_gt_glossary", type=str, default="",
                        help="ACL only: per-paper glossary JSON with zh translations for GT terms")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load glossary ---
    _log(f"Loading glossary from {args.glossary_json}")
    with open(args.glossary_json) as f:
        raw_glossary = json.load(f)

    if isinstance(raw_glossary, dict):
        glossary_items = []
        for k, v in raw_glossary.items():
            if isinstance(v, dict):
                term = v.get("term", k)
                zh = (v.get("translation", "")
                      or v.get("zh", "")
                      or v.get("target_translations", {}).get("zh", ""))
            else:
                term = k
                zh = str(v)
            glossary_items.append((term, zh))
    elif isinstance(raw_glossary, list):
        glossary_items = []
        for item in raw_glossary:
            term = item.get("term", "")
            zh = (item.get("translation", "")
                  or item.get("zh", "")
                  or item.get("target_translations", {}).get("zh", ""))
            glossary_items.append((term, zh))
    else:
        raise ValueError(f"Unexpected glossary format: {type(raw_glossary)}")

    if args.glossary_size > 0:
        glossary_items = glossary_items[:args.glossary_size]

    term_list = [t for t, _ in glossary_items]
    zh_list = [z for _, z in glossary_items]
    glossary_zh_map = {t.lower(): z for t, z in glossary_items}
    _log(f"Glossary: {len(term_list)} terms")

    # --- Parse dataset ---
    _log(f"Parsing {args.dataset_type} dataset: {args.dataset_path}")
    if args.dataset_type == "dev":
        chunks = parse_dev_dataset(args.dataset_path)
    else:
        gt_zh_map = None
        if args.acl_gt_glossary:
            _log(f"Loading ACL GT glossary from {args.acl_gt_glossary}")
            with open(args.acl_gt_glossary) as f:
                acl_gt = json.load(f)
            gt_zh_map = {}
            for key, val in acl_gt.items():
                zh = val.get("zh", "")
                if zh:
                    gt_zh_map[key] = zh
            _log(f"ACL GT glossary: {len(gt_zh_map)} terms with zh")
        chunks = parse_acl_dataset(args.dataset_path, glossary_zh_map, gt_zh_map)

    has_term_chunks = sum(1 for c in chunks if c["gt_terms"])
    _log(f"Chunks: {len(chunks)} total, {has_term_chunks} with GT terms")

    # --- Build model ---
    _log("Building retriever model...")
    retriever, text_encoder, _maxsim_score = build_model(device)
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    feat_ext = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    # --- Encode glossary ---
    _log("Encoding glossary...")
    text_embs = encode_glossary(term_list, text_encoder, tokenizer, device)
    _log(f"Text embeddings: {text_embs.shape}")

    # --- Encode audio and retrieve ---
    _log("Running retriever inference...")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    t_start = time.time()
    n_done = 0
    n_failed = 0

    audio_batch = []
    batch_indices = []

    with open(args.output_path, "w") as f_out:
        for i, chunk in enumerate(chunks):
            apath = chunk["audio_path"]
            if not os.path.isfile(apath):
                n_failed += 1
                chunk["retriever_top10"] = []
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                continue

            try:
                audio_arr = load_audio(apath)
            except Exception as e:
                _log(f"  WARN: failed to load {apath}: {e}")
                n_failed += 1
                chunk["retriever_top10"] = []
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                continue

            audio_batch.append(audio_arr)
            batch_indices.append(i)

            if len(audio_batch) >= AUDIO_ENCODE_BATCH:
                embs = encode_audio_batch(audio_batch, retriever, feat_ext, device)
                for bi, ci in enumerate(batch_indices):
                    e = embs[bi:bi+1]
                    idx_arr, sco_arr = retrieve_topk(e, text_embs, _maxsim_score)
                    results = []
                    for ti, sc in zip(idx_arr, sco_arr):
                        results.append({"term": term_list[ti], "zh": zh_list[ti], "score": round(float(sc), 6)})
                    chunks[ci]["retriever_top10"] = results
                    f_out.write(json.dumps(chunks[ci], ensure_ascii=False) + "\n")
                n_done += len(audio_batch)
                audio_batch.clear()
                batch_indices.clear()

                if n_done % 500 == 0:
                    _log(f"  {n_done}/{len(chunks)} chunks, {time.time()-t_start:.0f}s")

        if audio_batch:
            embs = encode_audio_batch(audio_batch, retriever, feat_ext, device)
            for bi, ci in enumerate(batch_indices):
                e = embs[bi:bi+1]
                idx_arr, sco_arr = retrieve_topk(e, text_embs, _maxsim_score)
                results = []
                for ti, sc in zip(idx_arr, sco_arr):
                    results.append({"term": term_list[ti], "zh": zh_list[ti], "score": round(float(sc), 6)})
                chunks[ci]["retriever_top10"] = results
                f_out.write(json.dumps(chunks[ci], ensure_ascii=False) + "\n")
            n_done += len(audio_batch)

    elapsed = time.time() - t_start
    _log(f"Done: {n_done} chunks encoded, {n_failed} failed, {elapsed:.0f}s")
    _log(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
