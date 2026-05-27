#!/usr/bin/env python3
"""
Benchmark MaxSim window scaling: cost vs quality ablation.

For each window config, measures:
  - encode_ms:  audio encoder forward + multi-scale pooling (per chunk)
  - score_ms:   _maxsim_score against text bank (per chunk)
  - dev recall@10 (GT-only bank)
  - ACL recall@10 at gs1000 and gs10000

Uses the EXISTING trained checkpoint — no retraining required.
We monkey-patch retriever.maxsim_windows before each config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from time import perf_counter

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from documents.code.train.term_train.qwen3_glossary_neg_train import (
    BgeM3TextEncoder,
    Qwen3OmniRetriever,
    _maxsim_score,
    DEFAULT_TEXT_MAX_LENGTH,
    MAXSIM_DEFAULT_STRIDE,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor

# ======Configuration=====
MODEL_PATH = (
    "/mnt/taurus/data/jiaxuanluo/train_outputs/"
    "q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"
)
AUDIO_MODEL_ID = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_MODEL_ID = "BAAI/bge-m3"

DEV_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_DEV_JSONL = (
    "/mnt/gemini/data2/jiaxuanluo/"
    "acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
)
WIKI_GLOSSARY = (
    "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/"
    "data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
)

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHUNK_SAMPLES = 30720
LORA_RANK = 128
LORA_ALPHA = 256
TARGET_DIM = 1024
TEMPERATURE = 0.03
TEXT_LORA_RANK = 128
TEXT_LORA_ALPHA = 256
TEXT_POOLING = "cls"
SPARSE_WEIGHT = 0.7
LORA_TARGET_MODULES = "q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2".split()
TEXT_LORA_TARGET_MODULES = "query key value dense".split()

EVAL_TOPK = 10
ENCODE_BATCH = 64
TEXT_ENCODE_BATCH = 256
WARMUP_RUNS = 3
TIMING_RUNS = 5
GLOSSARY_SIZES = [1000, 10000]

WINDOW_CONFIGS: List[Tuple[str, List[int]]] = [
    ("min_W1", [24]),
    ("2scale_W11", [6, 24]),
    ("3scale_W18", [6, 12, 24]),
    ("current_W24", [6, 10, 16, 24]),
    ("5scale_W34", [6, 8, 12, 18, 24]),
    ("7scale_W43", [6, 8, 10, 12, 16, 20, 24]),
    ("full_W100", list(range(6, 25))),
]
# ======Configuration=====


def _log(msg: str) -> None:
    print(f"[BENCH] {msg}", flush=True)


def count_windows(windows: List[int], T: int = 24, stride: int = 2) -> int:
    total = 0
    for w in windows:
        if w >= T:
            total += 1
        else:
            total += (T - w) // stride + 1
    return total


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


def build_model(device: torch.device):
    retriever = Qwen3OmniRetriever(
        model_id=AUDIO_MODEL_ID,
        target_dim=TARGET_DIM,
        use_lora=True,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_target_modules=LORA_TARGET_MODULES,
        temperature=TEMPERATURE,
        learn_temp=False,
        pooling_type="transformer",
        use_maxsim=True,
        maxsim_windows=[6, 10, 16, 24],
        maxsim_stride=MAXSIM_DEFAULT_STRIDE,
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
    return retriever, text_encoder


def load_dev_samples(jsonl_path: str, max_samples: int = 0) -> List[Dict]:
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line.strip())
            t = (d.get("term_key") or d.get("term") or "").strip().lower()
            audio_path = d.get("chunk_audio_path", "")
            if t and audio_path and os.path.isfile(audio_path):
                samples.append(d)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def load_wiki_terms(path: str) -> List[str]:
    raw = json.load(open(path))
    if isinstance(raw, list):
        return [e.get("term", "").strip().lower() for e in raw if e.get("term")]
    elif isinstance(raw, dict):
        return [v.get("term", k).strip().lower() for k, v in raw.items()]
    raise ValueError(f"Unexpected glossary format: {type(raw)}")


@torch.no_grad()
def encode_texts(terms: List[str], text_encoder, tokenizer, device) -> torch.Tensor:
    all_embs = []
    for start in range(0, len(terms), TEXT_ENCODE_BATCH):
        batch = terms[start:start + TEXT_ENCODE_BATCH]
        tok = tokenizer(batch, padding=True, truncation=True,
                        max_length=DEFAULT_TEXT_MAX_LENGTH, return_tensors="pt").to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = text_encoder(tok.input_ids, tok.attention_mask)
        all_embs.append(embs.float())
    return F.normalize(torch.cat(all_embs, dim=0), p=2, dim=-1)


@torch.no_grad()
def encode_audio_batch(audio_arrays: List[np.ndarray], retriever, feat_ext, device) -> torch.Tensor:
    inputs = feat_ext(audio_arrays, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False)
    features = inputs.input_features
    B, C, T_mel = features.shape
    input_features = features.transpose(0, 1).reshape(C, -1).to(device).to(torch.bfloat16)
    feature_lens = torch.full((B,), T_mel, dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        embs = retriever(input_features, feature_lens)
    return embs.float()


def compute_recall(speech_embs: torch.Tensor, bank_embs: torch.Tensor,
                   targets: torch.Tensor, k: int = EVAL_TOPK) -> float:
    if speech_embs.ndim == 3:
        logits = _maxsim_score(speech_embs, bank_embs)
    else:
        logits = speech_embs @ bank_embs.T
    targets = targets.to(logits.device)
    k_eff = min(k, logits.size(1))
    return (
        torch.topk(logits, k=k_eff, dim=1)
        .indices.eq(targets.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
    )


def benchmark_encode(
    retriever, feat_ext, audio_arrays: List[np.ndarray], device: torch.device,
    windows: List[int], n_warmup: int, n_runs: int,
) -> float:
    retriever.maxsim_windows = windows
    inputs = feat_ext(audio_arrays, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt", padding=False)
    features = inputs.input_features
    B, C, T_mel = features.shape
    input_features = features.transpose(0, 1).reshape(C, -1).to(device).to(torch.bfloat16)
    feature_lens = torch.full((B,), T_mel, dtype=torch.long, device=device)

    for _ in range(n_warmup):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = retriever(input_features, feature_lens)
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = retriever(input_features, feature_lens)
        torch.cuda.synchronize()
        times.append(perf_counter() - t0)

    mean_ms = np.mean(times) * 1000.0 / B
    return mean_ms


def benchmark_score(
    speech_embs: torch.Tensor, text_bank: torch.Tensor,
    n_warmup: int, n_runs: int,
) -> float:
    B = speech_embs.shape[0]
    for _ in range(n_warmup):
        _ = _maxsim_score(speech_embs, text_bank)
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = perf_counter()
        _ = _maxsim_score(speech_embs, text_bank)
        torch.cuda.synchronize()
        times.append(perf_counter() - t0)

    mean_ms = np.mean(times) * 1000.0 / B
    return mean_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="window_ablation_results.tsv")
    parser.add_argument("--max_dev_samples", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    _log(f"Device: {device}")

    # --- Build model ---
    _log("Building model...")
    retriever, text_encoder = build_model(device)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    feat_ext = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

    # --- Load dev data ---
    _log(f"Loading dev data: {DEV_JSONL}")
    dev_samples = load_dev_samples(DEV_JSONL, args.max_dev_samples)
    _log(f"  dev samples with term: {len(dev_samples)}")

    _log(f"Loading ACL dev data: {ACL_DEV_JSONL}")
    acl_samples = load_dev_samples(ACL_DEV_JSONL, args.max_dev_samples)
    _log(f"  ACL samples with term: {len(acl_samples)}")

    # --- Load wiki terms ---
    _log(f"Loading wiki glossary: {WIKI_GLOSSARY}")
    wiki_terms = load_wiki_terms(WIKI_GLOSSARY)
    _log(f"  wiki terms: {len(wiki_terms)}")

    # --- Encode text for dev (GT bank) ---
    _log("Encoding dev text bank...")
    dev_term_to_bank: Dict[str, int] = {}
    dev_term_list: List[str] = []
    for s in dev_samples:
        t = (s.get("term_key") or "").strip().lower()
        if t and t not in dev_term_to_bank:
            dev_term_to_bank[t] = len(dev_term_list)
            dev_term_list.append(t)
    dev_bank_embs = encode_texts(dev_term_list, text_encoder, tokenizer, device)
    _log(f"  dev GT bank: {dev_bank_embs.shape[0]} unique terms")

    dev_targets = torch.tensor(
        [dev_term_to_bank[(s.get("term_key") or "").strip().lower()] for s in dev_samples],
        dtype=torch.long,
    )

    # --- Encode text for ACL (GT bank + wiki expansion) ---
    _log("Encoding ACL text bank...")
    acl_term_to_bank: Dict[str, int] = {}
    acl_term_list: List[str] = []
    for s in acl_samples:
        t = (s.get("term_key") or s.get("term") or "").strip().lower()
        if t and t not in acl_term_to_bank:
            acl_term_to_bank[t] = len(acl_term_list)
            acl_term_list.append(t)
    acl_gt_bank_embs = encode_texts(acl_term_list, text_encoder, tokenizer, device)
    _log(f"  ACL GT bank: {acl_gt_bank_embs.shape[0]} unique terms")

    acl_targets = torch.tensor(
        [acl_term_to_bank[(s.get("term_key") or s.get("term") or "").strip().lower()] for s in acl_samples],
        dtype=torch.long,
    )

    acl_gt_set = set(acl_term_list)
    wiki_filtered = [t for t in wiki_terms if t not in acl_gt_set]
    _log(f"  Wiki filtered (excl ACL GT): {len(wiki_filtered)}")
    wiki_embs = encode_texts(wiki_filtered[:max(GLOSSARY_SIZES)], text_encoder, tokenizer, device)

    acl_expanded_banks: Dict[int, torch.Tensor] = {}
    for gs in GLOSSARY_SIZES:
        n_extra = gs - len(acl_term_list)
        if n_extra <= 0:
            acl_expanded_banks[gs] = acl_gt_bank_embs
        else:
            n_add = min(n_extra, wiki_embs.shape[0])
            acl_expanded_banks[gs] = torch.cat([acl_gt_bank_embs, wiki_embs[:n_add]], dim=0)
        _log(f"  ACL bank gs{gs}: {acl_expanded_banks[gs].shape[0]} terms")

    # --- Load audio arrays ---
    _log("Loading dev audio...")
    dev_audios = [load_audio(s["chunk_audio_path"]) for s in dev_samples]
    _log(f"  dev audio chunks: {len(dev_audios)}")

    _log("Loading ACL audio...")
    acl_audios = [load_audio(s["chunk_audio_path"]) for s in acl_samples]
    _log(f"  ACL audio chunks: {len(acl_audios)}")

    timing_batch_audios = dev_audios[:ENCODE_BATCH]
    _log(f"  Timing batch size: {len(timing_batch_audios)}")

    # --- Run ablation ---
    results = []
    for label, windows in WINDOW_CONFIGS:
        W = count_windows(windows)
        _log(f"\n{'='*60}")
        _log(f"Config: {label} | windows={windows} | W={W}")
        _log(f"{'='*60}")

        retriever.maxsim_windows = windows

        # -- Encode timing --
        encode_ms = benchmark_encode(
            retriever, feat_ext, timing_batch_audios, device,
            windows, WARMUP_RUNS, TIMING_RUNS,
        )
        _log(f"  encode: {encode_ms:.2f} ms/chunk")

        # -- Encode all dev speech --
        _log("  Encoding dev speech...")
        dev_speech_list = []
        for start in range(0, len(dev_audios), ENCODE_BATCH):
            batch = dev_audios[start:start + ENCODE_BATCH]
            embs = encode_audio_batch(batch, retriever, feat_ext, device)
            dev_speech_list.append(embs)
        dev_speech_embs = torch.cat(dev_speech_list, dim=0)

        # -- Encode all ACL speech --
        _log("  Encoding ACL speech...")
        acl_speech_list = []
        for start in range(0, len(acl_audios), ENCODE_BATCH):
            batch = acl_audios[start:start + ENCODE_BATCH]
            embs = encode_audio_batch(batch, retriever, feat_ext, device)
            acl_speech_list.append(embs)
        acl_speech_embs = torch.cat(acl_speech_list, dim=0)

        # -- Score timing (dev speech vs gs1000/gs10000 bank) --
        timing_speech = dev_speech_embs[:ENCODE_BATCH].to(device)
        score_ms_gs1k = benchmark_score(
            timing_speech, acl_expanded_banks[1000].to(device),
            WARMUP_RUNS, TIMING_RUNS,
        )
        score_ms_gs10k = benchmark_score(
            timing_speech, acl_expanded_banks[10000].to(device),
            WARMUP_RUNS, TIMING_RUNS,
        )
        _log(f"  score gs1k:  {score_ms_gs1k:.2f} ms/chunk")
        _log(f"  score gs10k: {score_ms_gs10k:.2f} ms/chunk")

        # -- Dev recall@10 (GT-only bank) --
        dev_recall = compute_recall(dev_speech_embs, dev_bank_embs, dev_targets)
        _log(f"  dev recall@10: {dev_recall:.4f}")

        # -- ACL recall@10 at different glossary sizes --
        acl_recalls = {}
        for gs in GLOSSARY_SIZES:
            r = compute_recall(acl_speech_embs, acl_expanded_banks[gs], acl_targets)
            acl_recalls[gs] = r
            _log(f"  ACL recall@10 gs{gs}: {r:.4f}")

        results.append({
            "label": label,
            "windows": str(windows),
            "W": W,
            "encode_ms": round(encode_ms, 2),
            "score_ms_gs1k": round(score_ms_gs1k, 2),
            "score_ms_gs10k": round(score_ms_gs10k, 2),
            "total_ms_gs1k": round(encode_ms + score_ms_gs1k, 2),
            "total_ms_gs10k": round(encode_ms + score_ms_gs10k, 2),
            "dev_recall10": round(dev_recall, 4),
            **{f"acl_recall10_gs{gs}": round(acl_recalls[gs], 4) for gs in GLOSSARY_SIZES},
        })

    # --- Output ---
    header = list(results[0].keys())
    with open(args.output, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in results:
            f.write("\t".join(str(r[h]) for h in header) + "\n")
    _log(f"\nResults saved to {args.output}")

    _log("\n" + "=" * 120)
    fmt = "{:15s} {:>5s} {:>10s} {:>10s} {:>12s} {:>12s} {:>12s} {:>12s} {:>15s} {:>15s}"
    _log(fmt.format(
        "Config", "W", "enc_ms", "sc1k_ms", "sc10k_ms",
        "tot1k_ms", "tot10k_ms", "dev_r@10",
        "acl_r@10_gs1k", "acl_r@10_gs10k",
    ))
    _log("-" * 120)
    for r in results:
        _log(fmt.format(
            r["label"], str(r["W"]),
            f"{r['encode_ms']:.2f}", f"{r['score_ms_gs1k']:.2f}",
            f"{r['score_ms_gs10k']:.2f}", f"{r['total_ms_gs1k']:.2f}",
            f"{r['total_ms_gs10k']:.2f}", f"{r['dev_recall10']:.4f}",
            f"{r['acl_recall10_gs1000']:.4f}", f"{r['acl_recall10_gs10000']:.4f}",
        ))
    _log("=" * 120)


if __name__ == "__main__":
    main()
