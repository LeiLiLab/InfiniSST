#!/usr/bin/env python3
"""
Generate term_map for Speech LLM training using the best MaxSim retriever.

Supports **variable-length** speech chunks (0.96s–12.52s).  The retriever
uses multi-scale MaxSim windows [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24]
frames, matching the TCM final v3 checkpoint training launch, which naturally
adapt to any encoder-output length T:

**Mel frontend (RAG only):** this script feeds the **MaxSim retriever** only.
It uses ``WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")``
by default to match ``qwen3_glossary_neg_train.py``.  This is independent of
the Speech LLM translation path, which uses the Omni checkpoint's own processor
or vLLM multimodal pipeline.
  - w >= T  →  global average pooling (short chunks)
  - w <  T  →  sliding window with stride 2 (long chunks)

Input:  cleaned JSONL (no term_map in user messages) + glossary JSON
Output: per-chunk retrieval results JSONL (one line per conversation)

Each output line mirrors the input but adds a `retriever_results_by_chunk` field:
    [ [{term, zh, score, duration_sec, multiplier}, ...], ... ]

This is a pure inference script — it does NOT construct the final term_map.
A separate script (rebuild_termmap.py) will combine GT + retriever results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ======Configuration=====
AUDIO_MODEL_ID = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_MODEL_ID = "BAAI/bge-m3"
RAG_FEATURE_EXTRACTOR_MODEL_ID = "openai/whisper-large-v3"
EXPECTED_SAMPLE_RATE = 16000
UNIT_DURATION_SEC = 0.96

TEXT_ENCODE_BATCH = 256
AUDIO_ENCODE_BATCH = 32
MAX_BATCH_SECONDS = 60.0
RETRIEVAL_DENSITY = 10

LORA_RANK = 128
LORA_ALPHA = 256
POOLING_TYPE = "transformer"
TEMPERATURE = 0.03
USE_MAXSIM = True
MAXSIM_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24]
MAXSIM_STRIDE = 2
TARGET_DIM = 1024

TEXT_LORA_RANK = 128
TEXT_LORA_ALPHA = 256
TEXT_POOLING = "cls"
SPARSE_WEIGHT = 0.7

LORA_TARGET_MODULES = "q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2".split()
TEXT_LORA_TARGET_MODULES = "query key value dense".split()
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
    from transformers import AutoTokenizer, WhisperFeatureExtractor

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

    return retriever, text_encoder, _maxsim_score


def load_checkpoint(
    retriever, text_encoder, model_path: str, device: torch.device
) -> None:
    ckpt = torch.load(model_path, map_location=device)

    def _strip(sd):
        return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

    retriever.load_state_dict(_strip(ckpt.get("model_state_dict", {})), strict=False)
    if "text_model_state_dict" in ckpt:
        text_encoder.load_state_dict(_strip(ckpt["text_model_state_dict"]), strict=False)

    retriever.eval()
    text_encoder.eval()


@torch.no_grad()
def encode_glossary(
    terms: List[str], text_encoder, tokenizer, device: torch.device
) -> torch.Tensor:
    all_embs = []
    for start in range(0, len(terms), TEXT_ENCODE_BATCH):
        batch = terms[start : start + TEXT_ENCODE_BATCH]
        tok = tokenizer(
            batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = text_encoder(tok.input_ids, tok.attention_mask)
        all_embs.append(embs.float())
        done = start + len(batch)
        if (start // TEXT_ENCODE_BATCH) % 10 == 0:
            _log(f"  encoded {done}/{len(terms)} text terms")
    return torch.cat(all_embs, dim=0)


def load_audio_variable(path: str) -> Tuple[np.ndarray, float]:
    """Load full audio without truncation. Returns (audio_array, duration_sec)."""
    audio, sr = sf.read(path)
    assert sr == EXPECTED_SAMPLE_RATE, f"Unexpected SR {sr} for {path}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    assert audio.size > 0, f"Empty audio: {path}"
    mx = float(np.max(np.abs(audio)))
    if mx > 0:
        audio = audio / mx
    duration_sec = audio.shape[0] / EXPECTED_SAMPLE_RATE
    return audio, duration_sec


def _compute_multiplier(duration_sec: float) -> int:
    return max(1, round(duration_sec / UNIT_DURATION_SEC))


def _compute_top_k(
    duration_sec: float,
    multiplier: int,
    top_k_mode: str,
) -> int:
    """Compute the candidate recall K before score-threshold filtering.

    ``max_top_k`` is intentionally applied *after* threshold filtering so
    no-term chunks can naturally return few terms while term-rich chunks still
    have enough recall before the final cap.
    """
    if top_k_mode == "duration_sec_cap":
        k = int(math.ceil(max(0.0, float(duration_sec)) * float(RETRIEVAL_DENSITY)))
        return max(1, k)
    return max(1, int(RETRIEVAL_DENSITY) * int(multiplier))


def _normalise_glossary_entry(
    key: str,
    entry: Any,
    target_lang: str,
    allow_copy_translation_fallback: bool = False,
) -> Optional[Tuple[str, str]]:
    if isinstance(entry, str):
        term = str(key).strip()
        zh = entry.strip()
    elif isinstance(entry, dict):
        term = str(entry.get("term") or entry.get("source") or key).strip()
        zh = str(entry.get("translation") or entry.get("target_translation") or "").strip()
        target_translations = entry.get("target_translations")
        if not zh and isinstance(target_translations, dict):
            zh = str(target_translations.get(target_lang) or "").strip()
        if not zh:
            zh = str(entry.get(target_lang) or "").strip()
        if not zh and allow_copy_translation_fallback:
            zh = term
    else:
        return None
    if not term or not zh:
        return None
    return term, zh


def load_glossary_terms(
    path: str,
    target_lang: str,
    allow_copy_translation_fallback: bool = False,
) -> Tuple[List[str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        glossary = json.load(f)

    raw_items: List[Tuple[str, Any]]
    if isinstance(glossary, dict):
        raw_items = [(str(k), v) for k, v in glossary.items()]
    elif isinstance(glossary, list):
        raw_items = [(str(i), v) for i, v in enumerate(glossary)]
    else:
        raise ValueError(f"Unsupported glossary JSON format: {path}")

    term_list: List[str] = []
    zh_list: List[str] = []
    seen = set()
    for key, entry in raw_items:
        normalised = _normalise_glossary_entry(
            key,
            entry,
            target_lang,
            allow_copy_translation_fallback=allow_copy_translation_fallback,
        )
        if normalised is None:
            continue
        term, zh = normalised
        dedup_key = term.casefold()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        term_list.append(term)
        zh_list.append(zh)
    if not term_list:
        raise ValueError(f"No valid glossary terms loaded from {path}")
    return term_list, zh_list


@torch.no_grad()
def encode_audio_batch(
    audio_arrays: List[np.ndarray], retriever, feat_ext, device: torch.device
) -> torch.Tensor:
    """Encode a batch of variable-length audio arrays into MaxSim embeddings.

    Extracts mel features per sample, concatenates valid frames (no padding
    waste), and passes real feature_lens so the model masks correctly.
    """
    # Most SFT chunks are cut at fixed 0.96s multiples, so batches are often
    # same-length. Use one batched WhisperFeatureExtractor call in that common
    # path; the per-sample fallback below is much slower and CPU-bound.
    audio_lens = [int(a.shape[0]) for a in audio_arrays]
    if audio_lens and len(set(audio_lens)) == 1:
        inp = feat_ext(
            audio_arrays,
            sampling_rate=EXPECTED_SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_features = inp.input_features.to(device, dtype=torch.bfloat16)
        feature_lens = torch.full(
            (len(audio_arrays),),
            int(input_features.shape[-1]),
            dtype=torch.long,
            device=device,
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = retriever(input_features, feature_lens)
        return embs.float()

    mel_list: List[torch.Tensor] = []
    mel_lens: List[int] = []
    for audio in audio_arrays:
        inp = feat_ext(
            [audio],
            sampling_rate=EXPECTED_SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        mel = inp.input_features.squeeze(0)  # [C, T_mel_i]
        mel_list.append(mel)
        mel_lens.append(mel.shape[-1])

    input_features = torch.cat(mel_list, dim=-1).to(device, dtype=torch.bfloat16)
    feature_lens = torch.tensor(mel_lens, dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        embs = retriever(input_features, feature_lens)
    return embs.float()


def retrieve_topk(
    speech_emb: torch.Tensor,
    text_embs: torch.Tensor,
    _maxsim_score,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k glossary terms for one audio chunk."""
    if speech_emb.ndim == 3:
        sim = _maxsim_score(speech_emb, text_embs)  # [1, N_text]
    else:
        sim = speech_emb @ text_embs.T  # [1, N_text]
    sim_np = sim.cpu().numpy().squeeze(0)
    n = min(k, sim_np.shape[0])
    top_idx = np.argpartition(-sim_np, n)[:n]
    top_sco = sim_np[top_idx]
    order = np.argsort(-top_sco)
    return top_idx[order], top_sco[order]


def _should_flush_batch(
    audio_batch: List[np.ndarray],
    batch_seconds: float,
) -> bool:
    """Flush when we hit sample-count OR total-seconds limit."""
    if len(audio_batch) >= AUDIO_ENCODE_BATCH:
        return True
    if batch_seconds >= MAX_BATCH_SECONDS:
        return True
    return False


def _flush_and_retrieve(
    audio_batch: List[np.ndarray],
    durations: List[float],
    audio_batch_indices: List[int],
    retriever_results: List[Optional[List[Dict]]],
    retriever, feat_ext, device,
    text_embs, _maxsim_score, term_list, zh_list,
    top_k_mode: str,
    max_top_k: int,
    score_threshold: float,
) -> int:
    """Encode a batch, retrieve top-k per chunk, store results. Returns chunk count."""
    if not audio_batch:
        return 0
    embs = encode_audio_batch(audio_batch, retriever, feat_ext, device)
    embs = F.normalize(embs, p=2, dim=-1) if embs.ndim == 2 else embs
    for bi, ci in enumerate(audio_batch_indices):
        dur = durations[bi]
        mult = _compute_multiplier(dur)
        top_k = _compute_top_k(dur, mult, top_k_mode=top_k_mode)
        e = embs[bi : bi + 1]
        idx_arr, sco_arr = retrieve_topk(e, text_embs, _maxsim_score, top_k)
        chunk_res = []
        for ti, sc in zip(idx_arr, sco_arr):
            if float(sc) < float(score_threshold):
                continue
            chunk_res.append({
                "term": term_list[ti],
                "zh": zh_list[ti],
                "score": round(float(sc), 6),
            })
        if max_top_k > 0:
            chunk_res = chunk_res[: int(max_top_k)]
        retriever_results[ci] = {
            "results": chunk_res,
            "duration_sec": round(dur, 4),
            "multiplier": mult,
        }
    count = len(audio_batch)
    audio_batch.clear()
    durations.clear()
    audio_batch_indices.clear()
    return count


def _flush_global_and_retrieve(
    audio_batch: List[np.ndarray],
    durations: List[float],
    audio_batch_indices: List[Tuple[int, int]],
    retriever_results_by_conv: List[List[Optional[List[Dict]]]],
    retriever, feat_ext, device,
    text_embs, _maxsim_score, term_list, zh_list,
    top_k_mode: str,
    max_top_k: int,
    score_threshold: float,
) -> int:
    """Encode a cross-conversation batch and store results by (conv_idx, chunk_idx)."""
    if not audio_batch:
        return 0
    embs = encode_audio_batch(audio_batch, retriever, feat_ext, device)
    embs = F.normalize(embs, p=2, dim=-1) if embs.ndim == 2 else embs
    for bi, (conv_idx, chunk_idx) in enumerate(audio_batch_indices):
        dur = durations[bi]
        mult = _compute_multiplier(dur)
        top_k = _compute_top_k(dur, mult, top_k_mode=top_k_mode)
        e = embs[bi : bi + 1]
        idx_arr, sco_arr = retrieve_topk(e, text_embs, _maxsim_score, top_k)
        chunk_res = []
        for ti, sc in zip(idx_arr, sco_arr):
            if float(sc) < float(score_threshold):
                continue
            chunk_res.append({
                "term": term_list[ti],
                "zh": zh_list[ti],
                "score": round(float(sc), 6),
            })
        if max_top_k > 0:
            chunk_res = chunk_res[: int(max_top_k)]
        retriever_results_by_conv[conv_idx][chunk_idx] = {
            "results": chunk_res,
            "duration_sec": round(dur, 4),
            "multiplier": mult,
        }
    count = len(audio_batch)
    audio_batch.clear()
    durations.clear()
    audio_batch_indices.clear()
    return count


def main():
    global RETRIEVAL_DENSITY, AUDIO_ENCODE_BATCH, MAX_BATCH_SECONDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_jsonl", required=True)
    parser.add_argument("--glossary_json", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--retrieval_density", type=int, default=RETRIEVAL_DENSITY,
                        help="top_k = retrieval_density * multiplier")
    parser.add_argument("--top_k_mode", choices=["legacy_multiplier", "duration_sec_cap"],
                        default="legacy_multiplier",
                        help="legacy_multiplier: top_k = retrieval_density * multiplier; "
                             "duration_sec_cap: recall_k = ceil(duration_sec * retrieval_density). "
                             "In both modes, --max_top_k caps results after --score_threshold filtering.")
    parser.add_argument("--max_top_k", type=int, default=0,
                        help="Optional final cap after score-threshold filtering; 0 = no cap.")
    parser.add_argument("--score_threshold", type=float, default=float("-inf"),
                        help="Keep only retrieved candidates whose score is >= this threshold.")
    parser.add_argument("--target_lang", default="zh",
                        help="Target language key for glossary entries with target_translations.")
    parser.add_argument("--allow_copy_translation_fallback", action="store_true",
                        help="If a glossary entry lacks a target-language translation, use term as zh. "
                             "Disabled by default to avoid silent copy-style term maps.")
    parser.add_argument("--max_conversations", type=int, default=0,
                        help="0 = all; >0 = limit for smoke test")
    parser.add_argument("--audio_encode_batch", type=int, default=AUDIO_ENCODE_BATCH,
                        help="Number of audio chunks per encoder forward.")
    parser.add_argument("--max_batch_seconds", type=float, default=MAX_BATCH_SECONDS,
                        help="Maximum total seconds per audio encoder batch.")
    parser.add_argument(
        "--batch_across_conversations",
        action="store_true",
        help=(
            "Batch audio chunks across conversations before retrieval. This keeps output order "
            "unchanged but avoids tiny per-conversation GPU batches."
        ),
    )
    parser.add_argument(
        "--rag_feature_extractor_model_id",
        type=str,
        default=RAG_FEATURE_EXTRACTOR_MODEL_ID,
        help=(
            "Hub id for WhisperFeatureExtractor (mel for MaxSim retriever only). "
            f"Default: {RAG_FEATURE_EXTRACTOR_MODEL_ID}, matching retriever training."
        ),
    )
    args = parser.parse_args()

    RETRIEVAL_DENSITY = args.retrieval_density
    AUDIO_ENCODE_BATCH = int(args.audio_encode_batch)
    MAX_BATCH_SECONDS = float(args.max_batch_seconds)

    device = torch.device(args.device)

    _log(f"Loading glossary from {args.glossary_json}")
    term_list, zh_list = load_glossary_terms(
        args.glossary_json,
        target_lang=args.target_lang,
        allow_copy_translation_fallback=args.allow_copy_translation_fallback,
    )
    _log(f"Glossary: {len(term_list)} terms")

    # --- Build model ---
    _log("Building model...")
    retriever, text_encoder, _maxsim_score = build_model(device)
    load_checkpoint(retriever, text_encoder, args.model_path, device)

    from transformers import AutoTokenizer, WhisperFeatureExtractor
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    rag_fe_id = args.rag_feature_extractor_model_id
    feat_ext = WhisperFeatureExtractor.from_pretrained(rag_fe_id)
    _log(f"Model loaded. RAG mel: WhisperFeatureExtractor.from_pretrained({rag_fe_id!r})")

    # --- Encode glossary ---
    _log("Encoding glossary text embeddings...")
    text_embs = encode_glossary(term_list, text_encoder, tokenizer, device)
    text_embs = F.normalize(text_embs, p=2, dim=-1)
    _log(f"Text embeddings shape: {text_embs.shape}")

    # --- Process conversations ---
    _log(f"Processing {args.cleaned_jsonl}")
    _log(f"  RETRIEVAL_DENSITY={RETRIEVAL_DENSITY}, "
         f"TOP_K_MODE={args.top_k_mode}, "
         f"MAX_TOP_K={args.max_top_k}, "
         f"SCORE_THRESHOLD={args.score_threshold}, "
         f"MAX_BATCH_SECONDS={MAX_BATCH_SECONDS}, "
         f"AUDIO_ENCODE_BATCH={AUDIO_ENCODE_BATCH}")
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_convs = 0
    total_chunks = 0
    total_audio_failed = 0
    duration_hist: Dict[int, int] = {}  # multiplier -> count
    t_start = time.time()

    if args.batch_across_conversations:
        _log("Using cross-conversation audio batching with same-sample-length buckets.")
        conversations: List[Dict[str, Any]] = []
        retriever_results_by_conv: List[List[Optional[Dict]]] = []
        audio_buckets: Dict[int, Dict[str, Any]] = {}

        def _flush_bucket(sample_len: int) -> None:
            nonlocal total_chunks
            bucket = audio_buckets.get(sample_len)
            if not bucket or not bucket["audio"]:
                return
            total_chunks += _flush_global_and_retrieve(
                bucket["audio"], bucket["durations"], bucket["indices"],
                retriever_results_by_conv,
                retriever, feat_ext, device,
                text_embs, _maxsim_score, term_list, zh_list,
                args.top_k_mode, args.max_top_k, args.score_threshold,
            )
            bucket["seconds"] = 0.0

        with open(args.cleaned_jsonl, "r", encoding="utf-8") as f_in:
            for line_idx, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue
                try:
                    conv = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if 0 < args.max_conversations <= total_convs:
                    break

                conv_idx = len(conversations)
                audio_paths = conv.get("audios", []) or []
                conversations.append(conv)
                retriever_results_by_conv.append([None] * len(audio_paths))

                for chunk_idx, apath in enumerate(audio_paths):
                    if not os.path.isfile(apath):
                        total_audio_failed += 1
                        retriever_results_by_conv[conv_idx][chunk_idx] = {
                            "results": [],
                            "duration_sec": 0.0,
                            "multiplier": 1,
                        }
                        continue

                    try:
                        audio_arr, dur = load_audio_variable(apath)
                    except Exception as e:
                        _log(f"  WARN: failed to load {apath}: {e}")
                        total_audio_failed += 1
                        retriever_results_by_conv[conv_idx][chunk_idx] = {
                            "results": [],
                            "duration_sec": 0.0,
                            "multiplier": 1,
                        }
                        continue

                    mult = _compute_multiplier(dur)
                    duration_hist[mult] = duration_hist.get(mult, 0) + 1

                    sample_len = int(audio_arr.shape[0])
                    bucket = audio_buckets.setdefault(
                        sample_len,
                        {"audio": [], "durations": [], "indices": [], "seconds": 0.0},
                    )
                    bucket["audio"].append(audio_arr)
                    bucket["durations"].append(dur)
                    bucket["indices"].append((conv_idx, chunk_idx))
                    bucket["seconds"] += dur

                    if _should_flush_batch(bucket["audio"], bucket["seconds"]):
                        _flush_bucket(sample_len)

                total_convs += 1
                if total_convs % 500 == 0:
                    elapsed = time.time() - t_start
                    _log(
                        f"Queued: {total_convs} convs, {total_chunks} chunks encoded, "
                        f"{total_audio_failed} audio_failed, {elapsed:.0f}s"
                    )

        for sample_len in list(audio_buckets):
            _flush_bucket(sample_len)

        output_dir = os.path.dirname(args.output_jsonl)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
            for conv, retriever_results in zip(conversations, retriever_results_by_conv):
                rr_by_chunk = []
                chunk_metadata = []
                for r in retriever_results:
                    assert r is not None, "Bug: retriever result placeholder not filled"
                    rr_by_chunk.append(r["results"])
                    chunk_metadata.append({
                        "duration_sec": r["duration_sec"],
                        "multiplier": r["multiplier"],
                    })
                conv["retriever_results_by_chunk"] = rr_by_chunk
                conv["chunk_metadata"] = chunk_metadata
                f_out.write(json.dumps(conv, ensure_ascii=False) + "\n")

    else:
      with open(args.cleaned_jsonl, "r", encoding="utf-8") as f_in, \
           open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        for line_idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError:
                continue

            if 0 < args.max_conversations <= total_convs:
                break

            audio_paths = conv.get("audios", []) or []
            retriever_results: List[Optional[Dict]] = [None] * len(audio_paths)

            audio_batch: List[np.ndarray] = []
            batch_durations: List[float] = []
            audio_batch_indices: List[int] = []
            batch_seconds = 0.0

            for chunk_idx, apath in enumerate(audio_paths):
                if not os.path.isfile(apath):
                    total_audio_failed += 1
                    retriever_results[chunk_idx] = {"results": [], "duration_sec": 0.0, "multiplier": 1}
                    continue

                try:
                    audio_arr, dur = load_audio_variable(apath)
                except Exception as e:
                    _log(f"  WARN: failed to load {apath}: {e}")
                    total_audio_failed += 1
                    retriever_results[chunk_idx] = {"results": [], "duration_sec": 0.0, "multiplier": 1}
                    continue

                mult = _compute_multiplier(dur)
                duration_hist[mult] = duration_hist.get(mult, 0) + 1

                audio_batch.append(audio_arr)
                batch_durations.append(dur)
                audio_batch_indices.append(chunk_idx)
                batch_seconds += dur

                if _should_flush_batch(audio_batch, batch_seconds):
                    total_chunks += _flush_and_retrieve(
                        audio_batch, batch_durations, audio_batch_indices,
                        retriever_results,
                        retriever, feat_ext, device,
                        text_embs, _maxsim_score, term_list, zh_list,
                        args.top_k_mode, args.max_top_k, args.score_threshold,
                    )
                    batch_seconds = 0.0

            # flush remaining
            total_chunks += _flush_and_retrieve(
                audio_batch, batch_durations, audio_batch_indices,
                retriever_results,
                retriever, feat_ext, device,
                text_embs, _maxsim_score, term_list, zh_list,
                args.top_k_mode, args.max_top_k, args.score_threshold,
            )

            # Convert structured results to the output format:
            # retriever_results_by_chunk: list of list-of-dicts (backward compatible)
            # chunk_metadata: list of {duration_sec, multiplier}
            rr_by_chunk = []
            chunk_metadata = []
            for r in retriever_results:
                assert r is not None, "Bug: retriever result placeholder not filled"
                rr_by_chunk.append(r["results"])
                chunk_metadata.append({
                    "duration_sec": r["duration_sec"],
                    "multiplier": r["multiplier"],
                })

            conv["retriever_results_by_chunk"] = rr_by_chunk
            conv["chunk_metadata"] = chunk_metadata
            f_out.write(json.dumps(conv, ensure_ascii=False) + "\n")
            total_convs += 1

            if total_convs % 500 == 0:
                elapsed = time.time() - t_start
                _log(
                    f"Progress: {total_convs} convs, {total_chunks} chunks, "
                    f"{total_audio_failed} audio_failed, {elapsed:.0f}s"
                )

    elapsed = time.time() - t_start
    _log(f"Done: {total_convs} convs, {total_chunks} chunks, "
         f"{total_audio_failed} audio_failed, {elapsed:.0f}s total")
    _log(f"Output: {args.output_jsonl}")

    # Duration distribution summary
    _log("Multiplier distribution:")
    for mult in sorted(duration_hist.keys()):
        example_duration = mult * UNIT_DURATION_SEC
        _log(f"  multiplier={mult:>2d} ({mult * UNIT_DURATION_SEC:>6.2f}s): "
             f"{duration_hist[mult]:>6d} chunks, "
             f"recall_k={_compute_top_k(example_duration, mult, args.top_k_mode):>3d}, "
             f"post_filter_cap={args.max_top_k if args.max_top_k > 0 else 'none'}")


if __name__ == "__main__":
    main()
