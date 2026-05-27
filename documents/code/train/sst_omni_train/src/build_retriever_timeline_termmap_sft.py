#!/usr/bin/env python3
"""Build Speech LLM SFT JSONL with timeline MaxSim retriever term_map entries.

V1 policy:
* keep the existing streaming speech chunks in the source JSONL;
* for each chunk, retrieve over [chunk_start - lookback_sec, chunk_end];
* only keep MaxSim evidence windows that overlap the current chunk timeline;
* filter by absolute MaxSim threshold and top-k;
* do not backfill missed GT terms;
* if a retrieved term matches chunk GT, use the chunk-specific GT translation.

For low-latency Speech LLM SFT, ``--min-context-sec`` can delay retriever calls
until the timeline buffer is long enough for the retriever training regime.  If
the current chunk is already at least ``min_context_sec`` long, we retrieve from
the current chunk only.  Otherwise, with 0.96s chunks, ``--lookback-sec 1.92
--min-context-sec 2.88`` makes lm=1 chunks 0 and 1 use an empty term map, then
chunk 2 retrieves over chunks 0..2 and keeps only evidence overlapping chunk 2.
``--max-context-sec`` can also cap the context at the longest duration seen by
the retriever.

The script is intentionally explicit about filtering and missing inputs.  It
should not silently replace missing audio, malformed rows, or missing indexes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(
    p for p in _HERE.parents
    if (p / "documents" / "code" / "train" / "term_train" / "qwen3_glossary_neg_train.py").is_file()
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TERM_TRAIN_DIR = _REPO_ROOT / "documents" / "code" / "train" / "term_train"
if str(_TERM_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TERM_TRAIN_DIR))

from qwen3_glossary_neg_train import (  # noqa: E402
    BgeM3TextEncoder,
    Qwen3OmniRetriever,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor  # noqa: E402


SYSTEM_PROMPT_BY_LANG = {
    "zh": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate "
        "the audio into Chinese text. Use the 'term_map' as a reference for "
        "terminology if provided."
    ),
    "de": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate "
        "the audio into German text. Use the 'term_map' as a reference for "
        "terminology if provided."
    ),
    "ja": (
        "You are a professional simultaneous interpreter. "
        "You will be given chunks of English audio and you need to translate "
        "the audio into Japanese text. Use the 'term_map' as a reference for "
        "terminology if provided."
    ),
}

AUDIO_MODEL_ID = "Atotti/Qwen3-Omni-AudioTransformer"
TEXT_MODEL_ID = "BAAI/bge-m3"
RAG_FEATURE_EXTRACTOR_MODEL_ID = "openai/whisper-large-v3"
EXPECTED_SAMPLE_RATE = 16000
UNIT_DURATION_SEC = 0.96
ENCODER_FPS = 12.5
FRAME_SEC = 1.0 / ENCODER_FPS

TARGET_DIM = 1024
LORA_RANK = 128
LORA_ALPHA = 256
TEXT_LORA_RANK = 128
TEXT_LORA_ALPHA = 256
TEXT_POOLING = "cls"
SPARSE_WEIGHT = 0.7
POOLING_TYPE = "transformer"
TEMPERATURE = 0.03
MAXSIM_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24]
MAXSIM_STRIDE = 2
LORA_TARGET_MODULES = "q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2".split()
TEXT_LORA_TARGET_MODULES = "query key value dense".split()


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on {path}:{lineno}")
            yield lineno, obj


def _term_key(term: str) -> str:
    return " ".join(str(term or "").casefold().split())


def _extract_translation(entry: Mapping[str, Any], lang_code: str) -> str:
    value = entry.get("translation") or entry.get("target_translation") or entry.get(lang_code)
    if value is None and isinstance(entry.get("target_translations"), Mapping):
        value = entry["target_translations"].get(lang_code)
    return str(value or "").strip()


def _normalize_glossary_entry(
    key: str,
    entry: Any,
    lang_code: str,
    *,
    allow_copy_translation_fallback: bool,
) -> Optional[Dict[str, Any]]:
    if isinstance(entry, str):
        term = str(key).strip()
        translation = entry.strip()
        raw = {"term": term, "translation": translation, "target_translations": {lang_code: translation}}
    elif isinstance(entry, Mapping):
        term = str(entry.get("term") or entry.get("source") or key).strip()
        translation = _extract_translation(entry, lang_code)
        if not translation and allow_copy_translation_fallback:
            translation = term
        raw = dict(entry)
        raw.setdefault("term", term)
        raw.setdefault("target_translations", {})
        if isinstance(raw["target_translations"], Mapping):
            raw["target_translations"] = dict(raw["target_translations"])
            raw["target_translations"].setdefault(lang_code, translation)
        else:
            raw["target_translations"] = {lang_code: translation}
        raw["translation"] = translation
    else:
        return None

    if not term or not translation:
        return None
    raw["term"] = term
    raw["key"] = raw.get("key") or _term_key(term)
    raw["translation"] = translation
    return raw


def load_glossary(
    path: Path,
    lang_code: str,
    *,
    allow_copy_translation_fallback: bool = False,
) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        glossary = json.load(f)

    if isinstance(glossary, Mapping):
        raw_items = [(str(k), v) for k, v in glossary.items()]
    elif isinstance(glossary, list):
        raw_items = [(str(i), v) for i, v in enumerate(glossary)]
    else:
        raise ValueError(f"Unsupported glossary format: {path}")

    out: List[Dict[str, Any]] = []
    seen = set()
    skipped = 0
    for key, entry in raw_items:
        item = _normalize_glossary_entry(
            key,
            entry,
            lang_code,
            allow_copy_translation_fallback=allow_copy_translation_fallback,
        )
        if item is None:
            skipped += 1
            continue
        dedup = _term_key(item["term"])
        if not dedup or dedup in seen:
            continue
        seen.add(dedup)
        out.append(item)

    if not out:
        raise ValueError(f"No valid translated glossary terms loaded from {path}")
    _log(f"Loaded glossary: kept={len(out)} skipped_missing_translation={skipped}")
    return out


def _strip_state_dict(sd: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}


def build_audio_retriever(device: torch.device):
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
        use_maxsim=True,
        maxsim_windows=MAXSIM_WINDOWS,
        maxsim_stride=MAXSIM_STRIDE,
    ).to(device)
    retriever.eval()
    return retriever


def build_text_encoder(device: torch.device):
    text_encoder = BgeM3TextEncoder(
        model_id=TEXT_MODEL_ID,
        lora_rank=TEXT_LORA_RANK,
        lora_alpha=TEXT_LORA_ALPHA,
        target_modules=TEXT_LORA_TARGET_MODULES,
        full_finetune=False,
        sparse_weight=SPARSE_WEIGHT,
        text_pooling=TEXT_POOLING,
    ).to(device)
    text_encoder.eval()
    return text_encoder


def load_checkpoint(model_path: Path, device: torch.device, retriever=None, text_encoder=None) -> None:
    ckpt = torch.load(str(model_path), map_location=device)
    if retriever is not None:
        retriever.load_state_dict(_strip_state_dict(ckpt.get("model_state_dict", {})), strict=False)
        retriever.eval()
    if text_encoder is not None and "text_model_state_dict" in ckpt:
        text_encoder.load_state_dict(_strip_state_dict(ckpt["text_model_state_dict"]), strict=False)
        text_encoder.eval()


@torch.no_grad()
def build_text_index(args: argparse.Namespace) -> None:
    args.text_index_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    glossary = load_glossary(
        args.glossary_json,
        args.lang_code,
        allow_copy_translation_fallback=args.allow_copy_translation_fallback,
    )
    terms = [x["term"] for x in glossary]
    text_encoder = build_text_encoder(device)
    load_checkpoint(args.model_path, device, text_encoder=text_encoder)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)

    embs: List[torch.Tensor] = []
    for start in range(0, len(terms), args.text_encode_batch):
        batch = terms[start : start + args.text_encode_batch]
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cur = text_encoder(tok.input_ids, tok.attention_mask)
        embs.append(F.normalize(cur.float(), p=2, dim=-1).cpu())
        if start == 0 or (start // args.text_encode_batch) % 20 == 0:
            _log(f"Encoded glossary terms {start + len(batch)}/{len(terms)}")

    text_embs = torch.cat(embs, dim=0)
    payload = {
        "text_embs": text_embs,
        "term_list": glossary,
        "glossary_json": str(args.glossary_json),
        "model_path": str(args.model_path),
        "lang_code": args.lang_code,
        "text_model_id": TEXT_MODEL_ID,
        "text_lora_rank": TEXT_LORA_RANK,
        "text_pooling": TEXT_POOLING,
        "created_at_unix": time.time(),
    }
    torch.save(payload, str(args.text_index_path))
    _log(f"Saved text index: {args.text_index_path} shape={tuple(text_embs.shape)}")


def load_audio_raw(path: str) -> np.ndarray:
    audio, sr = sf.read(path)
    if sr != EXPECTED_SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate {sr} for {path}; expected {EXPECTED_SAMPLE_RATE}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size <= 0:
        raise ValueError(f"Empty audio: {path}")
    return audio


def _audio_user_indices(messages: Sequence[Mapping[str, Any]]) -> List[int]:
    out = []
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user" and "<audio>" in str(msg.get("content") or ""):
            out.append(idx)
    return out


def _gt_terms(raw_terms: Any, lang_code: str) -> List[Dict[str, str]]:
    if raw_terms is None:
        return []
    if not isinstance(raw_terms, list):
        raise ValueError(f"gt_terms_by_chunk entry must be list, got {type(raw_terms).__name__}")
    out: List[Dict[str, str]] = []
    seen = set()
    for item in raw_terms:
        if not isinstance(item, Mapping):
            raise ValueError("gt_terms_by_chunk term entry must be an object")
        term = str(item.get("term") or item.get("source") or "").strip()
        translation = _extract_translation(item, lang_code)
        key = _term_key(term)
        if not term or not translation or not key or key in seen:
            continue
        seen.add(key)
        out.append({"term": term, "translation": translation, "key": key})
    return out


def _format_term_map(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "<audio>\n\nterm_map:NONE"
    lines = ["<audio>", "", "term_map:"]
    seen = set()
    for item in items:
        term = str(item.get("term") or "").strip()
        translation = str(item.get("translation") or item.get("zh") or "").strip()
        key = _term_key(term)
        if not term or not translation or not key or key in seen:
            continue
        seen.add(key)
        lines.append(f"{term}={translation}")
    if len(lines) == 3:
        return "<audio>\n\nterm_map:NONE"
    return "\n".join(lines)


def _compute_multiplier(duration_sec: float) -> int:
    return max(1, int(round(float(duration_sec) / UNIT_DURATION_SEC)))


def _build_window_time_ranges(maxsim_windows: List[int], maxsim_stride: int, t_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
    starts = []
    ends = []
    for w in maxsim_windows:
        if w >= t_frames:
            starts.append(0.0)
            ends.append(t_frames * FRAME_SEC)
        else:
            n_out = (t_frames - w) // maxsim_stride + 1
            for p in range(n_out):
                frame_start = p * maxsim_stride
                frame_end = frame_start + w
                starts.append(frame_start * FRAME_SEC)
                ends.append(frame_end * FRAME_SEC)
    return torch.tensor(starts, dtype=torch.float32), torch.tensor(ends, dtype=torch.float32)


@torch.no_grad()
def _encode_projected_seq_batch(
    audio_arrays: Sequence[np.ndarray],
    retriever,
    feat_ext,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inp = feat_ext(
        list(audio_arrays),
        sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_features = inp.input_features.to(device, dtype=torch.bfloat16)
    if hasattr(inp, "attention_mask") and inp.attention_mask is not None:
        feature_lens = inp.attention_mask.sum(dim=1).to(device=device, dtype=torch.long)
    else:
        feature_lens = torch.full(
            (len(audio_arrays),),
            input_features.shape[-1],
            dtype=torch.long,
            device=device,
        )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if input_features.ndim == 3:
            qwen_features = input_features.transpose(0, 1).reshape(input_features.shape[1], -1)
        else:
            qwen_features = input_features
        outputs = retriever.audio_encoder(qwen_features, feature_lens)
        hidden_states = outputs.last_hidden_state

        if hidden_states.ndim == 2:
            output_lens: List[int] = []
            for cur in feature_lens.tolist():
                reduced = cur
                for _ in range(3):
                    reduced = (reduced + 1) // 2
                output_lens.append(reduced)
            if sum(output_lens) != hidden_states.shape[0]:
                ratio = qwen_features.shape[1] / hidden_states.shape[0]
                output_lens = [max(1, round(x / ratio)) for x in feature_lens.tolist()]
                output_lens[-1] = hidden_states.shape[0] - sum(output_lens[:-1])

            from torch.nn.utils.rnn import pad_sequence

            parts = torch.split(hidden_states, output_lens, dim=0)
            hidden_states = pad_sequence(parts, batch_first=True)
            frame_lens = torch.tensor(output_lens, device=hidden_states.device)
        else:
            frame_lens = feature_lens

        batch_size, max_len, _ = hidden_states.shape
        mask = (
            torch.arange(max_len, device=hidden_states.device).expand(batch_size, max_len)
            < frame_lens.unsqueeze(1)
        )
        projected_seq = retriever.projector(hidden_states)
        projected_seq = projected_seq * mask.unsqueeze(-1).float()
    return projected_seq.float(), mask


@torch.no_grad()
def retrieve_timeline_batch(
    contexts: Sequence[Dict[str, Any]],
    retriever,
    feat_ext,
    text_embs: torch.Tensor,
    term_list: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    top_k: int,
    score_threshold: float,
) -> List[List[Dict[str, Any]]]:
    audio_arrays = [ctx["audio"] for ctx in contexts]
    projected_seq, mask = _encode_projected_seq_batch(audio_arrays, retriever, feat_ext, device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        window_embs = retriever._multiscale_pool(projected_seq, mask)
        window_embs = F.normalize(window_embs, p=2, dim=-1).float()

    t_frames = int(projected_seq.shape[1])
    rel_starts, rel_ends = _build_window_time_ranges(
        retriever.maxsim_windows,
        retriever.maxsim_stride,
        t_frames,
    )
    if int(rel_starts.numel()) != int(window_embs.shape[1]):
        raise RuntimeError(
            f"window range count mismatch: ranges={rel_starts.numel()} embs={window_embs.shape[1]}"
        )
    rel_starts = rel_starts.to(device)
    rel_ends = rel_ends.to(device)

    out: List[List[Dict[str, Any]]] = []
    text_f = text_embs.float()
    for bi, ctx in enumerate(contexts):
        actual_start = float(ctx["actual_start_sec"])
        actual_end = float(ctx["actual_end_sec"])
        actual_duration = max(1e-6, actual_end - actual_start)
        nominal_duration = max(float(rel_ends.max().item()), 1e-6)
        scale = actual_duration / nominal_duration
        abs_starts = actual_start + rel_starts * scale
        abs_ends = actual_start + rel_ends * scale
        current_start = float(ctx["chunk_start_sec"])
        current_end = float(ctx["chunk_end_sec"])
        valid_windows = (abs_ends > current_start) & (abs_starts < current_end)
        if int(valid_windows.sum().item()) == 0:
            out.append([])
            continue

        sim_by_window = window_embs[bi].float() @ text_f.T
        sim_by_window = sim_by_window.masked_fill(~valid_windows.unsqueeze(1), -float("inf"))
        scores, best_window_idx = sim_by_window.max(dim=0)
        finite = torch.isfinite(scores)
        if int(finite.sum().item()) == 0:
            out.append([])
            continue
        n = min(int(top_k), int(finite.sum().item()))
        masked_scores = scores.masked_fill(~finite, -float("inf"))
        top_sco, top_idx = torch.topk(masked_scores, k=n, largest=True, sorted=True)
        top_win = best_window_idx.gather(0, top_idx)
        top_start = abs_starts.gather(0, top_win)
        top_end = abs_ends.gather(0, top_win)

        chunk_items: List[Dict[str, Any]] = []
        for ti, sc, ts, te in zip(top_idx.tolist(), top_sco.tolist(), top_start.tolist(), top_end.tolist()):
            if float(sc) < float(score_threshold):
                continue
            entry = term_list[int(ti)]
            translation = _extract_translation(entry, ctx["lang_code"])
            if not translation:
                continue
            chunk_items.append({
                "term": str(entry.get("term") or ""),
                "translation": translation,
                "score": round(float(sc), 6),
                "time_start": round(float(ts), 4),
                "time_end": round(float(te), 4),
                "retrieval_mode": "timeline_train_v1",
            })
        out.append(chunk_items)
    return out


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(round((p / 100.0) * (len(xs) - 1)))
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


def _duration_bucket(duration_sec: float) -> str:
    if duration_sec <= 1.05:
        return "lm1"
    if duration_sec <= 3.90:
        return "lm2to4"
    if duration_sec <= 5.85:
        return "lm5to6"
    return "lm7plus"


def _update_bucket(stats: Dict[str, Any], bucket: str, has_gt: bool, gt_count: int, hit_count: int, map_size: int) -> None:
    bs = stats.setdefault("duration_buckets", {})
    cur = bs.setdefault(bucket, {
        "chunks": 0,
        "gt_chunks": 0,
        "gt_chunks_any_hit": 0,
        "gt_chunks_all_hit": 0,
        "gt_terms": 0,
        "gt_hits": 0,
        "term_map_entries": 0,
        "nonempty_term_maps": 0,
        "no_gt_chunks": 0,
        "no_gt_nonempty_term_maps": 0,
    })
    cur["chunks"] += 1
    cur["term_map_entries"] += map_size
    if map_size > 0:
        cur["nonempty_term_maps"] += 1
    if has_gt:
        cur["gt_chunks"] += 1
        cur["gt_terms"] += gt_count
        cur["gt_hits"] += hit_count
        if hit_count > 0:
            cur["gt_chunks_any_hit"] += 1
        if hit_count == gt_count:
            cur["gt_chunks_all_hit"] += 1
    else:
        cur["no_gt_chunks"] += 1
        if map_size > 0:
            cur["no_gt_nonempty_term_maps"] += 1


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.text_index_path.exists():
        raise FileNotFoundError(f"Missing text index: {args.text_index_path}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_json:
        args.sample_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sample_chunks_json:
        args.sample_chunks_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    _log(f"Loading text index: {args.text_index_path}")
    index_data = torch.load(str(args.text_index_path), map_location="cpu")
    text_embs = index_data["text_embs"].to(device)
    term_list = index_data["term_list"]
    if int(text_embs.shape[0]) != len(term_list):
        raise ValueError("Text index mismatch between text_embs and term_list")
    _log(f"Text index ready: {tuple(text_embs.shape)}")

    retriever = build_audio_retriever(device)
    load_checkpoint(args.model_path, device, retriever=retriever)
    feat_ext = WhisperFeatureExtractor.from_pretrained(RAG_FEATURE_EXTRACTOR_MODEL_ID)

    stats: Dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "glossary_json": str(args.glossary_json),
        "text_index_path": str(args.text_index_path),
        "model_path": str(args.model_path),
        "lang_code": args.lang_code,
        "top_k": args.top_k,
        "score_threshold": args.score_threshold,
        "lookback_sec": args.lookback_sec,
        "min_context_sec": args.min_context_sec,
        "max_context_sec": args.max_context_sec,
        "merge_multiplier_min": args.merge_multiplier_min,
        "merge_multiplier_max": args.merge_multiplier_max,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "input_rows_seen": 0,
        "rows_selected_by_shard": 0,
        "rows_filtered_by_merge_multiplier": 0,
        "rows_written": 0,
        "dropped_rows": 0,
        "dropped_reasons": Counter(),
        "audio_user_chunks": 0,
        "gt_chunks": 0,
        "no_gt_chunks": 0,
        "gt_terms_total": 0,
        "gt_terms_hit": 0,
        "gt_chunks_any_hit": 0,
        "gt_chunks_all_hit": 0,
        "term_map_entries_total": 0,
        "term_map_gt_entries": 0,
        "term_map_non_gt_entries": 0,
        "nonempty_term_map_chunks": 0,
        "no_gt_nonempty_term_map_chunks": 0,
        "rows_missing_gt_terms_by_chunk": 0,
        "rows_mismatched_audio_gt_counts": 0,
        "rows_mismatched_audio_message_counts": 0,
        "duration_buckets": {},
    }
    termmap_sizes: List[int] = []
    score_values: List[float] = []
    multiplier_hist: Counter = Counter()
    samples: List[Dict[str, Any]] = []
    sample_chunks: List[Dict[str, Any]] = []
    t0 = time.time()

    with args.output_jsonl.open("w", encoding="utf-8") as f_out:
        for line_idx, obj in _iter_jsonl(args.input_jsonl):
            stats["input_rows_seen"] += 1
            if args.num_shards > 1 and ((line_idx - 1) % args.num_shards) != args.shard_index:
                continue
            if 0 < args.max_conversations <= stats["rows_written"]:
                break
            stats["rows_selected_by_shard"] += 1

            try:
                messages = obj.get("messages")
                audios = obj.get("audios")
                merge_multiplier = int(obj.get("merge_multiplier") or -1)
                if args.merge_multiplier_min > 0 and merge_multiplier < args.merge_multiplier_min:
                    stats["rows_filtered_by_merge_multiplier"] += 1
                    continue
                if args.merge_multiplier_max > 0 and merge_multiplier > args.merge_multiplier_max:
                    stats["rows_filtered_by_merge_multiplier"] += 1
                    continue
                if not isinstance(messages, list) or not messages:
                    raise ValueError("missing non-empty messages")
                if not isinstance(audios, list) or not audios:
                    raise ValueError("missing non-empty audios")

                audio_user_idxs = _audio_user_indices(messages)
                if len(audio_user_idxs) != len(audios):
                    stats["rows_mismatched_audio_message_counts"] += 1
                    raise ValueError(
                        f"audio user messages={len(audio_user_idxs)} audios={len(audios)}"
                    )

                gt_by_chunk = obj.get("gt_terms_by_chunk")
                if gt_by_chunk is None:
                    stats["rows_missing_gt_terms_by_chunk"] += 1
                    gt_by_chunk = [[] for _ in audios]
                if not isinstance(gt_by_chunk, list):
                    raise ValueError("gt_terms_by_chunk is not a list")
                if len(gt_by_chunk) != len(audios):
                    stats["rows_mismatched_audio_gt_counts"] += 1
                    raise ValueError(f"gt_terms_by_chunk={len(gt_by_chunk)} audios={len(audios)}")

                if messages[0].get("role") == "system":
                    messages[0]["content"] = SYSTEM_PROMPT_BY_LANG[args.lang_code]

                chunk_arrays = [load_audio_raw(str(p)) for p in audios]
                chunk_lens = [int(x.shape[0]) for x in chunk_arrays]
                full_audio = np.concatenate(chunk_arrays)
                max_abs = float(np.max(np.abs(full_audio))) if full_audio.size else 0.0
                if max_abs > 0:
                    full_audio = full_audio / max_abs
                starts = np.cumsum([0] + chunk_lens[:-1]).astype(np.int64)
                ends = np.cumsum(chunk_lens).astype(np.int64)

                contexts: List[Dict[str, Any]] = []
                for chunk_idx, (start_samp, end_samp) in enumerate(zip(starts, ends)):
                    chunk_start_sec = float(start_samp) / EXPECTED_SAMPLE_RATE
                    chunk_end_sec = float(end_samp) / EXPECTED_SAMPLE_RATE
                    chunk_sec = max(0.0, chunk_end_sec - chunk_start_sec)
                    if args.min_context_sec > 0 and chunk_sec + 1e-6 >= float(args.min_context_sec):
                        encode_start_samp = int(start_samp)
                    else:
                        encode_start_samp = max(
                            0,
                            int(round((chunk_start_sec - args.lookback_sec) * EXPECTED_SAMPLE_RATE)),
                        )
                    if args.max_context_sec > 0:
                        max_context_samp = int(round(float(args.max_context_sec) * EXPECTED_SAMPLE_RATE))
                        encode_start_samp = max(encode_start_samp, int(end_samp) - max_context_samp, 0)
                    context_audio = np.asarray(full_audio[encode_start_samp:end_samp], dtype=np.float32)
                    if context_audio.size <= 0:
                        raise ValueError(f"empty context for chunk {chunk_idx}")
                    context_sec = float(context_audio.shape[0]) / EXPECTED_SAMPLE_RATE
                    contexts.append({
                        "chunk_idx": chunk_idx,
                        "audio": context_audio,
                        "context_len": int(context_audio.shape[0]),
                        "chunk_len": int(end_samp) - int(start_samp),
                        "retrieval_ready": context_sec + 1e-6 >= float(args.min_context_sec),
                        "actual_start_sec": float(encode_start_samp) / EXPECTED_SAMPLE_RATE,
                        "actual_end_sec": float(end_samp) / EXPECTED_SAMPLE_RATE,
                        "chunk_start_sec": chunk_start_sec,
                        "chunk_end_sec": chunk_end_sec,
                        "lang_code": args.lang_code,
                    })

                results_by_chunk: List[List[Dict[str, Any]]] = [[] for _ in audios]
                by_len: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
                for ctx in contexts:
                    if not bool(ctx.get("retrieval_ready", True)):
                        continue
                    by_len[int(ctx["context_len"])].append(ctx)
                for ctxs in by_len.values():
                    for start in range(0, len(ctxs), args.audio_batch_size):
                        batch = ctxs[start : start + args.audio_batch_size]
                        batch_results = retrieve_timeline_batch(
                            batch,
                            retriever,
                            feat_ext,
                            text_embs,
                            term_list,
                            device,
                            top_k=args.top_k,
                            score_threshold=args.score_threshold,
                        )
                        for ctx, chunk_results in zip(batch, batch_results):
                            results_by_chunk[int(ctx["chunk_idx"])] = chunk_results

                for chunk_idx, msg_idx in enumerate(audio_user_idxs):
                    gt_terms = _gt_terms(gt_by_chunk[chunk_idx], args.lang_code)
                    gt_by_key = {x["key"]: x for x in gt_terms}
                    final_items: List[Dict[str, Any]] = []
                    seen = set()
                    gt_hits = set()
                    for item in results_by_chunk[chunk_idx]:
                        term = str(item.get("term") or "").strip()
                        key = _term_key(term)
                        if not key or key in seen:
                            continue
                        seen.add(key)
                        out_item = dict(item)
                        if key in gt_by_key:
                            out_item["translation"] = gt_by_key[key]["translation"]
                            out_item["gt_translation_override"] = True
                            gt_hits.add(key)
                        final_items.append(out_item)
                        if item.get("score") is not None:
                            score_values.append(float(item["score"]))

                    messages[msg_idx]["content"] = _format_term_map(final_items)
                    chunk_duration = (ends[chunk_idx] - starts[chunk_idx]) / EXPECTED_SAMPLE_RATE
                    multiplier = _compute_multiplier(chunk_duration)
                    multiplier_hist[multiplier] += 1

                    map_size = len(final_items)
                    gt_count = len(gt_terms)
                    hit_count = len(gt_hits)
                    has_gt = gt_count > 0
                    stats["audio_user_chunks"] += 1
                    stats["term_map_entries_total"] += map_size
                    stats["term_map_gt_entries"] += hit_count
                    stats["term_map_non_gt_entries"] += max(0, map_size - hit_count)
                    if map_size > 0:
                        stats["nonempty_term_map_chunks"] += 1
                    if has_gt:
                        stats["gt_chunks"] += 1
                        stats["gt_terms_total"] += gt_count
                        stats["gt_terms_hit"] += hit_count
                        if hit_count > 0:
                            stats["gt_chunks_any_hit"] += 1
                        if hit_count == gt_count:
                            stats["gt_chunks_all_hit"] += 1
                    else:
                        stats["no_gt_chunks"] += 1
                        if map_size > 0:
                            stats["no_gt_nonempty_term_map_chunks"] += 1

                    termmap_sizes.append(map_size)
                    bucket = _duration_bucket(chunk_duration)
                    _update_bucket(stats, bucket, has_gt, gt_count, hit_count, map_size)

                    if len(sample_chunks) < args.sample_chunk_count and (map_size > 0 or has_gt):
                        sample_chunks.append({
                            "row_line": line_idx,
                            "utter_id": obj.get("utter_id"),
                            "chunk_idx": chunk_idx,
                            "chunk_duration_sec": round(chunk_duration, 4),
                            "context_sec": round(contexts[chunk_idx]["actual_end_sec"] - contexts[chunk_idx]["actual_start_sec"], 4),
                            "retrieval_ready": bool(contexts[chunk_idx].get("retrieval_ready", True)),
                            "gt_terms": gt_terms,
                            "term_map": final_items,
                        })

                obj["retriever_timeline_termmap_policy"] = {
                    "version": "v1",
                    "source": "timeline_maxsim_retriever",
                    "model_path": str(args.model_path),
                    "top_k": args.top_k,
                    "score_threshold": args.score_threshold,
                    "lookback_sec": args.lookback_sec,
                    "min_context_sec": args.min_context_sec,
                    "max_context_sec": args.max_context_sec,
                    "no_gt_backfill": True,
                    "gt_translation_override_on_retrieved_match": True,
                    "merge_multiplier_min": args.merge_multiplier_min,
                    "merge_multiplier_max": args.merge_multiplier_max,
                }
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                if len(samples) < args.sample_count:
                    samples.append({
                        "row_line": line_idx,
                        "utter_id": obj.get("utter_id"),
                        "audio_chunks": len(audios),
                        "merge_multiplier": obj.get("merge_multiplier"),
                        "first_user": messages[audio_user_idxs[0]]["content"],
                    })
                stats["rows_written"] += 1

            except Exception as exc:
                if args.drop_bad_rows:
                    stats["dropped_rows"] += 1
                    stats["dropped_reasons"][str(exc).splitlines()[0][:200]] += 1
                    continue
                raise RuntimeError(f"Failed processing {args.input_jsonl}:{line_idx}: {exc}") from exc

            if stats["rows_written"] and stats["rows_written"] % args.log_every == 0:
                elapsed = time.time() - t0
                _log(
                    f"Progress shard={args.shard_index}/{args.num_shards}: "
                    f"rows={stats['rows_written']} chunks={stats['audio_user_chunks']} "
                    f"elapsed={elapsed:.0f}s"
                )

    stats["dropped_reasons"] = dict(stats["dropped_reasons"])
    stats["multiplier_hist"] = dict(sorted(multiplier_hist.items()))
    stats["term_map_size_hist"] = dict(Counter(termmap_sizes).most_common(50))
    stats["gt_term_recall"] = (
        stats["gt_terms_hit"] / stats["gt_terms_total"] if stats["gt_terms_total"] else 0.0
    )
    stats["gt_chunk_any_hit_rate"] = (
        stats["gt_chunks_any_hit"] / stats["gt_chunks"] if stats["gt_chunks"] else 0.0
    )
    stats["gt_chunk_all_hit_rate"] = (
        stats["gt_chunks_all_hit"] / stats["gt_chunks"] if stats["gt_chunks"] else 0.0
    )
    stats["nonempty_term_map_rate"] = (
        stats["nonempty_term_map_chunks"] / stats["audio_user_chunks"] if stats["audio_user_chunks"] else 0.0
    )
    stats["no_gt_nonempty_term_map_rate"] = (
        stats["no_gt_nonempty_term_map_chunks"] / stats["no_gt_chunks"] if stats["no_gt_chunks"] else 0.0
    )
    stats["avg_term_map_entries_per_chunk"] = (
        stats["term_map_entries_total"] / stats["audio_user_chunks"] if stats["audio_user_chunks"] else 0.0
    )
    stats["p50_term_map_entries"] = _percentile([float(x) for x in termmap_sizes], 50)
    stats["p90_term_map_entries"] = _percentile([float(x) for x in termmap_sizes], 90)
    stats["p99_term_map_entries"] = _percentile([float(x) for x in termmap_sizes], 99)
    stats["score_p50"] = _percentile(score_values, 50)
    stats["score_p90"] = _percentile(score_values, 90)
    stats["score_p99"] = _percentile(score_values, 99)
    stats["elapsed_sec"] = round(time.time() - t0, 3)

    for bucket_stats in stats["duration_buckets"].values():
        chunks = max(1, int(bucket_stats["chunks"]))
        gt_terms = max(1, int(bucket_stats["gt_terms"]))
        gt_chunks = max(1, int(bucket_stats["gt_chunks"]))
        no_gt_chunks = max(1, int(bucket_stats["no_gt_chunks"]))
        bucket_stats["avg_term_map_entries"] = bucket_stats["term_map_entries"] / chunks
        bucket_stats["gt_term_recall"] = bucket_stats["gt_hits"] / gt_terms if bucket_stats["gt_terms"] else 0.0
        bucket_stats["gt_chunk_rate"] = bucket_stats["gt_chunks"] / chunks
        bucket_stats["gt_chunk_any_hit_rate"] = (
            bucket_stats["gt_chunks_any_hit"] / gt_chunks if bucket_stats["gt_chunks"] else 0.0
        )
        bucket_stats["gt_chunk_all_hit_rate"] = (
            bucket_stats["gt_chunks_all_hit"] / gt_chunks if bucket_stats["gt_chunks"] else 0.0
        )
        bucket_stats["no_gt_nonempty_term_map_rate"] = (
            bucket_stats["no_gt_nonempty_term_maps"] / no_gt_chunks
            if bucket_stats["no_gt_chunks"] else 0.0
        )
        bucket_stats["nonempty_term_map_rate"] = bucket_stats["nonempty_term_maps"] / chunks
        bucket_stats["gt_chunks_denominator"] = gt_chunks

    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sample_json:
        args.sample_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.sample_chunks_json:
        args.sample_chunks_json.write_text(
            json.dumps(sample_chunks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    _log(f"Wrote dataset: {args.output_jsonl}")
    _log(f"Wrote stats: {args.stats_json}")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--stats-json", type=Path)
    parser.add_argument("--sample-json", type=Path)
    parser.add_argument("--sample-chunks-json", type=Path)
    parser.add_argument("--glossary-json", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--text-index-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lang-code", choices=sorted(SYSTEM_PROMPT_BY_LANG), default="zh")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.73)
    parser.add_argument("--lookback-sec", type=float, default=1.92)
    parser.add_argument(
        "--min-context-sec",
        type=float,
        default=0.0,
        help="If >0, chunks whose retriever context is shorter than this use term_map:NONE.",
    )
    parser.add_argument(
        "--max-context-sec",
        type=float,
        default=0.0,
        help="If >0, cap retriever context to this many seconds ending at the current chunk end.",
    )
    parser.add_argument("--merge-multiplier-min", type=int, default=0)
    parser.add_argument("--merge-multiplier-max", type=int, default=0)
    parser.add_argument("--audio-batch-size", type=int, default=4)
    parser.add_argument("--text-encode-batch", type=int, default=256)
    parser.add_argument("--max-conversations", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--sample-chunk-count", type=int, default=80)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--drop-bad-rows", action="store_true")
    parser.add_argument("--allow-copy-translation-fallback", action="store_true")
    parser.add_argument("--build-index-only", action="store_true")
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.audio_batch_size <= 0:
        raise ValueError("--audio-batch-size must be positive")
    if not args.build_index_only:
        for name in ("input_jsonl", "output_jsonl", "stats_json"):
            if getattr(args, name) is None:
                raise ValueError(f"--{name.replace('_', '-')} is required unless --build-index-only")
    return args


def main() -> None:
    args = parse_args()
    if args.build_index_only:
        build_text_index(args)
    else:
        build_dataset(args)


if __name__ == "__main__":
    main()
