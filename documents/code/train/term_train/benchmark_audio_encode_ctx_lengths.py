#!/usr/bin/env python3
"""Benchmark Qwen3-Omni retriever audio encode latency for context lengths.

This intentionally does not run the full eval path or initialize WandB.  It
loads the same audio-side retriever configuration used by the referenced eval
launchers, preloads a small set of real ACL chunk WAVs, then times:

  1. pad/truncate + WhisperFeatureExtractor on CPU
  2. host-to-device tensor copy
  3. GPU audio encoder + projection + MaxSim pooling
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from documents.code.train.term_train.qwen3_glossary_neg_train import (  # noqa: E402
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_QWEN_AUDIO_MODEL_ID,
    Qwen3OmniRetriever,
)
from transformers import WhisperFeatureExtractor  # noqa: E402
import torch  # noqa: E402


DEFAULT_CKPT = (
    "/mnt/gemini/home/jiaxuanluo/train_outputs/"
    "q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_"
    "gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep3_"
    "bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt"
)
DEFAULT_JSONL = (
    "/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/"
    "acl6060_dev_dataset.jsonl"
)
DEFAULT_MAXSIM_WINDOWS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24]
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "proj1", "proj2"]


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * pct / 100.0
    lo = int(np.floor(k))
    hi = int(np.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def summarize(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean_ms": float(statistics.mean(values)) if values else float("nan"),
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "min_ms": min(values) if values else float("nan"),
        "max_ms": max(values) if values else float("nan"),
    }


def load_audio_paths(jsonl_path: str, limit: int) -> List[str]:
    paths: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if len(paths) >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = str(row.get("chunk_audio_path") or "").strip()
            if path and os.path.isfile(path):
                paths.append(path)
    if not paths:
        raise RuntimeError(f"No readable chunk_audio_path found in {jsonl_path}")
    return paths


def load_audio_arrays(paths: Sequence[str]) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    for path in paths:
        audio, sr = sf.read(path, dtype="float32")
        if sr != DEFAULT_AUDIO_SAMPLE_RATE:
            raise RuntimeError(f"Expected {DEFAULT_AUDIO_SAMPLE_RATE}Hz, got {sr}: {path}")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        max_abs = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if max_abs > 0:
            audio = audio / max_abs
        arrays.append(audio.astype(np.float32, copy=False))
    return arrays


def fixed_batch(
    audios: Sequence[np.ndarray],
    start: int,
    batch_size: int,
    fixed_samples: int,
) -> List[np.ndarray]:
    batch: List[np.ndarray] = []
    n = len(audios)
    for offset in range(batch_size):
        audio = audios[(start + offset) % n]
        if len(audio) < fixed_samples:
            audio = np.pad(audio, (0, fixed_samples - len(audio)), mode="constant")
        elif len(audio) > fixed_samples:
            audio = audio[:fixed_samples]
        batch.append(audio.astype(np.float32, copy=False))
    return batch


def strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if any(k.startswith("module.") for k in state):
        return {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
    return state


def load_retriever(args: argparse.Namespace, device: torch.device) -> Qwen3OmniRetriever:
    model = Qwen3OmniRetriever(
        model_id=args.audio_model_id,
        target_dim=args.target_dim,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        temperature=args.temperature,
        learn_temp=False,
        pooling_type="transformer",
        use_maxsim=True,
        maxsim_windows=args.maxsim_windows,
        maxsim_stride=args.maxsim_stride,
    ).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(
            strip_module_prefix(ckpt.get("model_state_dict", {})),
            strict=False,
        )
        print(
            f"[LOAD] checkpoint={args.checkpoint} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    model.eval()
    return model


def run_one_length(
    *,
    seconds: float,
    audios: Sequence[np.ndarray],
    feature_extractor: WhisperFeatureExtractor,
    retriever: Qwen3OmniRetriever,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    fixed_samples = int(round(seconds * DEFAULT_AUDIO_SAMPLE_RATE))
    total_iters = args.warmup_batches + args.measure_batches
    feature_ms: List[float] = []
    h2d_ms: List[float] = []
    encode_ms: List[float] = []
    end_to_end_ms: List[float] = []
    output_shape: List[int] = []
    input_feature_shape: List[int] = []

    with torch.inference_mode():
        for batch_idx in range(total_iters):
            base = (batch_idx * args.batch_size) % len(audios)
            t0 = time.perf_counter()
            batch_audio = fixed_batch(audios, base, args.batch_size, fixed_samples)
            t1 = time.perf_counter()
            inputs = feature_extractor(
                batch_audio,
                sampling_rate=DEFAULT_AUDIO_SAMPLE_RATE,
                return_tensors="pt",
                padding=False,
            )
            feats_cpu = inputs.input_features
            feat_lens_cpu = torch.full(
                (feats_cpu.size(0),),
                feats_cpu.size(-1),
                dtype=torch.long,
            )
            t2 = time.perf_counter()

            torch.cuda.synchronize(device)
            h0 = time.perf_counter()
            feats = feats_cpu.to(device, non_blocking=True).to(torch.bfloat16)
            feat_lens = feat_lens_cpu.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            h1 = time.perf_counter()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                torch.cuda.synchronize(device)
                e0 = time.perf_counter()
                out = retriever(feats, feat_lens)
                torch.cuda.synchronize(device)
                e1 = time.perf_counter()

            if batch_idx >= args.warmup_batches:
                feature_ms.append((t2 - t0) * 1000.0)
                h2d_ms.append((h1 - h0) * 1000.0)
                encode_ms.append((e1 - e0) * 1000.0)
                end_to_end_ms.append((e1 - t0) * 1000.0)
            output_shape = list(out.shape)
            input_feature_shape = list(feats_cpu.shape)

    result: Dict[str, Any] = {
        "seconds": seconds,
        "fixed_samples": fixed_samples,
        "batch_size": args.batch_size,
        "measure_batches": args.measure_batches,
        "input_features_shape": input_feature_shape,
        "output_shape": output_shape,
        "feature": summarize(feature_ms),
        "h2d": summarize(h2d_ms),
        "encode": summarize(encode_ms),
        "end_to_end": summarize(end_to_end_ms),
        "encode_samples_per_sec": (
            args.batch_size * 1000.0 / statistics.mean(encode_ms)
            if encode_ms else float("nan")
        ),
        "end_to_end_samples_per_sec": (
            args.batch_size * 1000.0 / statistics.mean(end_to_end_ms)
            if end_to_end_ms else float("nan")
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--audio-model-id", default=DEFAULT_QWEN_AUDIO_MODEL_ID)
    parser.add_argument("--seconds", type=float, nargs="+", default=[1.92, 5.76])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=20)
    parser.add_argument("--target-dim", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-target-modules", nargs="+", default=DEFAULT_LORA_TARGETS)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--maxsim-windows", type=int, nargs="+", default=DEFAULT_MAXSIM_WINDOWS)
    parser.add_argument("--maxsim-stride", type=int, default=2)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda:0")
    print(f"[ENV] cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"[ENV] device={torch.cuda.get_device_name(device)}")
    print(f"[DATA] jsonl={args.jsonl} num_samples={args.num_samples}")
    paths = load_audio_paths(args.jsonl, args.num_samples)
    audios = load_audio_arrays(paths)
    print(f"[DATA] loaded_audio={len(audios)} first={paths[0]}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    retriever = load_retriever(args, device)

    results: List[Dict[str, Any]] = []
    for seconds in args.seconds:
        print(f"[BENCH] seconds={seconds} batch_size={args.batch_size}", flush=True)
        result = run_one_length(
            seconds=seconds,
            audios=audios,
            feature_extractor=feature_extractor,
            retriever=retriever,
            device=device,
            args=args,
        )
        results.append(result)
        print(
            "[RESULT] "
            f"sec={seconds:.2f} "
            f"feat_mean={result['feature']['mean_ms']:.2f}ms "
            f"h2d_mean={result['h2d']['mean_ms']:.2f}ms "
            f"encode_mean={result['encode']['mean_ms']:.2f}ms "
            f"encode_p50={result['encode']['p50_ms']:.2f}ms "
            f"encode_p90={result['encode']['p90_ms']:.2f}ms "
            f"e2e_mean={result['end_to_end']['mean_ms']:.2f}ms "
            f"encode_sps={result['encode_samples_per_sec']:.2f} "
            f"shape={result['input_features_shape']}->{result['output_shape']}",
            flush=True,
        )

    by_sec = {round(r["seconds"], 4): r for r in results}
    if 1.92 in by_sec and 5.76 in by_sec:
        base = by_sec[1.92]
        long = by_sec[5.76]
        ratio = long["encode"]["mean_ms"] / base["encode"]["mean_ms"]
        e2e_ratio = long["end_to_end"]["mean_ms"] / base["end_to_end"]["mean_ms"]
        print(
            f"[RATIO] 5.76_vs_1.92 encode_mean={ratio:.3f}x "
            f"end_to_end_mean={e2e_ratio:.3f}x",
            flush=True,
        )

    payload = {
        "jsonl": args.jsonl,
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "num_samples": len(audios),
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "maxsim_windows": args.maxsim_windows,
        "maxsim_stride": args.maxsim_stride,
        "results": results,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fout:
            json.dump(payload, fout, indent=2, sort_keys=True)
        print(f"[WRITE] {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
