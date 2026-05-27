#!/usr/bin/env python3
"""
Compute TCR (Term Copy Rate) and FCR (False Copy Rate) for Speech LLM evaluation.

Can run in two modes:
  1. With vLLM inference (--run_inference): load model, generate translations, compute metrics
  2. Metrics-only (--predictions_dir): compute metrics from saved predictions

Strategies:
  - baseline:    always inject top-10 terms
  - chunk_gate:  inject top-10 if gap >= delta, else nothing
  - term_filter: keep terms with score >= tau

TCR = |GT terms whose zh appears in translation| / |GT terms in term_map|
FCR = |neg terms whose zh appears in translation| / |neg terms in term_map|
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ======Configuration=====
SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter. "
    "You will be given chunks of English audio and you need to translate the audio into Chinese text. "
    "Use the 'term_map' as a reference for terminology if provided."
)

CHUNK_GATE_DELTA = 0.09
TERM_FILTER_TAU = 0.65
TOP_K = 10
# ======Configuration=====


def _log(msg: str) -> None:
    print(f"[EVAL] {msg}", flush=True)


def load_eval_chunks(jsonl_path: str) -> List[Dict]:
    chunks = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_term_map_str(terms: List[Dict]) -> str:
    if not terms:
        return ""
    lines = ["term_map:"]
    seen = set()
    for t in terms:
        key = t["term"].lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{t['term']}={t['zh']}")
    return "\n".join(lines)


def apply_strategy(chunk: Dict, strategy: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Apply retrieval strategy. Returns (injected_terms, gt_terms_in_map).
    gt_terms_in_map: GT terms that ended up in the term_map (for TCR denominator).
    """
    retriever_results = chunk.get("retriever_top10", [])
    gt_terms = chunk.get("gt_terms", [])
    gt_keys = {t["term"].lower() for t in gt_terms}

    usable_results = [r for r in retriever_results if r.get("zh", "")]

    if strategy == "baseline":
        injected = usable_results[:TOP_K]

    elif strategy == "chunk_gate":
        if len(usable_results) >= 2:
            top1_score = usable_results[0]["score"]
            mean_rest = sum(r["score"] for r in usable_results[1:TOP_K]) / max(1, len(usable_results[1:TOP_K]))
            gap = top1_score - mean_rest
        else:
            gap = 0.0
        if gap >= CHUNK_GATE_DELTA:
            injected = usable_results[:TOP_K]
        else:
            injected = []

    elif strategy == "term_filter":
        injected = [r for r in usable_results[:TOP_K] if r["score"] >= TERM_FILTER_TAU]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    injected_keys = {t["term"].lower() for t in injected}
    gt_in_map = [t for t in gt_terms if t["term"].lower() in injected_keys]

    return injected, gt_in_map


def check_term_in_translation(zh_term: str, translation: str) -> bool:
    """Check if Chinese term appears in the translation."""
    if not zh_term or not translation:
        return False
    return zh_term in translation


def compute_metrics(
    chunks: List[Dict],
    predictions: Dict[str, str],
    strategy: str,
) -> Dict:
    """Compute TCR/FCR for one strategy."""
    gt_total = 0
    gt_correct = 0
    neg_total = 0
    neg_false_copy = 0

    chunks_with_termmap = 0
    chunks_empty_termmap = 0
    chunks_with_gt_in_map = 0

    for chunk in chunks:
        cid = chunk["chunk_id"]
        translation = predictions.get(cid, "")

        injected, gt_in_map = apply_strategy(chunk, strategy)
        gt_keys = {t["term"].lower() for t in chunk.get("gt_terms", [])}

        if not injected:
            chunks_empty_termmap += 1
            continue
        chunks_with_termmap += 1

        if gt_in_map:
            chunks_with_gt_in_map += 1

        for t in injected:
            is_gt = t["term"].lower() in gt_keys
            if is_gt:
                gt_total += 1
                if check_term_in_translation(t["zh"], translation):
                    gt_correct += 1
            else:
                neg_total += 1
                if check_term_in_translation(t["zh"], translation):
                    neg_false_copy += 1

    tcr = gt_correct / max(1, gt_total)
    fcr = neg_false_copy / max(1, neg_total)

    return {
        "strategy": strategy,
        "TCR": round(tcr, 4),
        "FCR": round(fcr, 4),
        "gt_correct": gt_correct,
        "gt_total": gt_total,
        "neg_false_copy": neg_false_copy,
        "neg_total": neg_total,
        "chunks_with_termmap": chunks_with_termmap,
        "chunks_empty_termmap": chunks_empty_termmap,
        "chunks_with_gt_in_map": chunks_with_gt_in_map,
    }


# ---------------------------------------------------------------------------
# vLLM inference
# ---------------------------------------------------------------------------

def run_vllm_inference(
    chunks: List[Dict],
    model_path: str,
    strategies: List[str],
    output_dir: str,
    tp_size: int = 2,
) -> Dict[str, Dict[str, str]]:
    """Run vLLM inference for all strategies. Returns {strategy: {chunk_id: translation}}."""
    from vllm import LLM, SamplingParams
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeProcessor

    _log(f"Loading vLLM model from {model_path} with tp={tp_size}")
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=tp_size,
        limit_mm_per_prompt={"audio": 1},
        max_num_seqs=1,
        max_model_len=4096,
        enforce_eager=True,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )
    _log("Model loaded.")

    all_predictions: Dict[str, Dict[str, str]] = {}

    for strategy in strategies:
        _log(f"--- Strategy: {strategy} ---")
        predictions: Dict[str, str] = {}
        t_start = time.time()

        for i, chunk in enumerate(chunks):
            cid = chunk["chunk_id"]
            apath = chunk["audio_path"]

            if not os.path.isfile(apath):
                predictions[cid] = ""
                continue

            injected, _ = apply_strategy(chunk, strategy)
            termmap_str = build_term_map_str(injected)

            user_text_parts = []
            if termmap_str:
                user_text_parts.append(f"\n\n{termmap_str}")
            user_content = [
                {"type": "audio", "audio": apath},
            ]
            if user_text_parts:
                user_content.append({"type": "text", "text": user_text_parts[0]})

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            audios, _, _ = process_mm_info(messages, use_audio_in_video=False)

            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"audio": audios},
                "mm_processor_kwargs": {"use_audio_in_video": False},
            }

            try:
                outputs = model.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
                translation = outputs[0].outputs[0].text.strip()
            except Exception as e:
                _log(f"  WARN: inference failed for {cid}: {e}")
                translation = ""

            predictions[cid] = translation

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                _log(f"  {i+1}/{len(chunks)} chunks, {elapsed:.0f}s")

        elapsed = time.time() - t_start
        _log(f"  Strategy {strategy}: {len(predictions)} chunks, {elapsed:.0f}s")

        pred_path = os.path.join(output_dir, f"predictions_{strategy}.jsonl")
        with open(pred_path, "w") as f:
            for cid, trans in predictions.items():
                f.write(json.dumps({"chunk_id": cid, "translation": trans}, ensure_ascii=False) + "\n")
        _log(f"  Saved: {pred_path}")

        all_predictions[strategy] = predictions

    return all_predictions


def load_predictions(output_dir: str, strategy: str) -> Dict[str, str]:
    pred_path = os.path.join(output_dir, f"predictions_{strategy}.jsonl")
    predictions = {}
    with open(pred_path) as f:
        for line in f:
            d = json.loads(line.strip())
            predictions[d["chunk_id"]] = d["translation"]
    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global CHUNK_GATE_DELTA, TERM_FILTER_TAU

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_jsonl", required=True,
                        help="Eval JSONL from eval_prepare_retriever.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strategies", nargs="+", default=["baseline", "chunk_gate", "term_filter"])
    parser.add_argument("--chunk_gate_delta", type=float, default=0.09)
    parser.add_argument("--term_filter_tau", type=float, default=0.65)

    parser.add_argument("--run_inference", action="store_true",
                        help="Run vLLM inference (requires GPU + Docker)")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--tp_size", type=int, default=2)

    parser.add_argument("--predictions_dir", type=str, default="",
                        help="Load predictions from dir (metrics-only mode)")
    args = parser.parse_args()

    CHUNK_GATE_DELTA = args.chunk_gate_delta
    TERM_FILTER_TAU = args.term_filter_tau

    os.makedirs(args.output_dir, exist_ok=True)

    _log(f"Loading eval data from {args.eval_jsonl}")
    chunks = load_eval_chunks(args.eval_jsonl)
    _log(f"Loaded {len(chunks)} chunks")

    has_gt = sum(1 for c in chunks if c.get("gt_terms"))
    _log(f"  with GT terms: {has_gt}, without: {len(chunks) - has_gt}")
    _log(f"  Strategies: {args.strategies}")
    _log(f"  chunk_gate delta={CHUNK_GATE_DELTA}, term_filter tau={TERM_FILTER_TAU}")

    if args.run_inference:
        assert args.model_path, "Must provide --model_path for inference"
        all_predictions = run_vllm_inference(
            chunks, args.model_path, args.strategies, args.output_dir, args.tp_size
        )
    elif args.predictions_dir:
        all_predictions = {}
        for strategy in args.strategies:
            all_predictions[strategy] = load_predictions(args.predictions_dir, strategy)
    else:
        raise ValueError("Must use either --run_inference or --predictions_dir")

    _log("Computing TCR/FCR metrics...")
    results = []
    for strategy in args.strategies:
        predictions = all_predictions[strategy]
        metrics = compute_metrics(chunks, predictions, strategy)
        results.append(metrics)
        _log(
            f"  {strategy:15s} | TCR={metrics['TCR']:.4f} ({metrics['gt_correct']}/{metrics['gt_total']}) | "
            f"FCR={metrics['FCR']:.4f} ({metrics['neg_false_copy']}/{metrics['neg_total']}) | "
            f"chunks_w_map={metrics['chunks_with_termmap']} empty={metrics['chunks_empty_termmap']}"
        )

    results_path = os.path.join(args.output_dir, "tcr_fcr_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    _log(f"Results saved: {results_path}")

    print("\n" + "=" * 80)
    print(f"{'Strategy':15s} | {'TCR':>8s} | {'FCR':>8s} | {'GT_hit':>8s} | {'GT_total':>8s} | {'NEG_fc':>8s} | {'NEG_total':>8s}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['strategy']:15s} | {r['TCR']:>8.4f} | {r['FCR']:>8.4f} | "
            f"{r['gt_correct']:>8d} | {r['gt_total']:>8d} | "
            f"{r['neg_false_copy']:>8d} | {r['neg_total']:>8d}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
