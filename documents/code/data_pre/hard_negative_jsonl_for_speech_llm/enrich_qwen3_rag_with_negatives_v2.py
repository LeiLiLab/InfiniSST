# ======Configuration=====
TERM_MAP_MAX_SIZE = 20
DEFAULT_MAX_NEG_PER_SEC = 4.5
DEFAULT_RAG_EVAL_MODE = "intersection"
DEFAULT_TTS_EMBEDDING_BATCH_SIZE = 32
DEFAULT_TTS_MAX_PROTOTYPES_PER_TERM = 8
DEFAULT_TTS_SIMILARITY_TOP_K = 10
DEFAULT_DEBUG_LOG_EVERY_N_FLUSHES = 20
DEFAULT_DEBUG_PER_CHUNK_LOG_LIMIT = 30
SUMMARY_FLOAT_ROUND_DIGITS = 4
# ======Configuration=====

import argparse
import json
import math
import os
import random
from typing import List, Dict, Optional, Tuple

import soundfile as sf
import torch
from tqdm import tqdm

# Import the retriever class
from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4

def generate_term_map_string(terms: List[Dict], target_lang_code: str) -> str:
    """
    Format the term list as a term_map string.
    """
    if not terms:
        return "term_map:NONE"
    
    seen = set()
    unique_terms = []
    for t in terms:
        # Use lower-cased term as key for deduplication
        key = t.get('key') or t.get('term', '').lower()
        if not key:
            continue
        if key not in seen:
            unique_terms.append(t)
            seen.add(key)
    
    if not unique_terms:
        return "term_map:NONE"
        
    lines = ["term_map:"]
    for t in unique_terms:
        term = t.get('term', '')
        # Handle different potential keys for translation
        translation = t.get("translation") or t.get(target_lang_code, "") or t.get("zh", "")
        if term and translation:
            lines.append(f"{term}={translation}")
            
    if len(lines) == 1:
        return "term_map:NONE"
        
    return "\n".join(lines)

def _pick_gt_translation(gt: dict, target_lang_code: str) -> str:
    v = gt.get(target_lang_code)
    if v:
        return str(v)
    v = gt.get("translation")
    if v:
        return str(v)
    v = gt.get("zh")
    if v:
        return str(v)
    return ""


def process_shard(
    input_path: str,
    output_path: str,
    index_path: str,
    model_path: str,
    target_lang_code: str,
    gpu_id: int,
    total_gpus: int,
    top_k: int,
    score_threshold: float,
    max_neg_per_sec: float,
    window_batch_size: int,
    rag_eval_mode: str,
    tts_terms_npy_path: str,
    tts_wav_dir: str,
    tts_embeddings_cache: str,
    tts_embedding_batch_size: int,
    tts_max_prototypes_per_term: int,
    tts_similarity_top_k: int,
    debug_log_every_n_flushes: int,
    debug_per_chunk_log_limit: int,
    no_random_neg: bool = False,
):
    device = "cuda:0"
    print(f"[INFO] Init retriever on {device} | gpu_id={gpu_id} total_gpus={total_gpus}")

    retriever = StreamingQwen3RAGRetrieverV4(
        index_path=index_path,
        model_path=model_path,
        device=device,
        lora_r=32,
        lora_alpha=64,
        text_lora_r=16,
        top_k=top_k,
        voting_k=top_k,
        score_threshold=score_threshold,
        rag_eval_mode=rag_eval_mode,
        tts_terms_npy_path=tts_terms_npy_path,
        tts_wav_dir=tts_wav_dir,
        tts_embeddings_cache=tts_embeddings_cache,
        tts_embedding_batch_size=tts_embedding_batch_size,
        tts_max_prototypes_per_term=tts_max_prototypes_per_term,
        tts_similarity_top_k=tts_similarity_top_k,
        verbose=False,
    )

    print(
        "[INFO] Retrieval mode="
        f"{rag_eval_mode} tts_terms_enabled="
        f"{bool(tts_terms_npy_path and tts_wav_dir)} "
        f"tts_terms_npy_path={tts_terms_npy_path or '<empty>'} "
        f"tts_wav_dir={tts_wav_dir or '<empty>'}"
    )

    def _safe_load_audio_16k(path: str) -> Optional[Tuple[torch.Tensor, float]]:
        if not os.path.exists(path):
            return None
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"[WARN] Failed reading audio {path}: {e}")
            return None
        if audio is None:
            return None
        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype("float32", copy=False)
        # Assume inputs are already 16k wav clips in this pipeline; avoid resampling overhead.
        if sr != 16000:
            print(f"[WARN] Unexpected sample rate {sr} for {path}, expected 16000. Skipping.")
            return None
        duration = float(len(audio)) / float(sr) if sr > 0 else 0.0
        return torch.from_numpy(audio), duration

    def _make_windows(audio_1d, chunk_samples: int, hop_samples: int) -> List:
        n = len(audio_1d)
        if n <= 0:
            return []
        windows = []
        start = 0
        while start + chunk_samples <= n:
            windows.append(audio_1d[start:start + chunk_samples])
            start += hop_samples
        # force-end pad (match retriever force_process behavior)
        if start < n:
            tail = audio_1d[start:]
            if len(tail) < chunk_samples:
                tail = torch.nn.functional.pad(tail, (0, chunk_samples - len(tail)))
            else:
                tail = tail[:chunk_samples]
            windows.append(tail)
        return windows

    # We batch windows across many audios to increase GPU utilization.
    pending_windows: List[torch.Tensor] = []
    pending_jobs: List[Tuple[str, int, List[Dict], set, float]] = []
    # job tuple:
    # (instance_key, message_index_in_instance, formatted_gt, gt_keys, duration)
    pending_window_ranges: List[Tuple[int, int]] = []
    pending_instances: Dict[str, dict] = {}
    pending_instance_remaining: Dict[str, int] = {}
    pending_instance_total: Dict[str, int] = {}
    pending_instance_done: Dict[str, int] = {}
    flush_counter = 0
    chunk_log_counter = 0
    summary_stats: Dict[str, float] = {
        "instances_seen": 0.0,
        "instances_with_audio": 0.0,
        "audio_messages_seen": 0.0,
        "audio_read_failed": 0.0,
        "audio_no_windows": 0.0,
        "chunks_with_retrieval": 0.0,
        "chunks_with_candidates": 0.0,
        "chunks_with_neg_pool": 0.0,
        "chunks_with_selected_negatives": 0.0,
        "sum_duration_sec": 0.0,
        "sum_gt_terms": 0.0,
        "sum_candidate_terms": 0.0,
        "sum_negative_pool_terms": 0.0,
        "sum_random_neg_target": 0.0,
        "sum_selected_negatives": 0.0,
        "sum_final_term_map_size": 0.0,
        "sum_max_negatives_cap": 0.0,
        "retrieval_windows_total": 0.0,
        "retrieval_avg_text_terms_x_windows": 0.0,
        "retrieval_avg_tts_terms_x_windows": 0.0,
        "retrieval_avg_overlap_terms_x_windows": 0.0,
        "retrieval_avg_final_terms_x_windows": 0.0,
    }

    def _maybe_write_instance(instance_key: str, f_out):
        total = pending_instance_total.get(instance_key)
        if total is None:
            return
        remaining = pending_instance_remaining.get(instance_key, 0)
        done = pending_instance_done.get(instance_key, 0)
        if remaining == 0 and done >= total:
            instance = pending_instances.get(instance_key)
            if instance is not None:
                f_out.write(json.dumps(instance, ensure_ascii=False) + "\n")
            pending_instances.pop(instance_key, None)
            pending_instance_remaining.pop(instance_key, None)
            pending_instance_total.pop(instance_key, None)
            pending_instance_done.pop(instance_key, None)

    def _flush_batch(f_out):
        nonlocal pending_windows, pending_jobs, pending_window_ranges, pending_instances, pending_instance_remaining, pending_instance_total, pending_instance_done, flush_counter, chunk_log_counter
        if not pending_windows:
            return
        flush_counter += 1
        queued_windows = len(pending_windows)
        queued_jobs = len(pending_jobs)
        # Run batch retrieval once
        window_res_list = retriever.retrieve_windows([w.detach().cpu().numpy() for w in pending_windows])
        retrieve_stats = getattr(retriever, "last_retrieve_stats", {})
        retrieved_windows = int(retrieve_stats.get("windows", 0) or 0)
        if retrieved_windows > 0:
            summary_stats["retrieval_windows_total"] += float(retrieved_windows)
            summary_stats["retrieval_avg_text_terms_x_windows"] += float(retrieve_stats.get("avg_text_terms", 0.0)) * float(retrieved_windows)
            summary_stats["retrieval_avg_tts_terms_x_windows"] += float(retrieve_stats.get("avg_tts_terms", 0.0)) * float(retrieved_windows)
            summary_stats["retrieval_avg_overlap_terms_x_windows"] += float(retrieve_stats.get("avg_overlap_terms", 0.0)) * float(retrieved_windows)
            summary_stats["retrieval_avg_final_terms_x_windows"] += float(retrieve_stats.get("avg_final_terms", 0.0)) * float(retrieved_windows)
        if (
            flush_counter <= 3
            or (debug_log_every_n_flushes > 0 and flush_counter % debug_log_every_n_flushes == 0)
        ):
            print(
                "[DEBUG] Flush="
                f"{flush_counter} queued_windows={queued_windows} queued_jobs={queued_jobs} "
                f"mode={retrieve_stats.get('mode', 'unknown')} "
                f"tts_enabled={retrieve_stats.get('tts_enabled', False)} "
                f"avg_text_terms={retrieve_stats.get('avg_text_terms', 0.0):.2f} "
                f"avg_tts_terms={retrieve_stats.get('avg_tts_terms', 0.0):.2f} "
                f"avg_overlap_terms={retrieve_stats.get('avg_overlap_terms', 0.0):.2f} "
                f"avg_final_terms={retrieve_stats.get('avg_final_terms', 0.0):.2f}"
            )

        # For each pending audio, aggregate max score per term across its windows
        for (instance_key, msg_idx, formatted_gt, gt_keys, duration), (w_s, w_e) in zip(pending_jobs, pending_window_ranges):
            instance = pending_instances.get(instance_key)
            if instance is None:
                continue
            term_scores: Dict[str, float] = {}
            for wr in window_res_list[w_s:w_e]:
                for term_lc, score in wr.items():
                    prev = term_scores.get(term_lc)
                    if prev is None or score > prev:
                        term_scores[term_lc] = score

            all_candidates = []
            for term_lc, score in term_scores.items():
                term_info = retriever.term_map.get(term_lc)
                if not term_info:
                    continue
                tt = term_info.get("target_translations") or {}
                translation = tt.get(target_lang_code) or tt.get("zh") or ""
                all_candidates.append({
                    "key": term_lc,
                    "term": term_info.get("term", ""),
                    "translation": translation,
                    "score": float(score),
                })

            all_candidates.sort(key=lambda x: x["score"], reverse=True)
            negatives = [c for c in all_candidates if c.get("key") not in gt_keys]

            neg_upper = max(0, int(math.ceil(duration * max_neg_per_sec)))
            gt_count = len(formatted_gt)
            if gt_count > 0:
                max_negatives = max(0, TERM_MAP_MAX_SIZE - gt_count)
            else:
                max_negatives = TERM_MAP_MAX_SIZE
            if no_random_neg:
                num_negatives = min(neg_upper, max_negatives)
            else:
                num_negatives = random.randint(0, min(neg_upper, max_negatives))
            selected_negatives = negatives[:num_negatives]
            combined_terms = formatted_gt + selected_negatives
            random.shuffle(combined_terms)
            summary_stats["chunks_with_retrieval"] += 1.0
            summary_stats["sum_duration_sec"] += float(duration)
            summary_stats["sum_gt_terms"] += float(gt_count)
            summary_stats["sum_candidate_terms"] += float(len(all_candidates))
            summary_stats["sum_negative_pool_terms"] += float(len(negatives))
            summary_stats["sum_random_neg_target"] += float(num_negatives)
            summary_stats["sum_selected_negatives"] += float(len(selected_negatives))
            summary_stats["sum_final_term_map_size"] += float(len(combined_terms))
            summary_stats["sum_max_negatives_cap"] += float(max_negatives)
            if len(all_candidates) > 0:
                summary_stats["chunks_with_candidates"] += 1.0
            if len(negatives) > 0:
                summary_stats["chunks_with_neg_pool"] += 1.0
            if len(selected_negatives) > 0:
                summary_stats["chunks_with_selected_negatives"] += 1.0
            if chunk_log_counter < debug_per_chunk_log_limit:
                print(
                    "[DEBUG] ChunkSelection "
                    f"instance={instance_key} msg_idx={msg_idx} "
                    f"duration_sec={duration:.2f} gt_terms={gt_count} "
                    f"candidate_terms={len(all_candidates)} negative_pool={len(negatives)} "
                    f"neg_target={num_negatives} no_random_neg={no_random_neg} "
                    f"selected_negatives={len(selected_negatives)} "
                    f"final_term_map_size={len(combined_terms)}"
                )
                chunk_log_counter += 1

            term_map_str = generate_term_map_string(combined_terms, target_lang_code=target_lang_code)
            instance["messages"][msg_idx]["content"] = f"<audio>\n\n{term_map_str}"

            # Mark this instance job as done; only write once all audio messages completed.
            if instance_key in pending_instance_remaining:
                pending_instance_remaining[instance_key] -= 1
                pending_instance_done[instance_key] = pending_instance_done.get(instance_key, 0) + 1
                _maybe_write_instance(instance_key, f_out)

        pending_windows = []
        pending_jobs = []
        pending_window_ranges = []

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        local_instance_counter = 0
        for idx, line in enumerate(tqdm(f_in, desc=f"Shard {gpu_id}")):
            if total_gpus > 1 and idx % total_gpus != gpu_id:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                instance = json.loads(line)
            except Exception:
                continue
            summary_stats["instances_seen"] += 1.0

            messages = instance.get("messages", []) or []
            instance["messages"] = messages
            audio_paths = instance.get("audios", []) or []
            gt_terms_by_chunk = instance.get("gt_terms_by_chunk", []) or []

            audio_msg_indices: List[int] = []
            for msg_idx, msg in enumerate(messages):
                if msg.get("role") == "user" and "<audio>" in (msg.get("content") or ""):
                    if len(audio_msg_indices) >= len(audio_paths):
                        break
                    audio_msg_indices.append(msg_idx)

            instance_key = f"{gpu_id}:{local_instance_counter}"

            if audio_msg_indices:
                summary_stats["instances_with_audio"] += 1.0
                pending_instances[instance_key] = instance
                pending_instance_remaining[instance_key] = 0
                pending_instance_total[instance_key] = len(audio_msg_indices)
                pending_instance_done[instance_key] = 0

            for audio_idx, msg_idx in enumerate(audio_msg_indices):
                summary_stats["audio_messages_seen"] += 1.0
                msg = messages[msg_idx]
                audio_path = audio_paths[audio_idx]
                gt_terms = gt_terms_by_chunk[audio_idx] if audio_idx < len(gt_terms_by_chunk) else []

                formatted_gt = []
                for gt in gt_terms:
                    if not isinstance(gt, dict):
                        continue
                    term = str(gt.get("term", "")).strip()
                    if not term:
                        continue
                    translation = _pick_gt_translation(gt, target_lang_code).strip()
                    formatted_gt.append({
                        "key": term.lower(),
                        "term": term,
                        "translation": translation,
                    })
                gt_keys = {t["key"] for t in formatted_gt if t.get("key")}

                loaded = _safe_load_audio_16k(audio_path)
                if loaded is None:
                    # No audio: keep only GT
                    summary_stats["audio_read_failed"] += 1.0
                    summary_stats["sum_gt_terms"] += float(len(formatted_gt))
                    summary_stats["sum_final_term_map_size"] += float(len(formatted_gt))
                    term_map_str = generate_term_map_string(formatted_gt, target_lang_code=target_lang_code)
                    msg["content"] = f"<audio>\n\n{term_map_str}"
                    pending_instance_done[instance_key] += 1
                    _maybe_write_instance(instance_key, f_out)
                    continue

                audio_1d, duration = loaded
                windows = _make_windows(audio_1d, retriever.chunk_samples, retriever.hop_samples)
                if not windows:
                    summary_stats["audio_no_windows"] += 1.0
                    summary_stats["sum_gt_terms"] += float(len(formatted_gt))
                    summary_stats["sum_final_term_map_size"] += float(len(formatted_gt))
                    term_map_str = generate_term_map_string(formatted_gt, target_lang_code=target_lang_code)
                    msg["content"] = f"<audio>\n\n{term_map_str}"
                    pending_instance_done[instance_key] += 1
                    _maybe_write_instance(instance_key, f_out)
                    continue

                w_start = len(pending_windows)
                pending_windows.extend(windows)
                w_end = len(pending_windows)
                pending_jobs.append((instance_key, msg_idx, formatted_gt, gt_keys, duration))
                pending_window_ranges.append((w_start, w_end))
                pending_instance_remaining[instance_key] += 1

                # Flush when enough windows accumulated
                if len(pending_windows) >= window_batch_size:
                    _flush_batch(f_out)

            # If there were no audio messages, write through immediately
            if not audio_msg_indices:
                f_out.write(json.dumps(instance, ensure_ascii=False) + "\n")
            else:
                _maybe_write_instance(instance_key, f_out)

            local_instance_counter += 1

        # Flush remaining
        _flush_batch(f_out)

    def _safe_ratio(num: float, den: float) -> float:
        if den <= 0.0:
            return 0.0
        return num / den

    chunks_with_retrieval = summary_stats["chunks_with_retrieval"]
    audio_messages_seen = summary_stats["audio_messages_seen"]
    retrieval_windows_total = summary_stats["retrieval_windows_total"]
    summary_payload = {
        "type": "hard_negative_summary",
        "gpu_id": int(gpu_id),
        "total_gpus": int(total_gpus),
        "output_path": output_path,
        "rag_eval_mode_requested": rag_eval_mode,
        "rag_eval_mode_effective": str(getattr(retriever, "rag_eval_mode", rag_eval_mode)),
        "tts_enabled_effective": bool(getattr(retriever, "_tts_enabled", False)),
        "instances_seen": int(summary_stats["instances_seen"]),
        "instances_with_audio": int(summary_stats["instances_with_audio"]),
        "audio_messages_seen": int(audio_messages_seen),
        "audio_read_failed": int(summary_stats["audio_read_failed"]),
        "audio_no_windows": int(summary_stats["audio_no_windows"]),
        "chunks_with_retrieval": int(chunks_with_retrieval),
        "chunks_with_candidates": int(summary_stats["chunks_with_candidates"]),
        "chunks_with_neg_pool": int(summary_stats["chunks_with_neg_pool"]),
        "chunks_with_selected_negatives": int(summary_stats["chunks_with_selected_negatives"]),
        "avg_duration_sec_per_retrieved_chunk": round(_safe_ratio(summary_stats["sum_duration_sec"], chunks_with_retrieval), SUMMARY_FLOAT_ROUND_DIGITS),
        "avg_gt_terms_per_audio_message": round(_safe_ratio(summary_stats["sum_gt_terms"], audio_messages_seen), SUMMARY_FLOAT_ROUND_DIGITS),
        "avg_candidate_terms_per_retrieved_chunk": round(_safe_ratio(summary_stats["sum_candidate_terms"], chunks_with_retrieval), SUMMARY_FLOAT_ROUND_DIGITS),
        "avg_negative_pool_terms_per_retrieved_chunk": round(_safe_ratio(summary_stats["sum_negative_pool_terms"], chunks_with_retrieval), SUMMARY_FLOAT_ROUND_DIGITS),
        "avg_random_neg_target_per_retrieved_chunk": round(_safe_ratio(summary_stats["sum_random_neg_target"], chunks_with_retrieval), SUMMARY_FLOAT_ROUND_DIGITS),
        "avg_selected_negatives_per_retrieved_chunk": round(_safe_ratio(summary_stats["sum_selected_negatives"], chunks_with_retrieval), SUMMARY_FLOAT_ROUND_DIGITS),
        "avg_final_term_map_size_per_audio_message": round(_safe_ratio(summary_stats["sum_final_term_map_size"], audio_messages_seen), SUMMARY_FLOAT_ROUND_DIGITS),
        "avg_max_negative_cap_per_retrieved_chunk": round(_safe_ratio(summary_stats["sum_max_negatives_cap"], chunks_with_retrieval), SUMMARY_FLOAT_ROUND_DIGITS),
        "selected_negative_chunk_ratio": round(_safe_ratio(summary_stats["chunks_with_selected_negatives"], chunks_with_retrieval), SUMMARY_FLOAT_ROUND_DIGITS),
        "retrieval_windows_total": int(retrieval_windows_total),
        "retrieval_avg_text_terms": round(_safe_ratio(summary_stats["retrieval_avg_text_terms_x_windows"], retrieval_windows_total), SUMMARY_FLOAT_ROUND_DIGITS),
        "retrieval_avg_tts_terms": round(_safe_ratio(summary_stats["retrieval_avg_tts_terms_x_windows"], retrieval_windows_total), SUMMARY_FLOAT_ROUND_DIGITS),
        "retrieval_avg_overlap_terms": round(_safe_ratio(summary_stats["retrieval_avg_overlap_terms_x_windows"], retrieval_windows_total), SUMMARY_FLOAT_ROUND_DIGITS),
        "retrieval_avg_final_terms": round(_safe_ratio(summary_stats["retrieval_avg_final_terms_x_windows"], retrieval_windows_total), SUMMARY_FLOAT_ROUND_DIGITS),
    }
    print(f"[SUMMARY] {json.dumps(summary_payload, ensure_ascii=False)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-gt-jsonl", required=True)
    parser.add_argument("--output-base", required=True, help="Base output path without shard suffix. Will write <base>_gpu{gpu_id}.jsonl")
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-lang-code", default="zh", help="zh/ja/de")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--total-gpus", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--max-neg-per-sec", type=float, default=DEFAULT_MAX_NEG_PER_SEC)
    parser.add_argument("--window-batch-size", type=int, default=512, help="Number of audio windows per batch retrieval.")
    parser.add_argument("--rag-eval-mode", default=DEFAULT_RAG_EVAL_MODE, choices=["text", "tts", "intersection"])
    parser.add_argument("--tts-terms-npy-path", default="")
    parser.add_argument("--tts-wav-dir", default="")
    parser.add_argument("--tts-embeddings-cache", default="", help="Pre-computed TTS embeddings .npz cache (skips wav encoding)")
    parser.add_argument("--tts-embedding-batch-size", type=int, default=DEFAULT_TTS_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--tts-max-prototypes-per-term", type=int, default=DEFAULT_TTS_MAX_PROTOTYPES_PER_TERM)
    parser.add_argument("--tts-similarity-top-k", type=int, default=DEFAULT_TTS_SIMILARITY_TOP_K)
    parser.add_argument("--debug-log-every-n-flushes", type=int, default=DEFAULT_DEBUG_LOG_EVERY_N_FLUSHES)
    parser.add_argument("--debug-per-chunk-log-limit", type=int, default=DEFAULT_DEBUG_PER_CHUNK_LOG_LIMIT)
    parser.add_argument("--no-random-neg", action="store_true", default=False,
                        help="Disable random negative count; use all negatives up to the per-sec / term-map cap.")
    args = parser.parse_args()

    output_path = f"{args.output_base}_gpu{args.gpu_id}.jsonl"
    process_shard(
        input_path=args.input_gt_jsonl,
        output_path=output_path,
        index_path=args.index_path,
        model_path=args.model_path,
        target_lang_code=args.target_lang_code,
        gpu_id=args.gpu_id,
        total_gpus=args.total_gpus,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        max_neg_per_sec=args.max_neg_per_sec,
        window_batch_size=args.window_batch_size,
        rag_eval_mode=args.rag_eval_mode,
        tts_terms_npy_path=args.tts_terms_npy_path,
        tts_wav_dir=args.tts_wav_dir,
        tts_embeddings_cache=args.tts_embeddings_cache,
        tts_embedding_batch_size=args.tts_embedding_batch_size,
        tts_max_prototypes_per_term=args.tts_max_prototypes_per_term,
        tts_similarity_top_k=args.tts_similarity_top_k,
        debug_log_every_n_flushes=args.debug_log_every_n_flushes,
        debug_per_chunk_log_limit=args.debug_per_chunk_log_limit,
        no_random_neg=args.no_random_neg,
    )

if __name__ == "__main__":
    main()

