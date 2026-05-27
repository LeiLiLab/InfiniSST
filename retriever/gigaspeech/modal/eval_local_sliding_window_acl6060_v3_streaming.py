#!/usr/bin/env python3
"""
Streaming Sliding Window Evaluation Script for ACL6060 Dataset

Simulates the streaming behavior of infinisst_omni_vllm_rag.py:
- Process audio in fixed-duration cycles (default: 1.92s per vLLM call)
- Within each cycle: sliding window -> max pooling -> top-N filtering
- Each cycle is INDEPENDENT (terms are NOT accumulated across cycles)
- top-N = ceil(cycle_duration * terms_per_second) = ceil(1.92 * 2.5) = 5

This differs from v2 which does top-N filtering on the ENTIRE segment.
"""

import os
import sys
import argparse
import json
import torch
import faiss
import pickle
import re
import glob
import math
import numpy as np
from tqdm import tqdm
from datetime import datetime
import random

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import model components
from Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder,
    Qwen2AudioTextEncoder,
    ContrastiveQwen2AudioModel,
)


def load_audio_from_wav(wav_path, target_sr=16000):
    """Load audio from wav file."""
    try:
        import soundfile as sf
        import librosa
        
        audio, sr = sf.read(wav_path)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)
            
    except Exception as e:
        print(f"[WARN] Failed to load audio {wav_path}: {e}")
        return None


def load_text_lines(txt_path):
    """Load text lines from plain text file."""
    results = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            results.append(line)
    return results


from flashtext import KeywordProcessor

def extract_gt_terms_from_text(text, glossary_terms, min_words=1, min_chars=3):
    """Extract GT terms from text using FlashText matching against glossary."""
    keyword_processor = KeywordProcessor(case_sensitive=False)
    
    for term in glossary_terms:
        word_count = len(term.split())
        if len(term) >= min_chars and word_count >= min_words:
            keyword_processor.add_keyword(term.lower())
    
    found_keywords = keyword_processor.extract_keywords(text)
    return set(k.lower() for k in found_keywords)


def load_wav_files_sorted(wav_dir):
    """Load wav file paths sorted by numeric sent_id."""
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    
    def extract_number(filename):
        basename = os.path.basename(filename)
        match = re.search(r'sent_(\d+)\.wav', basename)
        if match:
            return int(match.group(1))
        return 0
    
    wav_files = sorted(wav_files, key=extract_number)
    return wav_files


def create_sliding_window_chunks(audio, chunk_size=2.0, hop_size=1.0, sample_rate=16000):
    """Create sliding window chunks from audio."""
    chunk_samples = int(chunk_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)
    
    chunks = []
    start = 0
    
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        
        # Pad if necessary
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
        
        chunks.append(chunk)
        start += hop_samples
        
        if start >= len(audio):
            break
    
    return chunks


def encode_audio_batch(model, audio_tensors, device):
    """Encode a batch of audio tensors."""
    with torch.no_grad():
        processed = [torch.from_numpy(a).float().to(device) for a in audio_tensors]
        embeddings = model.encode_audio(processed)
        return embeddings.detach().cpu().float().numpy()


def l2_distance_to_score(distance: float) -> float:
    """Convert FAISS L2 distance into a score where higher is better."""
    d = float(distance)
    return 1.0 / (1.0 + d)


def retrieve_terms_for_chunks(model, retriever, chunks, device, top_k=5, batch_size=32):
    """
    Retrieve terms for all chunks and aggregate results using max score pooling.
    
    Returns:
        Dict[str, float]: term(lowercase) -> max_score across all windows
    """
    term2max_score = {}
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        try:
            audio_embs = encode_audio_batch(model, batch_chunks, device)
            
            for emb in audio_embs:
                emb = emb.reshape(1, -1)
                D, I = retriever.index.search(emb, top_k)
                
                for idx, dist in zip(I[0], D[0]):
                    if 0 <= idx < len(retriever.term_list):
                        term_entry = retriever.term_list[idx]
                        term = term_entry['term'] if isinstance(term_entry, dict) else term_entry
                        term_lc = str(term).lower()
                        score = l2_distance_to_score(dist)
                        prev = term2max_score.get(term_lc)
                        if prev is None or score > prev:
                            term2max_score[term_lc] = score
        except Exception as e:
            print(f"[WARN] Failed to process batch: {e}")
            continue
    
    return term2max_score


def filter_top_n(term2score: dict, n: int, threshold: float = 0.0) -> dict:
    """
    Filter to top-N terms by score, with optional threshold.
    
    Args:
        term2score: Dict of term -> score
        n: Maximum number of terms to keep
        threshold: Minimum score threshold (default: 0.0)
    
    Returns:
        Dict of top-N terms with score >= threshold
    """
    # First filter by threshold
    filtered = {t: s for t, s in term2score.items() if s >= threshold}
    
    # Then take top-N
    n = max(1, n)
    sorted_terms = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_terms[:n])


def process_segment_streaming(
    model, 
    retriever, 
    audio, 
    device, 
    cycle_duration=1.92,
    chunk_size=2.0,
    hop_size=1.0,
    top_k=5,
    terms_per_second=2.5,
    threshold=0.5,
    batch_size=32,
    sample_rate=16000,
):
    """
    Process a segment using streaming simulation.
    
    Simulates the behavior of infinisst_omni_vllm_rag.py:
    - Split audio into cycles of `cycle_duration` seconds
    - Within each cycle: sliding window -> max pooling -> threshold filter -> top-N
    - Each cycle is INDEPENDENT (no accumulation)
    
    Args:
        threshold: Minimum score threshold to filter terms before top-N
    
    Returns:
        List[Dict]: Per-cycle results, each containing:
            - cycle_idx: Cycle index
            - cycle_start/end: Time range
            - terms: Top-N terms for this cycle (after threshold filtering)
            - all_terms: All retrieved terms before filtering
    """
    cycle_samples = int(cycle_duration * sample_rate)
    top_n = math.ceil(cycle_duration * terms_per_second)  # Fixed top-N per cycle
    
    results = []
    cycle_idx = 0
    start = 0
    
    while start < len(audio):
        end = min(start + cycle_samples, len(audio))
        cycle_audio = audio[start:end]
        cycle_duration_actual = len(cycle_audio) / sample_rate
        
        # Create sliding window chunks within this cycle
        chunks = create_sliding_window_chunks(
            cycle_audio, 
            chunk_size=chunk_size, 
            hop_size=hop_size, 
            sample_rate=sample_rate
        )
        
        if len(chunks) == 0:
            start = end
            cycle_idx += 1
            continue
        
        # Retrieve terms with max pooling
        all_terms = retrieve_terms_for_chunks(
            model, retriever, chunks, device,
            top_k=top_k, batch_size=batch_size
        )
        
        # Apply threshold filter + top-N
        # Key: If no terms pass threshold, return empty dict (avoid poison)
        filtered_terms = filter_top_n(all_terms, top_n, threshold=threshold)
        
        results.append({
            'cycle_idx': cycle_idx,
            'cycle_start': start / sample_rate,
            'cycle_end': end / sample_rate,
            'cycle_duration': cycle_duration_actual,
            'num_chunks': len(chunks),
            'top_n': top_n,
            'threshold': threshold,
            'all_terms': all_terms,  # Before filtering
            'terms': filtered_terms,  # After threshold + top-N
        })
        
        start = end
        cycle_idx += 1
    
    return results


def aggregate_cycle_results(cycle_results):
    """
    Aggregate results from all cycles into a single set of retrieved terms.
    
    Uses UNION of all cycle terms (since each cycle outputs independently).
    For duplicate terms, keeps the max score.
    """
    aggregated = {}
    for cycle in cycle_results:
        for term, score in cycle['terms'].items():
            if term not in aggregated or score > aggregated[term]:
                aggregated[term] = score
    return aggregated


def compute_f_beta(precision: float, recall: float, beta: float) -> float:
    """Compute F-beta score."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def compute_hit_poison_rates(results: list, threshold: float = 0.0) -> dict:
    """
    Compute Hit Rate and Poison Rate for a given threshold.
    
    Args:
        results: List of result dicts with 'gt_terms', 'aggregated_terms'
        threshold: Score threshold for filtering
    
    Returns:
        Dict with hit_rate, poison_rate, and counts
    
    Definitions:
        - Hit Rate: Among non-empty outputs, proportion that contain at least one GT term
        - Poison Rate: Among all outputs, proportion that have NO GT and have FP terms
    """
    non_empty_count = 0
    hit_count = 0  # Non-empty outputs that contain at least one GT
    poison_count = 0  # Outputs with no GT but have FP terms
    total_count = len(results)
    
    for r in results:
        gt_set = set(r['gt_terms'])
        aggregated_terms = r.get('aggregated_terms', {})
        
        # Apply threshold filter
        terms_above_threshold = {t: s for t, s in aggregated_terms.items() if s >= threshold}
        
        if len(terms_above_threshold) > 0:
            non_empty_count += 1
            
            # Check if any GT term is in the output
            retrieved_set = set(terms_above_threshold.keys())
            has_gt = len(gt_set & retrieved_set) > 0
            
            if has_gt:
                hit_count += 1
            else:
                # No GT but has FP -> poison
                poison_count += 1
    
    hit_rate = hit_count / non_empty_count if non_empty_count > 0 else 0.0
    poison_rate = poison_count / total_count if total_count > 0 else 0.0
    
    return {
        'threshold': threshold,
        'hit_rate': hit_rate,
        'poison_rate': poison_rate,
        'hit_count': hit_count,
        'non_empty_count': non_empty_count,
        'poison_count': poison_count,
        'total_count': total_count,
    }


def _reconstruct_all_vectors(index) -> np.ndarray:
    """Reconstruct all vectors from a FAISS index."""
    n = int(index.ntotal)
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        vecs = index.reconstruct_n(0, n)
        return np.asarray(vecs, dtype=np.float32)
    except Exception:
        d = int(index.d)
        out = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            out[i] = index.reconstruct(i)
        return out


def restrict_index_to_terms(faiss_index, term_list, keep_terms_lc: set):
    """Restrict a FAISS index + term_list to a subset of terms."""
    if not keep_terms_lc:
        return faiss_index, term_list, 0

    term2idx = {}
    for i, entry in enumerate(term_list):
        term = entry.get("term") if isinstance(entry, dict) else entry
        if term is None:
            continue
        term_lc = str(term).lower()
        if term_lc not in term2idx:
            term2idx[term_lc] = i

    keep_indices = [term2idx[t] for t in keep_terms_lc if t in term2idx]
    keep_indices = sorted(set(keep_indices))
    if not keep_indices:
        return faiss_index, term_list, 0

    all_vecs = _reconstruct_all_vectors(faiss_index)
    if all_vecs.size == 0:
        return faiss_index, term_list, 0

    sub_vecs = all_vecs[keep_indices]
    new_index = faiss.IndexFlatL2(sub_vecs.shape[1])
    new_index.add(sub_vecs)
    new_term_list = [term_list[i] for i in keep_indices]
    return new_index, new_term_list, len(keep_indices)


def main():
    parser = argparse.ArgumentParser(description="Streaming Sliding Window Evaluation for ACL6060")
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--prebuilt_index', type=str, required=True, help='Path to prebuilt index (.pkl)')
    parser.add_argument('--glossary_path', type=str, required=True, help='Path to glossary json')
    
    # ACL6060-specific arguments
    parser.add_argument('--wav_dir', type=str, required=True, 
                       help='Directory containing segmented wav files (gold)')
    parser.add_argument('--txt_path', type=str, required=True,
                       help='Path to plain text file (one line per segment)')
    
    # Optional arguments
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help='Base model name')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--max_samples', type=int, default=0, help='Maximum samples to evaluate (0=all)')
    
    # Streaming simulation parameters
    parser.add_argument('--cycle_duration', type=float, default=1.92, 
                       help='Duration of each processing cycle in seconds (simulates vLLM call interval)')
    parser.add_argument('--chunk_size', type=float, default=2.0, help='Sliding window chunk size in seconds')
    parser.add_argument('--hop_size', type=float, default=1.0, help='Sliding window hop size in seconds')
    parser.add_argument('--top_k', type=int, default=5, help='Top-k terms to retrieve per chunk')
    parser.add_argument('--terms_per_second', type=float, default=2.5, 
                       help='Terms per second for top-N calculation (N = ceil(cycle_duration * terms_per_second))')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Minimum score threshold for term filtering (prevents poison output when no relevant terms)')
    
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for audio encoding')
    parser.add_argument('--min_words', type=int, default=1, help='Minimum word count for GT terms')
    parser.add_argument('--min_chars', type=int, default=3, help='Minimum character count for GT terms')
    parser.add_argument('--save_plot_dir', type=str, default=None, help='If set, save plots and results')
    parser.add_argument('--restrict_index_to_eval_terms', action='store_true', 
                       help='If set, restrict index/term_list to GT terms')
    parser.add_argument('--random_sample', action='store_true', 
                       help='If set, randomly sample max_samples from data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return 1
    
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    
    # Calculate fixed top-N
    fixed_top_n = math.ceil(args.cycle_duration * args.terms_per_second)
    print(f"\n[INFO] Streaming Simulation Parameters:")
    print(f"  Cycle duration: {args.cycle_duration}s (simulates vLLM call interval)")
    print(f"  Chunk size: {args.chunk_size}s")
    print(f"  Hop size: {args.hop_size}s")
    print(f"  Terms per second: {args.terms_per_second}")
    print(f"  Fixed top-N per cycle: ceil({args.cycle_duration} * {args.terms_per_second}) = {fixed_top_n}")
    print(f"  Score threshold: {args.threshold} (filter before top-N to avoid poison output)")
    
    # Load glossary
    print("\n" + "="*80)
    print("LOADING GLOSSARY")
    print("="*80)
    print(f"[INFO] Loading glossary from: {args.glossary_path}")
    with open(args.glossary_path, 'r', encoding='utf-8') as f:
        glossary = json.load(f)
    
    glossary_terms = set(k.lower() for k in glossary.keys())
    print(f"[INFO] Loaded {len(glossary_terms)} terms from glossary")
    
    # Load prebuilt index
    print("\n" + "="*80)
    print("LOADING PREBUILT INDEX")
    print("="*80)
    print(f"[INFO] Loading index from: {args.prebuilt_index}")
    with open(args.prebuilt_index, 'rb') as f:
        index_data = pickle.load(f)
    
    faiss_index = faiss.deserialize_index(index_data['faiss_index'])
    term_list = index_data['term_list']
    print(f"[INFO] Index contains {faiss_index.ntotal} vectors, {len(term_list)} terms")
    
    # Load ACL6060 data
    print("\n" + "="*80)
    print("LOADING ACL6060 DATA")
    print("="*80)
    print(f"[INFO] Loading wav files from: {args.wav_dir}")
    print(f"[INFO] Loading text from: {args.txt_path}")
    
    wav_files = load_wav_files_sorted(args.wav_dir)
    text_lines = load_text_lines(args.txt_path)
    
    print(f"[INFO] Found {len(wav_files)} wav files")
    print(f"[INFO] Found {len(text_lines)} text lines")
    
    if len(wav_files) != len(text_lines):
        print(f"[ERROR] Mismatch: {len(wav_files)} wav files vs {len(text_lines)} text lines!")
        return 1
    
    # Combine data and extract GT terms
    print(f"[INFO] Extracting GT terms from text using FlashText")
    samples = []
    for i, (wav_path, text) in enumerate(zip(wav_files, text_lines)):
        gt_terms = extract_gt_terms_from_text(
            text, 
            glossary_terms, 
            min_words=args.min_words, 
            min_chars=args.min_chars
        )
        samples.append({
            'id': os.path.basename(wav_path),
            'wav_path': wav_path,
            'text': text,
            'gt_terms': gt_terms,
        })
    
    # Optionally subsample
    if args.max_samples > 0 and args.max_samples < len(samples):
        if args.random_sample:
            random.seed(args.seed)
            samples = random.sample(samples, args.max_samples)
        else:
            samples = samples[:args.max_samples]
    
    print(f"[INFO] Using {len(samples)} samples for evaluation")
    
    # Optional: restrict index to eval terms
    if args.restrict_index_to_eval_terms:
        print("\n" + "="*80)
        print("RESTRICTING INDEX TO EVAL TERMS")
        print("="*80)
        eval_term_set = set()
        for s in samples:
            for t in s['gt_terms']:
                eval_term_set.add(str(t).lower())

        print(f"[INFO] Terms found in eval texts (unique): {len(eval_term_set)}")
        before_n = int(faiss_index.ntotal)
        faiss_index, term_list, kept = restrict_index_to_terms(faiss_index, term_list, eval_term_set)
        after_n = int(faiss_index.ntotal)
        print(f"[INFO] Index restricted: {before_n} -> {after_n} vectors (kept={kept})")
    
    # Initialize model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    speech_encoder = Qwen2AudioSpeechEncoder(model_name=args.model_name, device=device)
    text_encoder = Qwen2AudioTextEncoder(
        model_name=args.model_name,
        device=device,
        shared_model=speech_encoder.get_shared_model()
    )
    
    model = ContrastiveQwen2AudioModel(
        speech_encoder,
        text_encoder,
        proj_dim=512,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    ).to(device)
    
    print(f"[INFO] Loading weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("[INFO] Model loaded and set to eval mode")
    
    # Create retriever structure
    class SimpleRetrieverWrapper:
        def __init__(self, index, term_list):
            self.index = index
            self.term_list = term_list
    
    retriever = SimpleRetrieverWrapper(faiss_index, term_list)
    
    # Evaluation
    print("\n" + "="*80)
    print("RUNNING STREAMING SIMULATION EVALUATION")
    print("="*80)
    
    total_gt_terms = 0
    total_hits = 0
    total_fp = 0
    pos_scores = []
    neg_scores = []
    samples_with_terms = 0
    samples_processed = 0
    failed_audio = 0
    total_cycles = 0
    
    results = []
    
    for sample in tqdm(samples, desc="Evaluating"):
        gt_terms = list(sample['gt_terms'])
        
        if not gt_terms:
            continue
        
        samples_with_terms += 1
        
        # Load audio
        audio = load_audio_from_wav(sample['wav_path'])
        if audio is None:
            failed_audio += 1
            continue
        
        samples_processed += 1
        audio_duration = len(audio) / 16000.0
        
        # Process segment with streaming simulation
        cycle_results = process_segment_streaming(
            model, retriever, audio, device,
            cycle_duration=args.cycle_duration,
            chunk_size=args.chunk_size,
            hop_size=args.hop_size,
            top_k=args.top_k,
            terms_per_second=args.terms_per_second,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )
        
        total_cycles += len(cycle_results)
        
        # Aggregate all cycle results (union with max score)
        aggregated_terms = aggregate_cycle_results(cycle_results)
        
        # Calculate hits
        gt_set = set(gt_terms)
        retrieved_set = set(aggregated_terms.keys())
        hit_terms = sorted(gt_set & retrieved_set)
        hits = len(hit_terms)
        fp_terms = retrieved_set - gt_set
        
        total_gt_terms += len(gt_terms)
        total_hits += hits
        total_fp += len(fp_terms)
        
        # Collect scores
        for t in hit_terms:
            pos_scores.append(aggregated_terms[t])
        for t in fp_terms:
            neg_scores.append(aggregated_terms[t])
        
        results.append({
            'id': sample['id'],
            'text': sample['text'],
            'gt_terms': gt_terms,
            'hit_terms': hit_terms,
            'audio_duration': audio_duration,
            'num_cycles': len(cycle_results),
            'aggregated_terms': aggregated_terms,
            'cycle_results': cycle_results,
            'hits': hits,
            'fp': len(fp_terms),
        })
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    recall = total_hits / total_gt_terms if total_gt_terms > 0 else 0
    precision = total_hits / (total_hits + total_fp) if (total_hits + total_fp) > 0 else 0
    f1 = compute_f_beta(precision, recall, 1.0)
    f2 = compute_f_beta(precision, recall, 2.0)
    f3 = compute_f_beta(precision, recall, 3.0)
    
    print(f"[RESULTS] Total samples: {len(samples)}")
    print(f"[RESULTS] Samples with GT terms: {samples_with_terms}")
    print(f"[RESULTS] Samples processed: {samples_processed}")
    print(f"[RESULTS] Failed audio loads: {failed_audio}")
    print(f"[RESULTS] Total cycles processed: {total_cycles}")
    print(f"[RESULTS] Avg cycles per sample: {total_cycles / samples_processed:.2f}" if samples_processed > 0 else "N/A")
    print(f"[RESULTS] " + "-"*48)
    print(f"[RESULTS] Total GT terms: {total_gt_terms}")
    print(f"[RESULTS] Total hits (TP): {total_hits}")
    print(f"[RESULTS] Total false positives (FP): {total_fp}")
    print(f"[RESULTS] " + "-"*48)
    print(f"[RESULTS] Precision: {precision:.4f}")
    print(f"[RESULTS] Recall: {recall:.4f}")
    print(f"[RESULTS] F1-Score: {f1:.4f}")
    print(f"[RESULTS] F2-Score: {f2:.4f}")
    print(f"[RESULTS] F3-Score: {f3:.4f}")
    print(f"[RESULTS] " + "-"*48)
    
    if pos_scores:
        print(f"[RESULTS] Positive scores: n={len(pos_scores)}, min={np.min(pos_scores):.4f}, max={np.max(pos_scores):.4f}, avg={np.mean(pos_scores):.4f}")
    if neg_scores:
        print(f"[RESULTS] Negative scores: n={len(neg_scores)}, min={np.min(neg_scores):.4f}, max={np.max(neg_scores):.4f}, avg={np.mean(neg_scores):.4f}")
    
    print(f"[RESULTS] " + "-"*48)
    print(f"[RESULTS] === Streaming Parameters ===")
    print(f"[RESULTS] Cycle duration: {args.cycle_duration}s")
    print(f"[RESULTS] Fixed top-N per cycle: {fixed_top_n}")
    print(f"[RESULTS] Chunk size: {args.chunk_size}s")
    print(f"[RESULTS] Hop size: {args.hop_size}s")
    print(f"[RESULTS] Top-k per chunk: {args.top_k}")
    print(f"[RESULTS] Terms per second: {args.terms_per_second}")
    print(f"[RESULTS] Score threshold: {args.threshold}")
    print(f"[RESULTS] Index terms: {len(term_list)}")
    
    # ======== Compute Hit Rate vs Poison Rate curve (threshold sweep) ========
    print("\n" + "="*80)
    print("HIT RATE vs POISON RATE ANALYSIS (Threshold Sweep)")
    print("="*80)
    print(f"[INFO] Scanning thresholds from 0.3 to 0.8 to find optimal filtering point")
    print(f"[INFO] Note: This uses aggregated_terms (after cycle-level top-N filtering)")
    
    # Collect all scores from aggregated results
    all_scores = []
    for r in results:
        for s in r['aggregated_terms'].values():
            all_scores.append(s)
    
    hit_poison_results = []
    
    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        # Generate thresholds from 0.3 to 0.8 (or max_score if smaller)
        threshold_start = 0.3
        threshold_end = min(0.8, max_score)
        thresholds = np.linspace(threshold_start, threshold_end, 50)
        
        # Also include some key points
        key_thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        thresholds = np.concatenate([key_thresholds, thresholds])
        thresholds = sorted(set(thresholds))  # Remove duplicates
        
        for threshold in thresholds:
            hp = compute_hit_poison_rates(results, threshold)
            hit_poison_results.append(hp)
        
        # Print summary table
        print(f"\n{'Threshold':>10} | {'Hit Rate':>10} | {'Poison Rate':>12} | {'Non-Empty':>10} | {'Hits':>6} | {'Poison':>7}")
        print("-" * 70)
        for hp in hit_poison_results[::5]:  # Print every 5th row
            print(f"{hp['threshold']:>10.4f} | {hp['hit_rate']:>10.2%} | {hp['poison_rate']:>12.2%} | {hp['non_empty_count']:>10} | {hp['hit_count']:>6} | {hp['poison_count']:>7}")
        
        # Find optimal threshold (maximize hit rate while minimizing poison rate)
        # Target: Hit Rate > 90%, Poison Rate < 5%
        best_hp = None
        for hp in hit_poison_results:
            if hp['hit_rate'] >= 0.90 and hp['poison_rate'] <= 0.05:
                if best_hp is None or hp['hit_rate'] > best_hp['hit_rate']:
                    best_hp = hp
        
        if best_hp:
            print(f"\n[OPTIMAL] Found threshold meeting targets (Hit Rate >= 90%, Poison Rate <= 5%):")
            print(f"  Threshold: {best_hp['threshold']:.4f}")
            print(f"  Hit Rate: {best_hp['hit_rate']:.2%}")
            print(f"  Poison Rate: {best_hp['poison_rate']:.2%}")
        else:
            print(f"\n[INFO] No threshold found meeting both targets (Hit Rate >= 90%, Poison Rate <= 5%)")
            # Find best trade-off
            best_trade = max(hit_poison_results, key=lambda x: x['hit_rate'] - x['poison_rate'] * 2)
            print(f"  Best trade-off threshold: {best_trade['threshold']:.4f}")
            print(f"  Hit Rate: {best_trade['hit_rate']:.2%}")
            print(f"  Poison Rate: {best_trade['poison_rate']:.2%}")
    else:
        print("[WARN] No scores available for threshold sweep")
    
    # Print sample results
    print("\n" + "="*80)
    print("SAMPLE RESULTS (first 5)")
    print("="*80)
    for r in results[:5]:
        print(f"\n[Sample] {r['id']}")
        print(f"  Text: {r['text'][:100]}...")
        print(f"  Duration: {r['audio_duration']:.2f}s, Cycles: {r['num_cycles']}")
        print(f"  GT terms: {r['gt_terms']}")
        print(f"  Hit terms: {r['hit_terms']}")
        print(f"  Hits/Total GT: {r['hits']}/{len(r['gt_terms'])}")
        print(f"  FP count: {r['fp']}")
        
        # Show per-cycle breakdown
        for c in r['cycle_results'][:3]:  # First 3 cycles
            print(f"    Cycle {c['cycle_idx']}: [{c['cycle_start']:.2f}s - {c['cycle_end']:.2f}s], "
                  f"chunks={c['num_chunks']}, terms={list(c['terms'].keys())[:5]}...")

    # Save results if requested
    if args.save_plot_dir:
        os.makedirs(args.save_plot_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        meta_path = os.path.join(args.save_plot_dir, f"streaming_v3_meta_{ts}.json")
        scores_path = os.path.join(args.save_plot_dir, f"streaming_v3_scores_{ts}.json")
        hp_data_path = os.path.join(args.save_plot_dir, f"streaming_v3_hit_poison_data_{ts}.json")
        hp_curve_path = os.path.join(args.save_plot_dir, f"streaming_v3_hit_poison_curve_{ts}.png")
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": "acl6060_streaming_v3",
                "cycle_duration": args.cycle_duration,
                "fixed_top_n": fixed_top_n,
                "threshold": args.threshold,
                "chunk_size": args.chunk_size,
                "hop_size": args.hop_size,
                "top_k": args.top_k,
                "terms_per_second": args.terms_per_second,
                "samples_processed": samples_processed,
                "total_cycles": total_cycles,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "total_gt_terms": total_gt_terms,
                "total_hits": total_hits,
                "total_fp": total_fp,
            }, f, indent=2, ensure_ascii=False)
        
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump({
                "positive_scores": pos_scores,
                "negative_scores": neg_scores,
            }, f, indent=2, ensure_ascii=False)
        
        # Save Hit Rate vs Poison Rate data
        if hit_poison_results:
            with open(hp_data_path, "w", encoding="utf-8") as f:
                json.dump({
                    "thresholds": [hp['threshold'] for hp in hit_poison_results],
                    "hit_rates": [hp['hit_rate'] for hp in hit_poison_results],
                    "poison_rates": [hp['poison_rate'] for hp in hit_poison_results],
                    "hit_counts": [hp['hit_count'] for hp in hit_poison_results],
                    "poison_counts": [hp['poison_count'] for hp in hit_poison_results],
                    "non_empty_counts": [hp['non_empty_count'] for hp in hit_poison_results],
                }, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved hit/poison data to: {hp_data_path}")
        
        print(f"\n[INFO] Saved metadata to: {meta_path}")
        print(f"[INFO] Saved scores to: {scores_path}")
        
        # Plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            # Plot 1: Score distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            bins = 60
            if pos_scores:
                ax.hist(pos_scores, bins=bins, alpha=0.6, density=True, label="Positive (GT Hits)", color="green")
            if neg_scores:
                ax.hist(neg_scores, bins=bins, alpha=0.6, density=True, label="Negative (FP)", color="red")
            ax.set_title(f"Streaming v3: Score Distribution\n(cycle={args.cycle_duration}s, top-N={fixed_top_n}, threshold={args.threshold})")
            ax.set_xlabel("Similarity Score")
            ax.set_ylabel("Density")
            ax.legend()
            
            plot_path = os.path.join(args.save_plot_dir, f"streaming_v3_distribution_{ts}.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"[INFO] Saved score distribution plot to: {plot_path}")
            
            # Plot 2: Hit Rate vs Poison Rate curve
            if hit_poison_results and len(hit_poison_results) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                hit_rates = [hp['hit_rate'] for hp in hit_poison_results]
                poison_rates = [hp['poison_rate'] for hp in hit_poison_results]
                thresholds_plot = [hp['threshold'] for hp in hit_poison_results]
                
                # Plot as scatter with color gradient for threshold
                scatter = ax.scatter(poison_rates, hit_rates, c=thresholds_plot, cmap='viridis', s=50, alpha=0.8)
                ax.plot(poison_rates, hit_rates, 'b-', alpha=0.3, linewidth=1)
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Threshold')
                
                # Add target lines
                ax.axhline(y=0.90, color='green', linestyle='--', alpha=0.5, label='Target Hit Rate (90%)')
                ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='Target Poison Rate (5%)')
                
                # Highlight target region
                ax.fill_between([0, 0.05], [0.90, 0.90], [1.0, 1.0], alpha=0.1, color='green', label='Target Region')
                
                ax.set_xlabel('Poison Rate (No GT + FP / Total)', fontsize=12)
                ax.set_ylabel('Hit Rate (Has GT / Non-Empty)', fontsize=12)
                ax.set_title(f'Hit Rate vs Poison Rate Curve (Streaming v3)\n(cycle={args.cycle_duration}s, threshold sweep: 0.3-0.8)', fontsize=14)
                ax.set_xlim(-0.02, max(poison_rates) + 0.05)
                ax.set_ylim(min(hit_rates) - 0.05, 1.02)
                ax.legend(loc='lower left')
                ax.grid(True, alpha=0.3)
                
                # Annotate some key points
                for i in range(0, len(hit_poison_results), max(1, len(hit_poison_results) // 8)):
                    hp = hit_poison_results[i]
                    ax.annotate(f'{hp["threshold"]:.2f}', 
                               (hp['poison_rate'], hp['hit_rate']),
                               textcoords="offset points", 
                               xytext=(5, 5), 
                               fontsize=8, alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(hp_curve_path, dpi=150)
                plt.close()
                print(f"[INFO] Saved hit/poison curve to: {hp_curve_path}")
        except Exception as e:
            print(f"[WARN] Failed to save plot: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

