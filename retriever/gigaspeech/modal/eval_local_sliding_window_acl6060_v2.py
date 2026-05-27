#!/usr/bin/env python3
"""
Sliding Window Evaluation Script for ACL6060 Dataset

Evaluates the retriever using ACL6060 format:
- Load audio from segmented wav files (gold segments)
- Use sliding window with configurable chunk size and hop size
- Aggregate recall@k terms from all chunks using max score pooling
- Calculate hit rate against GT terms from tagged text
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
    """
    Load audio from wav file.
    
    Args:
        wav_path: Path to wav file
        target_sr: Target sample rate (default 16000)
    
    Returns:
        numpy array of audio samples
    """
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
    """
    Load text lines from plain text file.
    Each line corresponds to a segment.
    
    Args:
        txt_path: Path to plain text file
    
    Returns:
        List of text strings
    """
    results = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            results.append(line)
    
    return results


def extract_gt_terms_from_text(text, glossary_terms, min_words=1, min_chars=3):
    """
    Extract GT terms from text using FlashText matching against glossary.
    Case-insensitive matching.
    
    Args:
        text: Text to search
        glossary_terms: Set of glossary terms (lowercase)
        min_words: Minimum word count for terms
        min_chars: Minimum character count for terms
    
    Returns:
        Set of matched terms (lowercase)
    """
    keyword_processor = KeywordProcessor(case_sensitive=False)
    
    # Add glossary terms with filtering
    for term in glossary_terms:
        word_count = len(term.split())
        if len(term) >= min_chars and word_count >= min_words:
            # Add term with lowercase as the canonical form
            keyword_processor.add_keyword(term.lower())
    
    # Extract keywords (longest match first by default)
    found_keywords = keyword_processor.extract_keywords(text)
    
    # Return as lowercase set
    return set(k.lower() for k in found_keywords)


def load_wav_files_sorted(wav_dir):
    """
    Load wav file paths sorted by numeric sent_id.
    
    Args:
        wav_dir: Directory containing wav files
    
    Returns:
        List of wav file paths sorted by sent_id
    """
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    
    def extract_number(filename):
        """Extract number from filename like sent_123.wav -> 123"""
        basename = os.path.basename(filename)
        match = re.search(r'sent_(\d+)\.wav', basename)
        if match:
            return int(match.group(1))
        return 0
    
    wav_files = sorted(wav_files, key=extract_number)
    return wav_files


from flashtext import KeywordProcessor

def find_gt_terms_flashtext(text, glossary_terms, min_words=1, min_chars=3):
    """
    Use FlashText for fast multi-pattern matching with longest match priority.
    
    Args:
        text: Text to search
        glossary_terms: Set of glossary terms
        min_words: Minimum word count (default 1, set to 2 for multi-word only)
        min_chars: Minimum character count (default 3)
    """
    keyword_processor = KeywordProcessor(case_sensitive=False)
    
    # Filter glossary terms
    for term in glossary_terms:
        word_count = len(term.split())
        if len(term) >= min_chars and word_count >= min_words:
            keyword_processor.add_keyword(term)
            
    # extract_keywords uses Longest Match First by default
    found_keywords = keyword_processor.extract_keywords(text)
    
    # Deduplicate
    return list(set(found_keywords))


def create_sliding_window_chunks(audio, chunk_size=2.0, hop_size=1.0, sample_rate=16000):
    """
    Create sliding window chunks from audio.
    
    Args:
        audio: Audio samples (numpy array)
        chunk_size: Chunk size in seconds
        hop_size: Hop size in seconds (sliding window step)
        sample_rate: Sample rate
    
    Returns:
        List of audio chunks (numpy arrays)
    """
    chunk_samples = int(chunk_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)
    
    chunks = []
    start = 0
    
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        
        # Pad if necessary (last chunk might be shorter)
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
        
        chunks.append(chunk)
        start += hop_samples
        
        # Avoid creating chunks that are mostly padding
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
    """
    Convert FAISS L2 distance into a score where higher is better.

    We use score = 1 / (1 + distance) so that:
    - score is in (0, 1]
    - higher is better
    - max score corresponds to min distance
    """
    d = float(distance)
    return 1.0 / (1.0 + d)


def retrieve_terms_for_chunks(model, retriever, chunks, device, top_k=5, batch_size=32):
    """
    Retrieve terms for all chunks and aggregate results using max score pooling.
    
    Args:
        model: The contrastive model
        retriever: SimpleRetriever with index and term_list
        chunks: List of audio chunks (numpy arrays)
        device: torch device
        top_k: Number of terms to retrieve per chunk
        batch_size: Batch size for encoding
    
    Returns:
        Dict[str, float]: term(lowercase) -> max_score across all windows
    """
    term2max_score = {}
    
    # Encode chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        try:
            audio_embs = encode_audio_batch(model, batch_chunks, device)
            
            # Search for each embedding
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


def filter_top_n_by_duration(term2score: dict, audio_duration: float, terms_per_second: float = 2.5) -> dict:
    """
    Filter retrieved terms to top-N based on audio duration.
    
    Args:
        term2score: Dict of term -> score
        audio_duration: Audio duration in seconds
        terms_per_second: Number of terms per second (default 2.5)
    
    Returns:
        Dict of top-N terms by score
    """
    import math
    n = math.ceil(audio_duration * terms_per_second)
    n = max(1, n)  # At least 1
    
    # Sort by score descending and take top N
    sorted_terms = sorted(term2score.items(), key=lambda x: x[1], reverse=True)
    top_n_terms = dict(sorted_terms[:n])
    
    return top_n_terms


def compute_f_beta(precision: float, recall: float, beta: float) -> float:
    """Compute F-beta score."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def compute_hit_poison_rates(results: list, threshold: float = 0.0) -> dict:
    """
    Compute Hit Rate and Poison Rate for a given threshold.
    
    Args:
        results: List of result dicts with 'gt_terms', 'filtered_terms' (after top-N)
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
        filtered_terms = r.get('filtered_terms', {})
        
        # Apply threshold filter
        terms_above_threshold = {t: s for t, s in filtered_terms.items() if s >= threshold}
        
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
    """Reconstruct all vectors from a FAISS index as a float32 matrix [N, D]."""
    n = int(index.ntotal)
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        # Fast path for IndexFlat* and many other indexes
        vecs = index.reconstruct_n(0, n)
        return np.asarray(vecs, dtype=np.float32)
    except Exception:
        # Fallback: reconstruct one by one
        d = int(index.d)
        out = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            out[i] = index.reconstruct(i)
        return out


def restrict_index_to_terms(faiss_index, term_list, keep_terms_lc: set):
    """
    Restrict a FAISS index + term_list to a subset of terms.

    keep_terms_lc: lowercase terms to keep.
    Returns (new_index, new_term_list, kept_count).
    """
    if not keep_terms_lc:
        return faiss_index, term_list, 0

    term2idx = {}
    for i, entry in enumerate(term_list):
        term = entry.get("term") if isinstance(entry, dict) else entry
        if term is None:
            continue
        term_lc = str(term).lower()
        # keep first occurrence
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
    parser = argparse.ArgumentParser(description="Sliding Window Evaluation Script for ACL6060")
    
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
    parser.add_argument('--chunk_size', type=float, default=2.0, help='Chunk size in seconds')
    parser.add_argument('--hop_size', type=float, default=1.0, help='Hop size (sliding window step) in seconds')
    parser.add_argument('--top_k', type=int, default=5, help='Top-k terms to retrieve per chunk')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for audio encoding')
    parser.add_argument('--min_words', type=int, default=1, help='Minimum word count for GT terms')
    parser.add_argument('--min_chars', type=int, default=3, help='Minimum character count for GT terms')
    parser.add_argument('--save_plot_dir', type=str, default=None, help='If set, save score distribution plot')
    parser.add_argument('--restrict_index_to_eval_terms', action='store_true', 
                       help='If set, restrict index/term_list to GT terms')
    parser.add_argument('--random_sample', action='store_true', 
                       help='If set, randomly sample max_samples from data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--terms_per_second', type=float, default=2.5, 
                       help='Number of terms to keep per second of audio (for top-N filtering)')
    parser.add_argument('--enable_top_n_filter', action='store_true',
                       help='Enable top-N filtering based on duration (N = ceil(duration * terms_per_second))')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return 1
    
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    
    # Load glossary for additional term matching
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
    
    # Validate data alignment
    if len(wav_files) != len(text_lines):
        print(f"[ERROR] Mismatch: {len(wav_files)} wav files vs {len(text_lines)} text lines!")
        return 1
    
    # Combine data and extract GT terms using FlashText from glossary
    print(f"[INFO] Extracting GT terms from text using FlashText (glossary size: {len(glossary_terms)})")
    samples = []
    for i, (wav_path, text) in enumerate(zip(wav_files, text_lines)):
        # Extract GT terms by matching glossary terms in text (case-insensitive)
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
            'gt_terms': gt_terms,  # Extracted via FlashText from glossary
        })
    
    # Optionally subsample
    if args.max_samples > 0 and args.max_samples < len(samples):
        if args.random_sample:
            random.seed(args.seed)
            samples = random.sample(samples, args.max_samples)
        else:
            samples = samples[:args.max_samples]
    
    print(f"[INFO] Using {len(samples)} samples for evaluation")
    
    # Optional: restrict index to terms present in this eval subset
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
    
    # Load model weights
    print(f"[INFO] Loading weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DDP prefix
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
    print("RUNNING SLIDING WINDOW EVALUATION")
    print("="*80)
    print(f"[INFO] Chunk size: {args.chunk_size}s, Hop size: {args.hop_size}s, Top-k: {args.top_k}")
    if args.enable_top_n_filter:
        print(f"[INFO] Top-N filter ENABLED: terms_per_second={args.terms_per_second}")
    else:
        print(f"[INFO] Top-N filter DISABLED (using all retrieved terms)")
    
    total_gt_terms = 0
    total_hits = 0
    total_hits_filtered = 0  # Hits after top-N filtering
    gt_hit_scores = []
    pos_scores = []  # retrieved terms that are in GT
    neg_scores = []  # retrieved terms that are NOT in GT (false positives / noise)
    pos_scores_filtered = []  # After top-N filtering
    neg_scores_filtered = []  # After top-N filtering
    samples_with_terms = 0
    samples_processed = 0
    failed_audio = 0
    
    results = []
    
    for sample in tqdm(samples, desc="Evaluating"):
        # Get GT terms (already extracted from tagged text)
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
        audio_duration = len(audio) / 16000.0  # Assuming 16kHz
        
        # Create sliding window chunks
        chunks = create_sliding_window_chunks(
            audio, 
            chunk_size=args.chunk_size, 
            hop_size=args.hop_size
        )
        
        # Retrieve terms from all chunks with max score pooling
        retrieved_terms = retrieve_terms_for_chunks(
            model, retriever, chunks, device, 
            top_k=args.top_k, 
            batch_size=args.batch_size
        )
        
        # Apply top-N filter based on duration
        if args.enable_top_n_filter:
            filtered_terms = filter_top_n_by_duration(
                retrieved_terms, 
                audio_duration, 
                terms_per_second=args.terms_per_second
            )
        else:
            filtered_terms = retrieved_terms
        
        # Calculate hits (before filtering)
        gt_set = set(gt_terms)
        retrieved_set = set(retrieved_terms.keys())
        hit_terms = sorted(gt_set & retrieved_set)
        hits = len(hit_terms)
        total_gt_terms += len(gt_terms)
        total_hits += hits
        
        # Calculate hits (after filtering)
        filtered_set = set(filtered_terms.keys())
        hit_terms_filtered = sorted(gt_set & filtered_set)
        hits_filtered = len(hit_terms_filtered)
        total_hits_filtered += hits_filtered
        
        # Collect scores (before filtering - for distribution analysis)
        for t in hit_terms:
            gt_hit_scores.append(retrieved_terms[t])
            pos_scores.append(retrieved_terms[t])
        for t, s in retrieved_terms.items():
            if t not in gt_set:
                neg_scores.append(s)
        
        # Collect scores (after filtering)
        for t in hit_terms_filtered:
            pos_scores_filtered.append(filtered_terms[t])
        for t, s in filtered_terms.items():
            if t not in gt_set:
                neg_scores_filtered.append(s)
        
        import math
        top_n = math.ceil(audio_duration * args.terms_per_second)
        
        results.append({
            'id': sample['id'],
            'text': sample['text'],
            'gt_terms': gt_terms,
            'hit_terms': hit_terms,
            'hit_terms_filtered': hit_terms_filtered,
            'gt_term_scores': {t: retrieved_terms.get(t) for t in gt_terms},
            # All retrieved terms (before filtering)
            'retrieved_terms': retrieved_terms,
            # Filtered terms (after top-N)
            'filtered_terms': filtered_terms,
            # keep a small debug view: top 20 retrieved terms by score
            'retrieved_top': sorted(retrieved_terms.items(), key=lambda x: x[1], reverse=True)[:20],
            'hits': hits,
            'hits_filtered': hits_filtered,
            'num_chunks': len(chunks),
            'audio_duration': audio_duration,
            'top_n': top_n,
        })
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    accuracy = total_hits / total_gt_terms if total_gt_terms > 0 else 0
    accuracy_filtered = total_hits_filtered / total_gt_terms if total_gt_terms > 0 else 0
    
    if gt_hit_scores:
        gt_score_min = float(np.min(gt_hit_scores))
        gt_score_max = float(np.max(gt_hit_scores))
        gt_score_avg = float(np.mean(gt_hit_scores))
    else:
        gt_score_min = None
        gt_score_max = None
        gt_score_avg = None
    
    # Calculate Precision, Recall, F2, F3 (after top-N filtering)
    tp = total_hits_filtered
    fp = len(neg_scores_filtered)
    fn = total_gt_terms - total_hits_filtered
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = compute_f_beta(precision, recall, 1.0)
    f2 = compute_f_beta(precision, recall, 2.0)
    f3 = compute_f_beta(precision, recall, 3.0)
    
    print(f"[RESULTS] Total samples: {len(samples)}")
    print(f"[RESULTS] Samples with GT terms: {samples_with_terms}")
    print(f"[RESULTS] Samples processed (audio loaded): {samples_processed}")
    print(f"[RESULTS] Failed audio loads: {failed_audio}")
    print(f"[RESULTS] " + "-"*48)
    print(f"[RESULTS] Total GT terms: {total_gt_terms}")
    print(f"[RESULTS] Total hits (before filter): {total_hits}")
    print(f"[RESULTS] Total hits (after top-N filter): {total_hits_filtered}")
    print(f"[RESULTS] Recall (before filter): {accuracy:.2%}")
    print(f"[RESULTS] Recall (after top-N filter): {accuracy_filtered:.2%}")
    print(f"[RESULTS] " + "-"*48)
    print(f"[RESULTS] === After Top-N Filtering ===")
    print(f"[RESULTS] TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"[RESULTS] Precision: {precision:.4f}")
    print(f"[RESULTS] Recall: {recall:.4f}")
    print(f"[RESULTS] F1-Score: {f1:.4f}")
    print(f"[RESULTS] F2-Score: {f2:.4f}")
    print(f"[RESULTS] F3-Score: {f3:.4f}")
    print(f"[RESULTS] " + "-"*48)
    print(f"[RESULTS] GT term score stats (hits only, score=1/(1+L2_distance)):")
    print(f"[RESULTS]   - count: {len(gt_hit_scores)}")
    print(f"[RESULTS]   - min: {gt_score_min}")
    print(f"[RESULTS]   - max: {gt_score_max}")
    print(f"[RESULTS]   - avg: {gt_score_avg}")
    if pos_scores and neg_scores:
        print(f"[RESULTS] " + "-"*48)
        print(f"[RESULTS] Retrieved score distribution (before filter):")
        print(f"[RESULTS]   - positive (GT hits): n={len(pos_scores)}, min={float(np.min(pos_scores)):.4f}, max={float(np.max(pos_scores)):.4f}, avg={float(np.mean(pos_scores)):.4f}")
        print(f"[RESULTS]   - negative (FP/noise): n={len(neg_scores)}, min={float(np.min(neg_scores)):.4f}, max={float(np.max(neg_scores)):.4f}, avg={float(np.mean(neg_scores)):.4f}")
    if pos_scores_filtered and neg_scores_filtered:
        print(f"[RESULTS] Retrieved score distribution (after top-N filter):")
        print(f"[RESULTS]   - positive (GT hits): n={len(pos_scores_filtered)}, min={float(np.min(pos_scores_filtered)):.4f}, max={float(np.max(pos_scores_filtered)):.4f}, avg={float(np.mean(pos_scores_filtered)):.4f}")
        print(f"[RESULTS]   - negative (FP/noise): n={len(neg_scores_filtered)}, min={float(np.min(neg_scores_filtered)):.4f}, max={float(np.max(neg_scores_filtered)):.4f}, avg={float(np.mean(neg_scores_filtered)):.4f}")
    print(f"[RESULTS] " + "-"*48)
    print(f"[RESULTS] Chunk size: {args.chunk_size}s")
    print(f"[RESULTS] Hop size: {args.hop_size}s")
    print(f"[RESULTS] Top-k per chunk: {args.top_k}")
    print(f"[RESULTS] Top-N filter: {'ENABLED' if args.enable_top_n_filter else 'DISABLED'}")
    print(f"[RESULTS] Terms per second: {args.terms_per_second}")
    print(f"[RESULTS] Index terms: {len(term_list)}")
    print(f"[RESULTS] GT filter: min_words={args.min_words}, min_chars={args.min_chars}")
    print(f"[RESULTS] restrict_index_to_eval_terms: {args.restrict_index_to_eval_terms}")
    print(f"[RESULTS] random_sample: {args.random_sample}, seed: {args.seed}")
    if args.save_plot_dir:
        print(f"[RESULTS] save_plot_dir: {args.save_plot_dir}")
    
    # ======== Compute Hit Rate vs Poison Rate curve ========
    print("\n" + "="*80)
    print("HIT RATE vs POISON RATE ANALYSIS")
    print("="*80)
    
    # Collect all scores for threshold sweep
    all_scores = []
    for r in results:
        for s in r['filtered_terms'].values():
            all_scores.append(s)
    
    hit_poison_results = []  # Initialize outside condition
    
    if all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        # Generate thresholds
        thresholds = np.linspace(min_score, max_score, 50)
        thresholds = np.concatenate([[0.0], thresholds])  # Include 0.0
        
        hit_poison_results = []
        for threshold in thresholds:
            hp = compute_hit_poison_rates(results, threshold)
            hit_poison_results.append(hp)
            
        # Print summary table
        print(f"{'Threshold':>10} | {'Hit Rate':>10} | {'Poison Rate':>12} | {'Non-Empty':>10} | {'Hits':>6} | {'Poison':>7}")
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
    
    # Print some example results
    print("\n" + "="*80)
    print("SAMPLE RESULTS (first 5)")
    print("="*80)
    for r in results[:5]:
        print(f"\n[Sample] {r['id']}")
        print(f"  Text: {r['text'][:100]}...")
        print(f"  Duration: {r['audio_duration']:.2f}s, Top-N: {r['top_n']}")
        print(f"  GT terms: {r['gt_terms']}")
        if r['hit_terms_filtered']:
            hit_with_scores = [(t, r['filtered_terms'].get(t)) for t in r['hit_terms_filtered']]
            print(f"  Hit GT terms (after filter): {hit_with_scores}")
        else:
            print("  Hit GT terms (after filter): []")
        print(f"  Filtered terms (top-N): {sorted(r['filtered_terms'].items(), key=lambda x: x[1], reverse=True)[:10]}...")
        print(f"  Hits (before/after filter): {r['hits']}/{r['hits_filtered']} of {len(r['gt_terms'])} GT")
        print(f"  Chunks: {r['num_chunks']}")

    # Optional: save score distribution plot and raw scores
    if args.save_plot_dir:
        os.makedirs(args.save_plot_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(args.save_plot_dir, f"acl6060_v2_scores_{ts}.json")
        meta_path = os.path.join(args.save_plot_dir, f"acl6060_v2_meta_{ts}.json")
        plot_path = os.path.join(args.save_plot_dir, f"acl6060_v2_score_distribution_{ts}.png")
        hp_curve_path = os.path.join(args.save_plot_dir, f"acl6060_v2_hit_poison_curve_{ts}.png")
        hp_data_path = os.path.join(args.save_plot_dir, f"acl6060_v2_hit_poison_data_{ts}.json")

        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "positive_scores": pos_scores,
                    "negative_scores": neg_scores,
                    "positive_scores_filtered": pos_scores_filtered,
                    "negative_scores_filtered": neg_scores_filtered,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": "acl6060_v2",
                    "chunk_size": args.chunk_size,
                    "hop_size": args.hop_size,
                    "top_k": args.top_k,
                    "terms_per_second": args.terms_per_second,
                    "enable_top_n_filter": args.enable_top_n_filter,
                    "max_samples": args.max_samples,
                    "min_words": args.min_words,
                    "min_chars": args.min_chars,
                    "score_definition": "1/(1+L2_distance)",
                    "num_positive_scores": len(pos_scores),
                    "num_negative_scores": len(neg_scores),
                    "num_positive_scores_filtered": len(pos_scores_filtered),
                    "num_negative_scores_filtered": len(neg_scores_filtered),
                    "recall_before_filter": accuracy,
                    "recall_after_filter": accuracy_filtered,
                    "precision": precision,
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "gt_hits": total_hits,
                    "gt_hits_filtered": total_hits_filtered,
                    "gt_total": total_gt_terms,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save Hit Rate vs Poison Rate data
        if all_scores:
            with open(hp_data_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "thresholds": [hp['threshold'] for hp in hit_poison_results],
                        "hit_rates": [hp['hit_rate'] for hp in hit_poison_results],
                        "poison_rates": [hp['poison_rate'] for hp in hit_poison_results],
                        "hit_counts": [hp['hit_count'] for hp in hit_poison_results],
                        "poison_counts": [hp['poison_count'] for hp in hit_poison_results],
                        "non_empty_counts": [hp['non_empty_count'] for hp in hit_poison_results],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"[INFO] Saved hit/poison data to: {hp_data_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Plot 1: Score distribution
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Before filtering
            ax1 = axes[0]
            bins = 60
            if pos_scores:
                ax1.hist(pos_scores, bins=bins, alpha=0.6, density=True, label="Positive (GT Hits)", color="green")
            if neg_scores:
                ax1.hist(neg_scores, bins=bins, alpha=0.6, density=True, label="Negative (FP/Noise)", color="red")
            ax1.set_title(f"Before Top-N Filter")
            ax1.set_xlabel("Similarity Score")
            ax1.set_ylabel("Density")
            ax1.legend()
            
            # After filtering
            ax2 = axes[1]
            if pos_scores_filtered:
                ax2.hist(pos_scores_filtered, bins=bins, alpha=0.6, density=True, label="Positive (GT Hits)", color="green")
            if neg_scores_filtered:
                ax2.hist(neg_scores_filtered, bins=bins, alpha=0.6, density=True, label="Negative (FP/Noise)", color="red")
            ax2.set_title(f"After Top-N Filter (N = ceil(duration * {args.terms_per_second}))")
            ax2.set_xlabel("Similarity Score")
            ax2.set_ylabel("Density")
            ax2.legend()
            
            plt.suptitle(f"ACL6060 Score Distribution (chunk={args.chunk_size}s, hop={args.hop_size}s)")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"[INFO] Saved score distribution plot to: {plot_path}")

            # Plot 2: Hit Rate vs Poison Rate curve
            if all_scores and hit_poison_results:
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
                
                ax.set_xlabel('Poison Rate (FP with no GT / Total)', fontsize=12)
                ax.set_ylabel('Hit Rate (with GT / Non-Empty)', fontsize=12)
                ax.set_title(f'Hit Rate vs Poison Rate Curve\n(Top-N Filter: N = ceil(duration * {args.terms_per_second}))', fontsize=14)
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
            
            print(f"[INFO] Saved raw scores to: {raw_path}")
            print(f"[INFO] Saved metadata to: {meta_path}")
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

