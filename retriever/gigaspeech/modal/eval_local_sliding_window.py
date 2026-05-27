#!/usr/bin/env python3
"""
Sliding Window Evaluation Script

Evaluates the retriever using real chunk format:
- Load audio from TSV (opus format with offset/length)
- Use 2s fixed chunk with 1s sliding window
- Aggregate recall@5 terms from all chunks
- Calculate hit rate against GT terms from glossary
"""

import os
import sys
import argparse
import json
import torch
import faiss
import pickle
import re
import subprocess
import tempfile
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


def load_audio_from_opus(audio_spec, target_sr=16000):
    """
    Load audio from opus file with offset/length specification.
    
    Args:
        audio_spec: Format like "/path/to/file.opus:offset:length"
        target_sr: Target sample rate (default 16000)
    
    Returns:
        numpy array of audio samples
    """
    parts = audio_spec.split(":")
    if len(parts) == 3:
        audio_path, offset, length = parts[0], int(parts[1]), int(parts[2])
    else:
        audio_path = audio_spec
        offset, length = 0, -1
    
    try:
        import soundfile as sf
        
        # For opus files, we need to use ffmpeg to extract the segment
        if audio_path.endswith('.opus'):
            # Calculate time offset and duration in seconds
            # Assuming original sample rate of 16000 for opus
            start_time = offset / target_sr
            duration = length / target_sr if length > 0 else None
            
            # Use ffmpeg to extract and convert
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
                cmd = ['ffmpeg', '-y', '-i', audio_path]
                if start_time > 0:
                    cmd.extend(['-ss', str(start_time)])
                if duration:
                    cmd.extend(['-t', str(duration)])
                cmd.extend(['-ar', str(target_sr), '-ac', '1', '-f', 'wav', tmp.name])
                
                result = subprocess.run(cmd, capture_output=True, check=False)
                if result.returncode != 0:
                    return None
                
                audio, sr = sf.read(tmp.name)
                return audio.astype(np.float32)
        else:
            # For wav files, use soundfile directly
            audio, sr = sf.read(audio_path, start=offset, stop=offset+length if length > 0 else None)
            if sr != target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            return audio.astype(np.float32)
            
    except Exception as e:
        print(f"[WARN] Failed to load audio {audio_spec}: {e}")
        return None


from flashtext import KeywordProcessor

def find_gt_terms_flashtext(text, glossary_terms, min_words=1, min_chars=3):
    """
    使用 FlashText 进行极速多模匹配，自动处理最长匹配优先
    
    Args:
        text: 要搜索的文本
        glossary_terms: glossary 中的 term 集合
        min_words: 最少词数（默认1，设为2可只保留多词术语如人名）
        min_chars: 最少字符数（默认3）
    
    Note: 单词术语如 "sense", "lot", "got" 容易与普通词混淆，
          建议设置 min_words=2 只保留真正的专有名词（人名、地名等）
    """
    keyword_processor = KeywordProcessor(case_sensitive=False)
    
    # 过滤 glossary terms
    for term in glossary_terms:
        word_count = len(term.split())
        if len(term) >= min_chars and word_count >= min_words:
            keyword_processor.add_keyword(term)
            
    # extract_keywords 默认就是 Longest Match First
    found_keywords = keyword_processor.extract_keywords(text)
    
    # 去重
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
    Retrieve terms for all chunks and aggregate results.
    
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


def load_tsv_data(tsv_path, max_samples=1000, random_sample=False, seed=42):
    """
    Load data from TSV file.
    
    Args:
        tsv_path: Path to TSV file
        max_samples: Maximum number of samples to load
        random_sample: If True, randomly sample max_samples rows from the TSV (excluding header)
        seed: Random seed for reproducibility
    
    Returns:
        List of dicts with id, audio_spec, text
    """
    rng = random.Random(int(seed))

    def parse_line(line: str):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            return None
        return {
            "id": parts[0],
            "audio_spec": parts[1],
            "n_frames": int(parts[2]) if parts[2].isdigit() else 0,
            "speaker": parts[3],
            "text": parts[4],
            "lang": parts[5] if len(parts) > 5 else "en",
        }

    samples = []
    seen = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        _header = f.readline()
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            if not random_sample:
                samples.append(parsed)
                if len(samples) >= max_samples:
                    break
                continue

            # Reservoir sampling
            seen += 1
            if len(samples) < max_samples:
                samples.append(parsed)
            else:
                j = rng.randrange(seen)
                if j < max_samples:
                    samples[j] = parsed

    if random_sample:
        rng.shuffle(samples)
    return samples


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


def restrict_index_to_terms(faiss_index, term_list, keep_terms_lc: set[str]):
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
    parser = argparse.ArgumentParser(description="Sliding Window Evaluation Script")
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--prebuilt_index', type=str, required=True, help='Path to prebuilt index (.pkl)')
    parser.add_argument('--glossary_path', type=str, required=True, help='Path to glossary_used.json')
    parser.add_argument('--tsv_path', type=str, required=True, help='Path to TSV file with test data')
    
    # Optional arguments
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help='Base model name')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum samples to evaluate')
    parser.add_argument('--chunk_size', type=float, default=2.0, help='Chunk size in seconds')
    parser.add_argument('--hop_size', type=float, default=1.0, help='Hop size (sliding window step) in seconds')
    parser.add_argument('--top_k', type=int, default=5, help='Top-k terms to retrieve per chunk')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for audio encoding')
    parser.add_argument('--min_words', type=int, default=2, help='Minimum word count for GT terms (2=multi-word only, filters out common words like "sense", "lot")')
    parser.add_argument('--min_chars', type=int, default=3, help='Minimum character count for GT terms')
    parser.add_argument('--save_plot_dir', type=str, default=None, help='If set, save score distribution plot and raw scores to this directory')
    parser.add_argument('--restrict_index_to_eval_terms', action='store_true', help='If set, restrict index/term_list to GT terms found in the first max_samples TSV rows (speeds up evaluation)')
    parser.add_argument('--random_sample', action='store_true', help='If set, randomly sample max_samples rows from TSV instead of taking the first rows')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for TSV sampling and any randomized steps')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!")
        return 1
    
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    
    # Load glossary for GT term matching
    print("\n" + "="*80)
    print("LOADING GLOSSARY")
    print("="*80)
    print(f"[INFO] Loading glossary from: {args.glossary_path}")
    with open(args.glossary_path, 'r', encoding='utf-8') as f:
        glossary = json.load(f)
    
    glossary_terms = set(glossary.keys())  # Keys are already lowercase
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
    
    # Load TSV data
    print("\n" + "="*80)
    print("LOADING TSV DATA")
    print("="*80)
    print(f"[INFO] Loading data from: {args.tsv_path}")
    samples = load_tsv_data(
        args.tsv_path,
        max_samples=args.max_samples,
        random_sample=args.random_sample,
        seed=args.seed,
    )
    print(f"[INFO] Loaded {len(samples)} samples")

    # Optional: restrict index to terms present in this eval subset
    if args.restrict_index_to_eval_terms:
        print("\n" + "="*80)
        print("RESTRICTING INDEX TO EVAL TERMS")
        print("="*80)
        eval_term_set = set()
        for s in samples:
            gt_terms_tmp = find_gt_terms_flashtext(
                s.get("text", ""),
                glossary_terms,
                min_words=args.min_words,
                min_chars=args.min_chars,
            )
            for t in gt_terms_tmp:
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
    
    total_gt_terms = 0
    total_hits = 0
    gt_hit_scores = []
    pos_scores = []  # retrieved terms that are in GT
    neg_scores = []  # retrieved terms that are NOT in GT (false positives / noise)
    samples_with_terms = 0
    samples_processed = 0
    failed_audio = 0
    
    results = []
    
    for sample in tqdm(samples, desc="Evaluating"):
        # Find GT terms in text (use min_words=2 to filter out common words like "sense", "lot")
        gt_terms = find_gt_terms_flashtext(
            sample['text'], 
            glossary_terms, 
            min_words=args.min_words,
            min_chars=args.min_chars
        )

        if not gt_terms:
            continue
        
        samples_with_terms += 1
        
        # Load audio
        audio = load_audio_from_opus(sample['audio_spec'])
        if audio is None:
            failed_audio += 1
            continue
        
        samples_processed += 1
        
        # Create sliding window chunks
        chunks = create_sliding_window_chunks(
            audio, 
            chunk_size=args.chunk_size, 
            hop_size=args.hop_size
        )
        
        # Retrieve terms from all chunks
        retrieved_terms = retrieve_terms_for_chunks(
            model, retriever, chunks, device, 
            top_k=args.top_k, 
            batch_size=args.batch_size
        )
        
        # Calculate hits
        gt_set = set(gt_terms)
        retrieved_set = set(retrieved_terms.keys())
        hit_terms = sorted(gt_set & retrieved_set)
        hits = len(hit_terms)
        total_gt_terms += len(gt_terms)
        total_hits += hits
        for t in hit_terms:
            gt_hit_scores.append(retrieved_terms[t])
            pos_scores.append(retrieved_terms[t])
        for t, s in retrieved_terms.items():
            if t not in gt_set:
                neg_scores.append(s)
        
        results.append({
            'id': sample['id'],
            'text': sample['text'],
            'gt_terms': gt_terms,
            'hit_terms': hit_terms,
            'gt_term_scores': {t: retrieved_terms.get(t) for t in gt_terms},
            # keep a small debug view: top 20 retrieved terms by score
            'retrieved_top': sorted(retrieved_terms.items(), key=lambda x: x[1], reverse=True)[:20],
            'hits': hits,
            'num_chunks': len(chunks)
        })
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    accuracy = total_hits / total_gt_terms if total_gt_terms > 0 else 0
    if gt_hit_scores:
        gt_score_min = float(np.min(gt_hit_scores))
        gt_score_max = float(np.max(gt_hit_scores))
        gt_score_avg = float(np.mean(gt_hit_scores))
    else:
        gt_score_min = None
        gt_score_max = None
        gt_score_avg = None
    
    print(f"[RESULTS] Total samples in TSV: {len(samples)}")
    print(f"[RESULTS] Samples with GT terms: {samples_with_terms}")
    print(f"[RESULTS] Samples processed (audio loaded): {samples_processed}")
    print(f"[RESULTS] Failed audio loads: {failed_audio}")
    print(f"[RESULTS] " + "-"*48)
    print(f"[RESULTS] Total GT terms: {total_gt_terms}")
    print(f"[RESULTS] Total hits: {total_hits}")
    print(f"[RESULTS] Global Accuracy (Hit Rate): {accuracy:.2%}")
    print(f"[RESULTS] " + "-"*48)
    print(f"[RESULTS] GT term score stats (hits only, score=1/(1+L2_distance)):")
    print(f"[RESULTS]   - count: {len(gt_hit_scores)}")
    print(f"[RESULTS]   - min: {gt_score_min}")
    print(f"[RESULTS]   - max: {gt_score_max}")
    print(f"[RESULTS]   - avg: {gt_score_avg}")
    if pos_scores and neg_scores:
        print(f"[RESULTS] " + "-"*48)
        print(f"[RESULTS] Retrieved score distribution (max over windows):")
        print(f"[RESULTS]   - positive (GT hits): n={len(pos_scores)}, min={float(np.min(pos_scores))}, max={float(np.max(pos_scores))}, avg={float(np.mean(pos_scores))}")
        print(f"[RESULTS]   - negative (FP/noise): n={len(neg_scores)}, min={float(np.min(neg_scores))}, max={float(np.max(neg_scores))}, avg={float(np.mean(neg_scores))}")
    print(f"[RESULTS] " + "-"*48)
    print(f"[RESULTS] Chunk size: {args.chunk_size}s")
    print(f"[RESULTS] Hop size: {args.hop_size}s")
    print(f"[RESULTS] Top-k per chunk: {args.top_k}")
    print(f"[RESULTS] Index terms: {len(term_list)}")
    print(f"[RESULTS] GT filter: min_words={args.min_words}, min_chars={args.min_chars}")
    print(f"[RESULTS] restrict_index_to_eval_terms: {args.restrict_index_to_eval_terms}")
    print(f"[RESULTS] random_sample: {args.random_sample}, seed: {args.seed}")
    if args.save_plot_dir:
        print(f"[RESULTS] save_plot_dir: {args.save_plot_dir}")
    
    # Print some example results
    print("\n" + "="*80)
    print("SAMPLE RESULTS (first 5)")
    print("="*80)
    for r in results[:5]:
        print(f"\n[Sample] {r['id']}")
        print(f"  Text: {r['text'][:100]}...")
        print(f"  GT terms: {r['gt_terms']}")
        if r['hit_terms']:
            hit_with_scores = [(t, r['gt_term_scores'].get(t)) for t in r['hit_terms']]
            print(f"  Hit GT terms (term, score): {hit_with_scores}")
        else:
            print("  Hit GT terms (term, score): []")
        print(f"  Retrieved top (term, score): {r['retrieved_top'][:10]}...")
        print(f"  Hits: {r['hits']}/{len(r['gt_terms'])}")
        print(f"  Chunks: {r['num_chunks']}")

    # Optional: save score distribution plot and raw scores
    if args.save_plot_dir:
        os.makedirs(args.save_plot_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(args.save_plot_dir, f"sliding_scores_{ts}.json")
        meta_path = os.path.join(args.save_plot_dir, f"sliding_scores_meta_{ts}.json")
        plot_path = os.path.join(args.save_plot_dir, f"score_distribution_{ts}.png")

        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "positive_scores": pos_scores,
                    "negative_scores": neg_scores,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunk_size": args.chunk_size,
                    "hop_size": args.hop_size,
                    "top_k": args.top_k,
                    "max_samples": args.max_samples,
                    "min_words": args.min_words,
                    "min_chars": args.min_chars,
                    "score_definition": "1/(1+L2_distance)",
                    "num_positive_scores": len(pos_scores),
                    "num_negative_scores": len(neg_scores),
                    "accuracy": accuracy,
                    "gt_hits": total_hits,
                    "gt_total": total_gt_terms,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 4))
            bins = 60
            if pos_scores:
                plt.hist(pos_scores, bins=bins, alpha=0.6, density=True, label="Positive (GT Hits)", color="green")
            if neg_scores:
                plt.hist(neg_scores, bins=bins, alpha=0.6, density=True, label="Negative (FP/Noise)", color="red")
            plt.title("Score Distribution: Positive vs Negative Samples")
            plt.xlabel("Similarity Score (1/(1+L2_distance))")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"[INFO] Saved score distribution plot to: {plot_path}")
            print(f"[INFO] Saved raw scores to: {raw_path}")
            print(f"[INFO] Saved metadata to: {meta_path}")
        except Exception as e:
            print(f"[WARN] Failed to save plot: {e}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




















