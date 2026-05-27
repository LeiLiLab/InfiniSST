import json
import os
import glob
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm
from typing import List, Dict, Tuple
import re

# Global cache for base_id -> list of TextGrid paths
base_id_to_tgs_cache = {}

def get_all_tgs_for_id(base_id, textgrid_dir):
    if base_id in base_id_to_tgs_cache:
        return base_id_to_tgs_cache[base_id]
    pattern = os.path.join(textgrid_dir, f"{base_id}_S*.TextGrid")
    files = glob.glob(pattern)
    base_id_to_tgs_cache[base_id] = files
    return files

def parse_textgrid_robust(textgrid_path: str) -> List[Dict]:
    """Robust TextGrid parser for short format"""
    words = []
    if not os.path.exists(textgrid_path):
        return words
    try:
        with open(textgrid_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines()]
        
        words_tier_index = -1
        for i, line in enumerate(lines):
            if line == '"words"':
                if i > 0 and lines[i-1] == '"IntervalTier"':
                    words_tier_index = i
                    break
        
        if words_tier_index == -1:
            return words
            
        try:
            interval_count = int(lines[words_tier_index + 3])
        except (ValueError, IndexError):
            return words
            
        current_idx = words_tier_index + 4
        for _ in range(interval_count):
            if current_idx + 2 >= len(lines):
                break
            try:
                xmin = float(lines[current_idx])
                xmax = float(lines[current_idx + 1])
                text = lines[current_idx + 2].strip('"')
                
                if text and text not in ['<SIL>', 'SIL', '<s>', '</s>', 'sp', 'sil', '']:
                    words.append({
                        "word": text.lower(),
                        "start": xmin,
                        "end": xmax
                    })
                current_idx += 3
            except (ValueError, IndexError):
                current_idx += 3
                continue
    except Exception:
        pass
    return words

def find_term_time_spans(words: List[Dict], term: str) -> List[Tuple[float, float]]:
    """Find all time spans for a specific term in the words list"""
    spans = []
    clean_term = re.sub(r'[^\w\s]', '', term).lower().strip()
    term_words = clean_term.split()
    if not term_words:
        return spans
        
    word_texts = [w["word"] for w in words]
    for i in range(len(word_texts) - len(term_words) + 1):
        if word_texts[i:i+len(term_words)] == term_words:
            start_time = words[i]["start"]
            end_time = words[i + len(term_words) - 1]["end"]
            spans.append((start_time, end_time))
    return spans

def find_correct_textgrid(inst, textgrid_dir) -> Tuple[str, List[Dict]]:
    """Find the correct TextGrid by matching gt_terms with content"""
    utter_id = inst['utter_id']
    base_id = utter_id.split('_')[0]
    
    # Collect all GT terms from this instance
    all_gt_terms = []
    for chunk in inst.get('gt_terms_by_chunk', []):
        for t in chunk:
            clean = re.sub(r'[^\w\s]', '', t['term']).lower().strip()
            if clean: all_gt_terms.append(clean)
    
    if not all_gt_terms:
        return None, []
    
    # Deduplicate for faster matching
    unique_gt_terms = list(set(all_gt_terms))
    
    candidate_files = get_all_tgs_for_id(base_id, textgrid_dir)
    if not candidate_files:
        return None, []

    best_tg = None
    best_words = []
    max_matches = 0
    
    # To speed up, if many candidates, we might want to prioritize files near the segment index
    # but for now let's just search
    for tg_path in candidate_files:
        words = parse_textgrid_robust(tg_path)
        if not words: continue
        
        tg_text = " ".join([w['word'] for w in words])
        matches = sum(1 for term in unique_gt_terms if term in tg_text)
        
        if matches > max_matches:
            max_matches = matches
            best_tg = tg_path
            best_words = words
            
        # Early exit if we find a very good match
        if matches >= len(unique_gt_terms) * 0.9 and matches > 0:
            return tg_path, words
            
    # Require at least some threshold of matching terms
    if max_matches >= min(2, len(unique_gt_terms)) or (len(unique_gt_terms) == 1 and max_matches == 1):
        return best_tg, best_words
        
    return None, []

def apply_mask(audio_data: np.ndarray, sr: int, start_time: float, end_time: float):
    """Apply Gaussian white noise to the specified time range"""
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    start_sample = max(0, start_sample)
    end_sample = min(len(audio_data), end_sample)
    
    if start_sample < end_sample:
        noise_std = 0.05 
        audio_data[start_sample:end_sample] = np.random.normal(0, noise_std, end_sample - start_sample)

def process_audio_masking(input_jsonl, output_jsonl, textgrid_dir, output_audio_base, limit=None, sample_count=5600):
    print(f"Reading instances from {input_jsonl}...")
    instances = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
            
    print(f"Loaded {len(instances)} instances. Searching for matching TextGrids via content...")
    
    valid_candidates = []
    # Using a smaller subset for filtering if we're just searching for 5.6k anyway
    # to avoid parsing every single TextGrid in the repo.
    # But since we need 5.6k random ones, we should search until we find enough.
    
    random.shuffle(instances)
    
    os.makedirs(output_audio_base, exist_ok=True)
    results = []
    mask_stats = {"instances": 0, "chunks": 0, "terms_matched": 0, "terms_skipped": 0}
    
    pbar = tqdm(total=limit if limit else sample_count, desc="Finding and Masking")
    
    for inst in instances:
        if len(results) >= (limit if limit else sample_count):
            break
            
        tg_path, words = find_correct_textgrid(inst, textgrid_dir)
        if not tg_path:
            continue
            
        # Store the matched path in the instance for later display
        inst['matched_textgrid'] = tg_path
        
        audios = inst.get('audios', [])
        gt_terms_by_chunk = inst.get('gt_terms_by_chunk', [])
        new_audios = list(audios)
        hop_size = 0.96
        
        instance_modified = False
        for i in range(len(gt_terms_by_chunk)):
            if i >= len(audios): break
            terms = gt_terms_by_chunk[i]
            if not terms: continue
            
            audio_path = audios[i]
            if not os.path.exists(audio_path): continue
            
            try:
                audio_data, sr = sf.read(audio_path)
            except Exception:
                continue
                
            chunk_start_in_segment = i * hop_size
            chunk_modified = False
            
            for term_obj in terms:
                term_en = term_obj['term']
                spans = find_term_time_spans(words, term_en)
                
                if not spans:
                    mask_stats["terms_skipped"] += 1
                    continue
                
                for t_start, t_end in spans:
                    rel_start = t_start - chunk_start_in_segment
                    rel_end = t_end - chunk_start_in_segment
                    chunk_dur = len(audio_data) / sr
                    if rel_start < chunk_dur and rel_end > 0:
                        apply_mask(audio_data, sr, rel_start, rel_end)
                        chunk_modified = True
                        mask_stats["terms_matched"] += 1
            
            if chunk_modified:
                rel_path = os.path.relpath(audio_path, "/")
                masked_filename = rel_path.replace(".wav", "_masked.wav").replace("/", "_")
                masked_path = os.path.join(output_audio_base, masked_filename)
                sf.write(masked_path, audio_data, sr)
                new_audios[i] = masked_path
                mask_stats["chunks"] += 1
                instance_modified = True
        
        if instance_modified:
            mask_stats["instances"] += 1
            inst['audios'] = new_audios
            results.append(inst)
            pbar.update(1)
        else:
            # Even if not modified (matched TextGrid but times didn't align),
            # we count it as a result if we really want to reach the quota?
            # User said "from matching textgrid instances, sample 5.6k and do mask".
            # If masking fails due to time mismatch, maybe skip.
            pass

    pbar.close()
    print(f"\nMasking Statistics:")
    print(f"  - Instances modified: {mask_stats['instances']}")
    print(f"  - Chunks modified: {mask_stats['chunks']}")
    print(f"  - Terms matched & masked: {mask_stats['terms_matched']}")
    print(f"  - Terms missing in found TextGrid: {mask_stats['terms_skipped']}")

    print(f"Writing results to {output_jsonl}...")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for inst in results:
            f.write(json.dumps(inst, ensure_ascii=False) + '\n')
            
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    input_file = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_GT.jsonl"
    output_file = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_GT_masked_v2.jsonl"
    textgrid_dir = "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"
    output_audio_base = "/mnt/gemini/data1/jiaxuanluo/masked_audios_v4_content_match"
    
    limit = args.limit
    if args.test and not limit:
        limit = 10
        
    results = process_audio_masking(input_file, output_file, textgrid_dir, output_audio_base, limit=limit)
    
    if args.test or limit:
        print("\n=== Test Samples (Random 10) ===")
        sampled_results = random.sample(results, min(10, len(results)))
        for i, inst in enumerate(sampled_results):
            print(f"\nSample {i+1}:")
            print(f"Utter ID: {inst['utter_id']}")
            print(f"Matched TextGrid: {inst.get('matched_textgrid', 'N/A')}")
            masked = [a for a in inst['audios'] if '_masked.wav' in a]
            print(f"Masked Audio Paths: {masked}")
            non_empty_gt = [(idx, g) for idx, g in enumerate(inst['gt_terms_by_chunk']) if g]
            print(f"GT Terms (chunk_idx, terms): {non_empty_gt}")
