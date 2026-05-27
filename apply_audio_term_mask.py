import json
import os
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm
from typing import List, Dict, Tuple
import re

def parse_textgrid_robust(textgrid_path: str) -> List[Dict]:
    """Robust TextGrid parser for short format"""
    words = []
    if not os.path.exists(textgrid_path):
        return words
    try:
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # Find the "words" tier
        words_tier_index = -1
        for i, line in enumerate(lines):
            if line == '"words"':
                # Check if the previous line is '"IntervalTier"'
                if i > 0 and lines[i-1] == '"IntervalTier"':
                    words_tier_index = i
                    break
        
        if words_tier_index == -1:
            return words
            
        # After "words", we have xmin, xmax, and then interval count
        # xmin: lines[words_tier_index + 1]
        # xmax: lines[words_tier_index + 2]
        # interval_count: lines[words_tier_index + 3]
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
    except Exception as e:
        print(f"[ERROR] Failed to parse TextGrid {textgrid_path}: {e}")
    return words

def find_term_time_spans(words: List[Dict], term: str) -> List[Tuple[float, float]]:
    """Find all time spans for a specific term in the words list"""
    spans = []
    # Clean term: remove punctuation and lower case
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

def apply_mask(audio_data: np.ndarray, sr: int, start_time: float, end_time: float):
    """Apply Gaussian white noise to the specified time range"""
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    start_sample = max(0, start_sample)
    end_sample = min(len(audio_data), end_sample)
    
    if start_sample < end_sample:
        # Match power roughly or use fixed std
        noise_std = 0.05 
        audio_data[start_sample:end_sample] = np.random.normal(0, noise_std, end_sample - start_sample)

def get_textgrid_path(utter_id: str, base_dir: str) -> str:
    if '_' not in utter_id: return ""
    parts = utter_id.rsplit('_', 1)
    if len(parts) != 2: return ""
    base_id, segment_id = parts
    try:
        formatted_segment = f"S{int(segment_id):07d}"
        return os.path.join(base_dir, f"{base_id}_{formatted_segment}.TextGrid")
    except ValueError:
        return ""

def process_audio_masking(input_jsonl, output_jsonl, textgrid_dir, output_audio_base, limit=None, sample_count=5600):
    print(f"Reading instances from {input_jsonl}...")
    instances = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
            
    # Filter instances with TextGrid AND existing terms
    valid_candidates = []
    for inst in tqdm(instances, desc="Filtering candidates"):
        tg_path = get_textgrid_path(inst['utter_id'], textgrid_dir)
        if os.path.exists(tg_path):
            # Check if has any gt terms
            if any(inst.get('gt_terms_by_chunk', [])):
                valid_candidates.append(inst)
    
    print(f"Found {len(valid_candidates)} candidates with TextGrid.")
    
    # Sampling
    if limit:
        sampled_instances = valid_candidates[:limit]
    else:
        sampled_instances = random.sample(valid_candidates, min(sample_count, len(valid_candidates)))
    
    print(f"Processing {len(sampled_instances)} instances for masking...")
    
    os.makedirs(output_audio_base, exist_ok=True)
    
    results = []
    mask_stats = {"instances": 0, "chunks": 0, "terms_matched": 0, "terms_skipped": 0}
    
    for inst in tqdm(sampled_instances, desc="Masking audio"):
        utter_id = inst['utter_id']
        tg_path = get_textgrid_path(utter_id, textgrid_dir)
        words = parse_textgrid_robust(tg_path)
        
        audios = inst.get('audios', [])
        gt_terms_by_chunk = inst.get('gt_terms_by_chunk', [])
        new_audios = list(audios)
        
        # Assume 1.92s chunk, 0.96s hop
        hop_size = 0.96
        
        instance_modified = False
        for i in range(len(gt_terms_by_chunk)):
            if i >= len(audios): break
            terms = gt_terms_by_chunk[i]
            if not terms: continue
            
            audio_path = audios[i]
            if not os.path.exists(audio_path): continue
            
            audio_data, sr = sf.read(audio_path)
            chunk_start_in_segment = i * hop_size
            
            chunk_modified = False
            for term_obj in terms:
                term_en = term_obj['term']
                spans = find_term_time_spans(words, term_en)
                
                if not spans:
                    mask_stats["terms_skipped"] += 1
                    continue
                
                for t_start, t_end in spans:
                    # Translate segment time to chunk relative time
                    rel_start = t_start - chunk_start_in_segment
                    rel_end = t_end - chunk_start_in_segment
                    
                    # Check overlap with chunk [0, duration]
                    chunk_dur = len(audio_data) / sr
                    if rel_start < chunk_dur and rel_end > 0:
                        apply_mask(audio_data, sr, rel_start, rel_end)
                        chunk_modified = True
                        mask_stats["terms_matched"] += 1
            
            if chunk_modified:
                # Save masked audio
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

    print(f"\nMasking Statistics:")
    print(f"  - Instances modified: {mask_stats['instances']}")
    print(f"  - Chunks modified: {mask_stats['chunks']}")
    print(f"  - Terms matched & masked: {mask_stats['terms_matched']}")
    print(f"  - Terms not found in TextGrid: {mask_stats['terms_skipped']}")

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
    output_file = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_GT_masked.jsonl"
    textgrid_dir = "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"
    output_audio_base = "/mnt/gemini/data1/jiaxuanluo/masked_audios_v4"
    
    limit = args.limit
    if args.test and not limit:
        limit = 10
        
    results = process_audio_masking(input_file, output_file, textgrid_dir, output_audio_base, limit=limit)
    
    if args.test or limit:
        print("\n=== Test Samples (Random 10) ===")
        # Only sample from those that actually got masked
        masked_results = [r for r in results if any('_masked.wav' in a for a in r['audios'])]
        if not masked_results:
            print("No samples were masked. Check TextGrid content matching.")
            # Show one for debug
            if results:
                print(f"Debug Info for first result ({results[0]['utter_id']}):")
                tg = get_textgrid_path(results[0]['utter_id'], textgrid_dir)
                print(f"TextGrid: {tg}")
                w = parse_textgrid_robust(tg)
                print(f"Parsed Words: {[item['word'] for item in w[:10]]}")
                print(f"GT Terms: {results[0]['gt_terms_by_chunk']}")
        else:
            sampled_results = random.sample(masked_results, min(10, len(masked_results)))
            for i, inst in enumerate(sampled_results):
                print(f"\nSample {i+1}:")
                print(f"Utter ID: {inst['utter_id']}")
                masked = [a for a in inst['audios'] if '_masked.wav' in a]
                print(f"Masked Audio Paths: {masked}")
                non_empty_gt = [(idx, g) for idx, g in enumerate(inst['gt_terms_by_chunk']) if g]
                print(f"GT Terms (chunk_idx, terms): {non_empty_gt}")
