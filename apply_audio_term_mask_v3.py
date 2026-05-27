import json
import os
import glob
import random
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple
import re
import torch.multiprocessing as mp

# --- CONFIGURATION ---
TSV_PATH = "/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
LARGE_CLEANED_JSONL = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_ner_baseline_aligned_freq_k20_cleaned.jsonl"
# 基础合并文件，已包含 RAG 负样本
FINAL_MERGED_JSONL = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_final_merged_shuffled_with_uid.jsonl"
OUTPUT_FINAL_JSONL = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_final_v5_masked_enriched.jsonl"
TEXTGRID_DIR = "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"
MASKED_AUDIO_BASE = "/mnt/gemini/data1/jiaxuanluo/masked_audios_v5_final"

# 只允许对 pool_A_GT 进行 Mask
POOL_A_GT_FILE = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_GT.jsonl"

MASK_INSTANCE_LIMIT = 12000
MAX_MASKED_CLIPS_PER_INSTANCE = 2
MASK_PROBABILITY = 0.7

def apply_heavy_noise_mixing(audio_data: np.ndarray, sr: int, start_time: float, end_time: float, noise_ratio=0.7):
    start_sample = max(0, int(start_time * sr))
    end_sample = min(len(audio_data), int(end_time * sr))
    if start_sample < end_sample:
        segment = audio_data[start_sample:end_sample]
        noise = np.random.normal(0, 1, len(segment))
        signal_rms = np.sqrt(np.mean(segment**2)) + 1e-9
        noise = noise * (signal_rms / (np.sqrt(np.mean(noise**2)) + 1e-9))
        audio_data[start_sample:end_sample] = segment * (1 - noise_ratio) + noise * noise_ratio

def parse_textgrid_robust(textgrid_path: str) -> List[Dict]:
    words = []
    if not os.path.exists(textgrid_path): return words
    try:
        with open(textgrid_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines()]
        words_tier_index = -1
        for i, line in enumerate(lines):
            if line == '"words"':
                if i > 0 and lines[i-1] == '"IntervalTier"':
                    words_tier_index = i
                    break
        if words_tier_index == -1: return words
        interval_count = int(lines[words_tier_index + 3])
        current_idx = words_tier_index + 4
        for _ in range(interval_count):
            if current_idx + 2 >= len(lines): break
            try:
                xmin = float(lines[current_idx]); xmax = float(lines[current_idx + 1]); text = lines[current_idx + 2].strip('"')
                if text and text not in ['<SIL>', 'SIL', '<s>', '</s>', 'sp', 'sil', '']:
                    words.append({"word": text.lower(), "start": xmin, "end": xmax})
                current_idx += 3
            except: current_idx += 3; continue
    except: pass
    return words

def find_term_time_spans(words: List[Dict], term: str) -> List[Tuple[float, float]]:
    spans = []
    clean_term = re.sub(r'[^\w\s]', '', term).lower().strip()
    term_words = clean_term.split()
    if not term_words: return spans
    word_texts = [w["word"] for w in words]
    for i in range(len(word_texts) - len(term_words) + 1):
        if word_texts[i:i+len(term_words)] == term_words:
            spans.append((words[i]["start"], words[i + len(term_words) - 1]["end"]))
    return spans

def find_correct_textgrid_by_content(utter_id, src_text, textgrid_dir):
    base_id = utter_id.split('_')[0]
    pattern = os.path.join(textgrid_dir, f"{base_id}_S*.TextGrid")
    candidate_files = glob.glob(pattern)
    if not candidate_files: return None, []
    clean_src = re.sub(r'[^\w\s]', '', src_text).lower().strip()
    for tg_path in candidate_files:
        words = parse_textgrid_robust(tg_path)
        if not words: continue
        tg_text = " ".join([w['word'] for w in words])
        if tg_text in clean_src or clean_src in tg_text:
            return tg_path, words
    return None, []

def generate_term_map_string(terms: List[Dict]) -> str:
    if not terms: return "term_map:NONE"
    seen = set(); unique_terms = []
    for t in terms:
        k = t['term'].lower()
        if k not in seen: unique_terms.append(t); seen.add(k)
    lines = ["term_map:"]
    for t in unique_terms:
        zh = t.get('zh', t.get('translation', ''))
        lines.append(f"{t['term']}={zh}")
    return "\n".join(lines)

def worker_masking(rank, instances, gt_terms_map, metadata, neg_pool, output_audio_base):
    processed = []
    for inst in tqdm(instances, desc=f"Worker {rank}", position=rank):
        utter_id = inst['utter_id']
        src_text = metadata[utter_id]
        gt_terms_by_chunk = gt_terms_map[utter_id]
        
        tg_path, words = find_correct_textgrid_by_content(utter_id, src_text, TEXTGRID_DIR)
        
        audios = inst.get('audios', [])
        new_audios = list(audios)
        hop_size = 0.96
        
        masked_clips_in_instance = 0
        masked_chunk_indices = set()
        
        if tg_path:
            for i in range(len(gt_terms_by_chunk)):
                if i >= len(audios): break
                if masked_clips_in_instance >= MAX_MASKED_CLIPS_PER_INSTANCE: break
                
                terms = gt_terms_by_chunk[i]
                if not terms: continue
                
                # Candidate for masking: has multi-word term
                multi_word_terms = [t for t in terms if len(t['term'].split()) >= 2]
                if not multi_word_terms: continue
                
                # 70% probability to mask
                if random.random() >= MASK_PROBABILITY: continue
                
                target_term_obj = max(multi_word_terms, key=lambda x: len(x['term'].split()))
                
                audio_path = audios[i]
                if not os.path.exists(audio_path): continue
                try:
                    audio_data, sr = sf.read(audio_path)
                    spans = find_term_time_spans(words, target_term_obj['term'])
                    chunk_modified = False
                    for t_start, t_end in spans:
                        rel_start = t_start - (i * hop_size)
                        rel_end = t_end - (i * hop_size)
                        if rel_start < (len(audio_data)/sr) and rel_end > 0:
                            apply_heavy_noise_mixing(audio_data, sr, rel_start, rel_end)
                            chunk_modified = True
                    
                    if chunk_modified:
                        rel_path = os.path.relpath(audio_path, "/")
                        masked_filename = rel_path.replace(".wav", "_masked.wav").replace("/", "_")
                        masked_path = os.path.join(output_audio_base, masked_filename)
                        os.makedirs(os.path.dirname(masked_path), exist_ok=True)
                        sf.write(masked_path, audio_data, sr)
                        new_audios[i] = masked_path
                        masked_clips_in_instance += 1
                        masked_chunk_indices.add(i)
                except: continue
        
        inst['audios'] = new_audios
        
        # Enrichment logic: ONLY replace term_map for MASKED clips
        audio_chunk_idx = 0
        for m in inst['messages']:
            if m['role'] == 'user' and "<audio>" in m['content']:
                if audio_chunk_idx in masked_chunk_indices:
                    # ONLY for masked clips: replace with GT + 3-8 random negatives
                    gt_part = [{"term": t['term'], "zh": t['zh']} for t in gt_terms_by_chunk[audio_chunk_idx]]
                    num_neg = random.randint(3, 8)
                    negs = random.sample(neg_pool, min(num_neg, len(neg_pool)))
                    combined = gt_part + negs
                    random.shuffle(combined)
                    m['content'] = f"<audio>\n\n{generate_term_map_string(combined)}"
                # else: 保持原样 (merged文件中已经有 RAG 召回的 Top K 负样本)
                audio_chunk_idx += 1
        
        processed.append(inst)
    return processed

def load_metadata(path):
    print(f"Loading metadata from {path}...")
    df = pd.read_csv(path, sep='\t')
    return dict(zip(df['id'], df['src_text']))

def load_negative_terms_pool(path):
    print(f"Loading negative terms pool from {path}...")
    pool = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            for chunk in data.get('gt_terms_by_chunk', []):
                for t in chunk:
                    if t.get('term') and t.get('zh'): pool.append(t)
    return pool

def load_gt_terms_map(gt_file):
    print(f"Loading GT terms mapping for {os.path.basename(gt_file)}...")
    mapping = {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            mapping[data['utter_id']] = data.get('gt_terms_by_chunk', [])
    return mapping

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=MASK_INSTANCE_LIMIT)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    metadata = load_metadata(TSV_PATH)
    neg_pool = load_negative_terms_pool(LARGE_CLEANED_JSONL)
    # 核心：只加载 Ordinary GT 组的 GT 信息
    gt_terms_map = load_gt_terms_map(POOL_A_GT_FILE)
    
    print(f"Reading merged dataset from {FINAL_MERGED_JSONL}...")
    all_merged_inst = []
    with open(FINAL_MERGED_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): all_merged_inst.append(json.loads(line))
    
    # 筛选 pool_A_GT 中的候选者 (且包含多词术语)
    pool_a_gt_candidates = []
    for inst in all_merged_inst:
        uid = inst.get('utter_id')
        if uid in gt_terms_map and uid in metadata:
            # 检查是否有长词 (multi-word terms)
            has_multi = False
            for chunk in gt_terms_map[uid]:
                if any(len(t['term'].split()) >= 2 for t in chunk):
                    has_multi = True
                    break
            if has_multi:
                pool_a_gt_candidates.append(inst)
    
    print(f"Found {len(pool_a_gt_candidates)} Pool A GT candidates with multi-word terms.")
    
    sample_count = min(args.limit, len(pool_a_gt_candidates))
    random.seed(42)
    sampled_for_masking = random.sample(pool_a_gt_candidates, sample_count)
    
    print(f"Processing {len(sampled_for_masking)} instances for masking using {args.num_workers} workers...")
    os.makedirs(MASKED_AUDIO_BASE, exist_ok=True)
    
    chunks = np.array_split(sampled_for_masking, args.num_workers)
    with mp.Pool(args.num_workers) as pool:
        results_list = []
        for i in range(args.num_workers):
            res = pool.apply_async(worker_masking, args=(i, chunks[i].tolist(), gt_terms_map, metadata, neg_pool, MASKED_AUDIO_BASE))
            results_list.append(res)
        
        all_processed = []
        for r in results_list:
            all_processed.extend(r.get())
    
    processed_masked = {inst['utter_id']: inst for inst in all_processed}

    print(f"Writing final merged result to {OUTPUT_FINAL_JSONL}...")
    total_written = 0
    with open(OUTPUT_FINAL_JSONL, 'w', encoding='utf-8') as f_out:
        for inst in tqdm(all_merged_inst, desc="Writing"):
            uid = inst.get('utter_id')
            if uid in processed_masked:
                final_inst = processed_masked[uid]
            else:
                final_inst = inst
                # 普通实例或未选中的 pool_A_GT：确保 term_map 格式一致（如果缺失）
                for m in final_inst['messages']:
                    if m['role'] == 'user' and "<audio>" in m['content'] and "term_map:" not in m['content']:
                        m['content'] = m['content'].strip() + "\n\nterm_map:NONE"
            
            # 最终输出清理
            out_obj = {"messages": final_inst['messages'], "audios": final_inst['audios']}
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
            total_written += 1
    
    print(f"Done! Written {total_written} instances to {OUTPUT_FINAL_JSONL}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
