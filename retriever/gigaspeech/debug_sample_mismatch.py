#!/usr/bin/env python3
"""
调试样本和命名实体不匹配的问题
"""

import json
import os

def debug_sample_mismatch():
    # 读取预处理的样本文件
    samples_file = "data/samples/xl/term_preprocessed_samples_0_500000.json"
    ner_file = "data/named_entities_train_xl_split_0.json"
    tsv_file = "data/split_tsv/train_xl_split_0.tsv"
    
    if not os.path.exists(samples_file):
        print(f"[ERROR] Samples file not found: {samples_file}")
        return
    
    # 读取样本
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # 查找问题样本
    target_segment = "POD0000000302_S0000144"
    target_sample = None
    target_index = None
    
    for i, sample in enumerate(samples):
        if sample["segment_id"] == target_segment:
            target_sample = sample
            target_index = i
            break
    
    if not target_sample:
        print(f"[ERROR] Target sample {target_segment} not found in preprocessed samples")
        return
    
    print(f"[INFO] Found target sample at index {target_index}")
    print(f"[INFO] Sample text: {target_sample['text']}")
    print(f"[INFO] Ground truth terms: {target_sample['ground_truth_term']}")
    
    # 检查术语是否真的在文本中
    text = target_sample['text']
    for term in target_sample['ground_truth_term']:
        term_lower = term.lower()
        if term_lower in text:
            print(f"[CHECK] ✅ Term '{term}' (as '{term_lower}') found in text")
        else:
            print(f"[CHECK] ❌ Term '{term}' (as '{term_lower}') NOT found in text")
            # 检查是否是部分匹配
            term_words = term_lower.split()
            found_words = []
            for word in term_words:
                if word in text:
                    found_words.append(word)
            if found_words:
                print(f"[CHECK]   但找到了部分词汇: {found_words}")
    
    # 读取命名实体文件
    if os.path.exists(ner_file):
        with open(ner_file, 'r', encoding='utf-8') as f:
            named_entities_list = json.load(f)
        print(f"[INFO] NER file has {len(named_entities_list)} entries")
        
        # 检查对应的命名实体
        if target_index < len(named_entities_list):
            corresponding_ner = named_entities_list[target_index]
            print(f"[INFO] Corresponding NER: {corresponding_ner}")
            
            # 检查NER是否在文本中
            for ner in corresponding_ner:
                ner_lower = ner.lower()
                if ner_lower in text:
                    print(f"[NER CHECK] ✅ NER '{ner}' found in text")
                else:
                    print(f"[NER CHECK] ❌ NER '{ner}' NOT found in text")
        else:
            print(f"[ERROR] Index {target_index} out of range for NER file")
    
    # 读取原始TSV文件，查找该样本
    if os.path.exists(tsv_file):
        print(f"\n[INFO] Checking original TSV file...")
        with open(tsv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        found_in_tsv = False
        for i, line in enumerate(lines):
            if target_segment in line:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    segment_id, audio_path, n_frames, speaker, src_text = parts[:5]
                    print(f"[INFO] Found in TSV at line {i+1}")
                    print(f"[INFO] TSV segment_id: {segment_id}")
                    print(f"[INFO] TSV text: {src_text[:100]}...")
                    found_in_tsv = True
                    break
        
        if not found_in_tsv:
            print(f"[WARNING] Sample {target_segment} not found in TSV file")
    
    # 检查前几个样本看是否有模式
    print(f"\n[INFO] Checking first 3 samples for pattern...")
    for i in range(min(3, len(samples))):
        sample = samples[i]
        print(f"[SAMPLE {i}] ID: {sample['segment_id']}")
        print(f"[SAMPLE {i}] Text: {sample['text'][:50]}...")
        print(f"[SAMPLE {i}] Terms: {sample['ground_truth_term']}")
        print()

if __name__ == "__main__":
    debug_sample_mismatch() 