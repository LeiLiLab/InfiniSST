import json
import os
from tqdm import tqdm
import sys

def check_file(jsonl_path, textgrid_dir):
    utter_ids = set()
    print(f"\nProcessing {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading JSONL"):
            if not line.strip():
                continue
            try:
                instance = json.loads(line)
                utter_id = instance.get('utter_id')
                if utter_id:
                    utter_ids.add(utter_id)
            except json.JSONDecodeError:
                continue

    print(f"Found {len(utter_ids)} unique utter_ids.")

    match_count = 0
    missing_count = 0
    samples = []

    for utter_id in tqdm(utter_ids, desc="Checking TextGrids"):
        if '_' not in utter_id:
            missing_count += 1
            continue
            
        parts = utter_id.rsplit('_', 1)
        if len(parts) != 2:
            missing_count += 1
            continue
            
        base_id, segment_id = parts
        try:
            formatted_segment = f"S{int(segment_id):07d}"
            filename = f"{base_id}_{formatted_segment}.TextGrid"
            filepath = os.path.join(textgrid_dir, filename)
            
            if os.path.exists(filepath):
                match_count += 1
                if len(samples) < 3:
                    samples.append((utter_id, filename))
            else:
                missing_count += 1
        except ValueError:
            missing_count += 1

    print(f"--- Statistics for {os.path.basename(jsonl_path)} ---")
    print(f"Matched: {match_count} ({match_count/len(utter_ids)*100:.2f}%)")
    print(f"Missing: {missing_count} ({missing_count/len(utter_ids)*100:.2f}%)")
    if samples:
        for utter_id, filename in samples:
            print(f"  Sample: {utter_id} -> {filename}")

textgrid_dir = "/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"
check_file("/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_GT.jsonl", textgrid_dir)
check_file("/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_lagging.jsonl", textgrid_dir)

