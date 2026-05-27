import json
import os
import random
import argparse
from tqdm import tqdm

def merge_shuffle_sample(input_paths, output_base_path, sample_rate=1.0):
    all_instances = []
    
    print(f"Reading and processing instances...")
    for file_path in input_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping: {file_path}")
            continue
            
        print(f"Processing {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # KEEP utter_id for easy replacement later
                    final_obj = {
                        "utter_id": data.get("utter_id"),
                        "messages": data.get("messages", []),
                        "audios": data.get("audios", [])
                    }
                    all_instances.append(final_obj)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {file_path}")
                    continue

    total_count = len(all_instances)
    print(f"Total instances loaded: {total_count}")
    
    print(f"Shuffling {total_count} instances...")
    random.shuffle(all_instances)

    # Sampling
    num_to_sample = int(total_count * sample_rate)
    print(f"Sampling {num_to_sample} instances (rate: {sample_rate})...")
    sampled_instances = all_instances[:num_to_sample]

    # Generate output path
    output_path = f"{output_base_path}_with_uid.jsonl"

    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for inst in tqdm(sampled_instances, desc="Writing"):
            f_out.write(json.dumps(inst, ensure_ascii=False) + '\n')

    print(f"\nSuccessfully merged and shuffled.")
    print(f"Total instances written: {len(sampled_instances)}")
    print(f"Final file saved to: {output_path}")

if __name__ == "__main__":
    input_files = [
        "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_GT_enriched.jsonl",
        "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_lagging_enriched.jsonl",
        "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_B_baseline.jsonl",
        "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_Pool_B_distract.jsonl"
    ]
    output_base = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_final_merged_shuffled"
    random.seed(42)
    merge_shuffle_sample(input_files, output_base, 1.0)










