import json
import os
import random
import torch
import numpy as np
import librosa
import pickle
import torch.multiprocessing as mp
from tqdm import tqdm
from typing import List, Dict

# 导入检索器类
from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4

def generate_term_map_string(terms: List[Dict]) -> str:
    if not terms:
        return ""
    seen = set()
    unique_terms = []
    for t in terms:
        if t['key'] not in seen:
            unique_terms.append(t)
            seen.add(t['key'])
    
    lines = ["term_map:"]
    for t in unique_terms:
        lines.append(f"{t['term']}={t['translation']}")
    return "\n".join(lines)

def parse_existing_term_map(content: str) -> List[Dict]:
    if "term_map:" not in content:
        return []
    lines = content.split('\n')
    terms = []
    found_header = False
    for line in lines:
        if line.strip() == "term_map:":
            found_header = True
            continue
        if found_header and "=" in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                term, trans = parts
                terms.append({"key": term.lower(), "term": term, "translation": trans})
    return terms

def worker(rank, world_size, input_lines, output_path, config):
    """每个 GPU 对应的进程执行的任务"""
    device = f"cuda:{rank}"
    print(f"[Worker {rank}] Initializing on {device}...")
    
    retriever = StreamingQwen3RAGRetrieverV4(
        index_path=config['index_path'],
        model_path=config['model_path'],
        device=device,
        lora_r=32,
        lora_alpha=64,
        text_lora_r=16,
        top_k=20,
        voting_k=20,
        score_threshold=0.0,
        verbose=False
    )

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(input_lines, desc=f"GPU {rank}", position=rank):
            if not line.strip():
                f_out.write(line)
                continue
            
            instance = json.loads(line)
            messages = instance['messages']
            audio_paths = instance.get('audios', [])
            
            audio_idx = 0
            for i, msg in enumerate(messages):
                if msg['role'] == 'user' and "<audio>" in msg['content']:
                    if audio_idx < len(audio_paths):
                        audio_path = audio_paths[audio_idx]
                        if not os.path.exists(audio_path):
                            audio_idx += 1
                            continue
                        
                        try:
                            y, sr = librosa.load(audio_path, sr=16000)
                            duration = len(y) / sr
                            retriever.reset()
                            retriever.accumulate_audio(y, force_process=True)
                            
                            all_candidates = []
                            for term_lc, score in retriever._term_scores.items():
                                term_info = retriever.term_map.get(term_lc)
                                if term_info:
                                    all_candidates.append({
                                        "key": term_lc,
                                        "term": term_info['term'],
                                        "translation": term_info['target_translations'].get('zh', ''),
                                        "score": score
                                    })
                            all_candidates.sort(key=lambda x: x['score'], reverse=True)
                            
                            existing_terms = parse_existing_term_map(msg['content'])
                            existing_keys = {t['key'] for t in existing_terms}
                            negatives = [c for c in all_candidates if c['key'] not in existing_keys]
                            
                            k = random.randint(0, int(duration * 9))
                            selected_negatives = negatives[:k]
                            combined = existing_terms + selected_negatives
                            random.shuffle(combined)
                            
                            msg['content'] = f"<audio>\n\n{generate_term_map_string(combined)}"
                        except Exception as e:
                            print(f"Error processing {audio_path}: {e}")
                        
                        audio_idx += 1
            
            f_out.write(json.dumps(instance, ensure_ascii=False) + '\n')

def process_file_multigpu(input_path, output_path, config, num_gpus=8):
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    # 将任务平分给 8 个进程
    chunk_size = (len(all_lines) + num_gpus - 1) // num_gpus
    processes = []
    temp_outputs = []

    mp.set_start_method('spawn', force=True)

    for rank in range(num_gpus):
        start_idx = rank * chunk_size
        end_idx = min((rank + 1) * chunk_size, len(all_lines))
        chunk_lines = all_lines[start_idx:end_idx]
        
        temp_out = f"{output_path}.tmp_{rank}"
        temp_outputs.append(temp_out)
        
        p = mp.Process(target=worker, args=(rank, num_gpus, chunk_lines, temp_out, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 合并临时文件
    print(f"Merging results into {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for temp_out in temp_outputs:
            with open(temp_out, 'r', encoding='utf-8') as f_temp:
                f_out.write(f_temp.read())
            os.remove(temp_out)
    print(f"Finished {output_path}")

def main():
    config = {
        "index_path": "/mnt/gemini/data1/jiaxuanluo/pool_A_index_v4.pkl",
        "model_path": "/mnt/gemini/data2/jiaxuanluo/q3rag_unfrozen_lora-r32-tr16_bs4k_w1.0-0.0_sampled_best_snapshot_v2.pt"
    }
    
    input_files = [
        "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_lagging.jsonl",
        "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_pool_A_GT.jsonl"
    ]
    
    for input_path in input_files:
        output_path = input_path.replace(".jsonl", "_enriched.jsonl")
        process_file_multigpu(input_path, output_path, config, num_gpus=8)

if __name__ == "__main__":
    main()