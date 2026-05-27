#!/usr/bin/env python3
import os
import sys
import json
import torch
import faiss
import pickle
import math
import argparse
import re
import numpy as np
from tqdm import tqdm
import soundfile as sf
from transformers import WhisperFeatureExtractor

# Import common RAG evaluation utilities
from retriever.gigaspeech.acl_eval_utils import (
    run_acl_simulation,
    extract_gt_terms_from_text,
    sequential_match,
    build_keyword_processor
)

# 导入 Qwen3 相关组件 (V4 版本)
from agents.streaming_qwen3_rag_retriever_v4 import StreamingQwen3RAGRetrieverV4

def load_text_lines(txt_path):
    results = []
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(line.strip())
    return results

def print_simulation_summary(sim_results):
    if not sim_results:
        print("\n[WRN] No results to summarize.")
        return

    print("\n" + "="*60)
    print(f"{'FINAL SIMULATION SUMMARY':^60}")
    print("="*60)
    print(f"Total Samples: {sim_results['used_samples']}")
    print(f"Total GT:      {sim_results['total_gt']}")
    print(f"Total Hits:    {sim_results['total_hits']} (Ordered)")
    print(f"Total FPs:     {sim_results['total_fps']}")
    print("-" * 60)
    print(f"Recall:        {sim_results['recall']:.2%}")
    print(f"Precision:     {sim_results['precision']:.2%}")
    print("-" * 60)
    print(f"Pos Score Mean: {sim_results['pos_score_mean']:.4f}")
    print(f"Neg Score Mean: {sim_results['neg_score_mean']:.4f}")
    print(f"Margin:         {sim_results['margin']:.4f}")
    print(f"Top1-Top5 Gap:  {sim_results['gap_top1_top5_mean']:.4f}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Real-time Simulation RAG Evaluation for Qwen3-Omni (V4)")
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--index_path', type=str, required=True, help='Path to .pkl index')
    parser.add_argument('--glossary_path', type=str, required=True, help='Path to glossary.json')
    parser.add_argument('--wav_dir', type=str, required=True, help='Directory with gold wav files')
    parser.add_argument('--txt_path', type=str, required=True, help='Path to gold transcript txt')
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--top_k', type=int, default=5, help='Top K for each translation request')
    parser.add_argument('--rag_voting_k', type=int, default=20, help='Internal Top K for voting strategy')
    parser.add_argument('--rag_voting_min_votes', type=int, default=3, help='Minimum votes for voting strategy')
    parser.add_argument('--rag_chunk_size', type=float, default=1.92)
    parser.add_argument('--rag_hop_size', type=float, default=0.48)
    parser.add_argument('--score_threshold', type=float, default=0.0, help='Threshold to filter terms after Top-K retrieval')
    parser.add_argument('--vllm_interval', type=float, default=1.92, help='Interval to simulate vLLM request')
    parser.add_argument('--rag_strategy', type=str, default='voting', choices=['voting', 'max_pool'])
    parser.add_argument('--max_samples', type=int, default=0, help='Max samples to evaluate (0 for all)')

    # Debug controls for offline evaluation prints
    parser.add_argument('--debug_print_limit', type=int, default=0, help='Print first N segment previews (0 to disable)')
    parser.add_argument('--debug_miss_limit', type=int, default=0, help='Print first N missed samples (0 to disable)')
    parser.add_argument('--merge_plural_terms', action='store_true', default=False, help='Merge plural terms by canonicalizing to singular (eval-only)')
    
    # V4 specific arguments
    parser.add_argument('--rag_lora_r', type=int, default=32)
    parser.add_argument('--rag_text_lora_r', type=int, default=16)
    
    args = parser.parse_args()

    # 1. 初始化检索器 (V4)
    retriever = StreamingQwen3RAGRetrieverV4(
        index_path=args.index_path,
        model_path=args.model_path,
        device=args.device,
        lora_r=args.rag_lora_r,
        text_lora_r=args.rag_text_lora_r,
        top_k=args.top_k,
        voting_k=args.rag_voting_k,
        voting_min_votes=args.rag_voting_min_votes,
        score_threshold=args.score_threshold, 
        chunk_size=args.rag_chunk_size,
        hop_size=args.rag_hop_size,
        aggregation_strategy=args.rag_strategy,
        debug_audio_dir=None
    )

    # 2. 加载术语库
    with open(args.glossary_path, 'r', encoding='utf-8') as f:
        glossary = json.load(f)
    glossary_terms = set(k.lower() for k in glossary.keys())

    # 3. 准备数据
    import glob
    wav_files = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")), 
                      key=lambda x: int(re.search(r'sent_(\d+)', x).group(1)) if re.search(r'sent_(\d+)', x) else 0)
    text_lines = load_text_lines(args.txt_path)
    
    samples = []
    for wav, txt in zip(wav_files, text_lines):
        gt = extract_gt_terms_from_text(txt, glossary_terms)
        if gt: 
            samples.append({'wav': wav, 'text': txt, 'gt_terms': gt})

    if args.max_samples > 0: samples = samples[:args.max_samples]
    print(f"[INFO] Simulating {len(samples)} samples with real-time agent flow...")

    # 4. 执行仿真
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    
    # 重要：从 retriever 内部获取术语列表，以确保与 FAISS 索引顺序完全一致
    glossary_keys = [item["key"] for item in retriever.term_list]

    # Provide term canonical map for evaluation (optional).
    if args.merge_plural_terms:
        from retriever.gigaspeech.acl_eval_utils import _canonicalize_plural_english
        args.term_canonical_map = {k: _canonicalize_plural_english(k) for k in glossary_keys}
    
    # 获取 retriever 的内部模型和索引以适配 run_acl_simulation 接口
    model = retriever.model
    faiss_index = retriever.index
    
    # 设置必要的 batch size 参数供 run_acl_simulation 使用
    args.acl_audio_batch_size = 32 # 默认值

    sim_results = run_acl_simulation(
        retriever=model,
        faiss_index=faiss_index,
        glossary_terms=glossary_keys,
        wav_files=wav_files,
        text_lines=text_lines,
        args=args,
        device=args.device,
        feature_extractor=feature_extractor,
        limit=args.max_samples if args.max_samples > 0 else len(wav_files)
    )

    # 5. 输出汇总
    print_simulation_summary(sim_results)

if __name__ == "__main__":
    main()
