#!/usr/bin/env python3
import os
import sys
import json
import torch
import faiss
import pickle
import math
import argparse
import numpy as np
from tqdm import tqdm
from flashtext import KeywordProcessor
import soundfile as sf

# 导入 Qwen3 相关组件
from agents.streaming_qwen3_rag_retriever import StreamingQwen3RAGRetriever

def load_text_lines(txt_path):
    results = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(line.strip())
    return results

def extract_gt_terms_from_text(text, glossary_terms):
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for term in glossary_terms:
        if len(term) >= 3:
            keyword_processor.add_keyword(term.lower())
    found_keywords = keyword_processor.extract_keywords(text)
    return set(k.lower() for k in found_keywords)

def main():
    parser = argparse.ArgumentParser(description="Detailed Offline RAG Evaluation for Qwen3-Omni")
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pt checkpoint')
    parser.add_argument('--index_path', type=str, required=True, help='Path to .pkl index')
    parser.add_argument('--glossary_path', type=str, required=True, help='Path to glossary.json')
    parser.add_argument('--wav_dir', type=str, required=True, help='Directory with gold wav files')
    parser.add_argument('--txt_path', type=str, required=True, help='Path to gold transcript txt')
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--top_k', type=int, default=5, help='Top K for each window retrieval')
    parser.add_argument('--rag_voting_k', type=int, default=20, help='Internal Top K for voting strategy')
    parser.add_argument('--terms_per_second', type=float, default=2.5, help='For final top N = ceil(len * 2.5)')
    parser.add_argument('--rag_chunk_size', type=float, default=1.92)
    parser.add_argument('--rag_hop_size', type=float, default=0.96)
    parser.add_argument('--rag_strategy', type=str, default='voting', choices=['voting', 'max_pool'])
    parser.add_argument('--max_samples', type=int, default=10)
    
    args = parser.parse_args()

    # 1. 初始化检索器
    retriever = StreamingQwen3RAGRetriever(
        index_path=args.index_path,
        model_path=args.model_path,
        device=args.device,
        top_k=args.top_k,
        voting_k=args.rag_voting_k,
        score_threshold=0.0, 
        chunk_size=args.rag_chunk_size,
        hop_size=args.rag_hop_size,
        aggregation_strategy=args.rag_strategy,
        terms_per_second=args.terms_per_second,
        debug_audio_dir=None
    )

    # 2. 加载术语库
    with open(args.glossary_path, 'r', encoding='utf-8') as f:
        glossary = json.load(f)
    glossary_terms = set(k.lower() for k in glossary.keys())

    # 3. 准备数据
    import glob, re
    wav_files = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")), 
                      key=lambda x: int(re.search(r'sent_(\d+)', x).group(1)) if re.search(r'sent_(\d+)', x) else 0)
    text_lines = load_text_lines(args.txt_path)
    
    samples = []
    for wav, txt in zip(wav_files, text_lines):
        gt = extract_gt_terms_from_text(txt, glossary_terms)
        if gt: # 只评估有术语的句子
            samples.append({'wav': wav, 'text': txt, 'gt_terms': gt})

    if args.max_samples > 0: samples = samples[:args.max_samples]
    print(f"[INFO] Evaluating {len(samples)} samples with sliding window simulation...")

    # 4. 执行详细仿真
    print("\n" + "="*100)
    print(f"{'SLIDING WINDOW RAG EVALUATION':^100}")
    print("="*100)

    all_segment_results = []

    for s in samples:
        retriever.reset()
        audio, sr = sf.read(s['wav'])
        if sr != 16000: continue
        duration = len(audio) / 16000
        
        print(f"\n[ID]: {os.path.basename(s['wav'])} | Duration: {duration:.2f}s")
        print(f"[Transcript]: {s['text']}")
        print(f"[GT Terms]: {', '.join(sorted(s['gt_terms']))}")
        
        segment_max_scores = {} # 用于最终 Top N 聚合 (Max Pooling)
        
        # 模拟 0.96s 步长的增量音频流
        hop_samples = int(args.rag_hop_size * 16000)
        
        for start in range(0, len(audio), hop_samples):
            # 获取当前步长的增量音频 (0.96s)
            end = start + hop_samples
            chunk = audio[start:end]
            if len(chunk) == 0: break
            
            is_last = (end >= len(audio))
            
            # 清空单步得分以获取该窗口的独立结果展示
            retriever._term_scores = {}
            
            # 发送增量。检索器内部会 buffer 并在凑满 1.92s 时自动触发检索
            retriever.accumulate_audio(chunk, force_process=is_last)
            current_window_scores = retriever._term_scores.copy()
            
            # 合并到整句最高分
            for t, sc in current_window_scores.items():
                segment_max_scores[t] = max(segment_max_scores.get(t, 0), sc)
            
            # 打印该步产生的 RAG 结果
            if current_window_scores:
                top5 = sorted(current_window_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                top5_str = ", ".join([f"{t}({sc:.3f})" for t, sc in top5])
                print(f"  >>> Step @{start/16000:.2f}s | RAG Result: {top5_str}")

        # 5. 句子结束，计算段落级 Top N
        N = math.ceil(duration * args.terms_per_second)
        final_candidates = sorted(segment_max_scores.items(), key=lambda x: x[1], reverse=True)[:N]
        
        print(f"\n[Final Top {N} Candidates (Segment Aggregated)]:")
        hit_in_segment = []
        for term, score in final_candidates:
            is_hit = term in s['gt_terms']
            mark = "✅ [HIT]" if is_hit else "❌ [FP] "
            print(f"    {mark} {score:.4f} | {term}")
            if is_hit: hit_in_segment.append(term)
        
        missed = s['gt_terms'] - set(hit_in_segment)
        if missed:
            print(f"    ⚠️ [MISSED]: {', '.join(sorted(missed))}")
        
        print("-" * 80)
        
        all_segment_results.append({
            'gt': s['gt_terms'],
            'pred': segment_max_scores,
            'duration': duration
        })

    # 6. 最后打印总体指标表
    print_summary_table(all_segment_results, args.terms_per_second)

def print_summary_table(results, tps):
    print("\n" + "="*60)
    print(f"{'OVERALL PERFORMANCE VS THRESHOLD (TPS=' + str(tps) + ')':^60}")
    print("="*60)
    print(f"{'Threshold':>10} | {'Recall':>10} | {'Precision':>10} | {'Poison':>10}")
    print("-" * 60)
    
    # 调精细点：从 0.414 开始，步长 0.01
    thresholds = [0.0, 0.3, 0.4] + np.arange(0.41, 0.46, 0.05).tolist()
    for th in thresholds:
        total_gt = 0
        total_tp = 0
        total_fp = 0
        poison_count = 0
        
        for res in results:
            gt = res['gt']
            duration = res['duration']
            N = math.ceil(duration * tps)
            
            # 按分值过滤并取 Top N
            pred_scores = res['pred']
            top_preds = [t for t, s in sorted(pred_scores.items(), key=lambda x: x[1], reverse=True) if s >= th][:N]
            top_preds = set(top_preds)
            
            tp = len(top_preds & gt)
            fp = len(top_preds - gt)
            
            total_gt += len(gt)
            total_tp += tp
            total_fp += fp
            
            # Poison: 有输出但全是错的
            if len(top_preds) > 0 and tp == 0:
                poison_count += 1
                
        rec = total_tp / total_gt if total_gt > 0 else 0
        pre = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        poison = poison_count / len(results) if results else 0
        
        print(f"{th:>10.3f} | {rec:>10.2%} | {pre:>10.2%} | {poison:>10.2%}")

if __name__ == "__main__":
    main()
