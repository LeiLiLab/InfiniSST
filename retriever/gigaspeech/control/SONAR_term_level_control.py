#!/usr/bin/env python3
"""
Term-Level Control Group Evaluation
使用预训练SONAR编码器直接评估精准对齐的term-level chunks
不进行额外训练，提供纯净的baseline性能
"""

import torch
import numpy as np
import json
import argparse
import os
import sys
from tqdm import tqdm
import faiss
import random
from typing import List, Dict, Tuple

# 添加路径
sys.path.append('/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech')
sys.path.append('/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever')

from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from new_retrieve import Retriever


def load_glossary_terms(glossary_path: str) -> List[str]:
    """加载完整的术语表"""
    print(f"[INFO] Loading glossary from {glossary_path}")
    with open(glossary_path, "r") as f:
        glossary = json.load(f)
    
    # 提取所有术语，处理不同的数据格式
    terms = []
    if isinstance(glossary, list):
        for item in glossary:
            if isinstance(item, dict):
                # 如果是字典，尝试获取 'term' 或 'text' 字段
                term = item.get('term') or item.get('text') or item.get('word')
                if term:
                    terms.append(term.lower())
            elif isinstance(item, str):
                terms.append(item.lower())
    elif isinstance(glossary, dict):
        # 如果是字典格式，提取所有值
        for key, value in glossary.items():
            if isinstance(value, str):
                terms.append(value.lower())
            elif isinstance(value, dict) and 'term' in value:
                terms.append(value['term'].lower())
    
    # 去重并过滤
    terms = list(set(term for term in terms if term and len(term.strip()) >= 2))
    print(f"[INFO] Loaded {len(terms)} unique terms from glossary")
    return terms


def load_term_level_samples(samples_path: str) -> List[Dict]:
    """加载term-level chunk样本"""
    print(f"[INFO] Loading term-level samples from {samples_path}")
    with open(samples_path, "r") as f:
        all_samples = json.load(f)
    
    # 过滤有效样本
    valid_samples = []
    for s in all_samples:
        terms = s.get('term_chunk_audio_ground_truth_terms')
        if not (terms and isinstance(terms, list)):
            continue
        
        # 过滤术语
        filtered_terms = [
            t for t in terms
            if isinstance(t, str)
            and len(t) >= 3
            and sum(c.isdigit() for c in t) <= len(t) // 2
        ]
        if not filtered_terms:
            continue
        
        # 检查文件存在性
        audio_path = s.get('term_chunk_audio', '')
        if (
            s.get('term_chunk_text', '').strip()
            and audio_path
            and os.path.exists(audio_path)
        ):
            # 更新术语列表
            s = dict(s)
            s['term_chunk_audio_ground_truth_terms'] = filtered_terms
            valid_samples.append(s)
    
    print(f"[INFO] Filtered {len(valid_samples)} valid term-level samples from {len(all_samples)} total")
    return valid_samples


def encode_texts_in_batches(model, texts: List[str], batch_size: int = 512) -> torch.Tensor:
    """分批编码文本"""
    print(f"[DEBUG] encode_texts_in_batches called with {len(texts)} texts, batch_size={batch_size}")
    all_embeddings = []
    print(f"[INFO] Encoding {len(texts)} text terms in batches of {batch_size}")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"[INFO] Processing text batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        print(f"[DEBUG] Batch size: {len(batch)}")
        
        with torch.no_grad():
            try:
                emb = model.predict(batch, source_lang="eng_Latn")
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)
                all_embeddings.append(emb.cpu())
                print(f"[DEBUG] ✅ Text batch {i//batch_size + 1} encoded successfully, shape: {emb.shape}")
            except Exception as e:
                print(f"[ERROR] Failed to encode text batch {i//batch_size + 1}: {e}")
                raise
    
    print(f"[DEBUG] About to concatenate {len(all_embeddings)} text embedding batches")
    result = torch.cat(all_embeddings, dim=0)
    print(f"[DEBUG] ✅ Final text embeddings shape: {result.shape}")
    return result


def encode_audios_in_batches(model, audio_paths: List[str], batch_size: int = 32) -> torch.Tensor:
    """分批编码音频"""
    print(f"[DEBUG] encode_audios_in_batches called with {len(audio_paths)} audio files, batch_size={batch_size}")
    all_embeddings = []
    print(f"[INFO] Encoding {len(audio_paths)} audio files in batches of {batch_size}")
    
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        print(f"[INFO] Processing audio batch {i//batch_size + 1}/{(len(audio_paths) + batch_size - 1)//batch_size}")
        print(f"[DEBUG] Batch size: {len(batch_paths)}")
        print(f"[DEBUG] First audio path in batch: {batch_paths[0] if batch_paths else 'None'}")
        
        with torch.no_grad():
            try:
                print(f"[DEBUG] About to call model.predict for audio batch {i//batch_size + 1}")
                emb = model.predict(batch_paths)
                print(f"[DEBUG] ✅ Audio batch {i//batch_size + 1} predict() completed")
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)
                all_embeddings.append(emb.cpu())
                print(f"[DEBUG] ✅ Audio batch {i//batch_size + 1} encoded successfully, shape: {emb.shape}")
            except Exception as e:
                print(f"[ERROR] Failed to encode audio batch {i//batch_size + 1}: {e}")
                print(f"[INFO] Trying single file processing...")
                
                # 单个文件处理
                for j, single_path in enumerate(batch_paths):
                    try:
                        print(f"[DEBUG] Processing single audio file {j+1}/{len(batch_paths)}: {single_path}")
                        single_emb = model.predict([single_path])
                        if isinstance(single_emb, np.ndarray):
                            single_emb = torch.from_numpy(single_emb)
                        all_embeddings.append(single_emb.cpu())
                        print(f"[DEBUG] ✅ Single audio file encoded successfully")
                    except Exception as e2:
                        print(f"[ERROR] Failed to encode {single_path}: {e2}")
                        continue
    
    if not all_embeddings:
        raise RuntimeError("No audio files were successfully encoded")
    
    print(f"[DEBUG] About to concatenate {len(all_embeddings)} audio embedding batches")
    result = torch.cat(all_embeddings, dim=0)
    print(f"[DEBUG] ✅ Final audio embeddings shape: {result.shape}")
    return result


def extract_used_terms_from_samples(samples: List[Dict]) -> List[str]:
    """从样本中提取所有使用的术语"""
    used_terms = set()
    for sample in samples:
        terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        for term in terms:
            if isinstance(term, str) and len(term.strip()) >= 2:
                used_terms.add(term.lower())
    return list(used_terms)


def split_samples_train_test(samples: List[Dict], train_ratio: float = 0.99) -> Tuple[List[Dict], List[Dict]]:
    """分割样本为训练集和测试集（用于seen/unseen分析）"""
    random.seed(42)  # 固定随机种子
    samples_copy = samples.copy()
    random.shuffle(samples_copy)
    
    split_idx = int(len(samples_copy) * train_ratio)
    train_samples = samples_copy[:split_idx]
    test_samples = samples_copy[split_idx:]
    
    print(f"[INFO] Split samples: {len(train_samples)} train, {len(test_samples)} test")
    return train_samples, test_samples


def evaluate_term_level_recall(
    speech_encoder,
    text_encoder, 
    test_samples: List[Dict],
    glossary_terms: List[str],
    train_terms: List[str],
    top_ks: Tuple[int, ...] = (1, 5, 10),
    max_eval: int = 2000,
    audio_batch_size: int = 32,
    text_batch_size: int = 512
) -> Dict[int, List[float]]:
    """评估term-level recall性能"""
    
    print(f"\n{'='*60}")
    print("TERM-LEVEL CONTROL GROUP EVALUATION")
    print(f"{'='*60}")
    print(f"[DEBUG] Function started with parameters:")
    print(f"[DEBUG] - len(test_samples): {len(test_samples)}")
    print(f"[DEBUG] - len(vocabulary_terms): {len(glossary_terms)}")
    print(f"[DEBUG] - len(train_terms): {len(train_terms)}")
    print(f"[DEBUG] - top_ks: {top_ks}")
    print(f"[DEBUG] - max_eval: {max_eval}")
    print(f"[DEBUG] - audio_batch_size: {audio_batch_size}")
    print(f"[DEBUG] - text_batch_size: {text_batch_size}")
    
    # === 构建文本索引 ===
    print(f"[INFO] Building text index with {len(glossary_terms)} vocabulary terms...")
    print(f"[DEBUG] About to encode texts in batches...")
    text_embeddings = encode_texts_in_batches(text_encoder, glossary_terms, text_batch_size)
    print(f"[DEBUG] ✅ Text embeddings shape: {text_embeddings.shape}")
    
    # 创建FAISS索引
    print(f"[DEBUG] About to create FAISS index...")
    index = faiss.IndexFlatL2(text_embeddings.shape[1])
    print(f"[DEBUG] About to add embeddings to index...")
    index.add(text_embeddings.numpy())
    print(f"[INFO] FAISS index built with {index.ntotal} terms")
    
    # === 选择评估样本 ===
    print(f"[DEBUG] About to select evaluation samples...")
    eval_samples = test_samples[:max_eval] if len(test_samples) > max_eval else test_samples
    print(f"[INFO] Evaluating {len(eval_samples)} samples")
    
    # === 编码音频 ===
    print(f"[DEBUG] About to extract audio paths...")
    audio_paths = [sample['term_chunk_audio'] for sample in eval_samples]
    print(f"[DEBUG] ✅ Extracted {len(audio_paths)} audio paths")
    print(f"[DEBUG] About to encode audio in batches...")
    audio_embeddings = encode_audios_in_batches(speech_encoder, audio_paths, audio_batch_size)
    print(f"[DEBUG] ✅ Audio embeddings shape: {audio_embeddings.shape}")
    
    # === 计算recall ===
    recall_dict = {k: [] for k in top_ks}
    seen_terms_set = set(t.lower() for t in train_terms)
    
    print(f"[INFO] Computing recall for each sample...")
    for i, (sample, audio_emb) in enumerate(zip(eval_samples, audio_embeddings)):
        gt_terms = [t.lower() for t in sample['term_chunk_audio_ground_truth_terms']]
        
        # 搜索最相似的术语
        audio_emb_np = audio_emb.numpy().reshape(1, -1)
        
        for top_k in top_ks:
            D, I = index.search(audio_emb_np, top_k)
            retrieved_terms = [glossary_terms[idx].lower() for idx in I[0]]
            
            matched = sum(gt in retrieved_terms for gt in gt_terms)
            recall = matched / len(gt_terms) if gt_terms else 0.0
            recall_dict[top_k].append(recall)
        
        # 打印前几个样本的详细信息
        if i < 5:
            top_k_debug = top_ks[0]
            D, I = index.search(audio_emb_np, top_k_debug)
            retrieved_terms = [glossary_terms[idx].lower() for idx in I[0]]
            
            print(f"\n[DEBUG] Sample {i+1}:")
            print(f"[DEBUG] Audio: {os.path.basename(sample['term_chunk_audio'])}")
            print(f"[DEBUG] Text: {sample['term_chunk_text']}")
            print(f"[DEBUG] GT terms: {gt_terms}")
            print(f"[DEBUG] Retrieved@{top_k_debug}: {retrieved_terms}")
            print(f"[DEBUG] Recall@{top_k_debug}: {recall_dict[top_k_debug][-1]:.2%}")
    
    # === 打印结果 ===
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    for top_k in top_ks:
        recalls = recall_dict[top_k]
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"[RESULT] Average Recall@{top_k}: {avg_recall:.2%}")
        
        # 计算seen/unseen recall
        seen_recalls, unseen_recalls = [], []
        for recall_val, sample in zip(recalls, eval_samples):
            gt_terms = [t.lower() for t in sample['term_chunk_audio_ground_truth_terms']]
            if all(gt in seen_terms_set for gt in gt_terms):
                seen_recalls.append(recall_val)
            else:
                unseen_recalls.append(recall_val)
        
        avg_seen = sum(seen_recalls) / len(seen_recalls) if seen_recalls else 0.0
        avg_unseen = sum(unseen_recalls) / len(unseen_recalls) if unseen_recalls else 0.0
        total_samples = len(seen_recalls) + len(unseen_recalls)
        
        print(f"[RESULT] Seen Recall@{top_k}: {avg_seen:.2%} ({len(seen_recalls)}/{total_samples} samples)")
        print(f"[RESULT] Unseen Recall@{top_k}: {avg_unseen:.2%} ({len(unseen_recalls)}/{total_samples} samples)")
    
    return recall_dict


def main():
    print(f"[DEBUG] Script started at: {os.popen('date').read().strip()}")
    print(f"[DEBUG] Working directory: {os.getcwd()}")
    print(f"[DEBUG] Python executable: {sys.executable}")
    
    parser = argparse.ArgumentParser(description="Term-Level Control Group Evaluation")
    parser.add_argument('--samples_path', type=str, 
                       default='/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/xl_term_level_chunks_merged.json',
                       help="Path to term-level chunk samples")
    parser.add_argument('--glossary_path', type=str, 
                       default='/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_filtered.json',
                       help="Path to complete glossary")
    parser.add_argument('--train_ratio', type=float, default=0.99,
                       help="Ratio for train/test split (for seen/unseen analysis)")
    parser.add_argument('--max_eval', type=int, default=2000,
                       help="Maximum number of samples to evaluate")
    parser.add_argument('--audio_batch_size', type=int, default=32,
                       help="Batch size for audio encoding")
    parser.add_argument('--text_batch_size', type=int, default=512,
                       help="Batch size for text encoding")
    parser.add_argument('--output_dir', type=str, 
                       default='/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data',
                       help="Output directory for results")
    
    print(f"[DEBUG] Parsing arguments...")
    args = parser.parse_args()
    print(f"[DEBUG] Arguments parsed successfully")
    print(f"[DEBUG] samples_path: {args.samples_path}")
    print(f"[DEBUG] glossary_path: {args.glossary_path}")
    print(f"[DEBUG] max_eval: {args.max_eval}")
    
    print(f"[DEBUG] Checking CUDA availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}")
        print(f"[DEBUG] Current CUDA device: {torch.cuda.current_device()}")
    
    # === 初始化编码器（预训练，不进行额外训练） ===
    print(f"[INFO] Initializing pre-trained SONAR encoders...")
    print(f"[DEBUG] About to initialize speech encoder...")
    
    try:
        speech_encoder = SpeechToEmbeddingModelPipeline(
            encoder="sonar_speech_encoder_eng", 
            device=torch.device(device)
        )
        print(f"[DEBUG] ✅ Speech encoder initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize speech encoder: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"[DEBUG] About to initialize text encoder...")
    try:
        text_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device(device),
            dtype=torch.float32,
        )
        print(f"[DEBUG] ✅ Text encoder initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize text encoder: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"[INFO] ✅ Pre-trained encoders loaded successfully")
    
    # === 加载数据 ===
    print(f"\n[INFO] Loading data...")
    print(f"[DEBUG] About to load term-level samples from: {args.samples_path}")
    print(f"[DEBUG] Checking if samples file exists: {os.path.exists(args.samples_path)}")
    
    # 加载term-level样本
    all_samples = load_term_level_samples(args.samples_path)
    print(f"[DEBUG] ✅ Loaded {len(all_samples)} term-level samples")
    
    # 分割训练/测试集（用于seen/unseen分析）
    print(f"[DEBUG] About to split samples with train_ratio={args.train_ratio}")
    train_samples, test_samples = split_samples_train_test(all_samples, args.train_ratio)
    print(f"[DEBUG] ✅ Split complete: {len(train_samples)} train, {len(test_samples)} test")
    
    # 提取训练集术语（用于seen/unseen分析）
    print(f"[DEBUG] About to extract terms from training samples...")
    train_terms = extract_used_terms_from_samples(train_samples)
    print(f"[INFO] Training set contains {len(train_terms)} unique terms")
    
    # 从样本中提取所有unique的ground_truth_terms作为检索词汇表
    print(f"[DEBUG] About to extract all unique terms from samples...")
    all_sample_terms = extract_used_terms_from_samples(all_samples)
    print(f"[DEBUG] ✅ Extracted {len(all_sample_terms)} unique terms from samples")
    
    # 使用样本中的术语作为检索词汇表（比全量glossary小很多，速度更快）
    sample_terms = all_sample_terms  # 用样本中的terms替代glossary
    print(f"[INFO] Using {len(sample_terms)} terms from samples as retrieval vocabulary")
    
    # 分析术语覆盖情况
    train_terms_set = set(train_terms)
    sample_terms_set = set(sample_terms)
    overlap = train_terms_set.intersection(sample_terms_set)
    coverage = len(overlap) / len(train_terms_set) if train_terms_set else 0
    print(f"[INFO] Training terms covered in sample vocabulary: {len(overlap)}/{len(train_terms)} ({coverage:.1%})")
    
    # === 执行评估 ===
    print(f"[DEBUG] About to start evaluation...")
    print(f"[DEBUG] Evaluation parameters:")
    print(f"[DEBUG] - test_samples: {len(test_samples)}")
    print(f"[DEBUG] - sample_terms: {len(sample_terms)}")
    print(f"[DEBUG] - train_terms: {len(train_terms)}")
    print(f"[DEBUG] - max_eval: {args.max_eval}")
    print(f"[DEBUG] - audio_batch_size: {args.audio_batch_size}")
    print(f"[DEBUG] - text_batch_size: {args.text_batch_size}")
    
    recall_results = evaluate_term_level_recall(
        speech_encoder=speech_encoder,
        text_encoder=text_encoder,
        test_samples=test_samples,
        glossary_terms=sample_terms,  # 使用样本terms替代glossary
        train_terms=train_terms,
        top_ks=(1, 5, 10),
        max_eval=args.max_eval,
        audio_batch_size=args.audio_batch_size,
        text_batch_size=args.text_batch_size
    )
    
    print(f"[DEBUG] ✅ Evaluation completed successfully")
    
    # === 保存结果 ===
    results_path = os.path.join(args.output_dir, 'term_level_control_results.json')
    eval_summary = {
        'experiment_type': 'term_level_control_group',
        'description': 'Direct evaluation using pre-trained SONAR encoders on term-level chunks',
        'samples_path': args.samples_path,
        'vocabulary_source': 'extracted_from_samples',
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'test_samples': len(test_samples),
        'evaluated_samples': min(args.max_eval, len(test_samples)),
        'vocabulary_terms': len(sample_terms),
        'train_terms': len(train_terms),
        'train_terms_coverage_in_vocabulary': float(coverage),
        'results': {}
    }
    
    for top_k in [1, 5, 10]:
        if top_k in recall_results and recall_results[top_k]:
            avg_recall = sum(recall_results[top_k]) / len(recall_results[top_k])
            eval_summary['results'][f'recall@{top_k}'] = float(avg_recall)
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(eval_summary, f, indent=2)
    
    print(f"\n[INFO] Results saved to {results_path}")
    print(f"\n{'='*60}")
    print("TERM-LEVEL CONTROL GROUP EVALUATION COMPLETED")
    print(f"{'='*60}")
    print(f"[INFO] Key insights:")
    print(f"[INFO] - This is a clean baseline using pre-trained encoders")
    print(f"[INFO] - Each audio chunk is precisely aligned to one term via MFA")
    print(f"[INFO] - No additional training was performed")
    print(f"[INFO] - Results show the upper bound performance for perfect alignment")


if __name__ == "__main__":
    main() 