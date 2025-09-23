#!/usr/bin/env python3
"""
测试Qwen2-Audio模型的音频和文本embedding效果
测试单个音频chunk和对应文本的余弦相似度
"""

import torch
import numpy as np
import json
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import random

# 导入我们的Qwen2-Audio模型
from Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder, 
    Qwen2AudioTextEncoder, 
    ContrastiveQwen2AudioModel
)


def load_test_samples(samples_path, num_samples=10):
    """加载测试样本"""
    print(f"[INFO] Loading test samples from {samples_path}")
    
    with open(samples_path, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    
    # 过滤有效样本（有音频文件和文本，且音频文件存在）
    valid_samples = []
    for sample in all_samples:
        audio_path = sample.get('term_chunk_audio', '')
        chunk_text = sample.get('term_chunk_text', '')
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        
        if (audio_path and chunk_text.strip() and 
            os.path.exists(audio_path) and 
            ground_truth_terms):
            valid_samples.append(sample)
    
    print(f"[INFO] Found {len(valid_samples)} valid samples")
    
    # 随机选择指定数量的样本
    if len(valid_samples) > num_samples:
        random.seed(42)
        selected_samples = random.sample(valid_samples, num_samples)
    else:
        selected_samples = valid_samples
    
    print(f"[INFO] Selected {len(selected_samples)} samples for testing")
    return selected_samples


def test_single_sample(speech_encoder, text_encoder, sample, sample_idx):
    """测试单个样本的音频和文本embedding"""
    audio_path = sample['term_chunk_audio']
    chunk_text = sample['term_chunk_text']
    ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
    
    print(f"\n=== Sample {sample_idx + 1} ===")
    print(f"Audio: {os.path.basename(audio_path)}")
    print(f"Text: '{chunk_text}'")
    print(f"Ground truth terms: {ground_truth_terms}")
    
    try:
        # 编码音频
        print("[INFO] Encoding audio...")
        audio_embeddings = speech_encoder.predict([audio_path])
        audio_emb = audio_embeddings[0]  # [embedding_dim]
        
        # 编码文本
        print("[INFO] Encoding text...")
        text_embeddings = text_encoder.predict([chunk_text])
        text_emb = text_embeddings[0]  # [embedding_dim]
        
        # 编码ground truth terms
        print("[INFO] Encoding ground truth terms...")
        if ground_truth_terms:
            term_embeddings = text_encoder.predict(ground_truth_terms)
            term_embs = term_embeddings  # [num_terms, embedding_dim]
        else:
            term_embs = None
        
        # 计算余弦相似度
        # 1. 音频 vs 文本
        audio_text_sim = cosine_similarity([audio_emb], [text_emb])[0][0]
        print(f"[RESULT] Audio-Text similarity: {audio_text_sim:.4f}")
        
        # 2. 音频 vs ground truth terms
        if term_embs is not None and len(term_embs) > 0:
            audio_term_sims = cosine_similarity([audio_emb], term_embs)[0]
            max_audio_term_sim = np.max(audio_term_sims)
            best_term_idx = np.argmax(audio_term_sims)
            print(f"[RESULT] Audio-Terms similarity (max): {max_audio_term_sim:.4f} (term: '{ground_truth_terms[best_term_idx]}')")
            
            # 显示所有term的相似度
            for i, (term, sim) in enumerate(zip(ground_truth_terms, audio_term_sims)):
                print(f"  - '{term}': {sim:.4f}")
        
        # 3. 文本 vs ground truth terms
        if term_embs is not None and len(term_embs) > 0:
            text_term_sims = cosine_similarity([text_emb], term_embs)[0]
            max_text_term_sim = np.max(text_term_sims)
            best_term_idx = np.argmax(text_term_sims)
            print(f"[RESULT] Text-Terms similarity (max): {max_text_term_sim:.4f} (term: '{ground_truth_terms[best_term_idx]}')")
        
        # 返回结果统计
        results = {
            'audio_text_sim': audio_text_sim,
            'audio_term_sim_max': max_audio_term_sim if term_embs is not None else None,
            'text_term_sim_max': max_text_term_sim if term_embs is not None else None,
            'audio_emb_stats': {
                'mean': np.mean(audio_emb),
                'std': np.std(audio_emb),
                'norm': np.linalg.norm(audio_emb)
            },
            'text_emb_stats': {
                'mean': np.mean(text_emb),
                'std': np.std(text_emb),
                'norm': np.linalg.norm(text_emb)
            }
        }
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Failed to process sample {sample_idx + 1}: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_contrastive_model(model, samples):
    """测试完整的对比学习模型"""
    print(f"\n=== Testing Contrastive Model ===")
    
    # 准备数据
    audio_paths = [s['term_chunk_audio'] for s in samples]
    chunk_texts = [s['term_chunk_text'] for s in samples]
    
    try:
        # 编码音频和文本
        print("[INFO] Encoding audios through contrastive model...")
        audio_embs = model.encode_audio(audio_paths)  # [batch_size, proj_dim]
        
        print("[INFO] Encoding texts through contrastive model...")
        text_embs = model.encode_text(chunk_texts)    # [batch_size, proj_dim]
        
        # 转换为numpy
        if isinstance(audio_embs, torch.Tensor):
            audio_embs = audio_embs.detach().cpu().numpy()
        if isinstance(text_embs, torch.Tensor):
            text_embs = text_embs.detach().cpu().numpy()
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(audio_embs, text_embs)
        
        print(f"[RESULT] Similarity matrix shape: {similarity_matrix.shape}")
        print(f"[RESULT] Diagonal (correct pairs) similarities:")
        
        diagonal_sims = np.diag(similarity_matrix)
        for i, sim in enumerate(diagonal_sims):
            print(f"  Sample {i+1}: {sim:.4f}")
        
        print(f"[RESULT] Average diagonal similarity: {np.mean(diagonal_sims):.4f}")
        print(f"[RESULT] Average off-diagonal similarity: {np.mean(similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]):.4f}")
        
        # 计算top-k准确率
        top1_acc = np.mean(np.argmax(similarity_matrix, axis=1) == np.arange(len(samples)))
        print(f"[RESULT] Top-1 accuracy (audio->text): {top1_acc:.2%}")
        
        top1_acc_reverse = np.mean(np.argmax(similarity_matrix.T, axis=1) == np.arange(len(samples)))
        print(f"[RESULT] Top-1 accuracy (text->audio): {top1_acc_reverse:.2%}")
        
        return {
            'diagonal_sims': diagonal_sims,
            'avg_diagonal_sim': np.mean(diagonal_sims),
            'avg_off_diagonal_sim': np.mean(similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]),
            'top1_acc_audio2text': top1_acc,
            'top1_acc_text2audio': top1_acc_reverse
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to test contrastive model: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test Qwen2-Audio embedding quality")
    parser.add_argument('--samples_path', type=str, 
                       default='data/samples/xl/term_level_chunks_500000_1000000.json',
                       help='Path to test samples JSON file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to test')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-Audio-7B-Instruct',
                       help='Qwen2-Audio model name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--test_contrastive', action='store_true',
                       help='Also test the contrastive model')
    
    args = parser.parse_args()
    
    print("=== Qwen2-Audio Embedding Test ===")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Samples path: {args.samples_path}")
    print(f"Number of samples: {args.num_samples}")
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # 加载测试样本
    if not os.path.exists(args.samples_path):
        print(f"[ERROR] Samples file not found: {args.samples_path}")
        return
    
    samples = load_test_samples(args.samples_path, args.num_samples)
    if not samples:
        print("[ERROR] No valid samples found")
        return
    
    # 初始化编码器
    print(f"\n[INFO] Initializing Qwen2-Audio encoders...")
    try:
        speech_encoder = Qwen2AudioSpeechEncoder(
            model_name=args.model_name, 
            device=args.device
        )
        
        text_encoder = Qwen2AudioTextEncoder(
            model_name=args.model_name, 
            device=args.device
        )
        
        print("[INFO] Encoders initialized successfully")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize encoders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试单个样本
    print(f"\n=== Testing Individual Samples ===")
    all_results = []
    
    for i, sample in enumerate(samples):
        result = test_single_sample(speech_encoder, text_encoder, sample, i)
        if result:
            all_results.append(result)
    
    # 统计结果
    if all_results:
        print(f"\n=== Overall Statistics ===")
        audio_text_sims = [r['audio_text_sim'] for r in all_results]
        print(f"Audio-Text similarities: mean={np.mean(audio_text_sims):.4f}, std={np.std(audio_text_sims):.4f}")
        
        audio_term_sims = [r['audio_term_sim_max'] for r in all_results if r['audio_term_sim_max'] is not None]
        if audio_term_sims:
            print(f"Audio-Term similarities: mean={np.mean(audio_term_sims):.4f}, std={np.std(audio_term_sims):.4f}")
        
        text_term_sims = [r['text_term_sim_max'] for r in all_results if r['text_term_sim_max'] is not None]
        if text_term_sims:
            print(f"Text-Term similarities: mean={np.mean(text_term_sims):.4f}, std={np.std(text_term_sims):.4f}")
        
        # Embedding统计
        audio_norms = [r['audio_emb_stats']['norm'] for r in all_results]
        text_norms = [r['text_emb_stats']['norm'] for r in all_results]
        print(f"Audio embedding norms: mean={np.mean(audio_norms):.4f}, std={np.std(audio_norms):.4f}")
        print(f"Text embedding norms: mean={np.mean(text_norms):.4f}, std={np.std(text_norms):.4f}")
    
    # 测试对比学习模型（可选）
    if args.test_contrastive:
        print(f"\n=== Testing Contrastive Model ===")
        try:
            model = ContrastiveQwen2AudioModel(
                speech_encoder, text_encoder,
                hidden_dim=4096,
                proj_dim=512,
                unfreeze_layers=0
            ).to(args.device)
            
            contrastive_results = test_contrastive_model(model, samples)
            
        except Exception as e:
            print(f"[ERROR] Failed to test contrastive model: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== Test Completed ===")


if __name__ == "__main__":
    main()


