#!/usr/bin/env python3
"""
快速测试Qwen2-Audio的单个样本embedding
"""

import torch
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

# 导入我们的Qwen2-Audio模型
from Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder, 
    Qwen2AudioTextEncoder, 
    ContrastiveQwen2AudioModel
)


def quick_test():
    """快速测试单个样本"""
    
    # 测试样本（你可以修改这些路径）
    test_samples_path = "data/samples/xl/term_level_chunks_500000_1000000.json"
    
    print("=== Quick Qwen2-Audio Embedding Test ===")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载一个测试样本
    print(f"Loading samples from {test_samples_path}")
    with open(test_samples_path, 'r') as f:
        all_samples = json.load(f)
    
    # 找第一个有效样本
    test_sample = None
    for sample in all_samples[:100]:  # 只检查前100个
        audio_path = sample.get('term_chunk_audio', '')
        chunk_text = sample.get('term_chunk_text', '')
        ground_truth_terms = sample.get('term_chunk_audio_ground_truth_terms', [])
        
        if (audio_path and chunk_text.strip() and 
            os.path.exists(audio_path) and 
            ground_truth_terms):
            test_sample = sample
            break
    
    if not test_sample:
        print("ERROR: No valid test sample found!")
        return
    
    # 显示测试样本信息
    audio_path = test_sample['term_chunk_audio']
    chunk_text = test_sample['term_chunk_text']
    ground_truth_terms = test_sample['term_chunk_audio_ground_truth_terms']
    
    print(f"\n=== Test Sample ===")
    print(f"Audio: {os.path.basename(audio_path)}")
    print(f"Text: '{chunk_text}'")
    print(f"Ground truth terms: {ground_truth_terms}")
    
    # 初始化编码器
    print(f"\nInitializing Qwen2-Audio encoders...")
    try:
        speech_encoder = Qwen2AudioSpeechEncoder(device=device)
        text_encoder = Qwen2AudioTextEncoder(device=device)
        print("Encoders initialized successfully!")
        
    except Exception as e:
        print(f"ERROR: Failed to initialize encoders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试编码
    print(f"\n=== Testing Encodings ===")
    
    try:
        # 编码音频
        print("1. Encoding audio...")
        audio_embeddings = speech_encoder.predict([audio_path])
        audio_emb = audio_embeddings[0]
        print(f"   Audio embedding shape: {audio_emb.shape}")
        print(f"   Audio embedding stats: mean={np.mean(audio_emb):.4f}, std={np.std(audio_emb):.4f}, norm={np.linalg.norm(audio_emb):.4f}")
        
        # 编码文本
        print("2. Encoding text...")
        text_embeddings = text_encoder.predict([chunk_text])
        text_emb = text_embeddings[0]
        print(f"   Text embedding shape: {text_emb.shape}")
        print(f"   Text embedding stats: mean={np.mean(text_emb):.4f}, std={np.std(text_emb):.4f}, norm={np.linalg.norm(text_emb):.4f}")
        
        # 编码ground truth terms
        print("3. Encoding ground truth terms...")
        term_embeddings = text_encoder.predict(ground_truth_terms)
        print(f"   Terms embedding shape: {term_embeddings.shape}")
        
        # 计算相似度
        print(f"\n=== Similarity Results ===")
        
        # 音频 vs 文本
        audio_text_sim = cosine_similarity([audio_emb], [text_emb])[0][0]
        print(f"Audio-Text similarity: {audio_text_sim:.4f}")
        
        # 音频 vs terms
        audio_term_sims = cosine_similarity([audio_emb], term_embeddings)[0]
        print(f"Audio-Terms similarities:")
        for i, (term, sim) in enumerate(zip(ground_truth_terms, audio_term_sims)):
            print(f"  - '{term}': {sim:.4f}")
        max_audio_term_sim = np.max(audio_term_sims)
        best_term = ground_truth_terms[np.argmax(audio_term_sims)]
        print(f"  Best match: '{best_term}' ({max_audio_term_sim:.4f})")
        
        # 文本 vs terms
        text_term_sims = cosine_similarity([text_emb], term_embeddings)[0]
        print(f"Text-Terms similarities:")
        for i, (term, sim) in enumerate(zip(ground_truth_terms, text_term_sims)):
            print(f"  - '{term}': {sim:.4f}")
        max_text_term_sim = np.max(text_term_sims)
        best_term = ground_truth_terms[np.argmax(text_term_sims)]
        print(f"  Best match: '{best_term}' ({max_text_term_sim:.4f})")
        
        # 总结
        print(f"\n=== Summary ===")
        print(f"✅ Audio and text successfully encoded")
        print(f"📊 Audio-Text similarity: {audio_text_sim:.4f}")
        print(f"📊 Best Audio-Term similarity: {max_audio_term_sim:.4f}")
        print(f"📊 Best Text-Term similarity: {max_text_term_sim:.4f}")
        
        if audio_text_sim > 0.5:
            print("🎉 Good audio-text alignment!")
        elif audio_text_sim > 0.3:
            print("🤔 Moderate audio-text alignment")
        else:
            print("⚠️  Low audio-text alignment")
            
    except Exception as e:
        print(f"ERROR: Failed during encoding: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试对比学习模型
    print(f"\n=== Testing Contrastive Model ===")
    try:
        model = ContrastiveQwen2AudioModel(
            speech_encoder, text_encoder,
            hidden_dim=4096,
            proj_dim=512,
            unfreeze_layers=0
        ).to(device)
        
        # 编码通过投影层
        print("Testing projection layers...")
        audio_proj = model.encode_audio([audio_path])
        text_proj = model.encode_text([chunk_text])
        
        if isinstance(audio_proj, torch.Tensor):
            audio_proj = audio_proj.detach().cpu().numpy()
        if isinstance(text_proj, torch.Tensor):
            text_proj = text_proj.detach().cpu().numpy()
        
        proj_sim = cosine_similarity(audio_proj, text_proj)[0][0]
        print(f"Projected embeddings similarity: {proj_sim:.4f}")
        print(f"Projection layer effect: {proj_sim - audio_text_sim:+.4f}")
        
    except Exception as e:
        print(f"ERROR: Failed to test contrastive model: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== Test Completed ===")


if __name__ == "__main__":
    quick_test()


