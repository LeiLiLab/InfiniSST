#!/usr/bin/env python3
"""
测试修复后的term-level训练数值稳定性
"""

import torch
import os
import sys
import json

# 添加路径
sys.path.append('/home/jiaxuanluo/InfiniSST/retriever/gigaspeech')

from SONAR_term_level_train import TermLevelDataset, train_step, ContrastiveSpeechTextModel, is_audio_valid
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

def test_numerical_stability():
    print("🧪 Testing Term-Level Training Numerical Stability")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 检查数据加载
    data_path = "data/samples/xl/term_level_chunks_single_0_500000.json"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    print(f"✅ Data file found: {data_path}")
    
    try:
        dataset = TermLevelDataset(data_path, split="train", train_ratio=0.99)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # 2. 初始化模型
    try:
        speech_encoder = SpeechToEmbeddingModelPipeline(
            encoder="sonar_speech_encoder_eng", device=device
        )
        text_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=torch.float32,
        )
        
        model = ContrastiveSpeechTextModel(
            speech_encoder, text_encoder, unfreeze_layers=10
        ).to(device)
        
        print(f"✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return False
    
    # 3. 测试小batch训练步骤
    print("\n🔬 Testing training step with small batch...")
    
    # 创建小batch (4个样本)
    test_batch = []
    for i in range(min(4, len(dataset))):
        sample = dataset[i]
        if sample is not None:
            test_batch.append(sample)
    
    if len(test_batch) < 2:
        print("❌ Not enough valid samples for testing")
        return False
    
    print(f"Test batch size: {len(test_batch)}")
    
    # 打印batch信息和音频验证结果
    for i, (gt_terms, audio_path, chunk_text, has_target) in enumerate(test_batch):
        is_valid, reason = is_audio_valid(audio_path)
        print(f"  Sample {i}: GT={gt_terms}, Text='{chunk_text}'")
        print(f"    Audio: {audio_path}")
        print(f"    Valid: {is_valid} ({reason})")
    
    # 测试训练步骤
    try:
        print("\n⚡ Running train_step...")
        loss = train_step(model, test_batch, device, temperature=0.1)
        
        print(f"✅ Train step completed")
        print(f"Loss value: {loss.item():.6f}")
        print(f"Loss requires_grad: {loss.requires_grad}")
        print(f"Loss is NaN: {torch.isnan(loss).any()}")
        print(f"Loss is Inf: {torch.isinf(loss).any()}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ Loss contains NaN or Inf values!")
            return False
        else:
            print("✅ Loss is numerically stable")
            
    except Exception as e:
        print(f"❌ Train step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试梯度反传
    try:
        print("\n🔄 Testing gradient computation...")
        model.train()
        
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        loss = train_step(model, test_batch, device, temperature=0.1)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ Loss is NaN/Inf, skipping backward")
            return False
        
        loss.backward()
        
        # 检查梯度
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"✅ Gradient norm: {grad_norm:.6f}")
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("❌ Gradient norm is NaN/Inf!")
            return False
        
        optimizer.step()
        optimizer.zero_grad()
        
        print("✅ Gradient computation and optimization step successful")
        
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All numerical stability tests passed!")
    print("✅ Term-level training should now be stable")
    return True

if __name__ == "__main__":
    success = test_numerical_stability()
    sys.exit(0 if success else 1) 