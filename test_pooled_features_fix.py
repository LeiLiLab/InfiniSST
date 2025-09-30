#!/usr/bin/env python3
"""
测试 pooled_features 修复的脚本
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'retriever', 'gigaspeech', 'modal'))

import torch
import numpy as np

def test_pooled_features_fix():
    """测试 pooled_features 修复"""
    print("🔧 Testing pooled_features Fix")
    print("=" * 50)
    
    try:
        from Qwen2_Audio_train import Qwen2AudioSpeechEncoder, Qwen2AudioTextEncoder, ContrastiveQwen2AudioModel
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # 创建模型
        print("\n1️⃣ Creating model...")
        speech_encoder = Qwen2AudioSpeechEncoder(device=device)
        shared_model = speech_encoder.get_shared_model()
        text_encoder = Qwen2AudioTextEncoder(device=device, shared_model=shared_model)
        
        model = ContrastiveQwen2AudioModel(
            speech_encoder=speech_encoder,
            text_encoder=text_encoder,
            proj_dim=512,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        # 设置训练模式
        model.train()
        
        # 创建测试数据
        print("\n2️⃣ Creating test data...")
        batch_size = 4
        dummy_audios = [torch.randn(16000, dtype=torch.float32) for _ in range(batch_size)]
        
        # 测试音频编码
        print("\n3️⃣ Testing audio encoding...")
        audio_emb = model.encode_audio(dummy_audios)
        
        print(f"✅ Audio embeddings shape: {audio_emb.shape}")
        print(f"✅ Audio embeddings requires_grad: {audio_emb.requires_grad}")
        
        # 验证形状正确
        expected_shape = (batch_size, 512)  # proj_dim = 512
        if audio_emb.shape == expected_shape:
            print(f"✅ Shape is correct: {audio_emb.shape}")
        else:
            print(f"❌ Shape is wrong: got {audio_emb.shape}, expected {expected_shape}")
            return False
        
        # 验证梯度存在
        if audio_emb.requires_grad:
            print("✅ Gradients are enabled")
        else:
            print("❌ Gradients are disabled")
            return False
        
        # 测试反向传播
        print("\n4️⃣ Testing backward pass...")
        loss = audio_emb.sum()
        loss.backward()
        print("✅ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pooled_features_fix()
    if success:
        print("\n🎉 pooled_features fix successful!")
    else:
        print("\n💥 pooled_features fix failed!")
        sys.exit(1)
