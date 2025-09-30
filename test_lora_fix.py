#!/usr/bin/env python3
"""
测试LoRA修复的脚本
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'retriever', 'gigaspeech', 'modal'))

import torch
import numpy as np

def test_audio_encoding_shape():
    """测试音频编码形状修复"""
    print("🧪 Testing Audio Encoding Shape Fix")
    print("=" * 50)
    
    try:
        from Qwen2_Audio_train import Qwen2AudioSpeechEncoder, Qwen2AudioTextEncoder, ContrastiveQwen2AudioModel
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # 创建模型
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
        
        # 强制启用LoRA梯度
        print("\n🔧 Forcing LoRA gradients...")
        model.force_enable_lora_gradients()
        
        # 创建测试数据
        print("\n🎯 Creating test data...")
        batch_size = 4
        dummy_audios = [torch.randn(16000, dtype=torch.float32) for _ in range(batch_size)]
        dummy_texts = ["This is test text." for _ in range(batch_size)]
        
        # 测试编码
        print("\n🚀 Testing encoding...")
        audio_emb = model.encode_audio(dummy_audios)
        text_emb = model.encode_text(dummy_texts)
        
        print(f"✅ Audio embeddings shape: {audio_emb.shape}")
        print(f"✅ Text embeddings shape: {text_emb.shape}")
        print(f"✅ Audio embeddings requires_grad: {audio_emb.requires_grad}")
        print(f"✅ Text embeddings requires_grad: {text_emb.requires_grad}")
        
        # 计算损失
        print("\n💡 Testing loss calculation...")
        sim_matrix = (audio_emb @ text_emb.T) / 0.07
        labels = torch.arange(batch_size, dtype=torch.long, device=audio_emb.device)
        loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
        
        print(f"✅ Similarity matrix shape: {sim_matrix.shape}")
        print(f"✅ Labels shape: {labels.shape}")
        print(f"✅ Loss: {loss.item():.6f}")
        print(f"✅ Loss requires_grad: {loss.requires_grad}")
        
        # 反向传播
        print("\n⬅️ Testing backward pass...")
        loss.backward()
        
        # 检查梯度
        print("\n🔍 Checking gradients...")
        model.check_lora_gradients(step=1)
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_audio_encoding_shape()
    if success:
        print("\n🎉 LoRA fix verification successful!")
    else:
        print("\n💥 LoRA fix verification failed!")
        sys.exit(1)

