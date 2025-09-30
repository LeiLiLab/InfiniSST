#!/usr/bin/env python3
"""
测试LoRA日志功能的简单脚本
用于验证LoRA参数是否正确应用和更新
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'retriever', 'gigaspeech', 'modal'))

import torch
import numpy as np
from Qwen2_Audio_train import Qwen2AudioSpeechEncoder, Qwen2AudioTextEncoder, ContrastiveQwen2AudioModel

def test_lora_logging():
    """测试LoRA日志功能"""
    print("🧪 Testing LoRA Logging Functionality")
    print("=" * 60)
    
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # 初始化编码器
        print("\n🔧 Initializing encoders...")
        speech_encoder = Qwen2AudioSpeechEncoder(device=device)
        shared_model = speech_encoder.get_shared_model()
        text_encoder = Qwen2AudioTextEncoder(device=device, shared_model=shared_model)
        
        # 初始化对比学习模型（这里会自动打印详细的LoRA信息）
        print("\n🔧 Initializing ContrastiveQwen2AudioModel...")
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
        
        # 创建虚拟数据进行测试
        print("\n🎯 Creating dummy data for testing...")
        dummy_audio = torch.randn(16000, dtype=torch.float32)  # 1秒音频
        dummy_text = "This is a test sentence for LoRA verification."
        
        # 保存训练前的参数状态
        print("\n💾 Saving parameter state before training...")
        before_state = model.print_parameter_stats_before_after()
        
        # 创建优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 模拟几步训练
        print("\n🏃 Simulating training steps...")
        for step in range(3):
            print(f"\n--- Step {step + 1} ---")
            
            # 前向传播
            audio_emb = model.encode_audio([dummy_audio])
            text_emb = model.encode_text([dummy_text])
            
            # 计算简单的对比损失
            similarity = torch.mm(audio_emb, text_emb.T)
            labels = torch.arange(similarity.size(0), device=similarity.device)
            loss = torch.nn.CrossEntropyLoss()(similarity, labels)
            
            print(f"Loss: {loss.item():.6f}")
            
            # 反向传播
            loss.backward()
            
            # 检查梯度（每步都检查）
            model.check_lora_gradients(step=step + 1)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
        
        # 训练结束后比较参数变化
        print("\n📊 Comparing parameter changes after training...")
        model.print_parameter_stats_before_after(before_state)
        
        print("\n✅ LoRA logging test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_lora_logging()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)

