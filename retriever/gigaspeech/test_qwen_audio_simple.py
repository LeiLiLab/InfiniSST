#!/usr/bin/env python3
"""
简单的Qwen2-Audio模型测试脚本，用于验证CUDA assert修复
"""

import os
import sys
import torch
import numpy as np

# 启用CUDA调试
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# 设置GPU
gpu_id = "0"  # 使用单GPU测试
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def test_qwen_audio_basic():
    """基本的Qwen2-Audio模型测试"""
    try:
        print("[INFO] Testing Qwen2-Audio model loading...")
        
        # 导入模型
        from Qwen2_Audio_train import Qwen2AudioSpeechEncoder, Qwen2AudioTextEncoder
        
        device = torch.device("cuda:0")
        print(f"[INFO] Using device: {device}")
        
        # 初始化speech encoder
        print("[INFO] Initializing speech encoder...")
        speech_encoder = Qwen2AudioSpeechEncoder(
            model_name="Qwen/Qwen2-Audio-7B-Instruct", 
            device=device
        )
        print("[INFO] Speech encoder loaded successfully")
        
        # 初始化text encoder（共享模型）
        print("[INFO] Initializing text encoder...")
        text_encoder = Qwen2AudioTextEncoder(
            model_name="Qwen/Qwen2-Audio-7B-Instruct", 
            device=device,
            shared_model=speech_encoder.get_shared_model()
        )
        print("[INFO] Text encoder loaded successfully")
        
        # 测试文本编码
        print("[INFO] Testing text encoding...")
        test_texts = ["hello world", "test text"]
        try:
            text_emb = text_encoder.predict(test_texts)
            print(f"[INFO] Text encoding successful: {text_emb.shape}")
        except Exception as e:
            print(f"[ERROR] Text encoding failed: {e}")
            return False
        
        # 创建一个简单的音频文件用于测试
        print("[INFO] Creating test audio...")
        test_audio_path = "/tmp/test_audio.wav"
        import soundfile as sf
        
        # 生成1秒的正弦波音频
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
        
        sf.write(test_audio_path, audio_data, sample_rate)
        print(f"[INFO] Test audio created: {test_audio_path}")
        
        # 测试音频编码
        print("[INFO] Testing audio encoding...")
        try:
            audio_emb = speech_encoder.predict([test_audio_path])
            print(f"[INFO] Audio encoding successful: {audio_emb.shape}")
        except Exception as e:
            print(f"[ERROR] Audio encoding failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 清理测试文件
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
            print("[INFO] Test audio file cleaned up")
        
        print("[INFO] All tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Qwen2-Audio Basic Test ===")
    success = test_qwen_audio_basic()
    
    if success:
        print("[SUCCESS] Qwen2-Audio model test completed successfully!")
        sys.exit(0)
    else:
        print("[FAILURE] Qwen2-Audio model test failed!")
        sys.exit(1)

















