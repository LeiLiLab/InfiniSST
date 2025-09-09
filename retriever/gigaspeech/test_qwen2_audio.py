#!/usr/bin/env python3
"""
测试Qwen2-Audio模型集成的简单脚本
用于验证模型加载、音频和文本编码是否正常工作
"""

import torch
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

def test_qwen2_audio_integration():
    """测试Qwen2-Audio模型集成"""
    
    print("=== Qwen2-Audio Integration Test ===")
    
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    if device == "cuda":
        print(f"[INFO] CUDA devices: {torch.cuda.device_count()}")
        print(f"[INFO] Current device: {torch.cuda.current_device()}")
        print(f"[INFO] Device name: {torch.cuda.get_device_name()}")
    
    try:
        # 导入Qwen2-Audio模块
        print("\n[TEST 1] Importing Qwen2-Audio modules...")
        from Qwen2_Audio_train import (
            Qwen2AudioSpeechEncoder,
            Qwen2AudioTextEncoder, 
            ContrastiveQwen2AudioModel
        )
        print("✅ Import successful")
        
        # 测试文本编码器
        print("\n[TEST 2] Testing Text Encoder...")
        try:
            text_encoder = Qwen2AudioTextEncoder(device=device)
            test_texts = ["hello world", "artificial intelligence", "speech recognition"]
            text_embeddings = text_encoder.predict(test_texts)
            print(f"✅ Text encoding successful - Shape: {text_embeddings.shape}")
            print(f"   Sample embedding stats: mean={text_embeddings.mean():.4f}, std={text_embeddings.std():.4f}")
        except Exception as e:
            print(f"❌ Text encoding failed: {e}")
            return False
        
        # 测试音频编码器（如果有测试音频文件）
        print("\n[TEST 3] Testing Audio Encoder...")
        try:
            speech_encoder = Qwen2AudioSpeechEncoder(device=device)
            
            # 创建一个虚拟的音频文件用于测试
            import tempfile
            import soundfile as sf
            
            # 生成1秒的测试音频（16kHz采样率）
            sample_rate = 16000
            duration = 1.0
            test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, test_audio, sample_rate)
                test_audio_path = tmp_file.name
            
            try:
                audio_embeddings = speech_encoder.predict([test_audio_path])
                print(f"✅ Audio encoding successful - Shape: {audio_embeddings.shape}")
                print(f"   Sample embedding stats: mean={audio_embeddings.mean():.4f}, std={audio_embeddings.std():.4f}")
            finally:
                # 清理临时文件
                os.unlink(test_audio_path)
                
        except Exception as e:
            print(f"❌ Audio encoding failed: {e}")
            print("   This might be due to model loading issues or insufficient GPU memory")
            return False
        
        # 测试对比模型
        print("\n[TEST 4] Testing Contrastive Model...")
        try:
            model = ContrastiveQwen2AudioModel(
                speech_encoder, text_encoder,
                hidden_dim=4096, proj_dim=512, unfreeze_layers=0
            ).to(device)
            
            # 测试文本编码
            text_emb = model.encode_text(test_texts)
            print(f"✅ Model text encoding - Shape: {text_emb.shape}")
            
            # 测试音频编码（重新创建临时音频文件）
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, test_audio, sample_rate)
                test_audio_path = tmp_file.name
            
            try:
                audio_emb = model.encode_audio([test_audio_path])
                print(f"✅ Model audio encoding - Shape: {audio_emb.shape}")
                
                # 测试相似度计算
                similarity = torch.cosine_similarity(audio_emb[0:1], text_emb[0:1], dim=1)
                print(f"✅ Similarity computation - Value: {similarity.item():.4f}")
                
            finally:
                os.unlink(test_audio_path)
                
        except Exception as e:
            print(f"❌ Contrastive model test failed: {e}")
            return False
        
        # 测试参数统计
        print("\n[TEST 5] Model Parameter Statistics...")
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            print(f"✅ Parameter statistics:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%})")
            print(f"   Frozen parameters: {frozen_params:,} ({frozen_params/total_params:.1%})")
            
        except Exception as e:
            print(f"❌ Parameter statistics failed: {e}")
            return False
        
        print("\n🎉 All tests passed! Qwen2-Audio integration is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   Please ensure transformers, librosa, and other dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """检查必要的依赖包"""
    print("=== Dependency Check ===")
    
    required_packages = [
        'torch',
        'transformers', 
        'librosa',
        'soundfile',
        'numpy',
        'faiss'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All dependencies are available")
        return True


def main():
    """主函数"""
    print("Qwen2-Audio Integration Test Script")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages first.")
        sys.exit(1)
    
    print()
    
    # 运行集成测试
    if test_qwen2_audio_integration():
        print("\n🎉 Integration test completed successfully!")
        print("\nYou can now run the Qwen2-Audio training pipeline:")
        print("  bash Qwen2_Audio_term_level_pipeline.sh term true")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed.")
        print("\nPlease check the error messages above and fix the issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
