"""
测试Modal设置和基本功能
"""

import modal
import json
import os

# 创建测试App
app = modal.App("qwen2-audio-test")

# 简单的测试镜像
image = modal.Image.debian_slim(python_version="3.10").pip_install(["torch", "transformers"])

@app.function(image=image, timeout=300)
def test_environment():
    """测试Modal环境"""
    import torch
    import sys
    import os
    
    print(f"[TEST] Python version: {sys.version}")
    print(f"[TEST] PyTorch version: {torch.__version__}")
    print(f"[TEST] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[TEST] CUDA version: {torch.version.cuda}")
        print(f"[TEST] GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[TEST] GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return "Environment test completed"

@app.function(image=image, gpu=modal.gpu.A100(count=1), timeout=600)
def test_gpu():
    """测试GPU功能"""
    import torch
    
    print(f"[GPU TEST] CUDA available: {torch.cuda.is_available()}")
    print(f"[GPU TEST] GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        
        # 简单的GPU计算测试
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.matmul(x, y)
        
        print(f"[GPU TEST] Matrix multiplication result shape: {z.shape}")
        print(f"[GPU TEST] Result mean: {z.mean().item():.4f}")
        
        # 测试GPU内存
        print(f"[GPU TEST] GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"[GPU TEST] GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    return "GPU test completed"

@app.function(image=image.pip_install(["transformers"]), timeout=1200)
def test_model_loading():
    """测试模型加载"""
    from transformers import AutoTokenizer
    
    try:
        # 测试加载一个小模型
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # 测试tokenization
        text = "This is a test sentence."
        tokens = tokenizer(text, return_tensors="pt")
        
        print(f"[MODEL TEST] Tokenizer loaded successfully")
        print(f"[MODEL TEST] Input text: {text}")
        print(f"[MODEL TEST] Token IDs shape: {tokens['input_ids'].shape}")
        
        return "Model loading test completed"
        
    except Exception as e:
        print(f"[MODEL TEST ERROR] {e}")
        return f"Model loading test failed: {e}"

@app.local_entrypoint()
def main():
    """运行所有测试"""
    print("[INFO] Starting Modal environment tests...")
    
    # 测试基本环境
    print("\n=== Testing Basic Environment ===")
    result1 = test_environment.remote()
    print(f"Result: {result1}")
    
    # 测试GPU
    print("\n=== Testing GPU ===")
    result2 = test_gpu.remote()
    print(f"Result: {result2}")
    
    # 测试模型加载
    print("\n=== Testing Model Loading ===")
    result3 = test_model_loading.remote()
    print(f"Result: {result3}")
    
    print("\n[INFO] All tests completed!")
    print("\n[INFO] If all tests passed, your Modal setup is ready for Qwen2-Audio training.")
    print("[INFO] Next steps:")
    print("  1. Prepare your training data files")
    print("  2. Configure HuggingFace token secret")
    print("  3. Run: modal run modal_complete_training.py")

if __name__ == "__main__":
    main()
