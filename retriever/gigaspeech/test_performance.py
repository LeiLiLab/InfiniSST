#!/usr/bin/env python3
"""
测试Qwen2-Audio模型的性能瓶颈
分析训练慢的原因并提供优化建议
"""

import torch
import numpy as np
import json
import os
import time
import psutil
from tqdm import tqdm

# 导入我们的Qwen2-Audio模型
from Qwen2_Audio_train import (
    Qwen2AudioSpeechEncoder, 
    Qwen2AudioTextEncoder, 
    ContrastiveQwen2AudioModel
)


def test_model_loading_speed():
    """测试模型加载速度"""
    print("=== 测试模型加载速度 ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 测试speech encoder加载
    start_time = time.time()
    speech_encoder = Qwen2AudioSpeechEncoder(device=device)
    speech_load_time = time.time() - start_time
    print(f"Speech encoder加载时间: {speech_load_time:.2f}秒")
    
    # 测试text encoder加载
    start_time = time.time()
    text_encoder = Qwen2AudioTextEncoder(device=device)
    text_load_time = time.time() - start_time
    print(f"Text encoder加载时间: {text_load_time:.2f}秒")
    
    # 测试contrastive model初始化
    start_time = time.time()
    model = ContrastiveQwen2AudioModel(
        speech_encoder, text_encoder,
        hidden_dim=4096,
        proj_dim=512,
        unfreeze_layers=0
    ).to(device)
    model_init_time = time.time() - start_time
    print(f"Contrastive model初始化时间: {model_init_time:.2f}秒")
    
    total_time = speech_load_time + text_load_time + model_init_time
    print(f"总模型加载时间: {total_time:.2f}秒")
    
    return model, device


def test_single_audio_encoding_speed(model, device, num_tests=10):
    """测试单个音频编码速度"""
    print(f"\n=== 测试单个音频编码速度 ({num_tests}次) ===")
    
    # 加载一个测试样本
    test_samples_path = "data/samples/xl/term_level_chunks_500000_1000000.json"
    with open(test_samples_path, 'r') as f:
        all_samples = json.load(f)
    
    # 找到有效的音频文件
    test_audio_paths = []
    for sample in all_samples[:50]:  # 检查前50个
        audio_path = sample.get('term_chunk_audio', '')
        if audio_path and os.path.exists(audio_path):
            test_audio_paths.append(audio_path)
            if len(test_audio_paths) >= num_tests:
                break
    
    if len(test_audio_paths) == 0:
        print("ERROR: 没有找到有效的音频文件")
        return
    
    print(f"找到 {len(test_audio_paths)} 个有效音频文件")
    
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # 预热
    print("预热模型...")
    _ = raw_model.encode_audio([test_audio_paths[0]])
    
    # 测试单个音频编码速度
    encoding_times = []
    for i, audio_path in enumerate(test_audio_paths):
        start_time = time.time()
        try:
            audio_emb = raw_model.encode_audio([audio_path])
            encoding_time = time.time() - start_time
            encoding_times.append(encoding_time)
            print(f"音频 {i+1}: {encoding_time:.3f}秒 ({os.path.basename(audio_path)})")
        except Exception as e:
            print(f"音频 {i+1} 编码失败: {e}")
    
    if encoding_times:
        avg_time = np.mean(encoding_times)
        std_time = np.std(encoding_times)
        min_time = np.min(encoding_times)
        max_time = np.max(encoding_times)
        
        print(f"\n单个音频编码统计:")
        print(f"  平均时间: {avg_time:.3f}秒")
        print(f"  标准差: {std_time:.3f}秒")
        print(f"  最快: {min_time:.3f}秒")
        print(f"  最慢: {max_time:.3f}秒")
        
        # 估算batch处理时间
        batch_sizes = [8, 16, 32, 64]
        print(f"\n预估不同batch size的处理时间:")
        for batch_size in batch_sizes:
            estimated_time = avg_time * batch_size
            print(f"  Batch size {batch_size}: ~{estimated_time:.2f}秒")
    
    return test_audio_paths


def test_batch_encoding_speed(model, device, test_audio_paths):
    """测试批量编码速度"""
    print(f"\n=== 测试批量编码速度 ===")
    
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    batch_sizes = [1, 2, 4, 8]
    for batch_size in batch_sizes:
        if batch_size > len(test_audio_paths):
            continue
            
        batch_paths = test_audio_paths[:batch_size]
        
        # 测试3次取平均
        times = []
        for _ in range(3):
            start_time = time.time()
            try:
                audio_embs = raw_model.encode_audio(batch_paths)
                batch_time = time.time() - start_time
                times.append(batch_time)
            except Exception as e:
                print(f"Batch size {batch_size} 编码失败: {e}")
                break
        
        if times:
            avg_time = np.mean(times)
            per_sample_time = avg_time / batch_size
            print(f"Batch size {batch_size}: {avg_time:.3f}秒 ({per_sample_time:.3f}秒/样本)")


def test_memory_usage(model, device):
    """测试内存使用情况"""
    print(f"\n=== 测试内存使用情况 ===")
    
    # CPU内存
    process = psutil.Process()
    cpu_memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"CPU内存使用: {cpu_memory_mb:.1f} MB")
    
    # GPU内存
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        
        print(f"GPU内存分配: {gpu_memory_allocated:.1f} MB")
        print(f"GPU内存保留: {gpu_memory_reserved:.1f} MB")
        print(f"GPU总内存: {total_gpu_memory:.1f} MB")
        print(f"GPU内存使用率: {gpu_memory_reserved/total_gpu_memory:.1%}")


def test_text_encoding_speed(model, device):
    """测试文本编码速度"""
    print(f"\n=== 测试文本编码速度 ===")
    
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # 测试不同长度的文本
    test_texts = [
        "hello",
        "hello world",
        "this is a test sentence",
        "this is a longer test sentence with more words to see how performance changes",
        "this is an even longer test sentence with many more words to really test the performance of text encoding with various lengths"
    ]
    
    for i, text in enumerate(test_texts):
        times = []
        for _ in range(5):  # 测试5次取平均
            start_time = time.time()
            try:
                text_emb = raw_model.encode_text([text])
                encoding_time = time.time() - start_time
                times.append(encoding_time)
            except Exception as e:
                print(f"文本编码失败: {e}")
                break
        
        if times:
            avg_time = np.mean(times)
            print(f"文本 {i+1} ({len(text.split())} 词): {avg_time:.4f}秒")


def analyze_bottlenecks():
    """分析性能瓶颈并给出建议"""
    print(f"\n=== 性能瓶颈分析和优化建议 ===")
    
    print("可能的性能瓶颈:")
    print("1. 模型大小: Qwen2-Audio-7B是一个大模型，推理本身就比较慢")
    print("2. 音频预处理: 音频文件读取、重采样、特征提取都需要时间")
    print("3. GPU内存: 如果GPU内存不足，会降低批处理效率")
    print("4. 数据加载: 音频文件I/O可能是瓶颈")
    print("5. 模型未优化: 没有使用TensorRT、ONNX等推理优化")
    
    print("\n优化建议:")
    print("1. 减少batch size: 如果GPU内存不足，使用较小的batch size")
    print("2. 使用更快的存储: SSD比HDD快，内存缓存更快")
    print("3. 音频预处理优化: 预先转换音频格式，缓存特征")
    print("4. 模型量化: 使用FP16或INT8量化减少内存和计算")
    print("5. 并行处理: 使用多进程处理音频文件")
    print("6. 模型剪枝: 移除不必要的层或参数")
    print("7. 使用更小的模型: 考虑使用Qwen2-Audio-1.5B等较小版本")


def main():
    print("=== Qwen2-Audio 性能测试 ===")
    
    # 测试模型加载速度
    model, device = test_model_loading_speed()
    
    # 测试内存使用
    test_memory_usage(model, device)
    
    # 测试音频编码速度
    test_audio_paths = test_single_audio_encoding_speed(model, device, num_tests=5)
    
    if test_audio_paths:
        # 测试批量编码速度
        test_batch_encoding_speed(model, device, test_audio_paths)
    
    # 测试文本编码速度
    test_text_encoding_speed(model, device)
    
    # 分析瓶颈
    analyze_bottlenecks()
    
    print("\n=== 性能测试完成 ===")


if __name__ == "__main__":
    main()
