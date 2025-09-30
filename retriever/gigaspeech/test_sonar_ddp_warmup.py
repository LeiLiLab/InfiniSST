#!/usr/bin/env python3
"""
测试SONAR DDP预热下载机制
简化版测试脚本，只测试模型初始化阶段
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# 设置CUDA环境
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

def setup_ddp(rank, world_size):
    """设置DDP环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # 使用不同端口避免冲突
    
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '0'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    
    import datetime
    timeout = datetime.timedelta(minutes=5)
    
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
        if rank == 0:
            print(f"[INFO] Process group initialized successfully")
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] Failed to initialize process group: {e}")
        raise
    
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()

def warmup_model_download(rank, world_size):
    """Rank0 预热下载模型权重，避免多进程并发写缓存冲突"""
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rank 0 warming up model download...")
        warmup_start = time.time()
        
        try:
            # 使用CPU预热下载，避免GPU内存占用
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Downloading speech encoder...")
            speech_encoder_warmup = SpeechToEmbeddingModelPipeline(
                encoder="sonar_speech_encoder_eng", device="cpu"
            )
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Speech encoder downloaded successfully")
            
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Downloading text encoder...")
            text_encoder_warmup = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device="cpu",
                dtype=torch.float32,
            )
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Text encoder downloaded successfully")
            
            # 清理预热模型以释放内存
            del speech_encoder_warmup, text_encoder_warmup
            torch.cuda.empty_cache()
            
            warmup_time = time.time() - warmup_start
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Model warmup completed in {warmup_time:.2f}s")
            
        except Exception as e:
            print(f"[WARN] [{time.strftime('%H:%M:%S')}] Model warmup failed: {e}")
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Will proceed with normal initialization")
    
    # 等待rank0完成下载
    dist.barrier()
    
    if rank != 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rank {rank} proceeding after warmup barrier")

def test_ddp_warmup(rank, world_size):
    """测试DDP预热下载"""
    start_time = time.time()
    
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] DDP Test started with {world_size} GPUs")
    
    # === 设置DDP环境 ===
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] DDP setup completed")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Device: {device}")
    
    # === Rank0 预热模型下载 ===
    warmup_model_download(rank, world_size)
    
    # === 模型初始化 ===
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Starting model initialization...")
    
    try:
        speech_start = time.time()
        speech_encoder = SpeechToEmbeddingModelPipeline(
            encoder="sonar_speech_encoder_eng", device=device
        )
        speech_time = time.time() - speech_start
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Speech encoder initialized in {speech_time:.2f}s")

        text_start = time.time()
        text_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=torch.float32,
        )
        text_time = time.time() - text_start
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Text encoder initialized in {text_time:.2f}s")
        
        # 简单测试编码功能
        if rank == 0:
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Testing encoding functionality...")
            test_texts = ["hello world", "this is a test"]
            text_emb = text_encoder.predict(test_texts)
            print(f"[INFO] [{time.strftime('%H:%M:%S')}] Text encoding test successful: {text_emb.shape}")
        
        total_time = time.time() - start_time
        if rank == 0:
            print(f"[SUCCESS] [{time.strftime('%H:%M:%S')}] DDP warmup test completed successfully in {total_time:.2f}s")
        
    except Exception as e:
        if rank == 0:
            print(f"[ERROR] [{time.strftime('%H:%M:%S')}] Model initialization failed: {e}")
        raise
    
    cleanup_ddp()

def main():
    world_size = 2  # 使用2个GPU进行测试
    
    if torch.cuda.device_count() < world_size:
        print(f"[ERROR] Need at least {world_size} GPUs, but only {torch.cuda.device_count()} available")
        return
    
    print(f"[INFO] Starting DDP warmup test with {world_size} GPUs")
    
    # 启动多进程测试
    mp.spawn(test_ddp_warmup, args=(world_size,), nprocs=world_size, join=True)
    
    print("[INFO] DDP warmup test completed")

if __name__ == "__main__":
    main()
