#!/usr/bin/env python3
"""
简单的Qwen DDP测试脚本，用于验证DDP初始化是否正常工作
"""

import os
import sys
import time
import socket

# DDP相关导入
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# 设置CUDA环境
gpu_ids_arg = "0,2,3,4"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_arg
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch

def setup_ddp(rank, world_size):
    """设置DDP环境"""
    # 设置NCCL环境变量
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    
    # 打印调试信息
    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rank {rank}: MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
    
    # 设置超时时间
    import datetime
    timeout = datetime.timedelta(minutes=5)  # 5分钟超时
    
    # 初始化进程组
    try:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rank {rank}: Initializing process group...")
        dist.init_process_group(
            backend="nccl", 
            rank=rank, 
            world_size=world_size, 
            timeout=timeout
        )
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rank {rank}: Process group initialized successfully")
    except Exception as e:
        print(f"[ERROR] [{time.strftime('%H:%M:%S')}] Rank {rank}: Failed to initialize process group: {e}")
        print(f"[DEBUG] [{time.strftime('%H:%M:%S')}] Rank {rank}: Environment - MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
        import traceback
        traceback.print_exc()
        raise
    
    # 设置当前进程的GPU
    torch.cuda.set_device(rank)
    
    # 验证GPU设备
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Using device: cuda:{rank}")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Device name: {torch.cuda.get_device_name(rank)}")
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] Memory: {torch.cuda.get_device_properties(rank).total_memory / 1024**3:.1f} GB")

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()

def test_ddp(rank, world_size):
    """简单的DDP测试函数"""
    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Process {rank} started")
    
    # 设置DDP环境
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # 创建一个简单的模型
    model = torch.nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[rank])
    
    # 创建一些测试数据
    test_input = torch.randn(5, 10).to(device)
    
    # 前向传播
    output = model(test_input)
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rank {rank}: Test completed successfully")
    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Rank {rank}: Loss = {loss.item():.4f}")
    
    # 同步所有进程
    dist.barrier()
    
    if rank == 0:
        print(f"[INFO] [{time.strftime('%H:%M:%S')}] All processes completed successfully!")
    
    cleanup_ddp()

def main():
    world_size = 4  # 使用4个GPU
    
    print(f"[INFO] Starting DDP test with {world_size} GPUs")
    print(f"[INFO] GPU IDs: {gpu_ids_arg}")
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 预先设置一个固定端口，避免进程间通信问题
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    if 'MASTER_PORT' not in os.environ:
        master_port = find_free_port()
        os.environ['MASTER_PORT'] = str(master_port)
        print(f"[INFO] Pre-allocated master port: {master_port}")
    
    # 设置其他DDP环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    
    # 启动多进程测试
    try:
        mp.spawn(test_ddp, args=(world_size,), nprocs=world_size, join=True)
        print("[INFO] DDP test completed successfully!")
    except KeyboardInterrupt:
        print("[INFO] Test interrupted by user")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


















