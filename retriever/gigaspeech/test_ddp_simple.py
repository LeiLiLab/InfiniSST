#!/usr/bin/env python3
# 简单的DDP测试脚本

import os
import sys

# 设置CUDA环境（必须在导入torch之前）
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 只测试2个GPU

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def simple_ddp_test(rank, world_size):
    """简单的DDP测试函数"""
    print(f"[Rank {rank}] Starting DDP test...")
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    
    try:
        # 初始化进程组
        print(f"[Rank {rank}] Initializing process group...")
        import datetime
        timeout = datetime.timedelta(minutes=5)
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
        
        # 设置GPU设备
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        print(f"[Rank {rank}] Using device: {device}")
        print(f"[Rank {rank}] Device name: {torch.cuda.get_device_name(rank)}")
        
        # 创建简单的张量
        tensor = torch.ones(2, device=device) * rank
        print(f"[Rank {rank}] Before allreduce: {tensor}")
        
        # 执行allreduce操作
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] After allreduce: {tensor}")
        
        # 创建简单的模型测试DDP
        model = torch.nn.Linear(10, 1).to(device)
        ddp_model = DDP(model, device_ids=[rank])
        
        # 测试前向传播
        input_tensor = torch.randn(5, 10, device=device)
        output = ddp_model(input_tensor)
        loss = output.sum()
        
        # 测试反向传播
        loss.backward()
        print(f"[Rank {rank}] DDP model test successful!")
        
        # 清理
        dist.destroy_process_group()
        print(f"[Rank {rank}] Test completed successfully!")
        
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=== DDP简单测试 ===")
    
    # 检查CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    if torch.cuda.device_count() < 2:
        print("ERROR: Need at least 2 GPUs for DDP test!")
        return
    
    # 启动2个进程测试
    world_size = 2
    print(f"Starting DDP test with {world_size} GPUs...")
    
    try:
        mp.spawn(simple_ddp_test, args=(world_size,), nprocs=world_size, join=True)
        print("DDP test completed successfully!")
    except Exception as e:
        print(f"DDP test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
