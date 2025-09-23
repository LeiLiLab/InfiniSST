# DDP问题修复脚本 - 解决NCCL通信问题

import os
import sys
import torch
import torch.distributed as dist

def diagnose_ddp_environment():
    """诊断DDP环境问题"""
    print("=== DDP环境诊断 ===")
    
    # 1. 检查CUDA环境
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, Memory: {props.total_memory/1024**3:.1f}GB")
    
    # 2. 检查网络接口
    print("\n=== 网络接口检查 ===")
    import subprocess
    try:
        result = subprocess.run(['ip', 'addr'], capture_output=True, text=True)
        interfaces = []
        for line in result.stdout.split('\n'):
            if 'inet ' in line and ('127.0.0.1' in line or '192.168.' in line or '10.' in line):
                interfaces.append(line.strip())
        print("Available network interfaces:")
        for interface in interfaces[:5]:  # 只显示前5个
            print(f"  {interface}")
    except:
        print("Cannot check network interfaces")
    
    # 3. 检查NCCL版本
    print(f"\n=== NCCL信息 ===")
    try:
        print(f"NCCL version: {torch.cuda.nccl.version()}")
    except:
        print("NCCL version not available")
    
    # 4. 检查环境变量
    print(f"\n=== 环境变量 ===")
    nccl_vars = [k for k in os.environ.keys() if 'NCCL' in k or 'CUDA' in k]
    for var in sorted(nccl_vars):
        print(f"{var}={os.environ[var]}")

def test_simple_ddp():
    """测试简单的DDP通信"""
    print("\n=== 测试简单DDP通信 ===")
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # 使用不同的端口
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    
    def simple_worker(rank, world_size):
        try:
            print(f"[Rank {rank}] Initializing process group...")
            dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=torch.distributed.default_pg_timeout)
            
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
            
            print(f"[Rank {rank}] Creating tensor on device {device}")
            tensor = torch.ones(2, device=device) * rank
            
            print(f"[Rank {rank}] Before allreduce: {tensor}")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            print(f"[Rank {rank}] After allreduce: {tensor}")
            
            dist.destroy_process_group()
            print(f"[Rank {rank}] Success!")
            
        except Exception as e:
            print(f"[Rank {rank}] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试2个GPU的简单情况
    world_size = min(2, torch.cuda.device_count())
    if world_size >= 2:
        print(f"Testing with {world_size} GPUs...")
        import torch.multiprocessing as mp
        mp.spawn(simple_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("Need at least 2 GPUs for DDP test")

if __name__ == "__main__":
    diagnose_ddp_environment()
    test_simple_ddp()
