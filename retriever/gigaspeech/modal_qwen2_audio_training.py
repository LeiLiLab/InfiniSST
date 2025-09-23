"""
Modal部署脚本 - Qwen2-Audio Term-Level DDP训练
使用Modal云平台进行分布式训练
"""

import modal
import json
import os
from pathlib import Path

# 创建Modal App
app = modal.App("qwen2-audio-term-level-training")

# 定义容器镜像，包含所有必要的依赖
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git", 
        "wget", 
        "curl", 
        "ffmpeg",
        "libsndfile1-dev",
        "build-essential",
        "nvidia-cuda-toolkit"
    ])
    .pip_install([
        "torch==2.1.0",
        "torchvision==0.16.0", 
        "torchaudio==2.1.0",
        "transformers==4.36.0",
        "datasets==2.14.0",
        "accelerate==0.24.0",
        "peft==0.6.0",
        "faiss-gpu==1.7.4",
        "soundfile==0.12.1",
        "librosa==0.10.1",
        "numpy==1.24.3",
        "scipy==1.11.4",
        "scikit-learn==1.3.2",
        "tqdm==4.66.1",
        "tensorboard==2.15.1",
        "wandb==0.16.0",
        "sonar-space @ git+https://github.com/facebookresearch/SONAR.git#subdirectory=sonar"
    ])
    .run_commands([
        "pip install flash-attn==2.3.6 --no-build-isolation",
    ])
)

# 定义GPU资源配置 - 使用8个A100 GPU进行DDP训练
gpu_config = modal.gpu.A100(count=8)

# 定义存储卷用于数据和模型
volume = modal.Volume.from_name("qwen2-audio-training-data", create_if_missing=True)

# 数据上传函数
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,  # 1小时超时
)
def upload_data(data_files: dict):
    """
    上传本地数据文件到Modal存储卷
    
    Args:
        data_files: 字典，键为目标路径，值为本地文件内容
    """
    import json
    import os
    
    print(f"[INFO] Uploading {len(data_files)} data files to Modal volume...")
    
    for target_path, file_content in data_files.items():
        full_path = f"/data/{target_path}"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # 如果是JSON文件，直接写入
        if target_path.endswith('.json'):
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(file_content, f, indent=2, ensure_ascii=False)
        else:
            # 其他文件类型，假设是文本
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
        
        print(f"[INFO] Uploaded: {target_path}")
    
    # 提交卷的更改
    volume.commit()
    print(f"[INFO] All data files uploaded and committed to volume.")


# 主训练函数
@app.function(
    image=image,
    gpu=gpu_config,
    volumes={"/data": volume},
    timeout=86400,  # 24小时超时
    memory=256*1024,  # 256GB内存
    cpu=64,  # 64个CPU核心
    secrets=[
        modal.Secret.from_name("huggingface-token"),  # HuggingFace token用于下载模型
        modal.Secret.from_name("wandb-token")  # W&B token用于实验跟踪
    ]
)
def train_qwen2_audio_ddp(
    train_samples_path: str = "xl_term_level_chunks_merged.json",
    test_samples_path: str = "samples/xl/term_level_chunks_500000_1000000.json", 
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-4,
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    audio_text_loss_ratio: float = 0.3,
    audio_term_loss_ratio: float = 0.7,
    enable_hard_neg: bool = True,
    hard_neg_k: int = 10,
    enable_wandb: bool = True
):
    """
    使用DDP在多GPU上训练Qwen2-Audio模型
    """
    import subprocess
    import os
    import sys
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from pathlib import Path
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = "8"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_SHM_DISABLE"] = "0"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    
    # 设置HuggingFace缓存目录
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"
    
    # 初始化W&B（如果启用）
    if enable_wandb:
        import wandb
        wandb.init(
            project="qwen2-audio-term-level",
            name=f"ddp-training-{model_name.split('/')[-1]}",
            config={
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "audio_text_loss_ratio": audio_text_loss_ratio,
                "audio_term_loss_ratio": audio_term_loss_ratio,
                "enable_hard_neg": enable_hard_neg,
                "hard_neg_k": hard_neg_k,
            }
        )
    
    print(f"[INFO] Starting Qwen2-Audio DDP training on Modal")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] GPUs: {torch.cuda.device_count()}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Learning rate: {lr}")
    print(f"[INFO] LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # 检查数据文件
    train_path = f"/data/{train_samples_path}"
    test_path = f"/data/{test_samples_path}"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    print(f"[INFO] Training data: {train_path}")
    print(f"[INFO] Test data: {test_path}")
    
    # 将训练代码写入临时文件
    training_code = '''
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import faiss
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Union
import warnings
import librosa

warnings.filterwarnings("ignore")

# 这里需要包含所有的训练代码...
# 由于代码太长，这里省略具体实现
# 实际使用时需要将完整的训练代码包含在这里

def main():
    # DDP训练主函数
    pass

if __name__ == "__main__":
    main()
'''
    
    # 写入训练脚本
    script_path = "/tmp/qwen2_audio_ddp_train.py"
    with open(script_path, 'w') as f:
        f.write(training_code)
    
    # 构建训练命令
    cmd = [
        sys.executable, script_path,
        "--train_samples_path", train_path,
        "--test_samples_path", test_path,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--model_name", model_name,
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--audio_text_loss_ratio", str(audio_text_loss_ratio),
        "--audio_term_loss_ratio", str(audio_term_loss_ratio),
        "--save_path", "/data/qwen2_audio_term_level_modal.pt",
        "--best_model_path", "/data/qwen2_audio_term_level_best_modal.pt",
        "--gpu_ids", "0,1,2,3,4,5,6,7",
        "--filter_no_term"
    ]
    
    if enable_hard_neg:
        cmd.extend([
            "--enable_hard_neg",
            "--hard_neg_k", str(hard_neg_k)
        ])
    
    print(f"[INFO] Executing command: {' '.join(cmd)}")
    
    # 执行训练
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[INFO] Training completed successfully!")
        print("STDOUT:", result.stdout[-2000:])  # 最后2000字符
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed with return code {e.returncode}")
        print("STDOUT:", e.stdout[-2000:])  # 最后2000字符
        print("STDERR:", e.stderr[-2000:])  # 最后2000字符
        raise
    
    # 提交模型文件到卷
    volume.commit()
    
    if enable_wandb:
        wandb.finish()
    
    return "Training completed successfully!"


# 简化的训练函数，直接使用已有的DDP训练代码
@app.function(
    image=image,
    gpu=gpu_config,
    volumes={"/data": volume},
    timeout=86400,  # 24小时超时
    memory=256*1024,  # 256GB内存
    cpu=64,  # 64个CPU核心
    secrets=[
        modal.Secret.from_name("huggingface-token"),
    ]
)
def train_with_existing_code(
    train_samples_path: str = "xl_term_level_chunks_merged.json",
    test_samples_path: str = "samples/xl/term_level_chunks_500000_1000000.json",
    training_script_content: str = "",
    **training_args
):
    """
    使用已有的训练代码在Modal上训练
    """
    import subprocess
    import os
    import sys
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"
    
    print(f"[INFO] Starting training with {torch.cuda.device_count()} GPUs")
    
    # 写入训练脚本
    script_path = "/tmp/train_ddp.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(training_script_content)
    
    # 构建命令行参数
    cmd = [sys.executable, script_path]
    
    # 添加基本参数
    cmd.extend([
        "--train_samples_path", f"/data/{train_samples_path}",
        "--test_samples_path", f"/data/{test_samples_path}",
        "--save_path", "/data/qwen2_audio_term_level_modal.pt",
        "--best_model_path", "/data/qwen2_audio_term_level_best_modal.pt",
        "--gpu_ids", "0,1,2,3,4,5,6,7"
    ])
    
    # 添加其他训练参数
    for key, value in training_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"[INFO] Executing: {' '.join(cmd)}")
    
    # 执行训练
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出日志
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("[INFO] Training completed successfully!")
        else:
            print(f"[ERROR] Training failed with return code {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except Exception as e:
        print(f"[ERROR] Training execution failed: {e}")
        raise
    
    # 提交模型文件到卷
    volume.commit()
    print("[INFO] Model files committed to volume")
    
    return "Training completed successfully!"


# 本地入口点
@app.local_entrypoint()
def main():
    """
    本地入口点 - 上传数据并启动训练
    """
    import json
    from pathlib import Path
    
    # 定义本地数据文件路径
    local_data_files = {
        # 训练数据
        "xl_term_level_chunks_merged.json": "data/xl_term_level_chunks_merged.json",
        # 测试数据  
        "samples/xl/term_level_chunks_500000_1000000.json": "data/samples/xl/term_level_chunks_500000_1000000.json",
        # 词汇表
        "terms/glossary_filtered.json": "data/terms/glossary_filtered.json",
    }
    
    # 加载本地数据文件
    print("[INFO] Loading local data files...")
    data_to_upload = {}
    
    for target_path, local_path in local_data_files.items():
        if os.path.exists(local_path):
            print(f"[INFO] Loading {local_path} -> {target_path}")
            with open(local_path, 'r', encoding='utf-8') as f:
                data_to_upload[target_path] = json.load(f)
        else:
            print(f"[WARN] Local file not found: {local_path}")
    
    if not data_to_upload:
        print("[ERROR] No data files found to upload!")
        return
    
    # 上传数据到Modal
    print("[INFO] Uploading data to Modal...")
    upload_data.remote(data_to_upload)
    
    # 读取训练脚本内容
    training_script_path = "Qwen2_Audio_term_level_train_ddp.py"
    if os.path.exists(training_script_path):
        with open(training_script_path, 'r', encoding='utf-8') as f:
            training_script_content = f.read()
    else:
        print(f"[ERROR] Training script not found: {training_script_path}")
        return
    
    # 启动训练
    print("[INFO] Starting training on Modal...")
    
    training_args = {
        "epochs": 20,
        "batch_size": 128,
        "lr": 1e-4,
        "model_name": "Qwen/Qwen2-Audio-7B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "audio_text_loss_ratio": 0.3,
        "audio_term_loss_ratio": 0.7,
        "enable_hard_neg": True,
        "hard_neg_k": 10,
        "hard_neg_weight": 0.2,
        "hard_neg_margin": 0.1,
        "filter_no_term": True,
        "patience": 2
    }
    
    result = train_with_existing_code.remote(
        train_samples_path="xl_term_level_chunks_merged.json",
        test_samples_path="samples/xl/term_level_chunks_500000_1000000.json",
        training_script_content=training_script_content,
        **training_args
    )
    
    print(f"[INFO] Training result: {result}")


if __name__ == "__main__":
    main()
