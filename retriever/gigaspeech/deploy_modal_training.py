"""
简化的Modal部署脚本 - Qwen2-Audio Term-Level DDP训练
专门用于实际部署和运行
"""

import modal
import json
import os

# 创建Modal App
app = modal.App("qwen2-audio-ddp-training")

# 定义容器镜像
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git", "wget", "curl", "ffmpeg", "libsndfile1-dev", 
        "build-essential", "nvidia-cuda-toolkit"
    ])
    .pip_install([
        "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0",
        "transformers==4.36.0", "datasets==2.14.0", "accelerate==0.24.0",
        "peft==0.6.0", "faiss-gpu==1.7.4", "soundfile==0.12.1",
        "librosa==0.10.1", "numpy==1.24.3", "scipy==1.11.4",
        "scikit-learn==1.3.2", "tqdm==4.66.1"
    ])
    .run_commands([
        "pip install flash-attn==2.3.6 --no-build-isolation",
    ])
)

# 定义存储卷
volume = modal.Volume.from_name("qwen2-audio-data", create_if_missing=True)

# 数据上传函数
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
)
def upload_training_data(data_dict: dict):
    """上传训练数据到Modal卷"""
    import json
    import os
    
    print(f"[INFO] Uploading {len(data_dict)} files...")
    
    for file_path, content in data_dict.items():
        full_path = f"/data/{file_path}"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            if isinstance(content, (dict, list)):
                json.dump(content, f, indent=2, ensure_ascii=False)
            else:
                f.write(str(content))
        
        print(f"[INFO] Uploaded: {file_path}")
    
    volume.commit()
    print("[INFO] Data upload completed and committed.")

# 主训练函数
@app.function(
    image=image,
    gpu=modal.gpu.A100(count=8),
    volumes={"/data": volume},
    timeout=86400,  # 24小时
    memory=256*1024,  # 256GB
    cpu=64,
    secrets=[modal.Secret.from_name("huggingface-token")]
)
def run_ddp_training():
    """运行DDP训练"""
    import subprocess
    import os
    import sys
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "MASTER_ADDR": "localhost", 
        "MASTER_PORT": "12355",
        "WORLD_SIZE": "8",
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "1",
        "HF_HOME": "/data/hf_cache",
        "TRANSFORMERS_CACHE": "/data/hf_cache"
    })
    
    # 将训练脚本写入文件（这里需要包含完整的训练代码）
    training_script = '''
# 这里应该包含完整的Qwen2_Audio_term_level_train_ddp.py内容
# 为了简洁，这里省略...
import os
print("Training script placeholder - replace with actual training code")
'''
    
    script_path = "/tmp/train.py"
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    # 构建训练命令
    cmd = [
        sys.executable, script_path,
        "--train_samples_path", "/data/xl_term_level_chunks_merged.json",
        "--test_samples_path", "/data/samples/xl/term_level_chunks_500000_1000000.json",
        "--epochs", "20",
        "--batch_size", "128", 
        "--lr", "1e-4",
        "--model_name", "Qwen/Qwen2-Audio-7B-Instruct",
        "--save_path", "/data/model_final.pt",
        "--best_model_path", "/data/model_best.pt",
        "--gpu_ids", "0,1,2,3,4,5,6,7",
        "--enable_hard_neg",
        "--filter_no_term"
    ]
    
    print(f"[INFO] Running: {' '.join(cmd)}")
    
    # 执行训练
    process = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    
    # 实时输出
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"Training failed with code {process.returncode}")
    
    # 提交结果
    volume.commit()
    return "Training completed successfully"

# 本地入口点
@app.local_entrypoint() 
def main():
    """本地入口点"""
    # 加载本地数据
    data_files = {}
    
    # 训练数据
    train_path = "data/xl_term_level_chunks_merged.json"
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            data_files["xl_term_level_chunks_merged.json"] = json.load(f)
        print(f"[INFO] Loaded training data: {len(data_files['xl_term_level_chunks_merged.json'])} samples")
    
    # 测试数据
    test_path = "data/samples/xl/term_level_chunks_500000_1000000.json"
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            data_files["samples/xl/term_level_chunks_500000_1000000.json"] = json.load(f)
        print(f"[INFO] Loaded test data: {len(data_files['samples/xl/term_level_chunks_500000_1000000.json'])} samples")
    
    if not data_files:
        print("[ERROR] No data files found!")
        return
    
    # 上传数据
    print("[INFO] Uploading data to Modal...")
    upload_training_data.remote(data_files)
    
    # 启动训练
    print("[INFO] Starting DDP training...")
    result = run_ddp_training.remote()
    print(f"[INFO] Result: {result}")


if __name__ == "__main__":
    main()
