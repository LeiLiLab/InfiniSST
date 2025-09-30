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
    .apt_install(["git","wget","curl","ffmpeg","libsndfile1","build-essential","rsync"])
    # ✅ 明确安装 GPU 版 Torch (cu118)
    .pip_install(
        ["torch==2.3.1","torchvision==0.18.1","torchaudio==2.3.1"],
        extra_options="--index-url https://download.pytorch.org/whl/cu118"
    )
    # 安装支持Qwen2-Audio的transformers版本
    .pip_install([
        "transformers>=4.45.0",  # 需要更新版本支持Qwen2-Audio
        "accelerate==0.33.0",
        "datasets",
        "peft==0.11.1",
        "soundfile==0.12.1",
        "librosa==0.10.1",
        "numpy==1.26.4",
        "scipy==1.11.4",
        "scikit-learn==1.3.2",
        "tqdm==4.66.1",
        "huggingface_hub",
        "hf-transfer",
        "wandb==0.16.0"
    ])
    # ✅ 在 Modal 镜像里可用的 FAISS GPU 版本
    .pip_install(["faiss-gpu==1.7.2"])
    # （可选）Flash-Attn 如有编译/ABI 问题先注释
    .run_commands(["pip install --no-build-isolation --no-cache-dir 'flash-attn==2.3.6' || true"])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# 为大文件上传创建专用镜像，包含本地mmap目录（仅在需要时使用）
# image_with_mmap = image.add_local_dir("/mnt/gemini/data1/jiaxuanluo/mmap_shards", remote_path="/mnt/mmap_shards")

# 定义存储卷用于数据和模型
volume = modal.Volume.from_name("qwen2-audio-training-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
# 检查文件是否存在的函数
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,  # 10分钟超时
)
def check_existing_files(file_paths: list):
    """
    检查Modal存储卷中哪些文件已存在
    
    Args:
        file_paths: 要检查的文件路径列表
    
    Returns:
        dict: {file_path: exists}
    """
    import os
    
    result = {}
    for file_path in file_paths:
        full_path = f"/data/{file_path}"
        exists = os.path.exists(full_path)
        result[file_path] = exists
        print(f"[CHECK] {file_path}: {'EXISTS' if exists else 'NOT FOUND'}")
    
    return result


# 上传大文件到Volume的函数（已弃用 - 请使用Modal CLI）
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200,  # 2小时超时
    memory=8192,  # 8GB内存
)
def upload_large_files_to_volume(target_dir: str = "mmap_shards"):
    """
    从挂载的本地目录上传大文件到Modal Volume
    
    Args:
        target_dir: Volume中的目标目录
    """
    import os
    import shutil
    from pathlib import Path
    
    # 挂载的本地目录路径
    mounted_dir = "/mnt/mmap_shards"
    
    print(f"[INFO] Starting upload from mounted dir {mounted_dir} to /data/{target_dir}")
    
    # 确保挂载目录存在
    if not os.path.exists(mounted_dir):
        print(f"[ERROR] Mounted directory does not exist: {mounted_dir}")
        return False
    
    # 创建目标目录
    target_path = f"/data/{target_dir}"
    os.makedirs(target_path, exist_ok=True)
    
    # 统计文件
    files_to_upload = []
    total_size = 0
    for file_name in os.listdir(mounted_dir):
        if file_name.endswith(('.dat', '.index.npz')):
            mounted_file = os.path.join(mounted_dir, file_name)
            target_file = os.path.join(target_path, file_name)
            file_size = os.path.getsize(mounted_file)
            total_size += file_size
            
            # 检查文件是否已存在且大小相同
            if os.path.exists(target_file):
                target_size = os.path.getsize(target_file)
                if file_size == target_size:
                    print(f"[SKIP] File already exists with same size: {file_name}")
                    continue
                else:
                    print(f"[UPDATE] File size changed: {file_name} ({target_size} -> {file_size})")
            
            files_to_upload.append((mounted_file, target_file, file_name, file_size))
    
    if not files_to_upload:
        print("[INFO] No files need to be uploaded")
        return True
    
    upload_size = sum(f[3] for f in files_to_upload)
    print(f"[INFO] Uploading {len(files_to_upload)} files, total size: {upload_size / (1024**3):.2f} GB")
    
    # 上传文件
    for mounted_file, target_file, file_name, file_size in files_to_upload:
        try:
            print(f"[INFO] Uploading {file_name} ({file_size / (1024**3):.2f} GB)...")
            
            # 使用shutil.copy2保持文件元数据
            shutil.copy2(mounted_file, target_file)
            print(f"[SUCCESS] Uploaded {file_name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to upload {file_name}: {e}")
            return False
    
    # 提交Volume更改
    try:
        volume.commit()
        print(f"[SUCCESS] All {len(files_to_upload)} files uploaded and committed to volume")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to commit volume: {e}")
        return False


# 数据上传函数
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,  # 1小时超时
)
def upload_data(data_files: dict, force_upload: bool = False):
    """
    上传小文件（代码、配置等）到Modal存储卷，支持增量上传
    注意：此函数仅用于上传小文件，大文件请使用upload_large_files_to_volume函数
    
    Args:
        data_files: 字典，键为目标路径，值为本地文件内容（仅用于小文件）
        force_upload: 是否强制重新上传所有文件
    """
    import json
    import os
    import hashlib
    
    print(f"[INFO] Checking {len(data_files)} data files for upload...")
    
    files_to_upload = {}
    skipped_files = []
    
    for target_path, file_content in data_files.items():
        full_path = f"/data/{target_path}"
        
        # 检查文件是否已存在
        if not force_upload and os.path.exists(full_path):
            try:
                # 对于词汇表等静态文件，直接跳过检查
                if target_path in ['glossary_merged.json', 'glossary_filtered.json']:
                    skipped_files.append(target_path)
                    print(f"[SKIP] Static file skipped: {target_path}")
                    continue
                
                if target_path.endswith('.json'):
                    # 对于其他JSON文件，简单比较长度
                    with open(full_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    
                    if len(file_content) == len(existing_data):
                        skipped_files.append(target_path)
                        print(f"[SKIP] File unchanged (same length): {target_path}")
                        continue
                    else:
                        print(f"[UPDATE] File changed (length: {len(existing_data)} -> {len(file_content)}): {target_path}")
                else:
                    # 对于Python文件，比较文件大小
                    file_size = len(file_content)
                    existing_size = os.path.getsize(full_path)
                    
                    if file_size == existing_size:
                        skipped_files.append(target_path)
                        print(f"[SKIP] File unchanged (same size): {target_path}")
                        continue
                    else:
                        print(f"[UPDATE] File changed (size: {existing_size} -> {file_size}): {target_path}")
                    
            except Exception as e:
                print(f"[WARN] Failed to check existing file {target_path}: {e}, will upload")
        
        # 需要上传的文件
        files_to_upload[target_path] = file_content
    
    print(f"[INFO] Files to upload: {len(files_to_upload)}, skipped: {len(skipped_files)}")
    
    if not files_to_upload:
        print("[INFO] No files need to be uploaded.")
        return
    
    # 上传需要更新的文件
    for target_path, file_content in files_to_upload.items():
        full_path = f"/data/{target_path}"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # 根据文件类型处理
        if target_path.endswith('.json'):
            # JSON文件，直接写入
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(file_content, f, indent=2, ensure_ascii=False)
        else:
            # 其他文件类型（如Python代码），假设是文本
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
        
        print(f"[INFO] Uploaded: {target_path}")
    
    # 提交卷的更改
    volume.commit()
    print(f"[INFO] {len(files_to_upload)} files uploaded and committed to volume.")


@app.function(
    image=image,
    min_containers=1,
    gpu="H200:8",
    volumes={"/data": volume, "/root/.cache/huggingface": hf_cache_vol},
    timeout=86400,  # 24小时超时
    memory=512*1024,  # 512GB内存
    cpu=64,  # 64个CPU核心
    secrets=[
        modal.Secret.from_name("huggingface-token"),  # HuggingFace token用于下载模型
    ]
)
def train_ddp_modal(
      train_samples_path: str = "",
      test_samples_path: str = "",
      glossary_path: str = "",
      mmap_shard_dir: str = "",
      use_mount_directly: bool = False,
      **training_args
  ):
    """
    使用torchrun在Modal上进行8卡DDP训练
    """
    import subprocess
    import os
    import sys
    import time
    
    # 导入torch（在Modal环境中可用）
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch not available in Modal environment")
        raise
    
    # 数据路径策略：
    # - 默认：从 Volume 复制到本地 NVMe 以提高 I/O 性能
    # - 当 use_mount_directly=True：跳过本地化，直接使用 /data 挂载路径，便于快速测试
    if use_mount_directly:
        print("[INFO] Using mounted /data paths directly (skip localization)")
        def ensure_data_path(p: str) -> str:
            if not p:
                return p
            return p if p.startswith('/') else f"/data/{p}"
        train_samples_path = ensure_data_path(train_samples_path)
        glossary_path = ensure_data_path(glossary_path)
        test_samples_path = ensure_data_path(test_samples_path)
        if mmap_shard_dir:
            mmap_shard_dir = ensure_data_path(mmap_shard_dir)
    else:
        # 数据本地化 - 从 Volume 复制到本地 NVMe 以提高 I/O 性能
        local_root = "/workspace"
        os.makedirs(local_root, exist_ok=True)
        
        print("[INFO] Localizing data from Volume to NVMe for better I/O performance...")
        start_time = time.time()
        
        # 复制 mmap 分片数据
        if mmap_shard_dir and os.path.exists(mmap_shard_dir):
            local_mmap_dir = f"{local_root}/mmap_shards"
            print(f"[INFO] Copying mmap shards from {mmap_shard_dir} to {local_mmap_dir}")
            try:
                subprocess.run(["rsync", "-a", f"{mmap_shard_dir}/", f"{local_mmap_dir}/"], check=True)
                mmap_shard_dir = local_mmap_dir  # 更新为本地路径
                print(f"[INFO] mmap shards copied successfully")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Failed to copy mmap shards: {e}, using original path")
        
        # 复制 JSON 数据文件
        for file_name, local_var in [(train_samples_path, "train_samples_path"), (glossary_path, "glossary_path"), (test_samples_path, "test_samples_path")]:
            if file_name and os.path.exists(f"/data/{file_name}"):
                local_file = f"{local_root}/{file_name}"
                print(f"[INFO] Copying {file_name} to local storage")
                try:
                    subprocess.run(["cp", f"/data/{file_name}", local_file], check=True)
                    # 更新变量为本地路径
                    if local_var == "train_samples_path":
                        train_samples_path = local_file
                    elif local_var == "glossary_path":
                        glossary_path = local_file
                    elif local_var == "test_samples_path":
                        test_samples_path = local_file
                    print(f"[INFO] {file_name} copied successfully")
                except subprocess.CalledProcessError as e:
                    print(f"[WARN] Failed to copy {file_name}: {e}, using original path")
        
        copy_time = time.time() - start_time
        print(f"[INFO] Data localization completed in {copy_time:.1f}s")
    
    # 设置CUDA环境
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"
    
    # 设置DDP环境变量
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
    
    # 设置NCCL环境变量以适应Modal环境
    os.environ["NCCL_DEBUG"] = "WARN"  # 减少日志输出
    os.environ["NCCL_P2P_DISABLE"] = "0"  # 启用P2P通信
    os.environ["NCCL_IB_DISABLE"] = "1"   # 没有InfiniBand保持禁用
    os.environ["NCCL_SHM_DISABLE"] = "0"  # 启用共享内存
    # 移除SOCKET_IFNAME限制，让NCCL自动选择最佳接口
    os.environ.pop("NCCL_SOCKET_IFNAME", None)
    
    print(f"[INFO] Modal DDP Training with {torch.cuda.device_count()} GPUs")
    print(f"[INFO] Available GPUs: {list(range(torch.cuda.device_count()))}")
    
    # 使用上传到存储卷的训练脚本
    script_path = "/data/train_ddp_simplified.py"
    
    # 检查训练脚本是否存在
    if not os.path.exists(script_path):
        print(f"[ERROR] Training script not found in Modal storage: {script_path}")
        print("[INFO] Please ensure the training script is uploaded first")
        raise FileNotFoundError(f"Training script not found: {script_path}")
    
    # 需要将依赖的Python文件也复制到容器中
    # 这些文件应该在训练脚本的同一目录下
    dependency_files = [
        "/data/Qwen2_Audio_train.py",
        "/data/train_ddp_simplified.py",
        "/data/mmap_audio_reader.py"
    ]
    
    # 检查依赖文件是否存在
    print(f"[DEBUG] Checking dependency files in /data...")
    try:
        import subprocess as sp
        result = sp.run(['ls', '-la', '/data/'], capture_output=True, text=True)
        print(f"[DEBUG] /data/ directory contents:\n{result.stdout}")
    except:
        pass
    
    for dep_file in dependency_files:
        if not os.path.exists(dep_file):
            print(f"[WARN] Dependency file not found: {dep_file}")
            # 尝试等待一下，有时候Volume同步需要时间
            import time
            time.sleep(2)
            if os.path.exists(dep_file):
                print(f"[OK] Found dependency file after wait: {dep_file}")
            else:
                print(f"[ERROR] Still not found after wait: {dep_file}")
        else:
            print(f"[OK] Found dependency file: {dep_file}")
    
    # 设置Python路径以包含/data目录
    os.environ["PYTHONPATH"] = "/data:" + os.environ.get("PYTHONPATH", "")
    
    # 获取实际GPU数量并启动DDP训练
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Detected {num_gpus} GPUs, starting DDP training")
    
    cmd = [
        "python", "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_addr=127.0.0.1",
        "--master_port=29500",
        script_path
    ]
    
    # 添加基本参数（使用本地化后的路径）
    cmd.extend([
        "--train_samples_path", train_samples_path,
        "--save_path", "/data/qwen2_audio_term_level_modal.pt",
        "--glossary_path", glossary_path,
        "--best_model_path", "/data/qwen2_audio_term_level_best_modal.pt"
    ])
    
    # 只有当test_samples_path不为空时才添加
    if test_samples_path and test_samples_path.strip():
        cmd.extend(["--test_samples_path", test_samples_path])
    
    # 添加 mmap 分片目录参数（可能是本地化或直接挂载路径）
    if mmap_shard_dir:
        cmd.extend(["--mmap_shard_dir", mmap_shard_dir])
    
    # 添加其他训练参数
    for key, value in training_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"[INFO] Executing DDP command: {' '.join(cmd)}")
    
    # 执行DDP训练
    try:
        print(f"[DEBUG] Working directory: {os.getcwd()}")
        print(f"[DEBUG] Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
        print(f"[DEBUG] Contents of /data:")
        try:
            import subprocess as sp
            result = sp.run(['ls', '-la', '/data'], capture_output=True, text=True)
            print(result.stdout)
        except:
            pass
        
        # 执行DDP训练并捕获详细输出
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出日志
        stdout_output = []
        stderr_output = []
        
        import threading
        
        def read_stdout():
            for line in iter(process.stdout.readline, ''):
                line = line.rstrip()
                if line:
                    print(f"[STDOUT] {line}")
                    stdout_output.append(line)
        
        def read_stderr():
            for line in iter(process.stderr.readline, ''):
                line = line.rstrip()
                if line:
                    print(f"[STDERR] {line}")
                    stderr_output.append(line)
        
        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)
        
        stdout_thread.start()
        stderr_thread.start()
        
        process.wait()
        
        stdout_thread.join()
        stderr_thread.join()
        
        if process.returncode == 0:
            print("[INFO] DDP Training completed successfully!")
        else:
            print(f"[ERROR] DDP Training failed with return code {process.returncode}")
            print(f"[ERROR] Command: {' '.join(cmd)}")
            if stderr_output:
                print("[ERROR] Last stderr lines:")
                for line in stderr_output[-20:]:  # 显示最后20行错误
                    print(f"  {line}")
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except Exception as e:
        print(f"[ERROR] DDP Training execution failed: {e}")
        raise
    
    # 提交模型文件到卷
    volume.commit()
    print("[INFO] Model files committed to volume")
    
    return "DDP Training completed successfully!"


# 本地入口点
@app.local_entrypoint()
def main(skip_upload: bool = False, upload_large_files_only: bool = False, eval_only: bool = False, use_mount_directly: bool = False):
    """
    本地入口点 - 上传数据并启动训练或评估
    
    Args:
        skip_upload: 是否跳过所有文件上传步骤
        upload_large_files_only: 是否只上传大文件（mmap分片），跳过小文件和训练
        eval_only: 是否只进行评估（测试原始模型的recall效果）
    
    注意：大文件(mmap分片)应该通过Modal CLI上传：
    modal volume put qwen2-audio-training-data mmap_shards/ mmap_shards/
    """
    import json
    from pathlib import Path
    
    if not skip_upload:
        # 定义本地数据文件和依赖文件路径
        local_data_files = {
            # 训练数据
            "balanced_train_set.json": "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_train_set.json",
            "balanced_test_set.json": "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_test_set.json",
            # 词汇表
            "glossary_merged.json": "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_merged.json",
        }
        
        # 定义依赖的Python文件
        local_code_files = {
            "Qwen2_Audio_train.py": "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/Qwen2_Audio_train.py",
            "train_ddp_simplified.py": "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/Qwen2_Audio_term_level_train_ddp_simplified.py",
            "mmap_audio_reader.py": "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/mmap_audio_reader.py",
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
        
        # 加载依赖的Python文件
        print("[INFO] Loading Python dependency files...")
        for target_path, local_path in local_code_files.items():
            if os.path.exists(local_path):
                print(f"[INFO] Loading {local_path} -> {target_path}")
                with open(local_path, 'r', encoding='utf-8') as f:
                    data_to_upload[target_path] = f.read()
            else:
                print(f"[WARN] Local file not found: {local_path}")
        
        # 注意：mmap分片文件现在通过upload_large_files_to_volume单独处理
        # 不再将大文件添加到data_to_upload中
    
    if not skip_upload:
        # 检查是否只想上传大文件（现在通过CLI完成）
        if upload_large_files_only:
            print("[INFO] Large files should be uploaded via Modal CLI:")
            print("[INFO] modal volume put qwen2-audio-training-data mmap_shards/ mmap_shards/")
            return
        
        # 然后上传小文件（代码和配置）
        if data_to_upload:
            # 先检查哪些文件已经存在
            print("[INFO] Checking existing small files in Modal...")
            existing_files = check_existing_files.remote(list(data_to_upload.keys()))
            
            existing_count = sum(1 for exists in existing_files.values() if exists)
            print(f"[INFO] Found {existing_count}/{len(data_to_upload)} small files already exist in Modal")
            
            # 上传小文件到Modal（支持增量上传）
            print("[INFO] Uploading small files to Modal...")
            upload_data.remote(data_to_upload, force_upload=False)
        else:
            print("[INFO] No small files to upload")
    else:
        print("[INFO] Skipping file upload as requested")
    
    # 如果只是上传大文件模式，不执行训练
    if upload_large_files_only:
        return
    
    # 启动训练或评估（文件上传已同步完成）
    if eval_only:
        print("[INFO] Starting evaluation-only mode on Modal...")
        
        eval_args = {
            "model_name": "Qwen/Qwen2-Audio-7B-Instruct",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "eval_only": True,
            "eval_max_samples": 1000,  # 评估1000个样本
        }
        
        result = train_ddp_modal.remote(
            train_samples_path="balanced_train_set.json",
            test_samples_path="balanced_test_set.json",
            glossary_path="glossary_merged.json",
            mmap_shard_dir="/data/mmap_shards",  # 指向 Modal 中的 mmap 分片目录
            use_mount_directly=use_mount_directly,
            **eval_args
        )
        
        print(f"[INFO] Evaluation result: {result}")
    else:
        print("[INFO] Starting training on Modal...")
        
        training_args = {
            "epochs": 20,
            "batch_size": 256,
            "lr": 1e-4,
            "model_name": "Qwen/Qwen2-Audio-7B-Instruct",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "audio_text_loss_ratio": 0.3,
            "audio_term_loss_ratio": 0.7,
            "patience": 2,
            "gradient_accumulation_steps": 8,  # 降低累积步数，增大单次计算强度
        }
        
        result = train_ddp_modal.remote(
            train_samples_path="balanced_train_set.json",
            test_samples_path="balanced_test_set.json",
            glossary_path="glossary_merged.json",
            mmap_shard_dir="/data/mmap_shards",  # 指向 Modal 中的 mmap 分片目录
            use_mount_directly=use_mount_directly,
            **training_args
        )
        
        print(f"[INFO] Training result: {result}")


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    skip_upload = "--skip-upload" in sys.argv
    upload_large_files_only = "--upload-large-files-only" in sys.argv
    eval_only = "--eval-only" in sys.argv
    use_mount_directly = "--use-mount-directly" in sys.argv
    
    # 检查参数冲突
    if upload_large_files_only and skip_upload:
        print("[ERROR] Cannot use both --skip-upload and --upload-large-files-only")
        sys.exit(1)
    
    if eval_only and upload_large_files_only:
        print("[ERROR] Cannot use both --eval-only and --upload-large-files-only")
        sys.exit(1)
    
    main(skip_upload=skip_upload, upload_large_files_only=upload_large_files_only, eval_only=eval_only, use_mount_directly=use_mount_directly)
