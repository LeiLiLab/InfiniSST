"""
Modal部署脚本 - InfiniSST Stage1 GigaSpeech RAG训练
使用Modal云平台进行分布式训练，所有数据通过Volume挂载
"""
import modal
import os
from pathlib import Path

# 创建Modal App
app = modal.App("infinisst-stage1-gigaspeech-rag")

# === 关键修改点 ===
# 1) Torch 切到 cu124 对应的官方索引 (与 H100 + 你当前环境一致)
# 2) 预先固定 omegaconf/hydra/PyYAML，避免 fairseq 触发坏元数据版本
# 3) fairseq==0.12.2 用 --no-deps 安装，阻止它回拉 omegaconf 2.0.5
# 4) 其余依赖按你当前环境版本对齐
# 5) 默认 n_device=2，和 gpu="H100:2" 保持一致
# 6) PIP 重试/超时更健壮
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install([
        "git", "wget", "curl", "ffmpeg", "libsndfile1",
        "build-essential", "rsync", "ninja-build"
    ])
    # 安装 PyTorch (CUDA 12.4)
    .run_commands(["pip install 'pip<24.1'"])
    .pip_install(
        ["torch==2.5.1+cu124", "torchvision==0.20.1+cu124", "torchaudio==2.5.1+cu124"],
        extra_options="--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        ["huggingface_hub>=0.24.0"],
        extra_options="--retries 8 --timeout 120 --no-cache-dir"
    )
    # 预钉关键依赖，避免 fairseq 拉坏的 omegaconf
    .pip_install([
        "PyYAML>=5.1,<7.0",
        "omegaconf==2.0.6",
        "hydra-core==1.0.7",
        "setuptools<70",
        "wheel",
        # fairseq 依赖
        "bitarray",
        "cython",
        "regex",
        "sacrebleu",
        "tensorboardX",
        "cffi",
    ])
    # 仅安装 fairseq 本体，阻止自动解依赖
    .pip_install([
        "fairseq==0.12.2"
    ], extra_options="--no-deps")
    # 其余训练依赖（使用普通 Adam 优化器，无需 DeepSpeed）
    .pip_install([
        "transformers==4.47.0",
        "accelerate==1.7.0",
        "lightning==2.5.1.post0",
        "pytorch-lightning==2.5.1.post0",
        # DeepSpeed removed - using standard Adam optimizer instead
        "peft==0.15.2",
        "soundfile==0.13.1",
        "librosa==0.11.0",
        "numpy==1.24.4",
        "scipy==1.15.2",
        "scikit-learn==1.3.2",
        "tqdm==4.64.1",
        "tensorboard",
        "huggingface_hub",
        "hf-transfer==0.1.9",
        "wandb==0.19.10",
        "sentencepiece==0.2.0",
        "protobuf==6.31.1",
        "einops==0.8.1",
        "ray==2.48.0",
        "jieba==0.42.1",            # 中文分词库
        "torcheval==0.0.7",
        "torchdata==0.11.0",
    ])
    # Flash Attention（版本与 torch/cu124 匹配；失败不阻断构建）
    .run_commands([
        "pip install --no-build-isolation --no-cache-dir 'flash-attn==2.7.4.post1' || true"
    ])
    # FlashInfer（从预编译 wheel 安装，失败不阻断构建）
    .run_commands([
        "pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.5/ || true"
    ])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # 提升 pip 稳定性
        "PIP_DEFAULT_TIMEOUT": "120",
        "PIP_RETRIES": "8",
    })
)
# 定义存储卷用于数据和模型
data_volume = modal.Volume.from_name("infinisst-data", create_if_missing=True)
model_volume = modal.Volume.from_name("infinisst-models", create_if_missing=True)
output_volume = modal.Volume.from_name("infinisst-outputs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


code_volume = modal.Volume.from_name("infinisst-code", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/root/InfiniSST": code_volume},
    timeout=600,
)
def prepare_code(force_refresh=True):
    import os, subprocess, shutil, glob
    code_dir = "/root/InfiniSST"
    repo = "https://github.com/LeiLiLab/InfiniSST.git"
    branch = "jiaxuan/NE1.0"

    # 如果需要强制刷新，删除目录内的所有内容（而不是删除目录本身）
    if force_refresh and os.path.exists(code_dir) and os.listdir(code_dir):
        print(f"[INFO] Force refresh enabled, removing existing code in {code_dir}")
        # 删除目录内的所有文件和文件夹
        for item in glob.glob(f"{code_dir}/*") + glob.glob(f"{code_dir}/.*"):
            if os.path.basename(item) not in ['.', '..']:  # 跳过 . 和 ..
                try:
                    if os.path.islink(item) or os.path.isfile(item):
                        os.unlink(item)
                    elif os.path.isdir(item):
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"[WARN] Failed to remove {item}: {e}")
        print("[OK] Old code removed")

    if not os.listdir(code_dir):   # 空的才拉取
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, repo, code_dir],
            check=True
        )
        print("[OK] Code cloned")
    else:
        print("[INFO] Code already exists in volume, pulling latest changes")
        # 如果代码已存在，执行 git pull 更新
        try:
            subprocess.run(["git", "-C", code_dir, "pull", "origin", branch], check=True)
            print("[OK] Code updated")
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Failed to pull updates: {e}, using existing code")

    # 保存写入
    from modal import Volume
    code_volume.commit()

from huggingface_hub import snapshot_download
import os

@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],  # 私模需token
    timeout=60*60,
)
def warmup_hf_cache(model_id="Qwen/Qwen2.5-7B-Instruct", revision="main"):
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface")
    token = os.environ.get("HF_TOKEN")  # 从 secret 注入
    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir="/root/.cache/huggingface",
        token=token,
        local_files_only=False,         # 预热阶段允许联网
    )
    print("[OK] cached at:", path)
    return path



@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/models": model_volume,
        "/output": output_volume,
    },
    timeout=600,
)
def check_volume_structure():
    import os
    print("[INFO] Checking volume structure...")
    data_paths = [
        "/data/gigaspeech",
        "/models/Qwen2.5-7B-Instruct",
        "/models/wav2_vec_vox_960h_pl.pt",
    ]
    for path in data_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                files = os.listdir(path)[:10]
                print(f"[OK] {path} exists with {len(os.listdir(path))} items")
                print(f"     Sample items: {files}")
            else:
                size_mb = os.path.getsize(path) / (1024**2)
                print(f"[OK] {path} exists ({size_mb:.2f} MB)")
        else:
            print(f"[MISSING] {path}")
    return "Volume check completed"

code_volume = modal.Volume.from_name("infinisst-code", create_if_missing=True)


# 环境变量（全局，用于脚本执行）
os.environ["PYTHONPATH"] = "/root/InfiniSST"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

@app.function(
    image=image,
    gpu="H100:8",
    volumes={
        "/data": data_volume,
        "/models": model_volume,
        "/output": output_volume,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/InfiniSST": code_volume,   # ★★★ 新增：挂载代码卷
    },
    timeout=86400,
    memory=512*1024,
    cpu=64,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def train_infinisst(
    # Model paths
    llama_path: str = "Qwen/Qwen2.5-7B-Instruct",
    w2v2_path: str = "/models/wav2_vec_vox_960h_pl.pt",
    w2v2_type: str = "w2v2",
    ctc_finetuned: bool = True,

    # Data paths
    data_path: str = "/data/gigaspeech",
    data_split_train: str = "train_xl_case_ft-qwen2.5-32b-instruct_marked_mfa_punc_asr",
    data_split_eval: str = "dev_case_ft-qwen2.5-32b-instruct_marked_mfa_punc",

    # Languages
    source_lang: str = "English",
    target_lang: str = "Chinese",
    lang_code: str = "zh",

    # Model architecture
    model_type: str = "w2v2_qwen25",
    length_shrink_cfg: str = "[(1024,2,2)] * 2",
    block_size: int = 48,
    max_cache_size: int = 576,

    # LLM settings
    llm_freeze: bool = True,
    llm_emb_freeze: bool = True,
    llm_head_freeze: bool = True,
    use_flash_attn: bool = True,

    # Training settings
    trajectory: int = 9,
    trajectory_max_multiplier: int = 12,
    trajectory_prob_aug: float = 0.0,
    audio_normalize: bool = False,

    # Training hyperparameters
    seed: int = 998244353,
    stage: int = 1,
    train_bsz: int = 7200,            # H100x8: 4x from 1800 (L40S baseline)
    eval_bsz: int = 7200,             # H100x8: 4x from 1800 (L40S baseline)
    bsz_sent: int = 4,                # H100x8: doubled for better batching
    learning_rate: float = 2e-4,
    warmup_steps: int = 1000,

    # Training control
    run_name: str = "stage1_M=12_ls-cv-vp_norm0_qwen_rope_modal",
    n_device: int = 8,                # ★ 与 gpu="H100:8" 对齐
    max_epochs: int = 1,
    grad_acc_steps: int = 1,          # H100x8: minimal grad_acc for 8 GPUs
    clip_norm: float = 1.0,
    save_step: int = 2000,
    log_step: int = 100,
    eval_step: int = 1000,

    # Output settings
    output_dir: str = "/output/en-zh",

    # Options
    use_local_copy: bool = True,
    resume_training: bool = False,
):
    import subprocess, os, sys, time, torch, shutil

    print(f"[INFO] Starting InfiniSST Stage 1 Training on Modal")
    print(f"[INFO] Available GPUs: {torch.cuda.device_count()}")
    print(f"[INFO] GPU Names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    # 环境变量
    os.environ["PYTHONPATH"] = "/root/InfiniSST"
    os.environ["HF_HOME"] = "/root/.cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 训练期强制离线
    
    # H100 上 NCCL P2P/IB 性能良好，不需要禁用
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG", "INFO")

    # 本地化数据（NVMe）
    if use_local_copy:
        print("[INFO] Localizing data from Volume to local NVMe...")
        local_root = "/workspace"
        os.makedirs(local_root, exist_ok=True)
        start_time = time.time()
        local_data_path = f"{local_root}/gigaspeech"
        if os.path.exists(data_path) and not os.path.exists(local_data_path):
            print(f"[INFO] Copying dataset from {data_path} to {local_data_path}")
            try:
                subprocess.run(
                    ["rsync", "-a", "--info=progress2", f"{data_path}/", f"{local_data_path}/"],
                    check=True
                )
                data_path = local_data_path
                print(f"[OK] Dataset copied successfully")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Failed to copy dataset: {e}, using original path")
        copy_time = time.time() - start_time
        print(f"[INFO] Data localization completed in {copy_time:.1f}s")

    save_path = f"{output_dir}/runs/{run_name}"
    if not resume_training and os.path.exists(save_path):
        print(f"[INFO] Cleaning existing output directory: {save_path}")
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output will be saved to: {save_path}")

    train_script = "/root/InfiniSST/train/main.py"
    if not os.path.exists(train_script):
        print(f"[ERROR] Training script not found: {train_script}")
        raise FileNotFoundError(f"Training script not found: {train_script}")

    from huggingface_hub import snapshot_download
    def _resolve_hf_model_path(maybe_id_or_path: str) -> str:
        if os.path.isdir(maybe_id_or_path):
            return maybe_id_or_path
        return snapshot_download(
            repo_id=maybe_id_or_path,
            cache_dir="/root/.cache/huggingface",
            local_files_only=True,
        )

    llama_path = _resolve_hf_model_path(llama_path)
    print(f"[INFO] Resolved LLM path -> {llama_path}")
    # 训练命令
    cmd = [
        "python", train_script,
        "--model_type", model_type,
        "--w2v2_path", w2v2_path,
        "--w2v2_type", w2v2_type,
        "--ctc_finetuned", str(ctc_finetuned),
        "--length_shrink_cfg", length_shrink_cfg,
        "--block_size", str(block_size),
        "--max_cache_size", str(max_cache_size),

        "--llm_path", llama_path,
        "--llm_freeze", str(llm_freeze),
        "--llm_emb_freeze", str(llm_emb_freeze),
        "--llm_head_freeze", str(llm_head_freeze),
        "--use_flash_attn", str(use_flash_attn),

        "--data_path", data_path,
        "--data_split_train", data_split_train,
        "--data_split_eval", data_split_eval,
        "--source_lang", source_lang,
        "--target_lang", target_lang,
        "--trajectory", str(trajectory),
        "--trajectory_max_multiplier", str(trajectory_max_multiplier),
        "--trajectory_prob_aug", str(trajectory_prob_aug),
        "--audio_normalize", str(audio_normalize),

        "--seed", str(seed),
        "--stage", str(stage),
        "--train_bsz", str(train_bsz),
        "--eval_bsz", str(eval_bsz),
        "--bsz_sent", str(bsz_sent),
        "--learning_rate", str(learning_rate),
        "--warmup_steps", str(warmup_steps),
        "--run_name", run_name,

        "--n_device", str(n_device),
        "--max_epochs", str(max_epochs),
        "--grad_acc_steps", str(grad_acc_steps),
        "--clip_norm", str(clip_norm),
        "--save_dir", save_path,
        "--save_step", str(save_step),
        "--log_step", str(log_step),
        "--eval_step", str(eval_step),
    ]

    print(f"[INFO] Executing training command:")
    print(f"[CMD] {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True
        )
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            if line:
                print(line)
        process.wait()
        if process.returncode != 0:
            print(f"[ERROR] Training failed with return code {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, cmd)
        print("[SUCCESS] Training completed successfully!")
    except Exception as e:
        print(f"[ERROR] Training execution failed: {e}")
        raise

    # DeepSpeed ckpt 转换（可选，按你的脚本结构保留）
    print("[INFO] Converting DeepSpeed checkpoint to standard format...")
    try:
        zero_to_fp32_script = "/root/InfiniSST/train/zero_to_fp32.py"
        if os.path.exists(zero_to_fp32_script):
            last_ckpt = f"{save_path}/last.ckpt"
            output_bin = f"{save_path}/last.ckpt/pytorch_model.bin"
            subprocess.run(["python", zero_to_fp32_script, last_ckpt, output_bin], check=True)
            print(f"[OK] Checkpoint converted to: {output_bin}")
            prune_script = "/root/InfiniSST/train/prune_bin.py"
            if os.path.exists(prune_script):
                subprocess.run(["python", prune_script, output_bin], check=True)
                print(f"[OK] Model pruned")
        else:
            print(f"[WARN] Conversion script not found: {zero_to_fp32_script}")
    except Exception as e:
        print(f"[WARN] Failed to convert checkpoint: {e}")

    # 提交输出到Volume
    output_volume.commit()
    print("[INFO] Output files committed to volume")
    return {"status": "success", "save_path": save_path, "message": "Training completed successfully"}

@app.function(
    image=image,
    volumes={"/data": data_volume, "/models": model_volume},
    timeout=3600,
)
def upload_codebase(local_codebase_path: str = "/home/jiaxuanluo/InfiniSST"):
    import os
    print(f"[INFO] Uploading codebase from {local_codebase_path}")
    container_path = "/root/InfiniSST"
    os.makedirs(container_path, exist_ok=True)
    print(f"[INFO] Codebase structure prepared at {container_path}")
    return "Codebase upload completed"

# 本地入口点
@app.local_entrypoint()
def main(
    check_volume: bool = False,
    skip_training: bool = False,
    resume_training: bool = False,
    use_local_copy: bool = True,
):
    if check_volume:
        print("[INFO] Checking volume structure...")
        result = check_volume_structure.remote()
        print(f"[INFO] {result}")
        return
    if skip_training:
        print("[INFO] Skipping training as requested")
        return
    
    # 刷新代码到最新版本 only updated when needed
    print("[INFO] Refreshing codebase from GitHub...")
    prepare_code.remote(force_refresh=True)
    print("[INFO] Codebase refreshed successfully")
    
    print("[INFO] Starting InfiniSST Stage 1 Training on Modal...")
    result = train_infinisst.remote(
        run_name="stage1_M=12_ls-cv-vp_norm0_qwen_rope_modal",
        n_device=8,                    # ★ 与上方 gpu=H100:8 一致
        use_local_copy=use_local_copy,
        resume_training=resume_training,
    )
    print(f"[INFO] Training completed!")
    print(f"[INFO] Result: {result}")
    print(f"[INFO] Model saved to: {result['save_path']}")

if __name__ == "__main__":
    import sys
    check_volume = "--check-volume" in sys.argv
    skip_training = "--skip-training" in sys.argv
    resume_training = "--resume" in sys.argv
    use_local_copy = "--no-local-copy" not in sys.argv
    main(
        check_volume=check_volume,
        skip_training=skip_training,
        resume_training=resume_training,
        use_local_copy=use_local_copy,
    )