#!/bin/bash
#SBATCH --job-name=term_rag_train
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_term_rag_train.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_term_rag_train.err

set -euo pipefail

# ==================== Environment Setup ====================
# 彻底放弃 source conda.sh，改用手动环境变量注入，避免硬编码路径报错
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# 确保 Python 路径包含项目根目录，并清理可能 shadowing wandb 的本地路径
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# 防止当前目录下的 wandb/ 目录 shadowing 库文件
if [ -d "wandb" ] && [ ! -f "wandb/__init__.py" ]; then
    echo "[INFO] Found a non-package wandb directory, renaming to avoid shadowing..."
    mv wandb wandb_logs_$(date +%Y%m%d_%H%M%S)
fi

# WandB Setup
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online

# Cache directories
export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
export VLLM_CACHE="/mnt/gemini/data1/jiaxuanluo/vllm_cache"
export XDG_CACHE_HOME="/mnt/gemini/data1/jiaxuanluo/xdg_cache"
mkdir -p "$HF_HOME" "$VLLM_CACHE" "$XDG_CACHE_HOME"

# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 验证 Python 环境
which python
python --version
python -c "import wandb; print('wandb version:', wandb.__version__)"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech

# ==================== Configuration ====================
TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_dataset.jsonl"
SAVE_PATH="/mnt/gemini/data1/jiaxuanluo/term_rag_model.pt"
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=896         # 挑战极限物理 Batch Size。8卡下每卡 128 个。
LR=1e-4
GRAD_ACCUM=2            # 既然物理 Batch 已经 1024，就不再需要累积。
NUM_WORKERS=8

# LoRA settings
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.0

# Loss weights
AUDIO_TERM_RATIO=0.7
AUDIO_TEXT_RATIO=0.3

# Evaluation
EVAL_EVERY=1
EVAL_SAMPLES=1000
PATIENCE=3
SAVE_EVERY=500          # 每 500 step 保存一次

# TEST_LIMIT=2000
# WandB
WANDB_PROJECT="gigaspeech_zh"
WANDB_EXP_NAME="rag"

# ==================== GPU Detection & Setup ====================
# 参考 pipeline 逻辑：手动指定物理卡，跳过 0 和 1
PHYSICAL_GPUS=8
SKIP_GPUS=""

# 强制不使用 Slurm 分配的虚拟卡，直接控制物理卡
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID

GPU_LIST=()
for ((i=0; i<PHYSICAL_GPUS; i++)); do
    is_skip=false
    IFS=',' read -r -a SKIP_ARRAY <<< "${SKIP_GPUS}"
    for skip in "${SKIP_ARRAY[@]}"; do
        if [ "$i" == "$skip" ]; then is_skip=true; break; fi
    done
    if [ "$is_skip" = false ]; then
        GPU_LIST+=("$i")
    fi
done

# 将过滤后的物理卡 ID 重新组合成 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_LIST[*]}")
NUM_GPUS=${#GPU_LIST[@]}

if [ "${NUM_GPUS}" -eq 0 ]; then
    echo "[FATAL] No usable GPUs found after skipping ${SKIP_GPUS}."
    exit 1
fi

echo "[INFO] Using ${NUM_GPUS} GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Training data: ${TRAIN_JSONL}"
echo "[INFO] Save path: ${SAVE_PATH}"

# ==================== Training ====================
if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "[INFO] Starting multi-GPU DDP training with torchrun..."
    
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_addr=127.0.0.1 \
        --master_port=29600 \
        train_term_rag_local.py \
        --train_jsonl "${TRAIN_JSONL}" \
        --save_path "${SAVE_PATH}" \
        --model_name "${MODEL_NAME}" \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --num_workers ${NUM_WORKERS} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --audio_term_ratio ${AUDIO_TERM_RATIO} \
        --audio_text_ratio ${AUDIO_TEXT_RATIO} \
        --eval_every ${EVAL_EVERY} \
        --eval_samples ${EVAL_SAMPLES} \
        --patience ${PATIENCE} \
        --save_every_steps ${SAVE_EVERY} \
        --resume \
        --test_limit "${TEST_LIMIT:-0}" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_exp_name "${WANDB_EXP_NAME}"
else
    echo "[INFO] Starting single-GPU training..."
    
    python train_term_rag_local.py \
        --train_jsonl "${TRAIN_JSONL}" \
        --save_path "${SAVE_PATH}" \
        --model_name "${MODEL_NAME}" \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --num_workers ${NUM_WORKERS} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --audio_term_ratio ${AUDIO_TERM_RATIO} \
        --audio_text_ratio ${AUDIO_TEXT_RATIO} \
        --eval_every ${EVAL_EVERY} \
        --eval_samples ${EVAL_SAMPLES} \
        --patience ${PATIENCE} \
        --save_every_steps ${SAVE_EVERY} \
        --resume \
        --test_limit "${TEST_LIMIT:-0}" \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_exp_name "${WANDB_EXP_NAME}"
fi

echo "[INFO] Training completed!"



# --test_limit ${TEST_LIMIT} \