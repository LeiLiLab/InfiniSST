#!/bin/bash
#SBATCH --job-name=qwen3_unfrozen_text
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_qwen3_unfrozen_text_m.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_qwen3_unfrozen_text_m.err

set -euo pipefail

# ==================== 环境配置 ====================
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# 解决 NCCL 超时问题 (设置为 2 小时)
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO # 开启分布式调试日志
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # 异步错误处理，防止死等
export NCCL_DEBUG=WARN # 只显示警告级别的 NCCL 日志
export TORCH_NCCL_TRACE_BUFFER_SIZE=256 # 开启飞行记录器

# Cache / Logs: force to data2 to avoid data1 full disk / IO stalls
export WANDB_DIR="/mnt/gemini/data2/jiaxuanluo/wandb"
export HF_HOME="/mnt/gemini/data2/jiaxuanluo/huggingface_cache"
export XDG_CACHE_HOME="/mnt/gemini/data2/jiaxuanluo/xdg_cache"
mkdir -p "${WANDB_DIR}" "${HF_HOME}" "${XDG_CACHE_HOME}"

# Audio path remap: dataset JSONL may still point to data1, remap to data2 if exists
export AUDIO_PATH_REMAP_SRC="/mnt/gemini/data1/jiaxuanluo/"
export AUDIO_PATH_REMAP_DST="/mnt/gemini/data2/jiaxuanluo/"

# WandB 配置
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online

# ==================== Parameters ====================
# Usage:
#   sbatch run_qwen3_stage2_train_lora_unfrozen_text_m1.sh [M]
# where M is 1-4 (multiplier-merge / segment seconds: M*0.96).

M=${1:-4}
if ! [[ "${M}" =~ ^[1-4]$ ]]; then
  echo "[ERROR] Invalid M=${M}. Expected 1-4."
  exit 2
fi

# ==================== Paths ====================
#TRAIN_JSONL="/mnt/data2/jiaxuanluo/local_train_dataset.jsonl"
#DEV_JSONL="/mnt/data2/jiaxuanluo/local_dev_dataset.jsonl"
TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_dataset_m${M}.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_m${M}.jsonl"
PRECOMPUTED_DIR="/mnt/gemini/data2/jiaxuanluo/precomputed_text_embs_m${M}"
PRECOMPUTED_DEV_DIR="/mnt/gemini/data2/jiaxuanluo/precomputed_dev_embs_m${M}"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/qwen3_AuT_BGE_M3_train_lora_unfrozen_text.py"
# --- LoRA 消融实验配置区 ---
USE_LORA="true"
LORA_RANK=32
LORA_ALPHA=64
TEXT_LORA_RANK=16
TEXT_LORA_ALPHA=32

# 模块配置
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
#TEXT_TARGET_MODULES="query key value intermediate.dense output.dense"
TEXT_TARGET_MODULES="query key value"
MODULE_ABBR="all"

# -----------------------

# --- Batch Size 调整区 ---
# 注意：由于 Text Encoder 现在是在线计算并参与训练，显存消耗会显著增加。
# 如果 GPU 显存 (如 80G A100) 不够，请调小 PER_GPU_BATCH。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
PER_GPU_BATCH=512 # 相比冻结版本(1024)调小，以防 OOM
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH)) 
# -----------------------

LR=1e-4
TEST_LIMIT=0   
SAVE_STEPS=1000  
EVAL_STEPS_SAMPLE=200
EVAL_STEPS_FULL=1000
TERM_WEIGHT=1.0  
TRANS_WEIGHT=0.0
TEMPERATURE=0.03
LEARN_TEMP="false"
# Debug switch: set to "true" to bypass audio IO entirely and use dummy audio for every sample.
FORCE_DUMMY_AUDIO="false"
# --- 自动构建简短、唯一的保存前缀 ---
BS_ABBR=$((BATCH_SIZE / 1024))k
[ $((BATCH_SIZE % 1024)) -ne 0 ] && BS_ABBR="${BATCH_SIZE}"
MODE_NAME="unfrozen_lora-r${LORA_RANK}-tr${TEXT_LORA_RANK}"
[ "$LEARN_TEMP" = "true" ] && MODE_NAME="${MODE_NAME}_lt"
SAVE_NAME="m1_q3rag_${MODE_NAME}_bs${BS_ABBR}_w${TERM_WEIGHT}-${TRANS_WEIGHT}-ttm=${TEXT_TARGET_MODULES}-temperature=${TEMPERATURE}"
SAVE_NAME="m${M}_q3rag_${MODE_NAME}_bs${BS_ABBR}_w${TERM_WEIGHT}-${TRANS_WEIGHT}-ttm=${TEXT_TARGET_MODULES}-temperature=${TEMPERATURE}"
SAVE_PATH="/mnt/gemini/data2/jiaxuanluo/${SAVE_NAME}.pt"

echo "[INFO] Starting Qwen3-Omni Stage 2 Training (Unfrozen Text Encoder)..."
echo "[INFO] M: ${M}"
echo "[INFO] Save Path: ${SAVE_PATH}"
echo "[INFO] Audio LoRA Rank: ${LORA_RANK}, Text LoRA Rank: ${TEXT_LORA_RANK}"
echo "[INFO] Batch Size: ${BATCH_SIZE} (${NUM_GPUS} GPUs * ${PER_GPU_BATCH})"

# 构造可选标志
OPTS=""
if [ "$USE_LORA" = "true" ]; then
    OPTS="${OPTS} --use_lora"
fi
if [ "$TEST_LIMIT" -gt 0 ]; then
    OPTS="${OPTS} --test_limit ${TEST_LIMIT}"
fi
if [ "$LEARN_TEMP" = "true" ]; then
    OPTS="${OPTS} --learn_temp"
fi
if [ "$FORCE_DUMMY_AUDIO" = "true" ]; then
    OPTS="${OPTS} --force_dummy_audio"
fi

# 使用 torchrun 启动多卡 DDP
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=127.0.0.1 \
    --master_port=29905 \
    "${SCRIPT_PATH}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --dev_jsonl "${DEV_JSONL}" \
    --precomputed_dir "${PRECOMPUTED_DIR}" \
    --precomputed_dev_dir "${PRECOMPUTED_DEV_DIR}" \
    --save_path "${SAVE_PATH}" \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps_sample ${EVAL_STEPS_SAMPLE} \
    --eval_steps_full ${EVAL_STEPS_FULL} \
    --term_weight ${TERM_WEIGHT} \
    --temperature ${TEMPERATURE} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --text_lora_rank ${TEXT_LORA_RANK} \
    --text_lora_alpha ${TEXT_LORA_ALPHA} \
    --lora_target_modules ${TARGET_MODULES} \
    --text_lora_target_modules ${TEXT_TARGET_MODULES} \
    ${OPTS} \
    --epochs 20 \
    --num_workers 8 \
    --wandb_project "qwen3_rag" \
    --wandb_exp_name "stage2_${SAVE_NAME}"

echo "[INFO] Stage 2 Training completed!"

