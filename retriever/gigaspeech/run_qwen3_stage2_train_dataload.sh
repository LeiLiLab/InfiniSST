#!/bin/bash
#SBATCH --job-name=qwen3_stage2
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_qwen3_stage2.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_qwen3_stage2.err

set -euo pipefail

# ==================== 环境配置 ====================
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/mnt/taurus/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# WandB 配置
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online

# ==================== 路径与参数 ====================
TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_dataset.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_dataset.jsonl"

# --- WebDataset 配置 ---
TRAIN_SHARDS="/mnt/taurus/data2/jiaxuanluo/gigaspeech_webdataset_fbank_v1/train/shard_{00000..01667}.tar"
DEV_SHARDS="/mnt/taurus/data2/jiaxuanluo/gigaspeech_webdataset_fbank_v1/dev/shard_{00000..00012}.tar"

PRECOMPUTED_DIR="/mnt/gemini/data1/jiaxuanluo/precomputed_text_embs"
PRECOMPUTED_DEV_DIR="/mnt/gemini/data1/jiaxuanluo/precomputed_dev_embs"
SAVE_PATH="/mnt/gemini/data1/jiaxuanluo/qwen3_omni_retriever_stage2.pt"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/qwen3_AuT_BGE_M3_train_dataload.py"

# --- Batch Size 调整区 ---
NUM_GPUS=2
PER_GPU_BATCH=1024  # 从 1024 提升到 2048 以更好地利用显存并增强对比学习效果
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH)) 
# -----------------------

LR=1e-4
USE_LORA="true" 
TEST_LIMIT=1000    # 设为 0 跑全量，或者保持 100000 
SAVE_STEPS=2000  # 每 2000 步保存一次
EVAL_STEPS_SAMPLE=200
EVAL_STEPS_FULL=2000
TERM_WEIGHT=1.0  # Audio-Term 权重
TRANS_WEIGHT=0.0 # Audio-Transcript 权重
TEXT_LORA_RANK=16

# --- 自动构建简短、唯一的保存前缀 ---
BS_ABBR=$((BATCH_SIZE / 1024))k
MODE_ABBR=$([ "$USE_LORA" = "true" ] && echo "lora" || echo "lin")
# 结果示例: q3rag_lora_bs8k_w1.0-0.0
SAVE_NAME="q3rag_${MODE_ABBR}_bs${BS_ABBR}_w${TERM_WEIGHT}-${TRANS_WEIGHT}"
SAVE_PATH="/mnt/gemini/data1/jiaxuanluo/${SAVE_NAME}.pt"

echo "[INFO] Starting Qwen3-Omni Stage 2 Training..."
echo "[INFO] Save Path Base: ${SAVE_PATH}"
echo "[INFO] Using LoRA: ${USE_LORA}"
echo "[INFO] Using precomputed embeddings from: ${PRECOMPUTED_DIR}"
echo "[INFO] Evaluation dataset: ${DEV_JSONL}"

# 构造可选标志
OPTS=""
if [ "$USE_LORA" = "true" ]; then
    OPTS="${OPTS} --use_lora"
fi
if [ "$TEST_LIMIT" -gt 0 ]; then
    OPTS="${OPTS} --test_limit ${TEST_LIMIT}"
fi

# 使用 torchrun 启动多卡 DDP
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    "${SCRIPT_PATH}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --dev_jsonl "${DEV_JSONL}" \
    --train_shards "${TRAIN_SHARDS}" \
    --dev_shards "${DEV_SHARDS}" \
    --precomputed_dir "${PRECOMPUTED_DIR}" \
    --precomputed_dev_dir "${PRECOMPUTED_DEV_DIR}" \
    --save_path "${SAVE_PATH}" \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps_sample ${EVAL_STEPS_SAMPLE} \
    --eval_steps_full ${EVAL_STEPS_FULL} \
    --term_weight ${TERM_WEIGHT} \
    --trans_weight ${TRANS_WEIGHT} \
    --text_lora_rank ${TEXT_LORA_RANK} \
    ${OPTS} \
    --epochs 20 \
    --num_workers 6 \
    --wandb_project "qwen3_rag" \
    --wandb_exp_name "stage2_linear_probe"

echo "[INFO] Stage 2 Training completed!"

