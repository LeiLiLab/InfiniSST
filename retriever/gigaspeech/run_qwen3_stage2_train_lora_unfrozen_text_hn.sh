#!/bin/bash
#SBATCH --job-name=qwen3_unfrozen_text_hn
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_qwen3_unfrozen_text_hn.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_qwen3_unfrozen_text_hn.err

set -euo pipefail

# ==================== 环境配置 ====================
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# 解决 NCCL 超时问题 (设置为 2 小时)
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO # 开启分布式调试日志

# WandB 配置
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online

# ==================== 路径与参数 ====================
TRAIN_JSONL="/mnt/data2/jiaxuanluo/local_train_dataset.jsonl"
DEV_JSONL="/mnt/data2/jiaxuanluo/local_dev_dataset.jsonl"
# 新的 HN 文件路径 (由 LLM 生成)
HN_PATH="/mnt/gemini/data2/jiaxuanluo/llm_hard_negatives_v1.json"

# ACL 评估路径 (从 run_eval_rag_offline_aries_v4.sh 获取)
GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060.json"
WAV_DIR="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/segmented_wavs/gold"
TXT_PATH="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt"

SCRIPT_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/qwen3_AuT_BGE_M3_train_lora_unfrozen_text_hn.py"

# --- 权重加载 (从之前的 Stage 2 最佳模型开始) ---
# 请确认此路径是否为您想要继续训练的 Best PT
CHECKPOINT="/mnt/gemini/data2/jiaxuanluo/q3rag_unfrozen_lora-r32-tr16_bs4k_w1.0-0.0_sampled_best_snapshot_v2.pt"

# --- LoRA 配置 (与加载的 Checkpoint 必须严格一致) ---
USE_LORA="true"
LORA_RANK=32
LORA_ALPHA=64
TEXT_LORA_RANK=16
TEXT_LORA_ALPHA=32

TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
TEXT_TARGET_MODULES="query key value"

# --- Batch Size 调整 ---
# 注意：由于使用了 Hard Negatives，显存占用会增加。
# 如果显存不足，请调小 PER_GPU_BATCH。
NUM_GPUS=8
PER_GPU_BATCH=128 
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH)) 

LR=1e-5 # 加载预训练权重后，建议调小学习率进行微调
# 自动生成保存路径
SAVE_NAME="q3rag_unfrozen_hn_lora-r${LORA_RANK}-tr${TEXT_LORA_RANK}_bs${BATCH_SIZE}"
SAVE_PATH="/mnt/gemini/data2/jiaxuanluo/${SAVE_NAME}.pt"

echo "[INFO] Starting Qwen3 Stage 2 HN Training (Unfrozen Text Encoder)..."
echo "[INFO] Loading Checkpoint: ${CHECKPOINT}"
echo "[INFO] HN Path: ${HN_PATH}"
echo "[INFO] Save Path: ${SAVE_PATH}"
echo "[INFO] Audio LoRA Rank: ${LORA_RANK}, Text LoRA Rank: ${TEXT_LORA_RANK}"
echo "[INFO] Batch Size: ${BATCH_SIZE} (${NUM_GPUS} GPUs * ${PER_GPU_BATCH})"

# 使用 torchrun 启动多卡 DDP
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=127.0.0.1 \
    --master_port=29909 \
    "${SCRIPT_PATH}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --dev_jsonl "${DEV_JSONL}" \
    --hn_path "${HN_PATH}" \
    --glossary_path "${GLOSSARY_PATH}" \
    --wav_dir "${WAV_DIR}" \
    --txt_path "${TXT_PATH}" \
    --save_path "${SAVE_PATH}" \
    --checkpoint "${CHECKPOINT}" \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --epochs 200 \
    --eval_steps 10 \
    --num_hard_negs 10 \
    --hn_fallback_mode "random" \
    --hn_select_mode "random" \
    --neg_cache_max_size 0 \
    --neg_cache_refresh_steps 0 \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --text_lora_rank ${TEXT_LORA_RANK} \
    --text_lora_alpha ${TEXT_LORA_ALPHA} \
    --lora_target_modules ${TARGET_MODULES} \
    --text_lora_target_modules ${TEXT_TARGET_MODULES} \
    --use_lora \
    --num_workers 8 \
    --wandb_project "qwen3_rag_hn" \
    --wandb_exp_name "stage2_${SAVE_NAME}"

echo "[INFO] HN Training completed!"

