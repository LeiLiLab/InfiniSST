#!/bin/bash
#SBATCH --job-name=qwen3_tts_term_train
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_qwen3_tts_term_train.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_qwen3_tts_term_train.err

set -euo pipefail

# ======Configuration=====
# Environment
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# Use node-local tmp to avoid NFS ".nfs*" cleanup warnings at process exit.
LOCAL_TMP_DIR="/dev/shm/${USER}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"

# Distributed
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
CUDA_VISIBLE_GPU_LIST="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_GPU_LIST}"
NUM_GPUS=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29920

# WandB
export WANDB_API_KEY=8bffc48f5b611b7e51ef591ae7e9f139a3501828
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

# Paths
TRAIN_JSONL="/mnt/gemini/data/siqiouyang/term_train_dataset_final_with_tts.jsonl"
DEV_JSONL="/mnt/gemini/data/siqiouyang/term_dev_dataset_final_with_tts.jsonl"
TTS_ROOT_DIR="/mnt/gemini/data/siqiouyang/term_train_tts"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/tts_term_train.py"
SAVE_DIR="/mnt/gemini/data/jiaxuanluo"

# Model
USE_LORA="true"
LORA_RANK=32
LORA_ALPHA=64
TEXT_LORA_RANK=16
TEXT_LORA_ALPHA=32
TARGET_DIM=1024
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
TEXT_TARGET_MODULES="query key value"

# Training
PER_GPU_BATCH=512
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
EPOCHS=20
NUM_WORKERS=8
LR="1e-4"
SAVE_STEPS=500
EVAL_STEPS_SAMPLE=200
KEEP_CHECKPOINTS=3
TRAIN_LIMIT=0
RESUME_PATH=""
TEMPERATURE="0.03"
LEARN_TEMP="false"
TTS_LOSS_WEIGHT="1.0"
FORCE_DUMMY_AUDIO="false"
# =======================

BS_ABBR=$((BATCH_SIZE / 1024))k
if [ $((BATCH_SIZE % 1024)) -ne 0 ]; then
    BS_ABBR="${BATCH_SIZE}"
fi

MODE_NAME="tts_lora-r${LORA_RANK}-tr${TEXT_LORA_RANK}"
if [ "${LEARN_TEMP}" = "true" ]; then
    MODE_NAME="${MODE_NAME}_lt"
fi
VERSION="v2"
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_ttsw${TTS_LOSS_WEIGHT}_ttm=${TEXT_TARGET_MODULES}_temperature=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="stage2_${SAVE_NAME}"

echo "[INFO] Starting TTS term training on aries"
echo "[INFO] Save path: ${SAVE_PATH}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Batch size: ${BATCH_SIZE} (${NUM_GPUS} GPUs * ${PER_GPU_BATCH})"
echo "[INFO] Train JSONL: ${TRAIN_JSONL}"
echo "[INFO] Dev JSONL: ${DEV_JSONL}"
echo "[INFO] Resume path: ${RESUME_PATH}"
echo "[INFO] TTS loss weight: ${TTS_LOSS_WEIGHT}"

OPTS=""
if [ "${USE_LORA}" = "true" ]; then
    OPTS="${OPTS} --use_lora"
fi
if [ "${LEARN_TEMP}" = "true" ]; then
    OPTS="${OPTS} --learn_temp"
fi
if [ "${FORCE_DUMMY_AUDIO}" = "true" ]; then
    OPTS="${OPTS} --force_dummy_audio"
fi
if [ "${TRAIN_LIMIT}" -gt 0 ]; then
    OPTS="${OPTS} --train_limit ${TRAIN_LIMIT}"
fi
RESUME_ARGS=()
if [ -n "${RESUME_PATH}" ] && [ "${RESUME_PATH}" != "None" ] && [ "${RESUME_PATH}" != "none" ] && [ "${RESUME_PATH}" != "null" ]; then
    RESUME_ARGS=(--resume "${RESUME_PATH}")
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_PATH}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --dev_jsonl "${DEV_JSONL}" \
    --tts_root_dir "${TTS_ROOT_DIR}" \
    --save_path "${SAVE_PATH}" \
    "${RESUME_ARGS[@]}" \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_steps_sample "${EVAL_STEPS_SAMPLE}" \
    --keep_checkpoints "${KEEP_CHECKPOINTS}" \
    --temperature "${TEMPERATURE}" \
    --tts_loss_weight "${TTS_LOSS_WEIGHT}" \
    --target_dim "${TARGET_DIM}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --text_lora_rank "${TEXT_LORA_RANK}" \
    --text_lora_alpha "${TEXT_LORA_ALPHA}" \
    --lora_target_modules ${TARGET_MODULES} \
    --text_lora_target_modules ${TEXT_TARGET_MODULES} \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${WANDB_EXP_NAME}" \
    ${OPTS}

echo "[INFO] TTS term training on aries completed"

