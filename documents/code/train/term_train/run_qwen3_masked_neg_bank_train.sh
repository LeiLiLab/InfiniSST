#!/bin/bash
#SBATCH --job-name=qwen3_masked_neg
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_qwen3_masked_neg.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_qwen3_masked_neg.err

set -euo pipefail

# ======Configuration=====
# Environment
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

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
MASTER_PORT=29910

# WandB
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

# Paths
TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_dataset_final.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_final.jsonl"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_masked_neg_bank_train.py"
SAVE_DIR="/mnt/gemini/data/jiaxuanluo"
RESUME_PATH=""

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
TEMPERATURE="0.03"
LEARN_TEMP="false"
TRAIN_LIMIT=0
FORCE_DUMMY_AUDIO="false"

# Hard negative mining (set HARD_NEG_K>0 to enable, replaces random bank)
HARD_NEG_K=128
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=200
# Eval & checkpointing
SAVE_STEPS=500
EVAL_STEPS_SAMPLE=200
KEEP_CHECKPOINTS=3

# Multi-domain / glossary-scale eval
ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
EVAL_GLOSSARY_SIZES="1000 10000"
BEST_METRIC="eval_acl6060/recall@10_gs1000"
# =======================

BS_ABBR=$((BATCH_SIZE / 1024))k
if [ $((BATCH_SIZE % 1024)) -ne 0 ]; then
    BS_ABBR="${BATCH_SIZE}"
fi

MODE_NAME="masked_neg_lora-r${LORA_RANK}-tr${TEXT_LORA_RANK}"
if [ "${LEARN_TEMP}" = "true" ]; then
    MODE_NAME="${MODE_NAME}_lt"
fi
NEG_TAG=""
if [ "${HARD_NEG_K}" -gt 0 ]; then
    NEG_TAG="_hn${HARD_NEG_K}"
elif [ "${NEG_BANK_SIZE}" -gt 0 ]; then
    NEG_TAG="_nb${NEG_BANK_SIZE}"
fi
VERSION="v1"
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_ttm=${TEXT_TARGET_MODULES}_t=${TEMPERATURE}${NEG_TAG}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="stage2_${SAVE_NAME}"

echo "[INFO] Masked FN + Hard Neg Training"
echo "[INFO] Save path: ${SAVE_PATH}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Batch size: ${BATCH_SIZE} (${NUM_GPUS} GPUs * ${PER_GPU_BATCH})"
echo "[INFO] Hard neg: k=${HARD_NEG_K}  Random bank: size=${NEG_BANK_SIZE}  Refresh=${NEG_BANK_REFRESH_STEPS}"
echo "[INFO] ACL dev: ${ACL_DEV_JSONL}"
echo "[INFO] Eval wiki glossary: ${EVAL_WIKI_GLOSSARY}"
echo "[INFO] Eval glossary sizes: ${EVAL_GLOSSARY_SIZES}"
echo "[INFO] Best metric: ${BEST_METRIC}"
echo "[INFO] Resume: ${RESUME_PATH}"

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
    --save_path "${SAVE_PATH}" \
    "${RESUME_ARGS[@]}" \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --temperature "${TEMPERATURE}" \
    --target_dim "${TARGET_DIM}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --text_lora_rank "${TEXT_LORA_RANK}" \
    --text_lora_alpha "${TEXT_LORA_ALPHA}" \
    --lora_target_modules ${TARGET_MODULES} \
    --text_lora_target_modules ${TEXT_TARGET_MODULES} \
    --neg_bank_size "${NEG_BANK_SIZE}" \
    --neg_bank_refresh_steps "${NEG_BANK_REFRESH_STEPS}" \
    --hard_neg_k "${HARD_NEG_K}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_steps_sample "${EVAL_STEPS_SAMPLE}" \
    --keep_checkpoints "${KEEP_CHECKPOINTS}" \
    --acl_dev_jsonl "${ACL_DEV_JSONL}" \
    --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
    --eval_glossary_sizes ${EVAL_GLOSSARY_SIZES} \
    --best_metric "${BEST_METRIC}" \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${WANDB_EXP_NAME}" \
    ${OPTS}

echo "[INFO] Training completed"
