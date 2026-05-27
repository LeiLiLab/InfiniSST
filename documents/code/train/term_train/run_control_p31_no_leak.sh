#!/bin/bash
#SBATCH --job-name=ctrl_noleak
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:6
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_ctrl_noleak.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_ctrl_noleak.err

# Control experiment: old p31 1M dataset with inference terms removed.
# Replicates the best known result (0.783) but without data leakage.

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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Distributed
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
CUDA_VISIBLE_GPU_LIST="0,1,2,3,4,5"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_GPU_LIST}"
NUM_GPUS=6
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29920

# WandB
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

# Paths — OLD p31 1M dataset with inference terms REMOVED
TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_with_wiki_synth_normalized_no_leak.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
SAVE_DIR="/mnt/data/jiaxuanluo"
RESUME_PATH="/mnt/data/jiaxuanluo/q3rag_scale_lora-r32-tr128_bs4k_t=0.03_p31_1M_no_leak_ctrl_best.pt"

# Audio model (LoRA) — same as the 0.783 run
USE_LORA="true"
LORA_RANK=32
LORA_ALPHA=64
TARGET_DIM=1024
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"

# Text encoder: LoRA r128, frozen text embeddings
TEXT_FULL_FINETUNE="false"
TEXT_LR="0"
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
TEXT_TARGET_MODULES="query key value dense"

# Training — same hyperparams as the 0.783 baseline
PER_GPU_BATCH=682
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
EPOCHS=5
NUM_WORKERS=8
LR="1e-4"
TEMPERATURE="0.03"
LEARN_TEMP="false"
TRAIN_LIMIT=0
FORCE_DUMMY_AUDIO="false"
AUGMENT_SYNTH="false"

# Glossary negatives: disabled (same as baseline)
GLOSSARY_NEG_PATH=""
GLOSSARY_NEG_REFRESH_STEPS=0

# Legacy neg bank / hard neg (disabled)
HARD_NEG_K=0
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=0

# Eval & checkpointing
SAVE_STEPS=300
EVAL_STEPS_SAMPLE=100
KEEP_CHECKPOINTS=5
EVAL_TOPK=10

# Multi-domain / glossary-scale eval
ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
EVAL_GLOSSARY_SIZES="1000 10000"
BEST_METRIC="eval_acl6060/recall@10_gs1000"
BEST_METRIC_SECONDARY="eval_acl6060/recall@10_gs10000"
# ======Configuration=====

mkdir -p "${SAVE_DIR}"

BS_ABBR=$((BATCH_SIZE / 1024))k
if [ $((BATCH_SIZE % 1024)) -ne 0 ]; then
    BS_ABBR="${BATCH_SIZE}"
fi

if [ "${TEXT_FULL_FINETUNE}" = "true" ]; then
    TEXT_TAG="tff"
else
    TEXT_TAG="tr${TEXT_LORA_RANK}"
fi
MODE_NAME="scale_lora-r${LORA_RANK}-${TEXT_TAG}"
if [ "${LEARN_TEMP}" = "true" ]; then
    MODE_NAME="${MODE_NAME}_lt"
fi
VERSION="p31_1M_no_leak_ctrl"
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_t=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="${SAVE_NAME}"

echo "[TRAIN] Control experiment: p31 1M no data leakage"
echo "[TRAIN] Data: ${TRAIN_JSONL}"
echo "[TRAIN] Save: ${SAVE_PATH}"
echo "[TRAIN] Batch: ${BATCH_SIZE} (${NUM_GPUS} GPUs * ${PER_GPU_BATCH})"

OPTS=""
if [ "${USE_LORA}" = "true" ]; then
    OPTS="${OPTS} --use_lora"
fi
if [ "${TEXT_FULL_FINETUNE}" = "true" ]; then
    OPTS="${OPTS} --text_full_finetune"
fi
if [ "${LEARN_TEMP}" = "true" ]; then
    OPTS="${OPTS} --learn_temp"
fi
if [ "${FORCE_DUMMY_AUDIO}" = "true" ]; then
    OPTS="${OPTS} --force_dummy_audio"
fi
if [ "${AUGMENT_SYNTH}" = "true" ]; then
    OPTS="${OPTS} --augment_synth"
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
    --text_lr "${TEXT_LR}" \
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
    --glossary_neg_path "${GLOSSARY_NEG_PATH}" \
    --glossary_neg_refresh_steps "${GLOSSARY_NEG_REFRESH_STEPS}" \
    --neg_bank_size "${NEG_BANK_SIZE}" \
    --neg_bank_refresh_steps "${NEG_BANK_REFRESH_STEPS}" \
    --hard_neg_k "${HARD_NEG_K}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_steps_sample "${EVAL_STEPS_SAMPLE}" \
    --eval_topk "${EVAL_TOPK}" \
    --keep_checkpoints "${KEEP_CHECKPOINTS}" \
    --acl_dev_jsonl "${ACL_DEV_JSONL}" \
    --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
    --eval_glossary_sizes ${EVAL_GLOSSARY_SIZES} \
    --best_metric "${BEST_METRIC}" \
    --best_metric_secondary "${BEST_METRIC_SECONDARY}" \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${WANDB_EXP_NAME}" \
    ${OPTS}

echo "[TRAIN] Completed at $(date)"
