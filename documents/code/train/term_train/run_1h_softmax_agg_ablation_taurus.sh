#!/bin/bash
#SBATCH --job-name=q3_smax
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --gres=gpu:6
#SBATCH --time=0-01:30:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_smax.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_smax.err

# 1-hour ablation: MaxSim with softmax aggregation (vs hard_max baseline)
# Softmax tau anneals from 1.0 -> 0.05 over training.
# Match config: 6 GPUs, batch=9216, same as MaxSim-only 1h run.

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

LOCAL_TMP_DIR="/dev/shm/${USER}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export NCCL_TIMEOUT=7200
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
NUM_GPUS=6
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29966

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

export HF_HOME="/mnt/data/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/data/jiaxuanluo/cache"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
SAVE_DIR="/mnt/data/jiaxuanluo/train_outputs"

USE_LORA="true"
LORA_RANK=128
LORA_ALPHA=256
TARGET_DIM=1024
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
POOLING_TYPE="transformer"

USE_MAXSIM="true"
MAXSIM_WINDOWS="6 10 16 24"
MAXSIM_STRIDE=2
MAXSIM_AGG="softmax"
MAXSIM_SOFTMAX_TAU=1.0

# Text encoder: LoRA r128, CLS pooling, NO sparse (match MaxSim-only baseline)
TEXT_FULL_FINETUNE="false"
TEXT_LR="0"
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
TEXT_TARGET_MODULES="query key value dense"
TEXT_POOLING="cls"
SPARSE_WEIGHT="0.0"

PER_GPU_BATCH=1536
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
GRAD_CACHE_CHUNK_SIZE=256
EPOCHS=5
NUM_WORKERS=8
LR="1.7e-4"
TEMPERATURE="0.03"
TRAIN_LIMIT=0
WIKI_RANK=1000000
NOISY_RATIO=0.0
MARGIN=0.1
ONLINE_HARD_NEG_K=0

GLOSSARY_NEG_PATH=""
GLOSSARY_NEG_REFRESH_STEPS=0
HARD_NEG_K=0
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=0

SAVE_STEPS=100
EVAL_STEPS_SAMPLE=43
KEEP_CHECKPOINTS=3
EVAL_TOPK=10

ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
EVAL_GLOSSARY_SIZES="1000 10000"
BEST_METRIC="eval_acl6060/recall@10_gs1000"
BEST_METRIC_SECONDARY="eval_acl6060/recall@10_gs10000"
# ======Configuration=====

mkdir -p "${SAVE_DIR}"

VERSION="3var_clean_gc_wr$((WIKI_RANK / 1000))k_m${MARGIN}_maxsim_softmax"
TEXT_TAG="tr${TEXT_LORA_RANK}"
MODE_NAME="scale_lora-r${LORA_RANK}-${TEXT_TAG}"
BS_ABBR=$((BATCH_SIZE / 1024))k
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_t=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="stage2_${SAVE_NAME}"

echo "[TRAIN] MaxSim + softmax aggregation (1h ablation, tau ${MAXSIM_SOFTMAX_TAU} -> 0.05)"
echo "[TRAIN] Data: ${TRAIN_JSONL}"
echo "[TRAIN] Save: ${SAVE_PATH}"
echo "[TRAIN] Batch: ${BATCH_SIZE} (${NUM_GPUS} GPUs * ${PER_GPU_BATCH})"
echo "[TRAIN] MaxSim windows: ${MAXSIM_WINDOWS}, stride: ${MAXSIM_STRIDE}, agg: ${MAXSIM_AGG}"

OPTS="--use_maxsim --use_lora"
if [ "${WIKI_RANK}" -gt 0 ]; then OPTS="${OPTS} --wiki_rank ${WIKI_RANK}"; fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_PATH}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --dev_jsonl "${DEV_JSONL}" \
    --save_path "${SAVE_PATH}" \
    --lr "${LR}" \
    --text_lr "${TEXT_LR}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --temperature "${TEMPERATURE}" \
    --target_dim "${TARGET_DIM}" \
    --pooling_type "${POOLING_TYPE}" \
    --maxsim_windows ${MAXSIM_WINDOWS} \
    --maxsim_stride "${MAXSIM_STRIDE}" \
    --maxsim_agg "${MAXSIM_AGG}" \
    --maxsim_softmax_tau "${MAXSIM_SOFTMAX_TAU}" \
    --text_pooling "${TEXT_POOLING}" \
    --sparse_weight "${SPARSE_WEIGHT}" \
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
    --noisy_ratio "${NOISY_RATIO}" \
    --margin "${MARGIN}" \
    --online_hard_neg_k "${ONLINE_HARD_NEG_K}" \
    --grad_cache_chunk_size "${GRAD_CACHE_CHUNK_SIZE}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_steps_sample "${EVAL_STEPS_SAMPLE}" \
    --eval_topk "${EVAL_TOPK}" \
    --keep_checkpoints "${KEEP_CHECKPOINTS}" \
    --acl_dev_jsonl "${ACL_DEV_JSONL}" \
    --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
    --eval_glossary_sizes ${EVAL_GLOSSARY_SIZES} \
    --best_metric "${BEST_METRIC}" \
    --best_metric_secondary "${BEST_METRIC_SECONDARY}" \
    --eval_top100_samples 3 \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${WANDB_EXP_NAME}" \
    ${OPTS}

echo "[TRAIN] Completed at $(date)"
