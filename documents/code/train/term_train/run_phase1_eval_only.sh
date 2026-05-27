#!/bin/bash
# Phase 1 eval-only run: produce detection + threshold baselines on dev + ACL.
#
# Usage:
#   ./run_phase1_eval_only.sh [ckpt_path]
#
# Checkpoint hyper-parameters (temperature, pooling, maxsim window schedule)
# must match the CLI flags here; Config C was trained with T=0.07, maxsim
# windows "6 10 16 24" / stride 2, MFA-supervised maxsim.

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/mnt/taurus/data/jiaxuanluo/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/taurus/data/jiaxuanluo/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/taurus/data/jiaxuanluo/cache}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"

DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"

DEFAULT_CKPT="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"

TEMPERATURE="0.07"
MARGIN="0.1"
TARGET_DIM=1024
USE_LORA="true"
LORA_RANK=128
LORA_ALPHA=256
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
POOLING_TYPE="transformer"
USE_MAXSIM="true"
MAXSIM_WINDOWS="6 10 16 24"
MAXSIM_STRIDE=2
MFA_SUPERVISED="true"

TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
TEXT_TARGET_MODULES="query key value dense"
TEXT_POOLING="cls"
SPARSE_WEIGHT="0.0"

EVAL_BATCH_SIZE=32
EVAL_TOPK=10
EVAL_GLOSSARY_SIZES="1000 10000"
# ======Configuration=====

CKPT_PATH="${1:-${DEFAULT_CKPT}}"

assert_exists() {
    local path="$1"
    local label="$2"
    if [ ! -e "${path}" ]; then
        echo "[FATAL] ${label} not found: ${path}" >&2
        exit 2
    fi
}

assert_exists "${CKPT_PATH}" "checkpoint"
assert_exists "${DEV_JSONL}" "dev jsonl"
assert_exists "${ACL_DEV_JSONL}" "ACL jsonl"
assert_exists "${EVAL_WIKI_GLOSSARY}" "wiki glossary"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST_logs/phase1_eval"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/phase1_eval_${STAMP}.log"

echo "[PHASE1_EVAL] checkpoint=${CKPT_PATH}"
echo "[PHASE1_EVAL] log=${LOG_FILE}"

OPTS=""
if [ "${USE_LORA}" = "true" ]; then OPTS="${OPTS} --use_lora"; fi
if [ "${USE_MAXSIM}" = "true" ]; then OPTS="${OPTS} --use_maxsim"; fi
if [ "${MFA_SUPERVISED}" = "true" ]; then OPTS="${OPTS} --mfa_supervised_maxsim"; fi

torchrun \
    --nproc_per_node=1 \
    --master_addr="127.0.0.1" \
    --master_port="29997" \
    "${SCRIPT_PATH}" \
    --eval_only \
    --resume "${CKPT_PATH}" \
    --train_jsonl "${DEV_JSONL}" \
    --dev_jsonl "${DEV_JSONL}" \
    --acl_dev_jsonl "${ACL_DEV_JSONL}" \
    --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
    --eval_glossary_sizes ${EVAL_GLOSSARY_SIZES} \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --eval_topk "${EVAL_TOPK}" \
    --eval_top100_samples 0 \
    --save_path "/tmp/phase1_eval_only_${STAMP}.pt" \
    --temperature "${TEMPERATURE}" \
    --margin "${MARGIN}" \
    --target_dim "${TARGET_DIM}" \
    --pooling_type "${POOLING_TYPE}" \
    --maxsim_windows ${MAXSIM_WINDOWS} \
    --maxsim_stride "${MAXSIM_STRIDE}" \
    --text_pooling "${TEXT_POOLING}" \
    --sparse_weight "${SPARSE_WEIGHT}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --text_lora_rank "${TEXT_LORA_RANK}" \
    --text_lora_alpha "${TEXT_LORA_ALPHA}" \
    --lora_target_modules ${TARGET_MODULES} \
    --text_lora_target_modules ${TEXT_TARGET_MODULES} \
    --batch_size 1 \
    --epochs 1 \
    --num_workers 2 \
    ${OPTS} 2>&1 | tee "${LOG_FILE}"

echo "[PHASE1_EVAL] done. log=${LOG_FILE}"
