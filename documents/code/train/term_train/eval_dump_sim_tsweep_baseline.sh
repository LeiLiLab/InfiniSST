#!/bin/bash
#SBATCH --job-name=q3_sim_dump
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_sim_dump.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_sim_dump.err

# Eval-only dump of per-sample pos_sim / neg_sim distributions for the
# no-TCM tsweep baseline checkpoint (t=0.07, m=0.0). Produces NPZ files at
# DUMP_DIR for DEV + ACL6060, base bank + gs1000 + gs10000.

set -euo pipefail

export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

LOCAL_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-local}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export HF_HOME="/mnt/taurus/home/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/taurus/home/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/home/jiaxuanluo/cache"

export WANDB_MODE=disabled

SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
RESUME_PATH="/mnt/aries/data4/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs6k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_tsweep_best_acl6060_gs10000.pt"

# Same eval data as training run.
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
EVAL_GLOSSARY_SIZES="1000 10000"

# Where to write NPZ dumps.
DUMP_DIR="/mnt/taurus/home/jiaxuanluo/sim_dump/tsweep_bs6k_t0.07_m0.0_notcm"
mkdir -p "${DUMP_DIR}"

# Must match the architecture of the saved checkpoint (tsweep config).
LORA_RANK=128
LORA_ALPHA=256
TARGET_DIM=1024
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
POOLING_TYPE="transformer"
MAXSIM_WINDOWS="6 10 16 24"
MAXSIM_STRIDE=2
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
TEXT_TARGET_MODULES="query key value dense"
TEXT_POOLING="cls"
SPARSE_WEIGHT="0.0"
TEMPERATURE="0.07"
MARGIN="0.0"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
# SMOKE=1 → truncate DEV/ACL jsonl to first N lines under /tmp for smoke test.
SMOKE="${SMOKE:-0}"
SMOKE_N="${SMOKE_N:-200}"

if [ "${SMOKE}" = "1" ]; then
    SMOKE_DIR="/tmp/${USER}_sim_dump_smoke"
    mkdir -p "${SMOKE_DIR}"
    head -n "${SMOKE_N}" "${DEV_JSONL}" > "${SMOKE_DIR}/dev.jsonl"
    head -n "${SMOKE_N}" "${ACL_DEV_JSONL}" > "${SMOKE_DIR}/acl.jsonl"
    DEV_JSONL="${SMOKE_DIR}/dev.jsonl"
    ACL_DEV_JSONL="${SMOKE_DIR}/acl.jsonl"
    echo "[DUMP] SMOKE=1: truncated DEV/ACL to ${SMOKE_N} lines → ${SMOKE_DIR}"
fi

echo "[DUMP] Checkpoint: ${RESUME_PATH}"
echo "[DUMP] Output dir: ${DUMP_DIR}"
echo "[DUMP] EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"

# Clean up tmpdir on exit.
trap 'rm -rf "${LOCAL_TMP_DIR}" 2>/dev/null || true' EXIT

python "${SCRIPT_PATH}" \
    --eval_only \
    --resume "${RESUME_PATH}" \
    --train_jsonl "/dev/null" \
    --dev_jsonl "${DEV_JSONL}" \
    --acl_dev_jsonl "${ACL_DEV_JSONL}" \
    --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
    --eval_glossary_sizes ${EVAL_GLOSSARY_SIZES} \
    --save_path "/tmp/unused_save.pt" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --batch_size 8 \
    --epochs 1 \
    --num_workers 4 \
    --temperature "${TEMPERATURE}" \
    --target_dim "${TARGET_DIM}" \
    --pooling_type "${POOLING_TYPE}" \
    --use_maxsim \
    --mfa_supervised_maxsim \
    --maxsim_windows ${MAXSIM_WINDOWS} \
    --maxsim_stride "${MAXSIM_STRIDE}" \
    --text_pooling "${TEXT_POOLING}" \
    --sparse_weight "${SPARSE_WEIGHT}" \
    --use_lora \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --text_lora_rank "${TEXT_LORA_RANK}" \
    --text_lora_alpha "${TEXT_LORA_ALPHA}" \
    --lora_target_modules ${TARGET_MODULES} \
    --text_lora_target_modules ${TEXT_TARGET_MODULES} \
    --margin "${MARGIN}" \
    --eval_topk 10 \
    --dump_sim_distributions "${DUMP_DIR}" \
    --eval_minimal_metrics

echo "[DUMP] Done at $(date)"
ls -lh "${DUMP_DIR}"
