#!/bin/bash
#SBATCH --job-name=q3_variant_E_sweep
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_variant_E_sweep_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_variant_E_sweep_%x.err

# Post-hoc TCM threshold sweep on the variant-E best checkpoint.
#
# Answers three inference-gating questions against the trained retriever:
#
#   A. Single absolute threshold tau: for each tau in TCM_SWEEP_THRESHOLDS
#      we report P / R / F1 / pass_rate and two filter-after-retrieval
#      metrics (tcm_filtered_top1, tcm_filtered_recall{k}) on both the
#      in-distribution DEV set and the OOD ACL6060 set.
#   B. Dual-threshold policy is derived from the same sweep by picking
#      (T_reject, T_accept) pairs; no extra runtime cost.
#   C. Dev-calibrated single tau is the argmax F1@tau on DEV; we then read
#      the corresponding ACL6060 row to see how it transfers OOD.
#
# Runs with --eval_only, so it rebuilds the retriever from the ckpt
# args, loads DEV + ACL6060 dev loaders, makes a single forward pass per
# set, and dumps metrics via wandb + stdout.  1 GPU is enough because
# bank sizes stay <= 10k and audio counts are at most a few thousand.
#
# Usage:
#   sbatch --dependency=afterany:<main_jobid> run_hardneg_tcm_sweep_aries.sh
#   SWEEP_CKPT=/abs/path.pt sbatch run_hardneg_tcm_sweep_aries.sh

set -euo pipefail

# ======Configuration=====
MARGIN="0.0"
HCL_BETA="0.0"
TCM_LOSS_WEIGHT="1.0"
TCM_POS_THRESHOLD="0.85"
TCM_NEG_THRESHOLD="0.25"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"

# Thresholds to sweep at inference time.  0.25 and 0.85 are already
# computed by the existing tbeta/talpha path, so list only the extra
# points we want: a coarse grid from strict-reject to strict-accept.
TCM_SWEEP_THRESHOLDS="0.30 0.40 0.50 0.55 0.60 0.65 0.70 0.80"

# No bank mining / HCL at eval time.
HARD_NEG_K=0
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=50

export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

LOCAL_TMP_DIR="/dev/shm/${USER}/pytorch_tmp_sweep"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

NUM_GPUS=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29970

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

export HF_HOME="/mnt/aries/data4/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/aries/data4/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/aries/data4/jiaxuanluo/cache"

# Data paths.  Training JSONL is required by the script but the loader
# only reads it if --eval_only is off; we still point to a valid file.
TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
EVAL_GLOSSARY_SIZES="1000 10000"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
SAVE_DIR="/mnt/gemini/home/jiaxuanluo/train_outputs"

SAVE_NAME_STEM="q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_tcm_ep5"
DEFAULT_SWEEP_CKPT="${SAVE_DIR}/${SAVE_NAME_STEM}_best.pt"

USE_LORA="true"
LORA_RANK=128
LORA_ALPHA=256
TARGET_DIM=1024
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
POOLING_TYPE="transformer"

USE_MAXSIM="true"
MAXSIM_WINDOWS="6 10 16 24"
MAXSIM_STRIDE=2
MFA_SUPERVISED="true"

TEXT_LR="0"
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
TEXT_TARGET_MODULES="query key value dense"
TEXT_POOLING="cls"
SPARSE_WEIGHT="0.0"

# Eval-only config: eval_batch_size controls the only forward pass.
PER_GPU_BATCH=64
BATCH_SIZE=64
GRAD_CACHE_CHUNK_SIZE=64
EPOCHS=1
TRAIN_LIMIT=64
NUM_WORKERS=4
LR="1.7e-4"
TEMPERATURE="0.07"
WIKI_RANK=1000000
NOISY_RATIO=0.0
ONLINE_HARD_NEG_K=0

GLOSSARY_NEG_PATH=""
GLOSSARY_NEG_REFRESH_STEPS=0

EVAL_TOPK=10
EVAL_BATCH_SIZE=64
# ======Configuration=====

SWEEP_CKPT="${SWEEP_CKPT:-${DEFAULT_SWEEP_CKPT}}"
if [ ! -f "${SWEEP_CKPT}" ]; then
    echo "[SWEEP] checkpoint not found: ${SWEEP_CKPT}" >&2
    exit 1
fi
echo "[SWEEP] checkpoint = ${SWEEP_CKPT}"
echo "[SWEEP] thresholds = ${TCM_SWEEP_THRESHOLDS}"

# Honor SLURM's allocation: CUDA_VISIBLE_DEVICES is set by the scheduler,
# manually rescanning nvidia-smi on shared nodes is unsafe.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[PREFLIGHT] CUDA_VISIBLE_DEVICES is empty; SLURM did not allocate any GPU." >&2
    exit 1
fi
VISIBLE_COUNT="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)"
if [ "${VISIBLE_COUNT}" -lt "${NUM_GPUS}" ]; then
    echo "[PREFLIGHT] SLURM allocated ${VISIBLE_COUNT} GPUs but NUM_GPUS=${NUM_GPUS}." >&2
    exit 1
fi
echo "[PREFLIGHT] honoring SLURM allocation CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

SAVE_PATH="${SAVE_DIR}/${SAVE_NAME_STEM}__sweep.pt"
WANDB_EXP_NAME="variantE_sweep_$(basename "${SWEEP_CKPT}" .pt)"

OPTS=""
if [ "${USE_LORA}" = "true" ]; then OPTS="${OPTS} --use_lora"; fi
if [ "${USE_MAXSIM}" = "true" ]; then OPTS="${OPTS} --use_maxsim"; fi
if [ "${MFA_SUPERVISED}" = "true" ]; then OPTS="${OPTS} --mfa_supervised_maxsim"; fi
if [ "${WIKI_RANK}" -gt 0 ]; then OPTS="${OPTS} --wiki_rank ${WIKI_RANK}"; fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_PATH}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --dev_jsonl "${DEV_JSONL}" \
    --save_path "${SAVE_PATH}" \
    --resume "${SWEEP_CKPT}" \
    --eval_only \
    --lr "${LR}" \
    --text_lr "${TEXT_LR}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --train_limit "${TRAIN_LIMIT}" \
    --num_workers "${NUM_WORKERS}" \
    --temperature "${TEMPERATURE}" \
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
    --glossary_neg_path "${GLOSSARY_NEG_PATH}" \
    --glossary_neg_refresh_steps "${GLOSSARY_NEG_REFRESH_STEPS}" \
    --neg_bank_size "${NEG_BANK_SIZE}" \
    --neg_bank_refresh_steps "${NEG_BANK_REFRESH_STEPS}" \
    --hard_neg_k "${HARD_NEG_K}" \
    --noisy_ratio "${NOISY_RATIO}" \
    --margin "${MARGIN}" \
    --online_hard_neg_k "${ONLINE_HARD_NEG_K}" \
    --grad_cache_chunk_size "${GRAD_CACHE_CHUNK_SIZE}" \
    --eval_topk "${EVAL_TOPK}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --acl_dev_jsonl "${ACL_DEV_JSONL}" \
    --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
    --eval_glossary_sizes ${EVAL_GLOSSARY_SIZES} \
    --eval_top100_samples 0 \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${WANDB_EXP_NAME}" \
    --tcm_loss_weight "${TCM_LOSS_WEIGHT}" \
    --tcm_pos_threshold "${TCM_POS_THRESHOLD}" \
    --tcm_neg_threshold "${TCM_NEG_THRESHOLD}" \
    --tcm_loss_form "${TCM_LOSS_FORM}" \
    --tcm_reduction "${TCM_REDUCTION}" \
    --tcm_sweep_thresholds ${TCM_SWEEP_THRESHOLDS} \
    --hcl_beta "${HCL_BETA}" \
    --max_train_seconds 0 \
    ${OPTS}

echo "[SWEEP] completed at $(date)"
