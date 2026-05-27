#!/bin/bash
#SBATCH --job-name=q3_ablation
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ablation_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ablation_%x.err

# Hard-neg / MFA / Eval-minimal ablation launcher.
#
# Four variants are compared at a fixed max_steps budget (1500) with
# --eval_minimal_metrics so the ablation signal stays readable:
#
#   baseline       : hard_neg_k=64,  mfa_window_selection=hard_max
#   hardneg_k128   : hard_neg_k=128, mfa_window_selection=hard_max
#   mfa_smallest   : hard_neg_k=64,  mfa_window_selection=smallest
#   mfa_logsumexp  : hard_neg_k=64,  mfa_window_selection=logsumexp
#
# All other hyperparameters mirror run_hardneg_tcm_aries.sh for a strict
# controlled comparison against the existing variant E baseline.
#
# Usage:
#   sbatch run_hardneg_ablation_aries.sh <variant> [partition]
#     variant   : one of {baseline, hardneg_k128, mfa_smallest, mfa_logsumexp}
#     partition : optional override of SBATCH --partition via --export=VARIANT=...
#                 (default handled by sbatch -p)
#
# Example:
#   sbatch --export=ALL,ABLATION_VARIANT=mfa_smallest run_hardneg_ablation_aries.sh
#   sbatch -p gemini --export=ALL,ABLATION_VARIANT=hardneg_k128 run_hardneg_ablation_aries.sh

set -euo pipefail

# ======Configuration=====
# --- Ablation variant (injected via --export=ALL,ABLATION_VARIANT=<name>). ---
ABLATION_VARIANT="${ABLATION_VARIANT:-}"
if [ -z "${ABLATION_VARIANT}" ]; then
    echo "[CONFIG] ABLATION_VARIANT env var is required. "\
"Set via sbatch --export=ALL,ABLATION_VARIANT=<variant>." >&2
    exit 1
fi

# --- Step budget (identical for all variants). ---
MAX_STEPS=1500

# --- Hard-neg bank scale (A1 axis). ---
HARD_NEG_K_BASE=64
HARD_NEG_K_ABLATION=128

# --- MFA window selection (A2 axis). ---
MFA_MODE_BASE="hard_max"
MFA_MODE_SMALLEST="smallest"
MFA_MODE_LOGSUMEXP="logsumexp"
MFA_LSE_TEMPERATURE=1.0

# --- Eval cadence for ablation: emit more frequently so the 1500-step
#     training curve is legible. Default 43769 used 40; we bump to 80 so
#     the ablation gets ~19 eval points over 1500 steps.                 ---
EVAL_STEPS_SAMPLE=80
SAVE_STEPS=999999
KEEP_CHECKPOINTS=2
EVAL_TOPK=10

# --- Variant dispatch.
case "${ABLATION_VARIANT}" in
    baseline)
        HARD_NEG_K=${HARD_NEG_K_BASE}
        MFA_WINDOW_SELECTION=${MFA_MODE_BASE}
        ;;
    hardneg_k128)
        HARD_NEG_K=${HARD_NEG_K_ABLATION}
        MFA_WINDOW_SELECTION=${MFA_MODE_BASE}
        ;;
    mfa_smallest)
        HARD_NEG_K=${HARD_NEG_K_BASE}
        MFA_WINDOW_SELECTION=${MFA_MODE_SMALLEST}
        ;;
    mfa_logsumexp)
        HARD_NEG_K=${HARD_NEG_K_BASE}
        MFA_WINDOW_SELECTION=${MFA_MODE_LOGSUMEXP}
        ;;
    *)
        echo "[CONFIG] Unknown ABLATION_VARIANT=${ABLATION_VARIANT}. "\
"Expected one of: baseline, hardneg_k128, mfa_smallest, mfa_logsumexp" >&2
        exit 1
        ;;
esac

# --- Loss hyperparameters (mirrors variant E in run_hardneg_tcm_aries.sh) ---
MARGIN="0.0"
HCL_BETA="0.0"
TCM_LOSS_WEIGHT="1.0"
TCM_POS_THRESHOLD="0.85"
TCM_NEG_THRESHOLD="0.25"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"

NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=50

# --- Environment (fully qualified cross-node paths only) ---
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
# Disable user-site-packages: node-local ~/.local/ is only visible on
# specific nodes (e.g. on aries but not gemini) and has historically
# hijacked the regex/etc import paths with stale versions.  All deps must
# resolve from the shared conda env on /mnt/taurus.
export PYTHONNOUSERSITE=1

# Per-job isolated tmp on node-local /tmp (NOT /dev/shm).
#
# We previously used /dev/shm/${USER}/pytorch_tmp, but on aries the SLURM
# plugin/systemd tmpfs cleaner sporadically wipes user-owned subdirs of
# /dev/shm mid-run, causing DataLoader worker mkdtemp() failures like:
#     FileNotFoundError: '/dev/shm/jiaxuanluo_<jobid>/pytorch_tmp/pymp-XXXX'
# even when the parent was created by the launcher seconds earlier. /tmp is
# persistent for the lifetime of the job (cgroup-scoped cleanup happens on
# job exit, not while it is running) and is the standard default TMPDIR.
LOCAL_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-local}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Clean our per-job tmp on exit so /tmp does not accumulate stale dirs
# across repeated ablation submissions.
trap '[ -n "${LOCAL_TMP_DIR:-}" ] && rm -rf "${LOCAL_TMP_DIR}"' EXIT

export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO

NUM_GPUS=8
MASTER_ADDR="127.0.0.1"
# Pick a distinct master port per variant to allow concurrent jobs on different
# partitions without port collision (variant dispatch picks the offset).
case "${ABLATION_VARIANT}" in
    baseline)      MASTER_PORT=29970 ;;
    hardneg_k128)  MASTER_PORT=29971 ;;
    mfa_smallest)  MASTER_PORT=29972 ;;
    mfa_logsumexp) MASTER_PORT=29973 ;;
esac

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

# Point HF cache to the shared aries cross-node location where the Qwen3 Audio
# encoder is already materialized. Explicitly pin HF_HUB_CACHE and the two
# legacy aliases (HUGGINGFACE_HUB_CACHE / TRANSFORMERS_CACHE) so the download
# path resolution is not node-local (gemini's $HOME maps to a read-only path).
export HF_HOME="/mnt/aries/home/jiaxuanluo/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="/mnt/aries/data4/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/aries/data4/jiaxuanluo/cache"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
SAVE_DIR="/mnt/gemini/home/jiaxuanluo/train_outputs"

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

PER_GPU_BATCH=1536
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
GRAD_CACHE_CHUNK_SIZE=256

# Ablation runs are step-capped; give epochs a generous upper bound so the
# max_steps guard is the actual stop criterion. At bs=12288 on the 6.5M
# training set (~530 steps/epoch), 1500 steps spans ~2.83 epochs.
EPOCHS=5
MAX_TRAIN_SECONDS=0
NUM_WORKERS=4
LR="1.7e-4"
TEMPERATURE="0.07"
TRAIN_LIMIT=0
WIKI_RANK=1000000
NOISY_RATIO=0.0
ONLINE_HARD_NEG_K=0

GLOSSARY_NEG_PATH=""
GLOSSARY_NEG_REFRESH_STEPS=0

ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
EVAL_GLOSSARY_SIZES="1000 10000"
BEST_METRIC="eval_acl6060/recall@10_gs1000"
BEST_METRIC_SECONDARY="eval_acl6060/recall@10_gs10000"
# ======Configuration=====

mkdir -p "${SAVE_DIR}"

# Trust SLURM-provided CUDA_VISIBLE_DEVICES (full-node --gres=gpu:8 allocates
# all 8 GPUs).  We skip an explicit nvidia-smi preflight because on shared
# nodes it incorrectly flags other jobs' devices as busy even when SLURM has
# already filtered our view.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[PREFLIGHT] CUDA_VISIBLE_DEVICES unset by SLURM; defaulting to 0-$((NUM_GPUS - 1))" >&2
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi
echo "[PREFLIGHT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

BS_ABBR=$((BATCH_SIZE / 1024))k
if [ $((BATCH_SIZE % 1024)) -ne 0 ]; then
    BS_ABBR="${BATCH_SIZE}"
fi

TEXT_TAG="tr${TEXT_LORA_RANK}"
MODE_NAME="scale_lora-r${LORA_RANK}-${TEXT_TAG}"
VERSION="ablation_${ABLATION_VARIANT}_k${HARD_NEG_K}_mfa-${MFA_WINDOW_SELECTION}_steps${MAX_STEPS}"
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_t=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="ablation_${ABLATION_VARIANT}_${SAVE_NAME}"

echo "[TRAIN] ABLATION_VARIANT=${ABLATION_VARIANT}"
echo "[TRAIN] MAX_STEPS=${MAX_STEPS} EPOCHS=${EPOCHS} (step cap is authoritative)"
echo "[TRAIN] HARD_NEG_K=${HARD_NEG_K} MFA_WINDOW_SELECTION=${MFA_WINDOW_SELECTION} MFA_LSE_TEMPERATURE=${MFA_LSE_TEMPERATURE}"
echo "[TRAIN] Save: ${SAVE_PATH}"
echo "[TRAIN] Batch: ${BATCH_SIZE} (${NUM_GPUS} GPUs * ${PER_GPU_BATCH})"
echo "[TRAIN] TCM: lambda=${TCM_LOSS_WEIGHT} T_beta=${TCM_POS_THRESHOLD} T_alpha=${TCM_NEG_THRESHOLD} form=${TCM_LOSS_FORM} reduction=${TCM_REDUCTION}"

OPTS=""
if [ "${USE_LORA}" = "true" ]; then OPTS="${OPTS} --use_lora"; fi
if [ "${USE_MAXSIM}" = "true" ]; then OPTS="${OPTS} --use_maxsim"; fi
if [ "${MFA_SUPERVISED}" = "true" ]; then OPTS="${OPTS} --mfa_supervised_maxsim"; fi
if [ "${WIKI_RANK}" -gt 0 ]; then OPTS="${OPTS} --wiki_rank ${WIKI_RANK}"; fi

# Defensive re-mkdir for TMPDIR right before torchrun spawns workers. On aries
# we have seen tmpfs mount-namespace oddities where a dir created at bash
# script start disappears inside the DataLoader worker process; re-creating it
# here ensures the inode exists in whatever ns torchrun's children inherit.
mkdir -p "${LOCAL_TMP_DIR}"
if [ ! -d "${LOCAL_TMP_DIR}" ]; then
    echo "[PREFLIGHT][FATAL] TMPDIR ${LOCAL_TMP_DIR} is not writable" >&2
    exit 1
fi
echo "[PREFLIGHT] TMPDIR=${LOCAL_TMP_DIR} exists=$(stat -c %F "${LOCAL_TMP_DIR}")"

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
    --max_steps "${MAX_STEPS}" \
    --train_limit "${TRAIN_LIMIT}" \
    --num_workers "${NUM_WORKERS}" \
    --temperature "${TEMPERATURE}" \
    --target_dim "${TARGET_DIM}" \
    --pooling_type "${POOLING_TYPE}" \
    --maxsim_windows ${MAXSIM_WINDOWS} \
    --maxsim_stride "${MAXSIM_STRIDE}" \
    --mfa_window_selection "${MFA_WINDOW_SELECTION}" \
    --mfa_lse_temperature "${MFA_LSE_TEMPERATURE}" \
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
    --eval_top100_samples 0 \
    --eval_minimal_metrics \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${WANDB_EXP_NAME}" \
    --tcm_loss_weight "${TCM_LOSS_WEIGHT}" \
    --tcm_pos_threshold "${TCM_POS_THRESHOLD}" \
    --tcm_neg_threshold "${TCM_NEG_THRESHOLD}" \
    --tcm_loss_form "${TCM_LOSS_FORM}" \
    --tcm_reduction "${TCM_REDUCTION}" \
    --hcl_beta "${HCL_BETA}" \
    --max_train_seconds "${MAX_TRAIN_SECONDS}" \
    ${OPTS}

echo "[TRAIN] Ablation variant=${ABLATION_VARIANT} completed at $(date)"
