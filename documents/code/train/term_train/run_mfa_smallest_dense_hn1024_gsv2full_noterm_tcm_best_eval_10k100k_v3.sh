#!/bin/bash
#SBATCH --job-name=q3_ntcm_eval100k
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_q3_ntcm_eval100k_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_q3_ntcm_eval100k_%x.err

set -euo pipefail

TASK_OFFSET="${TASK_OFFSET:-0}"
TASK_ID_RAW="${SLURM_ARRAY_TASK_ID:-0}"
TASK_ID=$((TASK_ID_RAW + TASK_OFFSET))
CONFIG_PATH="${CONFIG_PATH:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/noterm_tcm_best_eval_10k100k_grid_v3.tsv}"
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[FATAL] CONFIG_PATH not found: ${CONFIG_PATH}" >&2
    exit 1
fi
LINE="$(awk -v n="$((TASK_ID + 2))" 'NR == n {print}' "${CONFIG_PATH}")"
if [ -z "${LINE}" ]; then
    echo "[FATAL] no config row for TASK_ID=${TASK_ID} in ${CONFIG_PATH}" >&2
    exit 1
fi
IFS=$'\t' read -r GRID_VARIANT TRAIN_RUN_ID CKPT_PATH TCM_POS TCM_NEG TCM_POS_W TCM_NEG_W INFER_TAU <<< "${LINE}"
if [ ! -f "${CKPT_PATH}" ]; then
    echo "[FATAL] checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-512}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="eval100k_${GRID_VARIANT}"
export VERSION="3var_gsv2full_gsfix_mfa_ntcm_eval100k_v3_${GRID_VARIANT}_${COMPUTE_TAG}_smallest_dense"
export WANDB_EXP_NAME="ntcm_eval100k_v3_${GRID_VARIANT}_${COMPUTE_TAG}"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_noterm_tcm_best_eval_10k100k_v3.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
export DEV_JSONL="/mnt/gemini/home/jiaxuanluo/noterm_dev/term_dev_with_gs_noterm_wiki_v3_balanced.jsonl"
export RESUME="${CKPT_PATH}"

export EVAL_ONLY=true
export TASK_TAG="eval"
export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=0
export NEG_BANK_SIZE=0
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT="${TCM_POS_W}"
export TCM_NEG_LOSS_WEIGHT="${TCM_NEG_W}"
export TCM_POS_THRESHOLD="${TCM_POS}"
export TCM_NEG_THRESHOLD="${TCM_NEG}"
export TCM_LOSS_FORM="hinge"
export TCM_REDUCTION="mean_viol"
export TCM_NEG_SCOPE="all"
export TCM_NEG_TOPK=0
export TCM_WARMUP_STEPS=0

export GRAD_CACHE_CHUNK_SIZE=256
export EPOCHS=1
export SCHEDULER_EPOCHS=1
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false
export MASTER_PORT=$((30220 + TASK_ID))

export DATA_TAG="3variant_gsv2full_gsfix_mfa_devv3_eval100k"
export EXPERIMENT_FAMILY="sst_tcm_noterm"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG}"
export BASELINE_RUN_IDS="us4obwe3 ${TRAIN_RUN_ID}"
export SELECT_CLEAN_GPUS=true

export ACL_DEV_JSONL=""
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json"
export EVAL_GLOSSARY_SIZES="10000 100000"
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/recall@10_gs100000"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="${INFER_TAU}"
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="Eval-only 10k/100k for ${GRID_VARIANT}: source_run=${TRAIN_RUN_ID}, tau=${INFER_TAU}, pos_w=${TCM_POS_W}, neg_w=${TCM_NEG_W}."

echo "[NTCM_EVAL100K] config=${CONFIG_PATH} raw_task=${TASK_ID_RAW} offset=${TASK_OFFSET} row=${TASK_ID} variant=${VARIANT_TAG}"
echo "[NTCM_EVAL100K] source_run=${TRAIN_RUN_ID} ckpt=${CKPT_PATH}"
echo "[NTCM_EVAL100K] pos=${TCM_POS} neg=${TCM_NEG} weights pos=${TCM_POS_W} neg=${TCM_NEG_W} tau=${INFER_TAU}"
echo "[NTCM_EVAL100K] compute=${COMPUTE_TAG} batch=$((NUM_GPUS * PER_GPU_BATCH))"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
