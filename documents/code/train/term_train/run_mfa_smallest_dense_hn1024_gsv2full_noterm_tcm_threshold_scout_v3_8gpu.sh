#!/bin/bash
#SBATCH --job-name=q3_ntcm_thr_v3
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=8:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_q3_ntcm_thr_v3_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_q3_ntcm_thr_v3_%x.err

set -euo pipefail

TASK_OFFSET="${TASK_OFFSET:-0}"
TASK_ID_RAW="${SLURM_ARRAY_TASK_ID:-0}"
TASK_ID=$((TASK_ID_RAW + TASK_OFFSET))
CONFIG_PATH="${CONFIG_PATH:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/noterm_tcm_threshold_scout_grid_v3.tsv}"
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "[FATAL] CONFIG_PATH not found: ${CONFIG_PATH}" >&2
    exit 1
fi
LINE="$(awk -v n="$((TASK_ID + 2))" 'NR == n {print}' "${CONFIG_PATH}")"
if [ -z "${LINE}" ]; then
    echo "[FATAL] no config row for TASK_ID=${TASK_ID} in ${CONFIG_PATH}" >&2
    exit 1
fi
IFS=$'\t' read -r GRID_VARIANT THRESHOLD_ROLE TCM_POS TCM_NEG TCM_POS_W TCM_NEG_W INFER_TAU <<< "${LINE}"

BASELINE_STEP="${BASELINE_STEP:-2650}"
SCOUT_STEPS="${SCOUT_STEPS:-250}"
TARGET_MAX_STEPS=$((BASELINE_STEP + SCOUT_STEPS))
export NUM_GPUS="${NUM_GPUS:-8}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1536}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="${GRID_VARIANT}"
export VERSION="3var_gsv2full_gsfix_mfa_ntcm_thr_v3_${VARIANT_TAG}_s${SCOUT_STEPS}_${COMPUTE_TAG}_smallest_dense"
export WANDB_EXP_NAME="ntcm_thr_v3_${VARIANT_TAG}_s${SCOUT_STEPS}_${COMPUTE_TAG}"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_noterm_tcm_threshold_scouts_v3.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
export DEV_JSONL="/mnt/gemini/home/jiaxuanluo/noterm_dev/term_dev_with_gs_noterm_wiki_v3_balanced.jsonl"
export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_r3auto1m_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt"

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=1024
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

export GRAD_CACHE_CHUNK_SIZE=512
export EPOCHS=20
export SCHEDULER_EPOCHS=20
export MAX_STEPS="${TARGET_MAX_STEPS}"
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=true
export MASTER_PORT=$((30020 + TASK_ID))

export DATA_TAG="3variant_gsv2full_gsfix_mfa_devv3"
export EXPERIMENT_FAMILY="sst_tcm_noterm"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG}"
export BASELINE_RUN_IDS="us4obwe3 pacexfw7 9o32xd0k"
export SELECT_CLEAN_GPUS=true

export ACL_DEV_JSONL=""
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
export EVAL_GLOSSARY_SIZES="10000"
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/recall@10_gs10000"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=50
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="${INFER_TAU}"
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="Threshold scout v3: T_beta=${TCM_POS}, T_alpha=${TCM_NEG}, inference_tau=${INFER_TAU}, pos_w=${TCM_POS_W}, neg_w=${TCM_NEG_W}."

echo "[NTCM_THR_V3] config=${CONFIG_PATH} raw_task=${TASK_ID_RAW} offset=${TASK_OFFSET} row=${TASK_ID} variant=${VARIANT_TAG}"
echo "[NTCM_THR_V3] thresholds pos=${TCM_POS} neg=${TCM_NEG} weights pos=${TCM_POS_W} neg=${TCM_NEG_W} infer_tau=${INFER_TAU}"
echo "[NTCM_THR_V3] target_step=${TARGET_MAX_STEPS} compute=${COMPUTE_TAG}"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
