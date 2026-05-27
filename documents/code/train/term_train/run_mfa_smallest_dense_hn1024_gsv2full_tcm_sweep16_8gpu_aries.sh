#!/bin/bash
#SBATCH --job-name=q3_tcm_sweep16
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=18:00:00
#SBATCH --array=0-15%1
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_q3_tcm_sweep16_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_q3_tcm_sweep16_%x.err

# Continuing sweep from the full GSV2 k=1024 TCM-off epoch-2 checkpoint.

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
POS_THRESHOLDS=(0.85 0.80 0.75 0.70)
NEG_THRESHOLDS=(0.70 0.60 0.50 0.40)
WEIGHTS=(1 2 4 8)

PAIR_IDX=$((TASK_ID / ${#WEIGHTS[@]}))
WEIGHT_IDX=$((TASK_ID % ${#WEIGHTS[@]}))
TCM_POS="${POS_THRESHOLDS[$PAIR_IDX]}"
TCM_NEG="${NEG_THRESHOLDS[$PAIR_IDX]}"
TCM_W="${WEIGHTS[$WEIGHT_IDX]}"

fmt_decimal() {
    local value="$1"
    value="${value#0.}"
    value="${value//./p}"
    echo "${value}"
}

POS_TAG="$(fmt_decimal "${TCM_POS}")"
NEG_TAG="$(fmt_decimal "${TCM_NEG}")"
W_TAG="$(fmt_decimal "${TCM_W}")"
export VARIANT_TAG="tcm_p${POS_TAG}_n${NEG_TAG}_w${W_TAG}"
export VERSION="3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_${VARIANT_TAG}_cont1_bs12k_smallest_dense_normAGGR_8gpu_aries"
export WANDB_EXP_NAME="variantE_hn1024_gsv2full_${VARIANT_TAG}_cont1_8gpu_aries"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_tcm_sweep16_aries.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
export RESUME="${RESUME:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep3_bs12k_smallest_dense_normAGGR_8gpu_aries_epoch_2.pt}"

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=1024
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT="${TCM_W}"
export TCM_NEG_LOSS_WEIGHT="${TCM_W}"
export TCM_POS_THRESHOLD="${TCM_POS}"
export TCM_NEG_THRESHOLD="${TCM_NEG}"
export TCM_LOSS_FORM="hinge"
export TCM_REDUCTION="mean_viol"
export TCM_NEG_SCOPE="all"
export TCM_NEG_TOPK=0
export TCM_WARMUP_STEPS=100

export NUM_GPUS=8
export PER_GPU_BATCH=1536
export GRAD_CACHE_CHUNK_SIZE=512
# Resumed epoch_2 has start_epoch=3; EPOCHS=4 trains exactly one more epoch.
export EPOCHS=4
export SCHEDULER_EPOCHS=4
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export MASTER_PORT=$((29990 + TASK_ID))
export DATA_TAG="3variant_gsv2full_gsfix_mfa"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:aries-8gpu"
export BASELINE_RUN_IDS="ly6sc2mr fma3wmh2 x68wsne9 d7ij3to1"
export SELECT_CLEAN_GPUS=true

# Dev-only selection with untrained P31 distractors.
export ACL_DEV_JSONL=""
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
export EVAL_GLOSSARY_SIZES="10000"
export BEST_METRIC="eval_dev/recall@10_gs10000"
export BEST_METRIC_SECONDARY="eval_dev/topk10_filtered_recall@tau_0p80_gs10000"
export EVAL_STEPS_SAMPLE=80
export TCM_SWEEP_THRESHOLDS="0.85 0.80 0.75 0.70"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
