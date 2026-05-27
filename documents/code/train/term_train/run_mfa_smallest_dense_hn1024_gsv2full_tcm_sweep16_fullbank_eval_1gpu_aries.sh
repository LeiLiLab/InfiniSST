#!/bin/bash
#SBATCH --job-name=q3_tcm_full_eval
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-15%1
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_q3_tcm_full_eval_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_q3_tcm_full_eval_%x.err

# One-shot dev fullbank eval for each TCM continuation.

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
BASE_VARIANT_TAG="tcm_p${POS_TAG}_n${NEG_TAG}_w${W_TAG}"

export VARIANT_TAG="fullbank_${BASE_VARIANT_TAG}"
export VERSION="3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_${VARIANT_TAG}_eval1_bs1536_smallest_dense_normAGGR_1gpu_aries"
export WANDB_EXP_NAME="eval_hn1024_gsv2full_${VARIANT_TAG}_dev1m_aries"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_tcm_sweep16_fullbank_eval_aries.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
export RESUME="${RESUME:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_${BASE_VARIANT_TAG}_cont1_bs12k_smallest_dense_normAGGR_8gpu_aries.pt}"

export EVAL_ONLY=true
export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=0
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_POS_THRESHOLD="${TCM_POS}"
export TCM_NEG_THRESHOLD="${TCM_NEG}"
export TCM_WARMUP_STEPS=0

export NUM_GPUS=1
export PER_GPU_BATCH=1536
export BATCH_SIZE=1536
export GRAD_CACHE_CHUNK_SIZE=0
export EPOCHS=1
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export TRAIN_LIMIT=1
export MASTER_PORT=$((30020 + TASK_ID))
export DATA_TAG="3variant_gsv2full_gsfix_mfa"
export TASK_TAG="eval"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:aries-1gpu"
export BASELINE_RUN_IDS="ly6sc2mr fma3wmh2 x68wsne9 d7ij3to1"
export SELECT_CLEAN_GPUS=true

export ACL_DEV_JSONL=""
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json"
export EVAL_GLOSSARY_SIZES="1000000"
export BEST_METRIC="eval_dev/recall@10_gs1000000"
export BEST_METRIC_SECONDARY="eval_dev/topk10_filtered_recall@tau_0p80_gs1000000"
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="0.85 0.80 0.75 0.70"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
