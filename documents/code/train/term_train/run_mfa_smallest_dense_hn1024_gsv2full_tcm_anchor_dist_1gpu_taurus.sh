#!/bin/bash
#SBATCH --job-name=q3_tcm_anchor_dist
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_q3_tcm_anchor_dist_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_q3_tcm_anchor_dist_%x.err

# Eval-only TCM-off score distribution dump for general unseen P31 banks.

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

case "${TASK_ID}" in
  0)
    BANK_TAG="10k"
    GS="10000"
    GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
    ;;
  1)
    BANK_TAG="100k"
    GS="100000"
    GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json"
    ;;
  *)
    echo "[FATAL] unsupported TASK_ID=${TASK_ID}" >&2
    exit 1
    ;;
esac

RUN_STAMP="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}_${BANK_TAG}"
export VARIANT_TAG="tcmanc_dist${BANK_TAG}"
export VERSION="3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_${VARIANT_TAG}_bs1gpu_smallest_dense_normAGGR_taurus"
export WANDB_EXP_NAME="variantE_hn1024_gsv2full_${VARIANT_TAG}_1gpu_taurus"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_tcm_anchor_dist_eval.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_r3auto1m_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt"

export EVAL_ONLY=true
export TASK_TAG="eval"
export NUM_GPUS=1
export PER_GPU_BATCH=512
export GRAD_CACHE_CHUNK_SIZE=256
export EPOCHS=1
export SCHEDULER_EPOCHS=1
export RESET_SCHEDULER=false
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export MASTER_PORT=$((29940 + TASK_ID))

export DATA_TAG="3variant_gsv2full_gsfix_mfa"
export EXPERIMENT_FAMILY="sst_tcm_anchor"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:taurus-1gpu"
export BASELINE_RUN_IDS="us4obwe3 pacexfw7 la8vkt47 k6e9askw"
export SELECT_CLEAN_GPUS=true

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=0
export NEG_BANK_SIZE=0
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_POS_THRESHOLD=0.80
export TCM_NEG_THRESHOLD=0.60
export TCM_WARMUP_STEPS=0

export ACL_DEV_JSONL=""
export EVAL_WIKI_GLOSSARY="${GLOSSARY}"
export EVAL_GLOSSARY_SIZES="${GS}"
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/recall@10_gs${GS}"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS=""
export DUMP_SIM_DISTRIBUTIONS="/mnt/gemini/home/jiaxuanluo/tcm_anchor_distributions/${RUN_STAMP}"
export RUN_VERDICT="Eval-only score distribution dump for TCM anchor tuning (${BANK_TAG} general unseen P31 bank)."

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"

