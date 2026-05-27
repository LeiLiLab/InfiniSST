#!/bin/bash
#SBATCH --job-name=q3_hn1024_eval1m
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_eval1m_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_eval1m_%x.err

# One-shot 1M fullbank eval for the resumed TCM-off baseline best checkpoint.
# Keep this outside DDP training to avoid NCCL timeout while rank 0 evaluates.

set -euo pipefail

export VARIANT_TAG="${VARIANT_TAG:-hn1024_gsv2full_r3best_eval1m}"
export VERSION="${VERSION:-3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_r3best_eval1m_bs1gpu_smallest_dense_normAGGR_taurus}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn1024_gsv2full_tcmoff_r3best_eval1m_1gpu_taurus}"
export NOTES_FILE="${NOTES_FILE:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_tcmoff_r3best_1m_eval_aries.md}"
export TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl}"
export RESUME="${RESUME:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_r3trk1m_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt}"

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
export MASTER_PORT=29990
export DATA_TAG="3variant_gsv2full_gsfix_mfa"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn1024_gsv2full_r3best_eval1m compute:taurus-1gpu}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-058tdx9a k6e9askw ly6sc2mr fma3wmh2 x68wsne9}"
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
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json"
export EVAL_GLOSSARY_SIZES="1000000"
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/recall@10_gs1000000"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
# Disable threshold sweep diagnostics for 1M eval to avoid all-query x fullbank materialization.
export TCM_SWEEP_THRESHOLDS=""
export RUN_VERDICT="${RUN_VERDICT:-Eval-only 1M fullbank recall for the resumed TCM-off baseline best checkpoint.}"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
