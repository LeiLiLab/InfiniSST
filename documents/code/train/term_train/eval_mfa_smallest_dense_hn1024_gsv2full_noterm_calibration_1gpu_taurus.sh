#!/bin/bash
#SBATCH --job-name=q3_noterm_calib
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_noterm_calib_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_noterm_calib_%x.err

# Eval-only no-term raw topK calibration dump for the step-2650 TCM-off baseline.

set -euo pipefail

RUN_STAMP="${SLURM_JOB_ID:-manual}"
export VARIANT_TAG="ntcm_calib10k_v2"
export VERSION="3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_noterm_calib10k_v2_bs1gpu_smallest_dense_normAGGR"
export WANDB_EXP_NAME="variantE_hn1024_gsv2full_noterm_calib10k_v2_1gpu"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_noterm_calibration_eval.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
export DEV_JSONL="/mnt/gemini/home/jiaxuanluo/noterm_dev/term_dev_with_gs_noterm_wiki_v2.jsonl"
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
export MASTER_PORT=29941

export DATA_TAG="3variant_gsv2full_gsfix_mfa"
export EXPERIMENT_FAMILY="sst_tcm_noterm"
export COMPUTE_TAG="${SLURM_JOB_PARTITION:-taurus}-1gpu"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG}"
export BASELINE_RUN_IDS="us4obwe3 pacexfw7 la8vkt47 rh7rl01w 0uupruft"
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
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
export EVAL_GLOSSARY_SIZES="10000"
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/recall@10_gs10000"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS=""
export DUMP_SIM_DISTRIBUTIONS="/mnt/gemini/home/jiaxuanluo/noterm_tcm_calibration/${RUN_STAMP}"
export RUN_VERDICT="Eval-only glossary-conditioned no-term raw topK calibration dump for TCM-off step-2650 baseline on dev v2."

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
