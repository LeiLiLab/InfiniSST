#!/bin/bash
#SBATCH --job-name=q3_ntcm_t75best
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ntcm_t75best_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ntcm_t75best_%x.err

set -euo pipefail

RESUME_STEP="${RESUME_STEP:-3500}"
TARGET_MAX_STEPS="${TARGET_MAX_STEPS:-6000}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1536}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="${VARIANT_TAG:-t75best_p1n4}"
export VERSION="3var_gsv2full_gsfix_mfa_ntcm_t75best_v3_${VARIANT_TAG}_from${RESUME_STEP}_${COMPUTE_TAG}_smallest_dense"
export WANDB_EXP_NAME="ntcm_t75best_v3_${VARIANT_TAG}_from${RESUME_STEP}_${COMPUTE_TAG}"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_noterm_tcm_tau075_besttrack_resume3500_v3.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
export DEV_JSONL="/mnt/gemini/home/jiaxuanluo/noterm_dev/term_dev_with_gs_noterm_wiki_v3_balanced.jsonl"
export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_ntcm_final_v3_n64_p1n4_s2000_aries-8gpu_smallest_dense_smoke4650_best_eval_dev_full_recallat10_gs100000.pt"

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=1024
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=1
export TCM_NEG_LOSS_WEIGHT=4
export TCM_POS_THRESHOLD=0.8500
export TCM_NEG_THRESHOLD=0.6400
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
export RESET_BEST_ON_RESUME=true
export RESUME_COSINE_DECAY_TO_MAX_STEPS=true
export MASTER_PORT="${MASTER_PORT:-30350}"

export DATA_TAG="3variant_gsv2full_gsfix_mfa_devv3"
export EXPERIMENT_FAMILY="sst_tcm_noterm"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG}"
export BASELINE_RUN_IDS="tau6iuo3 us4obwe3 aamk3dok iie3967j"
export SELECT_CLEAN_GPUS=true

export ACL_DEV_JSONL=""
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
export EVAL_GLOSSARY_SIZES="10000"
export FULL_EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json"
export FULL_EVAL_GLOSSARY_SIZES="100000"
export FULL_EVAL_EVERY_N_EVALS=10
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/topk10_filtered_recall@tau_0p75_gs10000"
export BEST_METRIC_SECONDARY="eval_dev/topk10_filtered_precision_micro@tau_0p75_gs10000"
export EVAL_STEPS_SAMPLE=50
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="0.7500"
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="Tau0.75 best-track resume from step ${RESUME_STEP}: track filtered recall and micro precision at gs10000."

echo "[NTCM_T75BEST] resume_step=${RESUME_STEP} target_max_steps=${TARGET_MAX_STEPS} compute=${COMPUTE_TAG}"
echo "[NTCM_T75BEST] primary=${BEST_METRIC}"
echo "[NTCM_T75BEST] secondary=${BEST_METRIC_SECONDARY}"
echo "[NTCM_T75BEST] resume=${RESUME}"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
