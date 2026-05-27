#!/bin/bash
#SBATCH --job-name=q3_ntcm_fs_n64p1n4
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=20:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ntcm_fs_n64p1n4_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ntcm_fs_n64p1n4_%x.err

set -euo pipefail

export NUM_GPUS="${NUM_GPUS:-8}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1536}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="fs_n64_p1n4"
export VERSION="3var_gsv2full_gsfix_mfa_ntcm_fromscratch_v3_n64_p1n4_${COMPUTE_TAG}_smallest_dense"
export WANDB_EXP_NAME="ntcm_fromscratch_v3_n64_p1n4_${COMPUTE_TAG}"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_noterm_tcm_fromscratch_n64_p1n4_v3.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
export DEV_JSONL="/mnt/gemini/home/jiaxuanluo/noterm_dev/term_dev_with_gs_noterm_wiki_v3_balanced.jsonl"
export RESUME=""

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=1024
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=1
export TCM_NEG_LOSS_WEIGHT=4
export TCM_POS_THRESHOLD=0.85
export TCM_NEG_THRESHOLD=0.64
export TCM_LOSS_FORM="hinge"
export TCM_REDUCTION="mean_viol"
export TCM_NEG_SCOPE="all"
export TCM_NEG_TOPK=0
export TCM_WARMUP_STEPS=0

export GRAD_CACHE_CHUNK_SIZE=512
export EPOCHS=20
export SCHEDULER_EPOCHS=20
export MAX_STEPS=4650
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false
export MASTER_PORT="${MASTER_PORT:-30360}"

export DATA_TAG="3variant_gsv2full_gsfix_mfa_devv3_fromscratch"
export EXPERIMENT_FAMILY="sst_tcm_noterm"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG}"
export BASELINE_RUN_IDS="us4obwe3 iie3967j z2jjzw4p tau6iuo3 j0e4un3k"
export SELECT_CLEAN_GPUS=true

export ACL_DEV_JSONL=""
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
export EVAL_GLOSSARY_SIZES="10000"
export FULL_EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json"
export FULL_EVAL_GLOSSARY_SIZES="100000"
export FULL_EVAL_EVERY_N_EVALS=10
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/recall@10_gs10000"
export BEST_METRIC_SECONDARY="eval_dev_full/recall@10_gs100000"
export EVAL_STEPS_SAMPLE=50
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="0.75"
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="From-scratch n64_p1n4 training: T_alpha=0.64, T_beta=0.85, pos_w=1, neg_w=4, 10k primary and 100k secondary eval."

echo "[NTCM_FS] variant=${VARIANT_TAG}"
echo "[NTCM_FS] TCM pos=${TCM_POS_THRESHOLD} neg=${TCM_NEG_THRESHOLD} weights pos=${TCM_POS_LOSS_WEIGHT} neg=${TCM_NEG_LOSS_WEIGHT} tau=${TCM_SWEEP_THRESHOLDS}"
echo "[NTCM_FS] max_steps=${MAX_STEPS} compute=${COMPUTE_TAG} batch=$((NUM_GPUS * PER_GPU_BATCH))"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
