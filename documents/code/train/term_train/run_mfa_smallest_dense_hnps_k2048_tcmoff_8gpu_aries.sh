#!/bin/bash
#SBATCH --job-name=q3_hn2048_off
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=07:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn2048_off_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn2048_off_%x.err

# HN-depth scout: k=2048, TCM fully off, same smallest+dense recipe.

set -euo pipefail

export VARIANT_TAG="hnps_k2048_tcmoff_smallest_dense_normAGGR_8gpu_scout"
export VERSION="3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k2048_tcmoff_smallest_dense_normAGGR_8gpu_scout"
export WANDB_EXP_NAME="variantE_hnps_k2048_tcmoff_smallest_dense_normAGGR_8gpu_scout"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hnps_k2048_tcmoff_scout.md"

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=2048
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_WARMUP_STEPS=0
export NUM_GPUS=8
export PER_GPU_BATCH=1536
export GRAD_CACHE_CHUNK_SIZE=256
export EPOCHS=1
export MAX_STEPS=200
export MAX_TRAIN_SECONDS=18000
export MASTER_PORT=29976
export EXTRA_WANDB_TAGS="variant:hnps_k2048_tcmoff_smallest_dense_normAGGR_8gpu_scout compute:aries-8gpu"
export BASELINE_RUN_IDS="fma3wmh2"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
