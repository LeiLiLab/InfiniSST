#!/bin/bash
#SBATCH --job-name=q3_hn8192_b23
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=240G
#SBATCH --gres=gpu:6
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn8192_b23_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn8192_b23_%x.err

# Final HN-size scout: k=8192, TCM off, 6-GPU Taurus.

set -euo pipefail

export VARIANT_TAG="hnps_k8192_bs2304_tcmoff_sd_6gpu"
export VERSION="3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k8192_tcmoff_bs2304_smallest_dense_normAGGR_6gpu_scout"
export WANDB_EXP_NAME="variantE_hnps_k8192_tcmoff_bs2304_smallest_dense_normAGGR_6gpu_scout"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hnps_k8192_tcmoff_bs2304_taurus_scout.md"

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=8192
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_WARMUP_STEPS=0
export NUM_GPUS=6
export PER_GPU_BATCH=384
export GRAD_CACHE_CHUNK_SIZE=128
export EPOCHS=1
export MAX_STEPS=1080
export MAX_TRAIN_SECONDS=36000
export MASTER_PORT=29978
export EXTRA_WANDB_TAGS="variant:hnps_k8192_bs2304_tcmoff_sd_6gpu compute:taurus-6gpu"
export BASELINE_RUN_IDS="6s3jr70q iaiyi1m8"
export SELECT_CLEAN_GPUS=true

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
