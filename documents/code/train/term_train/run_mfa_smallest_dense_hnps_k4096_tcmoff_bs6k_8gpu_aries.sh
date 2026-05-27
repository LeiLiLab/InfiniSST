#!/bin/bash
#SBATCH --job-name=q3_hn4096_b6
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=07:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn4096_b6_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn4096_b6_%x.err

# HN-depth scout: k=4096, TCM off, half local batch to fit MaxSim memory.

set -euo pipefail

export VARIANT_TAG="hnps_k4096_bs6k_tcmoff_sd_8gpu"
export VERSION="3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k4096_tcmoff_bs6k_smallest_dense_normAGGR_8gpu_scout"
export WANDB_EXP_NAME="variantE_hnps_k4096_tcmoff_bs6k_smallest_dense_normAGGR_8gpu_scout"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hnps_k4096_tcmoff_bs6k_scout.md"

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=4096
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_WARMUP_STEPS=0
export NUM_GPUS=8
export PER_GPU_BATCH=768
export GRAD_CACHE_CHUNK_SIZE=256
export EPOCHS=1
export MAX_STEPS=400
export MAX_TRAIN_SECONDS=21600
export MASTER_PORT=29977
export EXTRA_WANDB_TAGS="variant:hnps_k4096_bs6k_tcmoff_sd_8gpu compute:aries-8gpu"
export BASELINE_RUN_IDS="6s3jr70q"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
