#!/bin/bash
#SBATCH --job-name=q3_hn4096_res900
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=240G
#SBATCH --gres=gpu:6
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn4096_res900_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn4096_res900_%x.err

# Continue iaiyi1m8 (k=4096, TCM off) on Taurus 6 GPU before any TCM curriculum.

set -euo pipefail

export VARIANT_TAG="hn4096_resume900_tcmoff_taurus6"
export VERSION="3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k4096_tcmoff_resume900_taurus6"
export WANDB_EXP_NAME="variantE_hnps_k4096_tcmoff_resume900_taurus6"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hnps_k4096_tcmoff_resume900_taurus.md"

export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs6k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k4096_tcmoff_bs6k_smallest_dense_normAGGR_8gpu_scout_smoke400.pt"
export CONSTANT_LR=0.0001
export RESET_SCHEDULER=false

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=4096
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_WARMUP_STEPS=0

export NUM_GPUS=6
export PER_GPU_BATCH=768
export GRAD_CACHE_CHUNK_SIZE=256
export EPOCHS=2
export MAX_STEPS=900
export MAX_TRAIN_SECONDS=28800
export MASTER_PORT=29980
export EXTRA_WANDB_TAGS="variant:hn4096_resume900_tcmoff_taurus6 compute:taurus-6gpu"
export BASELINE_RUN_IDS="iaiyi1m8"
export SELECT_CLEAN_GPUS=true

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
