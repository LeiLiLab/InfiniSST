#!/bin/bash
#SBATCH --job-name=q3_hn4096_l4c5e5
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=36:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn4096_l4c5e5_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn4096_l4c5e5_%x.err

# Continue from l4i457ih best_acl6060_gs10000 with a stable constant LR.

set -euo pipefail

export VARIANT_TAG="hn4096_l4best_c5e5_aries8"
export VERSION="3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k4096_tcmoff_l4best_c5e5_resume3000_aries8"
export WANDB_EXP_NAME="variantE_hnps_k4096_tcmoff_l4best_c5e5_resume3000_aries8"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hnps_k4096_tcmoff_resume3000_from_l4_best_const5e5_aries.md"

export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs6k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k4096_tcmoff_resume3000_aries8_smoke3000_best_acl6060_gs10000.pt"
export CONSTANT_LR=0.00005

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=4096
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_WARMUP_STEPS=0

export NUM_GPUS=8
export PER_GPU_BATCH=768
export GRAD_CACHE_CHUNK_SIZE=256
export EPOCHS=4
export MAX_STEPS=3000
export MAX_TRAIN_SECONDS=0
export MASTER_PORT=29983
export EXTRA_WANDB_TAGS="variant:hn4096_l4best_c5e5_aries8 compute:aries-8gpu"
export BASELINE_RUN_IDS="iaiyi1m8 l4i457ih"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
