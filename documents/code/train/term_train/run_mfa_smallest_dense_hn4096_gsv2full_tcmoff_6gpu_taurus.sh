#!/bin/bash
#SBATCH --job-name=q3_hn4096_gsv2full
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=240G
#SBATCH --gres=gpu:6
#SBATCH --time=07:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn4096_gsv2full_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn4096_gsv2full_%x.err

# Full speaker-diverse GSV2 scout: same global-batch k=4096 TCM-off recipe as
# iaiyi1m8/yx52spnl, but with clean GSV2 wiki_synth shards 0-31.

set -euo pipefail

export VARIANT_TAG="hn4096_gsv2full_gsfix"
export VERSION="3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn4096_fbmax_bs6k_smallest_dense_normAGGR_6gpu_taurus_scout"
export WANDB_EXP_NAME="variantE_hn4096_gsv2full_gsfix_6gpu_taurus_scout"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn4096_gsv2full_tcmoff_scout.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=4096
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_WARMUP_STEPS=0
export NUM_GPUS=6
export PER_GPU_BATCH=1024
export GRAD_CACHE_CHUNK_SIZE=256
export EPOCHS=1
export MAX_STEPS=400
export MAX_TRAIN_SECONDS=21600
export MASTER_PORT=29982
export DATA_TAG="3variant_gsv2full_gsfix_mfa"
export EXTRA_WANDB_TAGS="variant:hn4096_gsv2full_gsfix compute:taurus-6gpu"
export BASELINE_RUN_IDS="iaiyi1m8,yx52spnl"
export SELECT_CLEAN_GPUS=true

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
