#!/bin/bash
#SBATCH --job-name=q3_hn4096_p020ow
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=240G
#SBATCH --gres=gpu:6
#SBATCH --time=07:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn4096_p020ow_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn4096_p020ow_%x.err

# Speaker-diversity scout with old clean wiki supplement. Same global batch and
# k=4096 TCM-off recipe as the partial GSV2 run, but active wiki count is
# restored to the iaiyi1m8 baseline.

set -euo pipefail

export VARIANT_TAG="hn4096_gsv2p020_oldwiki_fbmax"
export VERSION="3var_gsv2p020_oldwiki_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn4096_fbmax_bs6k_smallest_dense_normAGGR_6gpu_taurus_scout"
export WANDB_EXP_NAME="variantE_hn4096_gsv2p020_oldwiki_fbmax_6gpu_taurus_scout"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn4096_gsv2p020_oldwiki_tcmoff_scout.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_partial0_20_oldwiki_clean_mfa.jsonl"

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
export MASTER_PORT=29981
export DATA_TAG="3variant_gsv2p020_oldwiki_mfa"
export EXTRA_WANDB_TAGS="variant:hn4096_gsv2p020_oldwiki_fbmax compute:taurus-6gpu"
export BASELINE_RUN_IDS="iaiyi1m8,yx52spnl"
export SELECT_CLEAN_GPUS=true

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
