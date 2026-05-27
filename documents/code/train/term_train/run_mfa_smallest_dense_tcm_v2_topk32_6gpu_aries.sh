#!/bin/bash
#SBATCH --job-name=q3_mfa_tcmv2_topk32
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:6
#SBATCH --time=28:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_mfa_tcmv2_topk32_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_mfa_tcmv2_topk32_%x.err

set -euo pipefail

TCM_NEG_THRESHOLD="0.78"
TAU_STAR_TAG="tau0p80"

NEG_TAG="${TCM_NEG_THRESHOLD//./p}"
VARIANT_TAG="hnps_k512_tcmv2_topk32_smallest_dense_normAGGR_6gpu"
VERSION="3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k512_tcmv2_topk32_${TAU_STAR_TAG}_neg${NEG_TAG}_pos0p74_ep3_cold_smallest_dense_normAGGR_6gpu"
WANDB_EXP_NAME="variantE_hnps_k512_tcmv2_topk32_${TAU_STAR_TAG}_neg${NEG_TAG}_smallest_dense_normAGGR_6gpu"
NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_tcm_v2_e1_topk.md"

TCM_LOSS_WEIGHT="0.0"
TCM_POS_LOSS_WEIGHT="0.25"
TCM_NEG_LOSS_WEIGHT="1.0"
TCM_POS_THRESHOLD="0.74"
TCM_LOSS_FORM="hinge"
TCM_REDUCTION="mean_viol"
TCM_NEG_SCOPE="topk"
TCM_NEG_TOPK="32"
TCM_WARMUP_STEPS="100"

EXTRA_WANDB_TAGS="variant:hnps_k512_tcmv2_topk32_smallest_dense_normAGGR_6gpu compute:aries-6gpu"
BASELINE_RUN_IDS="tys70s0y"
MASTER_PORT="29966"

export VARIANT_TAG VERSION WANDB_EXP_NAME NOTES_FILE
export TCM_LOSS_WEIGHT TCM_POS_LOSS_WEIGHT TCM_NEG_LOSS_WEIGHT
export TCM_POS_THRESHOLD TCM_NEG_THRESHOLD TCM_LOSS_FORM TCM_REDUCTION
export TCM_NEG_SCOPE TCM_NEG_TOPK TCM_WARMUP_STEPS
export EXTRA_WANDB_TAGS BASELINE_RUN_IDS MASTER_PORT

bash "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_tcm_v2_common_6gpu_aries.sh"
