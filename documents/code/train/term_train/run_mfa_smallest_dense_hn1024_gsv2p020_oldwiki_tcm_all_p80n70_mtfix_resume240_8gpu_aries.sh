#!/bin/bash
#SBATCH --job-name=q3_hn1024_mtfix
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=1-12:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_mtfix_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_mtfix_%x.err

# Continue the GSV2p020 oldwiki all-scope TCM run from its best gs10000
# checkpoint, using the multi-term chunk HN/MFA masking fix.

set -euo pipefail

export VARIANT_TAG="hn1024_p020ow_tcmall_mtfix_r240"
export VERSION="3var_gsv2p020_oldwiki_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmall_p80n70_mtfix_r240_bs12k_smallest_dense_normAGGR_8gpu_aries"
export WANDB_EXP_NAME="variantE_hn1024_gsv2p020_oldwiki_tcmall_p80n70_mtfix_resume240_8gpu_aries"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2p020_oldwiki_tcm_all_p80n70_mtfix_resume240_aries.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_partial0_20_oldwiki_clean_mfa.jsonl"
export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2p020_oldwiki_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmall_p80n70_bs12k_smallest_dense_normAGGR_8gpu_aries_smoke530_best_acl6060_gs10000.pt"

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=1024

export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.10
export TCM_NEG_LOSS_WEIGHT=0.50
export TCM_POS_THRESHOLD=0.80
export TCM_NEG_THRESHOLD=0.70
export TCM_LOSS_FORM="hinge"
export TCM_REDUCTION="mean_viol"
export TCM_NEG_SCOPE="all"
export TCM_NEG_TOPK=0
export TCM_WARMUP_STEPS=100

export NUM_GPUS=8
export PER_GPU_BATCH=1536
export GRAD_CACHE_CHUNK_SIZE=512
export EPOCHS=5
# Do not use max_steps: train the full 5-epoch wall budget. MAX_STEPS=0 means
# "unlimited" in the common launcher (the torchrun arg is omitted when 0).
export MAX_STEPS=0
# No Python-side max walltime; rely on the SLURM time limit only.
export MAX_TRAIN_SECONDS=0
export MASTER_PORT=29985
export DATA_TAG="3variant_gsv2p020_oldwiki_mfa"
export EXTRA_WANDB_TAGS="variant:hn1024_p020ow_tcmall_mtfix_r240 compute:aries-8gpu"
export BASELINE_RUN_IDS="fma3wmh2 yx52spnl tys70s0y d7ij3to1"
export SELECT_CLEAN_GPUS=true

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
