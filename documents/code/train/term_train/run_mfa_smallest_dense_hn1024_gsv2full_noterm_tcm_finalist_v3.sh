#!/bin/bash
#SBATCH --job-name=q3_ntcm_final
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ntcm_final_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ntcm_final_%x.err

set -euo pipefail

BASELINE_STEP="${BASELINE_STEP:-2650}"
FINALIST_STEPS="${FINALIST_STEPS:-2000}"
TARGET_MAX_STEPS=$((BASELINE_STEP + FINALIST_STEPS))
export NUM_GPUS="${NUM_GPUS:-8}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1536}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="${VARIANT_TAG:-final_n64_p1n4}"
export VERSION="3var_gsv2full_gsfix_mfa_ntcm_final_v3_${VARIANT_TAG}_s${FINALIST_STEPS}_${COMPUTE_TAG}_smallest_dense"
export WANDB_EXP_NAME="ntcm_final_v3_${VARIANT_TAG}_s${FINALIST_STEPS}_${COMPUTE_TAG}"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_noterm_tcm_finalists_v3.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
export DEV_JSONL="/mnt/gemini/home/jiaxuanluo/noterm_dev/term_dev_with_gs_noterm_wiki_v3_balanced.jsonl"
export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_r3auto1m_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt"
