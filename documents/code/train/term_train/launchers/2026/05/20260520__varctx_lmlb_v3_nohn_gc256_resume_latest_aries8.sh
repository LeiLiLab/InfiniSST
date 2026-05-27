#!/bin/bash
#SBATCH --job-name=q3_nohn_resume_latest_a8
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_nohn_resume_latest_a8_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_nohn_resume_latest_a8_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
NOHN_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/launchers/2026/05/20260519__varctx_lmlb_v3_nohn_gc1024_aries8.sh"
RESUME_CKPT="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8192_gc256_wr1000k_m0.0_maxsim_mfa_variantE_nohn_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu01234567_aries_best.pt"

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[ERROR] resume checkpoint missing: ${RESUME_CKPT}" >&2
    exit 2
fi

export RESUME="${RESUME:-${RESUME_CKPT}}"
export HARD_NEG_K="${HARD_NEG_K:-0}"
export HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-0}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-256}"
export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-8192}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-8192}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-false}"
export MASTER_PORT="${MASTER_PORT:-29993}"

export SAVE_LATEST_ON_EVAL="${SAVE_LATEST_ON_EVAL:-true}"
export EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-0}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.75}"
export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs10000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-eval_acl6060/recall@10}"

export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260520__varctx_lmlb_v3_nohn_gc256_resume_latest_aries8.md}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_nohn_gsv2full_gsdedup_varctx576_v3_bs8192_gc256_resume_latest_aries8}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:nohn_varctx576_v3 compute:aries-8gpu ablation:hard_neg_off source:lh1b88kw resume_of:5dtpt842}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-5dtpt842 lh1b88kw ah9u1bao dxwrgbln}"
export RUN_VERDICT="${RUN_VERDICT:-Resume no-HN ablation from W&B run 5dtpt842 step-240 best checkpoint on aries 8GPU; writes an overwrite-only latest checkpoint after every eval for intermittent resume.}"

source "${NOHN_LAUNCHER}"
