#!/bin/bash
set -euo pipefail

# Resume the GigaSpeech-only HN1024 varctx ablation from the primary best
# checkpoint saved by the 4-GPU Aries run g49qabuf, but occupy all 8 Aries GPUs.

export RUN_STAMP="${RUN_STAMP:-20260525T1902_gsonly_hn1024_varctx576_resume_best_aries8}"
export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export WAIT_FOR_CUDA_DEVICES="${WAIT_FOR_CUDA_DEVICES:-true}"
export WAIT_FOR_CUDA_TIMEOUT_SEC="${WAIT_FOR_CUDA_TIMEOUT_SEC:-43200}"
export WAIT_FOR_CUDA_POLL_SEC="${WAIT_FOR_CUDA_POLL_SEC:-60}"

export RESUME="${RESUME:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_gsonly_bs8192_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_4gpu_aries_best.pt}"
export VARIANT_TAG="${VARIANT_TAG:-hn1024_varctx576_gsonly_resume_best_8gpu}"
export DATA_TAG="${DATA_TAG:-gsv2full_gsdedup_varctx576_gsonly}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_gsonly_bs8192_gc128_resume_best_g49qabuf_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn1024_gsonly_varctx576_resume_best_g49qabuf_aries8_${RUN_STAMP}}"
export NOTES_FILE="${NOTES_FILE:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes/2026/05/20260525__varctx576_hn1024_gsonly_resume_best_aries8.md}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn1024_gsonly_resume_best compute:aries-8gpu resume:from-best-g49qabuf ablation:data-gsonly}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-g49qabuf lh1b88kw ah9u1bao dxwrgbln}"

export PER_GPU_BATCH="${PER_GPU_BATCH:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-8192}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-128}"
export SAVE_LATEST_STEPS="${SAVE_LATEST_STEPS:-50}"
export MASTER_PORT="${MASTER_PORT:-20158}"
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/tmp/jx_gsonly8_${RUN_STAMP}}"

if [ ! -f "${RESUME}" ]; then
  echo "[ERROR] resume checkpoint missing: ${RESUME}" >&2
  exit 2
fi

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/launchers/2026/05/20260525__varctx576_hn1024_gsonly_tcmoff_ep6_aries4_wait.sh"
