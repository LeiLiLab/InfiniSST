#!/bin/bash
set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/launchers/2026/05/20260522__varctx_lmlb_v3_hn512_gc256_taurus6.sh"

export PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin:${PATH}"

export RUN_STAMP="${RUN_STAMP:-hn512_gc256_resume640_aries8_fast_latest50_20260524T2235Z}"
export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-8192}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-8192}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-256}"
export HARD_NEG_K="${HARD_NEG_K:-0}"
export HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-512}"

export RESUME="${RESUME:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu234567_taurus_latest.pt}"
export RESET_SCHEDULER="${RESET_SCHEDULER:-false}"
export RESET_BEST_ON_RESUME="${RESET_BEST_ON_RESUME:-false}"

export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export WAIT_FOR_CLEAN_GPUS="${WAIT_FOR_CLEAN_GPUS:-true}"
export GPU_CLEAN_THRESHOLD_MIB="${GPU_CLEAN_THRESHOLD_MIB:-500}"
export GPU_WAIT_INTERVAL_SEC="${GPU_WAIT_INTERVAL_SEC:-60}"
export GPU_WAIT_TIMEOUT_SEC="${GPU_WAIT_TIMEOUT_SEC:-172800}"
export MASTER_PORT="${MASTER_PORT:-29999}"
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/tmp/hn512_aries8_jiaxuanluo_20260524T2235Z}"

export EVAL_STEPS_SAMPLE="${EVAL_STEPS_SAMPLE:-100}"
export SAVE_LATEST_STEPS="${SAVE_LATEST_STEPS:-50}"
export SAVE_LATEST_ON_EVAL="${SAVE_LATEST_ON_EVAL:-true}"
export SAVE_STEPS="${SAVE_STEPS:-999999}"
export KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"
export NUM_WORKERS="${NUM_WORKERS:-0}"

export VARIANT_TAG="${VARIANT_TAG:-hn512_varctx576_v3_gc256_aries8_fast_latest50}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_bs8192_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu01234567_aries_fastlatest50}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260524__varctx_lmlb_v3_hn512_gc256_resume640_aries8_fast_latest50.md}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn512_gsv2full_gsdedup_varctx576_v3_bs8192_gc256_resume640_aries8_fast_latest50_20260524T2235Z}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn512_varctx576_v3 compute:aries-8gpu ablation:hard_neg512 source:lh1b88kw resume_of:bkcnqlg9 resume_after:bkcnqlg9 gradcache:256 eval_steps:100 latest_steps:50 gpu:01234567}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw 5fwrs7rh gasqw118 gsjheh6r yp0rmgrl bkcnqlg9 e981df6j 40fgbr2y bgz7akb6 ah9u1bao dxwrgbln}"
export RUN_VERDICT="${RUN_VERDICT:-HN512 fast 8-GPU Aries resume from bkcnqlg9 step640 latest; eval every 100 steps; overwrite latest every 50 train steps; preserve scheduler and best trackers.}"

cd "${REPO_ROOT}"
exec bash "${BASE_LAUNCHER}"
