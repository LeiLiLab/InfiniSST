#!/bin/bash
#SBATCH --job-name=q3_hn256_latest_t8
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn256_latest_t8_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn256_latest_t8_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh"
RESUME_CKPT="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_aries_latest.pt"

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[ERROR] resume checkpoint missing: ${RESUME_CKPT}" >&2
    exit 2
fi

export RESUME="${RESUME:-${RESUME_CKPT}}"
export HARD_NEG_K="${HARD_NEG_K:-0}"
export HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-256}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-256}"
export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-8192}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-8192}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_STEPS="${MAX_STEPS:-0}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export MASTER_PORT="${MASTER_PORT:-29995}"

export SAVE_LATEST_ON_EVAL="${SAVE_LATEST_ON_EVAL:-true}"
export EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-0}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.75}"
export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs10000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-eval_acl6060/recall@10}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS:-2}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/dev/shm/q3_hn256_latest_t8_${USER}_${RUN_STAMP}}"
export WANDB_DIR="${WANDB_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/aries/data4/jiaxuanluo/cache/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/aries/data4/jiaxuanluo/cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/mnt/aries/data4/jiaxuanluo/config}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
mkdir -p "${LOCAL_TMP_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}"

export VARIANT_TAG="${VARIANT_TAG:-hn256_varctx576_v3_gc${GRAD_CACHE_CHUNK_SIZE}_resume_latest_gpu01234567_tcmoff_ep6}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_bs8192_gc${GRAD_CACHE_CHUNK_SIZE}_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep${EPOCHS}_v3_smallest_dense_normAGGR_gpu01234567_taurus_resume_latest}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn256_gsv2full_gsdedup_varctx576_v3_bs8192_gc${GRAD_CACHE_CHUNK_SIZE}_resume_latest_taurus8}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260522__varctx_lmlb_v3_hn256_gc256_resume_latest_taurus8.md}"

export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn256_varctx576_v3 compute:taurus-8gpu ablation:hard_neg256 source:lh1b88kw resume_of:e981df6j resume_ckpt:latest}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-e981df6j lh1b88kw 5fwrs7rh 40fgbr2y bgz7akb6 ah9u1bao dxwrgbln}"
export RUN_VERDICT="${RUN_VERDICT:-Resume HN256 ablation from e981df6j latest checkpoint step 800 on taurus 8GPU; hard_neg_k_per_sample=256, grad_cache_chunk_size=${GRAD_CACHE_CHUNK_SIZE}, exact global batch 8192, and overwrite-only latest checkpoint after every eval.}"

source "${BASE_LAUNCHER}"
