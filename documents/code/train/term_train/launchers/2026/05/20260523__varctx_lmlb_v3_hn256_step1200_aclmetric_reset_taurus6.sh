#!/bin/bash
#SBATCH --job-name=q3_hn256_s1200_aclreset_t6
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:6
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn256_s1200_aclreset_t6_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn256_s1200_aclreset_t6_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh"
RESUME_CKPT="/mnt/gemini/home/jiaxuanluo/train_outputs/lrdx14pm_hn256_bestsec_acl6060r10_0p9924_step1200_tie1280_frozen_20260523.pt"

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[ERROR] resume checkpoint missing: ${RESUME_CKPT}" >&2
    exit 2
fi

export RESUME="${RESUME:-${RESUME_CKPT}}"
export RESET_BEST_ON_RESUME="${RESET_BEST_ON_RESUME:-true}"
export RESET_SCHEDULER="${RESET_SCHEDULER:-false}"
export HARD_NEG_K="${HARD_NEG_K:-0}"
export HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-256}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-256}"

# GPU 4/5 were not clean at launch time; keep this wrapper resumable with a clean
# six-GPU default while preserving the HN256 objective and near-8k global batch.
export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,6,7}"
export NUM_GPUS="${NUM_GPUS:-6}"
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-8190}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1365}"
export BATCH_SIZE="${BATCH_SIZE:-8190}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_STEPS="${MAX_STEPS:-0}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export MASTER_PORT="${MASTER_PORT:-29996}"

export SAVE_LATEST_ON_EVAL="${SAVE_LATEST_ON_EVAL:-true}"
export EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-0}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.75}"
export BEST_METRIC="${BEST_METRIC:-eval_acl6060/top1}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-eval_acl6060/recall@10_gs10000}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS:-2}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/dev/shm/q3_hn256_s1200_aclreset_t6_${USER}_${RUN_STAMP}}"
export WANDB_DIR="${WANDB_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/aries/data4/jiaxuanluo/cache/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/aries/data4/jiaxuanluo/cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/mnt/aries/data4/jiaxuanluo/config}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
mkdir -p "${LOCAL_TMP_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}"

export VARIANT_TAG="${VARIANT_TAG:-hn256_varctx576_v3_gc${GRAD_CACHE_CHUNK_SIZE}_step1200_aclmetric_reset_gpu012367_tcmoff_ep6}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_bs8190_gc${GRAD_CACHE_CHUNK_SIZE}_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep${EPOCHS}_v3_smallest_dense_normAGGR_gpu012367_taurus_step1200_aclmetric_reset}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn256_gsv2full_gsdedup_varctx576_v3_bs8190_gc${GRAD_CACHE_CHUNK_SIZE}_step1200_aclmetric_reset_taurus6}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260523__varctx_lmlb_v3_hn256_step1200_aclmetric_reset_taurus6.md}"

export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn256_varctx576_v3 compute:taurus-6gpu ablation:hard_neg256 source:lh1b88kw resume_of:lrdx14pm resume_ckpt:step1200_frozen metric_reset:acl_top1_acl_gs10000}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lrdx14pm e981df6j lh1b88kw 5fwrs7rh 40fgbr2y bgz7akb6 ah9u1bao dxwrgbln}"
export RUN_VERDICT="${RUN_VERDICT:-Resume HN256 from frozen lrdx14pm step-1200 checkpoint, reset best trackers, and use user-requested ACL readout metrics: primary eval_acl6060/top1, secondary eval_acl6060/recall@10_gs10000.}"

source "${BASE_LAUNCHER}"
