#!/bin/bash
#SBATCH --job-name=q3_varctx_nohn_taurus8
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/gemini/home/jiaxuanluo/logs/%j_q3_varctx_nohn_taurus8_%x.out
#SBATCH --error=/mnt/gemini/home/jiaxuanluo/logs/%j_q3_varctx_nohn_taurus8_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh"

export HARD_NEG_K="${HARD_NEG_K:-0}"
export HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-0}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-1024}"
export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}"
CUDA_DEVICE_COUNT="$(python3 - "${CUDA_DEVICE_LIST}" <<'PYEOF'
import re
import sys

parts = [p.strip() for p in re.split(r"[,\s]+", sys.argv[1]) if p.strip()]
if not parts:
    raise SystemExit("CUDA_DEVICE_LIST is empty")
print(len(parts))
PYEOF
)"
CUDA_DEVICE_TAG="$(tr -cd '0-9' <<< "${CUDA_DEVICE_LIST}")"
export NUM_GPUS="${NUM_GPUS:-${CUDA_DEVICE_COUNT}}"
# Use the exact 8k global batch when CUDA_DEVICE_LIST has 8 GPUs. For other GPU
# counts, use the closest no-over equal per-rank batch.
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-8192}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-$((TARGET_GLOBAL_BATCH / NUM_GPUS))}"
export BATCH_SIZE="${BATCH_SIZE:-$((PER_GPU_BATCH * NUM_GPUS))}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_STEPS="${MAX_STEPS:-0}"

export VARIANT_TAG="${VARIANT_TAG:-nohn_varctx576_v3_gc${GRAD_CACHE_CHUNK_SIZE}_gpu${CUDA_DEVICE_TAG}_tcmoff_ep6}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_bs${BATCH_SIZE}_gc${GRAD_CACHE_CHUNK_SIZE}_wr1000k_m0.0_maxsim_mfa_variantE_nohn_tcmoff_ep${EPOCHS}_v3_smallest_dense_normAGGR_gpu${CUDA_DEVICE_TAG}_aries}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_nohn_gsv2full_gsdedup_varctx576_v3_bs${BATCH_SIZE}_gc${GRAD_CACHE_CHUNK_SIZE}_tcmoff_ep${EPOCHS}_gpu${CUDA_DEVICE_TAG}_aries}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260519__varctx_lmlb_v3_nohn_gc1024_aries8.md}"

export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:nohn_varctx576_v3 compute:taurus-8gpu ablation:hard_neg_off source:lh1b88kw}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw ah9u1bao dxwrgbln}"
export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs10000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-eval_acl6060/recall@10}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS:-2}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json}"
export RUN_VERDICT="${RUN_VERDICT:-No-HN ablation from lh1b88kw recipe: hard_neg_k_per_sample=0, grad_cache_chunk_size=${GRAD_CACHE_CHUNK_SIZE}, target global batch 8192 with effective equal-rank batch ${BATCH_SIZE}; primary best metric is eval_dev/recall@10_gs10000 and secondary best metric is eval_acl6060/recall@10.}"

source "${BASE_LAUNCHER}"
