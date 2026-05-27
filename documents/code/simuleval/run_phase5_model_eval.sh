#!/usr/bin/env bash
set -euo pipefail

# Phase 5: run per-paper eval for a single Speech LLM model under test.
#
# Used by phase456_orchestrator.sh. Reuses run_one_density_eval.sh with
# per-paper glossary, lm=1, stride=1.92 (no sliding window, clean A/B vs the
# d5_cap baseline eval), and a DENSITY_TAG that is unique per model variant
# so the output directories do not collide with existing d5_*_k*_per_paper
# runs.
#
# Required env:
#   DENSITY_TAG     e.g. "5_cap" or "5_cap_adv"
#   MODEL_NAME      absolute path to the HF checkpoint directory
#
# Optional env (defaults below):
#   OUTPUT_BASE     default /mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed
#   GPUS            default "7,5,6"  (taurus: 5/6/7 free, 0-4 shared)
#   RAG_TOP_K       default 10
#   LATENCY_MULTIPLIER  default 1
#   RAG_RETRIEVE_STRIDE_SEC  default 1.92  (no overlap; matches Phase 0.5)
#   VLLM_DISABLE_CUSTOM_ALL_REDUCE  default 1 (taurus P2P workaround)
#
# All user-facing strings are in English.

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
RUN_SCRIPT="${ROOT_DIR}/documents/code/simuleval/run_one_density_eval.sh"

RAG_MODEL="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"

OUTPUT_BASE_DEFAULT="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed"
GPUS_DEFAULT="7,5,6"
RAG_TOP_K_DEFAULT="10"
LATENCY_MULTIPLIER_DEFAULT="1"
RAG_RETRIEVE_STRIDE_SEC_DEFAULT="1.92"
VLLM_DISABLE_CUSTOM_ALL_REDUCE_DEFAULT="1"
# ======Configuration=====

DENSITY_TAG="${DENSITY_TAG:?DENSITY_TAG is required}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"

OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_BASE_DEFAULT}}"
GPUS="${GPUS:-${GPUS_DEFAULT}}"
RAG_TOP_K="${RAG_TOP_K:-${RAG_TOP_K_DEFAULT}}"
LATENCY_MULTIPLIER="${LATENCY_MULTIPLIER:-${LATENCY_MULTIPLIER_DEFAULT}}"
RAG_RETRIEVE_STRIDE_SEC="${RAG_RETRIEVE_STRIDE_SEC:-${RAG_RETRIEVE_STRIDE_SEC_DEFAULT}}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-${VLLM_DISABLE_CUSTOM_ALL_REDUCE_DEFAULT}}"

assert_dir() {
  local d="$1"
  local tag="$2"
  if [[ ! -d "${d}" ]]; then
    echo "[ERROR] ${tag} directory not found: ${d}" >&2
    exit 2
  fi
}
assert_file() {
  local f="$1"
  local tag="$2"
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] ${tag} file not found: ${f}" >&2
    exit 2
  fi
}

assert_dir "${MODEL_NAME}" "HF model"
assert_file "${RAG_MODEL}" "RAG model"
assert_file "${RUN_SCRIPT}" "run_one_density_eval.sh"

echo "[INFO] ============================================================"
echo "[INFO] Phase 5 per-paper eval (single model)"
echo "[INFO] DENSITY_TAG=${DENSITY_TAG}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] GPUS=${GPUS} RAG_TOP_K=${RAG_TOP_K} LM=${LATENCY_MULTIPLIER}"
echo "[INFO] RAG_RETRIEVE_STRIDE_SEC=${RAG_RETRIEVE_STRIDE_SEC}"
echo "[INFO] VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE}"
echo "[INFO] ============================================================"

export DENSITY="${DENSITY_TAG}"
export MODEL_NAME
export RAG_TOP_K
export RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL}"
export OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}"
export GPU_SLOT_OVERRIDE="${GPUS}"
export LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIER}"
export SKIP_PHASE1_TAGGED="1"
export RAG_RETRIEVE_STRIDE_SEC_OVERRIDE="${RAG_RETRIEVE_STRIDE_SEC}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE

# Activate spaCyEnv so the Phase 3 offline eval in run_one_density_eval.sh can
# find simuleval/sacrebleu. run_one_density_eval.sh invokes python3 directly
# for the combine+extracted_by_paper stage; without this, it picks up the base
# miniconda python and fails at stream_laal_term.py import.
SPACY_ENV_BIN="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin"
if [[ -x "${SPACY_ENV_BIN}/python3" ]]; then
  export PATH="${SPACY_ENV_BIN}:${PATH}"
  export CONDA_PREFIX="${SPACY_ENV_BIN%/bin}"
  export CONDA_DEFAULT_ENV="spaCyEnv"
  echo "[INFO] spaCyEnv prepended to PATH: ${SPACY_ENV_BIN}"
else
  echo "[ERROR] spaCyEnv not found at ${SPACY_ENV_BIN}" >&2
  exit 2
fi

bash "${RUN_SCRIPT}"
