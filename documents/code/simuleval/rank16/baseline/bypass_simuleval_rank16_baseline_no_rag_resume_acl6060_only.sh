#!/usr/bin/env bash
set -euo pipefail

# Resume-only runner (baseline, NO RAG) for ACL6060 glossary.
#
# - Uses the baseline no-rag sweep script, so retrieval/term_map injection will NOT run.
# - Skips completed runs (non-empty instances.log), re-runs only missing/empty ones.
#
# All user-facing strings are in English.

# ======Configuration=====
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

ROOT_DIR="/home/jiaxuanluo/InfiniSST"
BASE_SCRIPT="${ROOT_DIR}/documents/code/simuleval/rank16/baseline/bypass_simuleval_rank32_iter_0000452_hf_baseline_no_rag_sweep.sh"

# Only run the ACL6060 glossary (used for output naming / bookkeeping).
ACL6060_GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"

# Baseline model (override required for your use-case)
MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE:-}"

# Output base (recommend a baseline-tagged directory)
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rank16_baseline_no_rag_acl6060_only}"

# Single setting by default; use "${VAR-...}" so empty string means "no override"
LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIERS_OVERRIDE-2}"
RAG_K2_VALUES_OVERRIDE="${RAG_K2_VALUES_OVERRIDE-10}"  # naming compatibility

# GPU override (physical ids)
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-}"

# Language override (optional)
LANG_CODE_OVERRIDE="${LANG_CODE_OVERRIDE:-}"
TARGET_LANG_OVERRIDE="${TARGET_LANG_OVERRIDE:-}"

# Determinism / decode overrides (pass-through to base script)
DEFAULT_GEN_SEED="998244353"
SEED_OVERRIDE="${SEED_OVERRIDE:-}"
TEMPERATURE_OVERRIDE="${TEMPERATURE_OVERRIDE:-}"

# Resume toggles (must match base script options)
DEFAULT_RESUME_MODE="1"
DEFAULT_CLEAN_OUTPUT_DIR_OVERRIDE="0"
DEFAULT_BACKUP_PARTIAL_RUNS="1"

# Respect external env vars; fall back to defaults above.
RESUME_MODE="${RESUME_MODE:-${DEFAULT_RESUME_MODE}}"
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-${DEFAULT_CLEAN_OUTPUT_DIR_OVERRIDE}}"
BACKUP_PARTIAL_RUNS="${BACKUP_PARTIAL_RUNS:-${DEFAULT_BACKUP_PARTIAL_RUNS}}"
# ======Configuration=====

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[ERROR] Base script not found: ${BASE_SCRIPT}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
if [[ ! -f "${ACL6060_GLOSSARY_PATH}" ]]; then
  echo "[ERROR] ACL6060 glossary not found: ${ACL6060_GLOSSARY_PATH}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ -z "${MODEL_NAME_OVERRIDE}" ]]; then
  echo "[ERROR] MODEL_NAME_OVERRIDE is required for baseline runs." >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

export RESUME_MODE
export CLEAN_OUTPUT_DIR_OVERRIDE
export BACKUP_PARTIAL_RUNS
export GLOSSARY_PATHS_OVERRIDE="${ACL6060_GLOSSARY_PATH}"
export MODEL_NAME_OVERRIDE
export OUTPUT_BASE_OVERRIDE
export LATENCY_MULTIPLIERS_OVERRIDE
export RAG_K2_VALUES_OVERRIDE
export CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE
export LANG_CODE_OVERRIDE
export TARGET_LANG_OVERRIDE
export SEED_OVERRIDE
export TEMPERATURE_OVERRIDE

echo "[INFO] Running baseline NO-RAG resume-only for ACL6060 glossary."
echo "[INFO] BASE_SCRIPT=${BASE_SCRIPT}"
echo "[INFO] MODEL_NAME_OVERRIDE=${MODEL_NAME_OVERRIDE}"
echo "[INFO] OUTPUT_BASE_OVERRIDE=${OUTPUT_BASE_OVERRIDE}"
echo "[INFO] CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE=${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-<none>}"
echo "[INFO] LANG_CODE_OVERRIDE=${LANG_CODE_OVERRIDE:-<none>} TARGET_LANG_OVERRIDE=${TARGET_LANG_OVERRIDE:-<none>}"
echo "[INFO] LATENCY_MULTIPLIERS_OVERRIDE=${LATENCY_MULTIPLIERS_OVERRIDE} RAG_K2_VALUES_OVERRIDE=${RAG_K2_VALUES_OVERRIDE}"
echo "[INFO] SEED_OVERRIDE=${SEED_OVERRIDE:-<default ${DEFAULT_GEN_SEED}>} TEMPERATURE_OVERRIDE=${TEMPERATURE_OVERRIDE:-<default>}"
echo "[INFO] RESUME_MODE=${RESUME_MODE} CLEAN_OUTPUT_DIR_OVERRIDE=${CLEAN_OUTPUT_DIR_OVERRIDE} BACKUP_PARTIAL_RUNS=${BACKUP_PARTIAL_RUNS}"

bash "${BASE_SCRIPT}"

