#!/usr/bin/env bash
set -euo pipefail

# Resume-only runner for the "ACL6060 raw glossary" half of:
#   bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh
#
# It will:
# - skip runs whose output dir already contains a non-empty instances.log
# - re-run only missing/empty ones
# - never delete completed results
#
# All user-facing strings are in English.

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
BASE_SCRIPT="${ROOT_DIR}/documents/code/simuleval/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh"

# Only run the ACL6060 glossary.
ACL6060_GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"

# SST HF model (rank16)
MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107/keep1.0_r16/v3-20260121-021342-hf}"

# Output base (keep it separate from rank32 outputs)
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank16_v3-20260121-021342-hf_zh_acl6060_only}"

# Run a single setting by default.
# Important: use "${VAR-...}" (not "${VAR:-...}") so that an explicit empty string means "no override"
# and will run the full sweep defined in the base script.
LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIERS_OVERRIDE-2}"
RAG_K2_VALUES_OVERRIDE="${RAG_K2_VALUES_OVERRIDE-10}"

# Resume toggles (must match the base script options)
RESUME_MODE="1"
CLEAN_OUTPUT_DIR_OVERRIDE="0"
BACKUP_PARTIAL_RUNS="1"
# ======Configuration=====

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[ERROR] Base script not found: ${BASE_SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${ACL6060_GLOSSARY_PATH}" ]]; then
  echo "[ERROR] ACL6060 glossary not found: ${ACL6060_GLOSSARY_PATH}" >&2
  exit 3
fi

export RESUME_MODE
export CLEAN_OUTPUT_DIR_OVERRIDE
export BACKUP_PARTIAL_RUNS
export GLOSSARY_PATHS_OVERRIDE="${ACL6060_GLOSSARY_PATH}"
export MODEL_NAME_OVERRIDE
export OUTPUT_BASE_OVERRIDE
export LATENCY_MULTIPLIERS_OVERRIDE
export RAG_K2_VALUES_OVERRIDE

echo "[INFO] Running resume-only for ACL6060 glossary."
echo "[INFO] BASE_SCRIPT=${BASE_SCRIPT}"
echo "[INFO] GLOSSARY_PATHS_OVERRIDE=${GLOSSARY_PATHS_OVERRIDE}"
echo "[INFO] MODEL_NAME_OVERRIDE=${MODEL_NAME_OVERRIDE}"
echo "[INFO] OUTPUT_BASE_OVERRIDE=${OUTPUT_BASE_OVERRIDE}"
echo "[INFO] LATENCY_MULTIPLIERS_OVERRIDE=${LATENCY_MULTIPLIERS_OVERRIDE} RAG_K2_VALUES_OVERRIDE=${RAG_K2_VALUES_OVERRIDE}"
echo "[INFO] RESUME_MODE=${RESUME_MODE} CLEAN_OUTPUT_DIR_OVERRIDE=${CLEAN_OUTPUT_DIR_OVERRIDE} BACKUP_PARTIAL_RUNS=${BACKUP_PARTIAL_RUNS}"

bash "${BASE_SCRIPT}"




