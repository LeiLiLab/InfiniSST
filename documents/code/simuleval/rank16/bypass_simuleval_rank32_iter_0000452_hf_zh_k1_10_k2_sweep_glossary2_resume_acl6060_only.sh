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
BASE_SCRIPT="${ROOT_DIR}/documents/code/simuleval/rank16/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh"

# Default glossary (single glossary run). Can be overridden.
DEFAULT_GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
# If set, overrides the glossary used for this resume-only run (single path).
GLOSSARY_PATH_OVERRIDE="${GLOSSARY_PATH_OVERRIDE:-}"
# If set, overrides the full GLOSSARY_PATHS list passed to the base sweep script (space-separated).
GLOSSARY_PATHS_OVERRIDE_OVERRIDE="${GLOSSARY_PATHS_OVERRIDE:-}"

# RAG toggle (default: enabled in base script). Set to "0" to disable RAG.
RAG_ENABLED_OVERRIDE="${RAG_ENABLED_OVERRIDE:-}"

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

GLOSSARY_PATHS_OVERRIDE_VALUE="${DEFAULT_GLOSSARY_PATH}"
if [[ -n "${GLOSSARY_PATHS_OVERRIDE_OVERRIDE}" ]]; then
  GLOSSARY_PATHS_OVERRIDE_VALUE="${GLOSSARY_PATHS_OVERRIDE_OVERRIDE}"
elif [[ -n "${GLOSSARY_PATH_OVERRIDE}" ]]; then
  GLOSSARY_PATHS_OVERRIDE_VALUE="${GLOSSARY_PATH_OVERRIDE}"
fi

if [[ -z "${GLOSSARY_PATHS_OVERRIDE_VALUE}" ]]; then
  echo "[ERROR] Glossary override is empty (set GLOSSARY_PATH_OVERRIDE or GLOSSARY_PATHS_OVERRIDE)." >&2
  exit 2
fi

# Validate all glossary paths exist.
export GLOSSARY_PATHS_OVERRIDE_VALUE
python3 - <<'PY'
import os
from pathlib import Path

val = (os.environ.get("GLOSSARY_PATHS_OVERRIDE_VALUE", "") or "").strip()
paths = [p for p in val.split() if p.strip()]
missing = [p for p in paths if not Path(p).is_file()]
if not paths:
    raise SystemExit("[ERROR] No glossary paths provided.")
if missing:
    raise SystemExit("[ERROR] Missing glossary path(s): " + "; ".join(missing))
PY

export RESUME_MODE
export CLEAN_OUTPUT_DIR_OVERRIDE
export BACKUP_PARTIAL_RUNS
export GLOSSARY_PATHS_OVERRIDE="${GLOSSARY_PATHS_OVERRIDE_VALUE}"
export MODEL_NAME_OVERRIDE
export OUTPUT_BASE_OVERRIDE
export LATENCY_MULTIPLIERS_OVERRIDE
export RAG_K2_VALUES_OVERRIDE
export RAG_ENABLED_OVERRIDE

echo "[INFO] Running resume-only for selected glossary path(s)."
echo "[INFO] BASE_SCRIPT=${BASE_SCRIPT}"
echo "[INFO] GLOSSARY_PATHS_OVERRIDE=${GLOSSARY_PATHS_OVERRIDE}"
echo "[INFO] MODEL_NAME_OVERRIDE=${MODEL_NAME_OVERRIDE}"
echo "[INFO] OUTPUT_BASE_OVERRIDE=${OUTPUT_BASE_OVERRIDE}"
echo "[INFO] LATENCY_MULTIPLIERS_OVERRIDE=${LATENCY_MULTIPLIERS_OVERRIDE} RAG_K2_VALUES_OVERRIDE=${RAG_K2_VALUES_OVERRIDE}"
echo "[INFO] RAG_ENABLED_OVERRIDE=${RAG_ENABLED_OVERRIDE:-<default>}"
echo "[INFO] RESUME_MODE=${RESUME_MODE} CLEAN_OUTPUT_DIR_OVERRIDE=${CLEAN_OUTPUT_DIR_OVERRIDE} BACKUP_PARTIAL_RUNS=${BACKUP_PARTIAL_RUNS}"

bash "${BASE_SCRIPT}"




