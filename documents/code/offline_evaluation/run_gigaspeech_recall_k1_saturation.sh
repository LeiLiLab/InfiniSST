#!/usr/bin/env bash
set -euo pipefail

# Run offline evaluation: recall@K1 saturation on GigaSpeech dev term dataset.
# All user-facing strings are in English.

# ======Configuration=====
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"

PY_SCRIPT="/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/gigaspeech_recall_k1_saturation.py"

# Direct Python execution (no conda activation needed).
OFFLINE_EVAL_PYTHON_BIN_DEFAULT="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"
OFFLINE_EVAL_PYTHON_BIN="${OFFLINE_EVAL_PYTHON_BIN:-${OFFLINE_EVAL_PYTHON_BIN_DEFAULT}}"
# ======Configuration=====

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Script not found: ${PY_SCRIPT}" >&2
  exit 2
fi

echo "[INFO] OFFLINE_EVAL_PYTHON_BIN=${OFFLINE_EVAL_PYTHON_BIN}"
if [[ ! -x "${OFFLINE_EVAL_PYTHON_BIN}" ]]; then
  echo "[ERROR] Python interpreter not found or not executable: ${OFFLINE_EVAL_PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONPATH="/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

"${OFFLINE_EVAL_PYTHON_BIN}" "${PY_SCRIPT}"


