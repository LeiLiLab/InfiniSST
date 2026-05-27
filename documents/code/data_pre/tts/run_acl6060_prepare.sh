#!/usr/bin/env bash
set -euo pipefail

# ======Configuration=====
REPO_DIR="/home/jiaxuanluo/InfiniSST"
PY_SCRIPT="${REPO_DIR}/documents/code/offline_evaluation/tts/prepare_acl6060_dev_dataset.py"
CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
# ======Configuration=====

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "[run_acl6060_prepare] Starting ACL6060 dev dataset preparation"
echo "[run_acl6060_prepare] PYTHON_BIN=${PYTHON_BIN}"
echo "[run_acl6060_prepare] PY_SCRIPT=${PY_SCRIPT}"

"${PYTHON_BIN}" "${PY_SCRIPT}"
