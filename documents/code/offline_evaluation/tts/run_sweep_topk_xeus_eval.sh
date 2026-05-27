#!/usr/bin/env bash
set -euo pipefail

# ======Configuration=====
REPO_DIR="/home/jiaxuanluo/InfiniSST"
PY_SCRIPT="${REPO_DIR}/documents/code/offline_evaluation/tts/sweep_topk_xeus_eval.py"

CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

CUDA_GPU_ID="${CUDA_GPU_ID:-0}"

# Comma-separated K values to sweep; override via env var or edit here.
SWEEP_K="${SWEEP_K:-1,2,3,5,7,10,15,20}"
# ======Configuration=====

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export NLTK_DATA="/mnt/gemini/data/jiaxuanluo/nltk_data"

echo "[sweep_topk] Sweep K=${SWEEP_K} on CUDA_GPU_ID=${CUDA_GPU_ID}"
echo "[sweep_topk] PYTHON_BIN=${PYTHON_BIN}"

CUDA_VISIBLE_DEVICES="${CUDA_GPU_ID}" \
    OFFLINE_EVAL_DEVICE="cuda:0" \
    OFFLINE_EVAL_SWEEP_K="${SWEEP_K}" \
    "${PYTHON_BIN}" "${PY_SCRIPT}"
