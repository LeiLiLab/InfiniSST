#!/usr/bin/env bash
set -euo pipefail

# ======Configuration=====
REPO_DIR="/home/jiaxuanluo/InfiniSST"
PY_SCRIPT="${REPO_DIR}/documents/code/offline_evaluation/tts/dual_vs_single_model_eval.py"
PYTHON_BIN="/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python"
CUDA_GPU_ID="${CUDA_GPU_ID:-0}"
# ======Configuration=====

echo "[run_dual_vs_single] Starting dual vs single model evaluation"
echo "[run_dual_vs_single] CUDA_GPU_ID=${CUDA_GPU_ID}"
echo "[run_dual_vs_single] PYTHON_BIN=${PYTHON_BIN}"
echo "[run_dual_vs_single] PY_SCRIPT=${PY_SCRIPT}"

env -i \
    HOME="${HOME}" \
    PATH="${PATH}" \
    CUDA_VISIBLE_DEVICES="${CUDA_GPU_ID}" \
    OFFLINE_EVAL_DEVICE="cuda:0" \
    "${PYTHON_BIN}" "${PY_SCRIPT}"
