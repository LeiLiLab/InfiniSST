#!/usr/bin/env bash
set -euo pipefail

# ======Configuration=====
REPO_DIR="/home/jiaxuanluo/InfiniSST"
PY_SCRIPT="${REPO_DIR}/documents/code/offline_evaluation/tts/dual_model_text_tts_intersection_eval.py"
PYTHON_BIN="/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python"
CUDA_GPU_ID="${CUDA_GPU_ID:-0}"
# ======Configuration=====

echo "[run_dual_model_eval] Starting dual-model text/TTS intersection evaluation"
echo "[run_dual_model_eval] CUDA_GPU_ID=${CUDA_GPU_ID}"
echo "[run_dual_model_eval] PYTHON_BIN=${PYTHON_BIN}"
echo "[run_dual_model_eval] PY_SCRIPT=${PY_SCRIPT}"

env -i \
    HOME="${HOME}" \
    PATH="${PATH}" \
    CUDA_VISIBLE_DEVICES="${CUDA_GPU_ID}" \
    OFFLINE_EVAL_DEVICE="cuda:0" \
    "${PYTHON_BIN}" "${PY_SCRIPT}"
