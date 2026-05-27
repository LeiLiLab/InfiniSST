#!/usr/bin/env bash
set -euo pipefail

# ======Configuration=====
REPO_DIR="/home/jiaxuanluo/InfiniSST"
PY_SCRIPT="${REPO_DIR}/documents/code/offline_evaluation/tts/xeus_tts_text_intersection_eval.py"

# spaCyEnv has both espnet (XEUS) and transformers (Qwen3-Omni)
CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

CUDA_GPU_ID="${CUDA_GPU_ID:-0}"

# Top-K for retrieval; override via env var or edit here.
# Example: TOP_K=5 CUDA_GPU_ID=0 bash run_xeus_eval.sh
TOP_K="${TOP_K:-10}"
# ======Configuration=====

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export NLTK_DATA="/mnt/gemini/data/jiaxuanluo/nltk_data"

echo "[run_xeus_eval] Dual-encoder evaluation: Qwen3-Omni (text) + XEUS (TTS)"
echo "[run_xeus_eval] CUDA_GPU_ID=${CUDA_GPU_ID}  TOP_K=${TOP_K}"
echo "[run_xeus_eval] PYTHON_BIN=${PYTHON_BIN}"

CUDA_VISIBLE_DEVICES="${CUDA_GPU_ID}" \
    OFFLINE_EVAL_DEVICE="cuda:0" \
    OFFLINE_EVAL_TOP_K="${TOP_K}" \
    "${PYTHON_BIN}" "${PY_SCRIPT}"
