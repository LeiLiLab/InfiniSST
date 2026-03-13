#!/usr/bin/env bash
#SBATCH --job-name=gigaspeech_thr_k1_10
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_gigaspeech_thr_k1_10.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_gigaspeech_thr_k1_10.err

set -euo pipefail

# Run offline evaluation: threshold sweep for F1/F2/F3 (K1=10) on GigaSpeech dev term dataset.
# All user-facing strings are in English.

# ======Configuration=====
REPO_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
PY_SCRIPT="${REPO_DIR}/documents/code/offline_evaluation/gigaspeech_threshold_ablation_k1_10.py"

# Direct Python execution (no conda activation).
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"
OFFLINE_EVAL_PYTHON_BIN_DEFAULT="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"
OFFLINE_EVAL_PYTHON_BIN="${OFFLINE_EVAL_PYTHON_BIN:-${OFFLINE_EVAL_PYTHON_BIN_DEFAULT}}"

# Offline-eval overrides (same as your provided command block).
OFFLINE_EVAL_DEVICE="${OFFLINE_EVAL_DEVICE:-cuda:0}"
OFFLINE_EVAL_MODEL_PATH="${OFFLINE_EVAL_MODEL_PATH:-/mnt/gemini/data/jiaxuanluo/bidirectional_loss_snap_1.pt}"
OFFLINE_EVAL_OUTPUT_DIR="${OFFLINE_EVAL_OUTPUT_DIR:-/mnt/gemini/data2/jiaxuanluo/tmp_threshold_ablation}"
OFFLINE_EVAL_THRESHOLD_STEPS="${OFFLINE_EVAL_THRESHOLD_STEPS:-51}"
# ======Configuration=====

echo "[INFO] OFFLINE_EVAL_PYTHON_BIN=${OFFLINE_EVAL_PYTHON_BIN}"
echo "[INFO] PY_SCRIPT=${PY_SCRIPT}"
echo "[INFO] OFFLINE_EVAL_DEVICE=${OFFLINE_EVAL_DEVICE}"
echo "[INFO] OFFLINE_EVAL_MODEL_PATH=${OFFLINE_EVAL_MODEL_PATH}"
echo "[INFO] OFFLINE_EVAL_OUTPUT_DIR=${OFFLINE_EVAL_OUTPUT_DIR}"
echo "[INFO] OFFLINE_EVAL_THRESHOLD_STEPS=${OFFLINE_EVAL_THRESHOLD_STEPS}"

if [[ ! -x "${OFFLINE_EVAL_PYTHON_BIN}" ]]; then
  echo "[ERROR] Python interpreter not found or not executable: ${OFFLINE_EVAL_PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Script not found: ${PY_SCRIPT}" >&2
  exit 2
fi

export OFFLINE_EVAL_DEVICE
export OFFLINE_EVAL_MODEL_PATH
export OFFLINE_EVAL_OUTPUT_DIR
export OFFLINE_EVAL_THRESHOLD_STEPS

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

"${OFFLINE_EVAL_PYTHON_BIN}" "${PY_SCRIPT}"

echo "[INFO] Done."
