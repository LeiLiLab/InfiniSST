#!/usr/bin/env bash
#SBATCH --job-name=gigaspeech_k1_saturation
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_gigaspeech_k1_saturation.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_gigaspeech_k1_saturation.err

set -euo pipefail

# Run offline evaluation: recall@K1 saturation on GigaSpeech dev term dataset.
# All user-facing strings are in English.

# ======Configuration=====
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"

PY_SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/gigaspeech_recall_k1_saturation.py"

# Optional overrides
OFFLINE_EVAL_DEVICE="${OFFLINE_EVAL_DEVICE:-cuda:0}"
OFFLINE_EVAL_MODEL_PATH_DEFAULT="/mnt/gemini/data/jiaxuanluo/bidirectional_loss_snap_1.pt"
OFFLINE_EVAL_MODEL_PATH="${OFFLINE_EVAL_MODEL_PATH:-${OFFLINE_EVAL_MODEL_PATH_DEFAULT}}"

# Direct Python execution (no conda activation needed on Slurm nodes).
OFFLINE_EVAL_PYTHON_BIN_DEFAULT="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"
OFFLINE_EVAL_PYTHON_BIN="${OFFLINE_EVAL_PYTHON_BIN:-${OFFLINE_EVAL_PYTHON_BIN_DEFAULT}}"
# ======Configuration=====

echo "[INFO] OFFLINE_EVAL_DEVICE=${OFFLINE_EVAL_DEVICE}"
export OFFLINE_EVAL_DEVICE

echo "[INFO] OFFLINE_EVAL_MODEL_PATH=${OFFLINE_EVAL_MODEL_PATH}"
export OFFLINE_EVAL_MODEL_PATH

echo "[INFO] OFFLINE_EVAL_PYTHON_BIN=${OFFLINE_EVAL_PYTHON_BIN}"
if [[ ! -x "${OFFLINE_EVAL_PYTHON_BIN}" ]]; then
  echo "[ERROR] Python interpreter not found or not executable: ${OFFLINE_EVAL_PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

"${OFFLINE_EVAL_PYTHON_BIN}" "${PY_SCRIPT}"


