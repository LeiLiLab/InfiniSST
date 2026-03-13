#!/usr/bin/env bash
#SBATCH --job-name=gigaspeech_thr_k1_10
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
##SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/tts/tts_sbatch_gigaspeech_threshold_ablation_k1_10_aries_%j.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/tts/tts_sbatch_gigaspeech_threshold_ablation_k1_10_aries_%j.err

set -euo pipefail

# Run offline evaluation: threshold sweep for F1/F2/F3 (K1=10) on GigaSpeech dev term dataset.
# All user-facing strings are in English.

# ======Configuration=====
REPO_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
PY_SCRIPT="${REPO_DIR}/documents/code/offline_evaluation/tts/tts_gigaspeech_threshold_ablation_k1_10.py"

# Direct Python execution (no conda activation).
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"
OFFLINE_EVAL_PYTHON_BIN_DEFAULT="${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python"
OFFLINE_EVAL_PYTHON_BIN="${OFFLINE_EVAL_PYTHON_BIN:-${OFFLINE_EVAL_PYTHON_BIN_DEFAULT}}"

# Offline-eval overrides (same as your provided command block).
CUDA_VISIBLE_GPU_ID="${CUDA_VISIBLE_GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_GPU_ID}"
OFFLINE_EVAL_DEVICE="${OFFLINE_EVAL_DEVICE:-cuda:0}"
#OFFLINE_EVAL_MODEL_PATH="${OFFLINE_EVAL_MODEL_PATH:-/mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_v2_step_4000.pt}"
#OFFLINE_EVAL_MODEL_PATH="${OFFLINE_EVAL_MODEL_PATH:-/mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_v2.pt}"
OFFLINE_EVAL_MODEL_PATH="${OFFLINE_EVAL_MODEL_PATH:-/mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_v2_epoch_5.pt}"
OFFLINE_EVAL_THRESHOLD_STEPS="${OFFLINE_EVAL_THRESHOLD_STEPS:-51}"
OFFLINE_EVAL_MODE="${OFFLINE_EVAL_MODE:-intersection}"
OFFLINE_EVAL_TTS_ROOT_DIR="${OFFLINE_EVAL_TTS_ROOT_DIR:-/mnt/gemini/data/siqiouyang/term_dev_tts}"
OFFLINE_EVAL_SWEEP_TYPE="${OFFLINE_EVAL_SWEEP_TYPE:-relative_margin}"
OFFLINE_EVAL_MARGIN_MIN="${OFFLINE_EVAL_MARGIN_MIN:-0.00}"
OFFLINE_EVAL_MARGIN_MAX="${OFFLINE_EVAL_MARGIN_MAX:-0.50}"
OFFLINE_EVAL_MARGIN_STEPS="${OFFLINE_EVAL_MARGIN_STEPS:-51}"

# Bind output directory to model path by default:
# /mnt/gemini/data/jiaxuanluo/offline_eval_threshold_ablation_tts_k1_10/<model_stem>
if [[ -z "${OFFLINE_EVAL_OUTPUT_DIR:-}" ]]; then
  MODEL_BASENAME="$(basename "${OFFLINE_EVAL_MODEL_PATH}")"
  MODEL_STEM="${MODEL_BASENAME%.*}"
  OFFLINE_EVAL_OUTPUT_DIR="/mnt/gemini/data/jiaxuanluo/offline_eval_threshold_ablation_tts_k1_10_v3/${MODEL_STEM}"
fi
# ======Configuration=====

echo "[INFO] OFFLINE_EVAL_PYTHON_BIN=${OFFLINE_EVAL_PYTHON_BIN}"
echo "[INFO] PY_SCRIPT=${PY_SCRIPT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] OFFLINE_EVAL_DEVICE=${OFFLINE_EVAL_DEVICE}"
echo "[INFO] OFFLINE_EVAL_MODEL_PATH=${OFFLINE_EVAL_MODEL_PATH}"
echo "[INFO] OFFLINE_EVAL_OUTPUT_DIR=${OFFLINE_EVAL_OUTPUT_DIR}"
echo "[INFO] OFFLINE_EVAL_THRESHOLD_STEPS=${OFFLINE_EVAL_THRESHOLD_STEPS}"
echo "[INFO] OFFLINE_EVAL_MODE=${OFFLINE_EVAL_MODE}"
echo "[INFO] OFFLINE_EVAL_TTS_ROOT_DIR=${OFFLINE_EVAL_TTS_ROOT_DIR}"
echo "[INFO] OFFLINE_EVAL_SWEEP_TYPE=${OFFLINE_EVAL_SWEEP_TYPE}"
echo "[INFO] OFFLINE_EVAL_MARGIN_MIN=${OFFLINE_EVAL_MARGIN_MIN}"
echo "[INFO] OFFLINE_EVAL_MARGIN_MAX=${OFFLINE_EVAL_MARGIN_MAX}"
echo "[INFO] OFFLINE_EVAL_MARGIN_STEPS=${OFFLINE_EVAL_MARGIN_STEPS}"

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
export OFFLINE_EVAL_MODE
export OFFLINE_EVAL_TTS_ROOT_DIR
export OFFLINE_EVAL_SWEEP_TYPE
export OFFLINE_EVAL_MARGIN_MIN
export OFFLINE_EVAL_MARGIN_MAX
export OFFLINE_EVAL_MARGIN_STEPS

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

"${OFFLINE_EVAL_PYTHON_BIN}" "${PY_SCRIPT}"

echo "[INFO] Done."
