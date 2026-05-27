#!/usr/bin/env bash
set -euo pipefail

# Run SimulEval for a fixed HF checkpoint (zh only).
# This is a minimal wrapper around:
#   documents/code/run_simuleval_rag_aries_v4_final_result_taurus.sh
#
# Usage:
#   bash documents/code/auto_simuleval_rank32_iter_0000452_hf_zh.sh
#

# ======Configuration=====
ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
SIM_SCRIPT="${ROOT_DIR}/documents/code/run_simuleval_rag_aries_v4_final_result_taurus.sh"

# Fixed model (rank32, iter_0000452 exported to HF)
MODEL_NAME_OVERRIDE="/mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107/keep1.0_r32/v3-20260108-045345/iter_0000452-hf"

# try 4 epoch results
MODEL_NAME_OVERRIDE_4_epochs="/mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107/keep1.0_r32/v3-20260108-045345-hf"

# Language
ONLY_LANG="zh"

# Output base (dedicated folder to avoid mixing with other experiments)
OUTPUT_BASE_OVERRIDE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh"

# Optional: blacklist physical GPU ids (as in CUDA_VISIBLE_DEVICES set by Slurm).
# Example: "1,2,3"
DISABLE_GPU_IDS="${DISABLE_GPU_IDS:-}"
# =======================

echo "[INFO] ONLY_LANG=${ONLY_LANG}"
echo "[INFO] MODEL_NAME_OVERRIDE=${MODEL_NAME_OVERRIDE}"
echo "[INFO] OUTPUT_BASE_OVERRIDE=${OUTPUT_BASE_OVERRIDE}"
echo "[INFO] SIM_SCRIPT=${SIM_SCRIPT}"
echo "[INFO] DISABLE_GPU_IDS=${DISABLE_GPU_IDS}"

if [[ ! -f "${SIM_SCRIPT}" ]]; then
  echo "[ERROR] Simuleval script not found: ${SIM_SCRIPT}" >&2
  exit 2
fi

if [[ ! -d "${MODEL_NAME_OVERRIDE}" ]]; then
  echo "[ERROR] HF model dir not found: ${MODEL_NAME_OVERRIDE}" >&2
  exit 2
fi

ONLY_LANG="${ONLY_LANG}" \
MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE}" \
DISABLE_GPU_IDS="${DISABLE_GPU_IDS}" \
sbatch "${SIM_SCRIPT}"

echo "[INFO] Submitted: ${SIM_SCRIPT}"


