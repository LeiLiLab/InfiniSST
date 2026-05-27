#!/usr/bin/env bash
#SBATCH --job-name=export_slm_v2_hf
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_export_slm_v2_hf.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_export_slm_v2_hf.err

set -euo pipefail

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"

MCORE_ADAPTERS="/mnt/gemini/data/jiaxuanluo/speech_llm_tcm_filtered_wiki100kgt_tau075_v2_sourcefinal_gtzh/keep1.0_r32/v0-20260507-103419"
HF_OUTPUT_DIR="/mnt/gemini/data2/jiaxuanluo/speech_llm_tcm_filtered_wiki100kgt_tau075_v2_sourcefinal_gtzh/keep1.0_r32/v0-20260507-103419-hf"

if [[ -n "${HOST_GPU_DEVICES_OVERRIDE_CSV:-}" ]]; then
  HOST_GPU_DEVICES_OVERRIDE="${HOST_GPU_DEVICES_OVERRIDE_CSV//:/,}"
fi
HOST_GPU_DEVICES_OVERRIDE="${HOST_GPU_DEVICES_OVERRIDE:-0,1,2,3}"

for p in "${WRAPPER}" "${MCORE_ADAPTERS}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

echo "[INFO] MCORE_ADAPTERS=${MCORE_ADAPTERS}"
echo "[INFO] HF_OUTPUT_DIR=${HF_OUTPUT_DIR}"
echo "[INFO] HOST_GPU_DEVICES=${HOST_GPU_DEVICES_OVERRIDE}"

unset CUDA_VISIBLE_DEVICES || true
unset NVIDIA_VISIBLE_DEVICES || true

HOST_GPU_DEVICES="${HOST_GPU_DEVICES_OVERRIDE}" \
INNER_SCRIPT_REL="documents/code/train/sst_omni_train/export_mcore_checkpoint_to_hf.sh" \
MCORE_ADAPTERS="${MCORE_ADAPTERS}" \
HF_OUTPUT_DIR="${HF_OUTPUT_DIR}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
bash "${WRAPPER}"
