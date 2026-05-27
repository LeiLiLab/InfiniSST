#!/usr/bin/env bash
#SBATCH --job-name=hf2mc_origin
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_hf2mc_origin_bsz4.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_hf2mc_origin_bsz4.err

set -euo pipefail

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"
INNER_SCRIPT_REL="documents/code/train/sst_omni_train/src/export_hf_to_mcore.sh"
NOTES_FILE="${ROOT_DIR}/documents/code/train/sst_omni_train/notes/2026/05/20260519__export_origin_bsz4_hf_to_mcore.md"

HF_MODEL_DIR="${HF_MODEL_DIR_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4}"
MCORE_OUTPUT_DIR="${MCORE_OUTPUT_DIR_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/gigaspeech_zh_s_origin_bsz4_mcore}"

for p in "${WRAPPER}" "${ROOT_DIR}/${INNER_SCRIPT_REL}" "${NOTES_FILE}" "${HF_MODEL_DIR}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

if [[ -n "${HOST_GPU_DEVICES_OVERRIDE_CSV:-}" ]]; then
  HOST_GPU_DEVICES_OVERRIDE="${HOST_GPU_DEVICES_OVERRIDE_CSV//:/,}"
fi
ALLOCATED_GPUS="${HOST_GPU_DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0}}"
IFS=',' read -r -a GPU_ARR <<< "${ALLOCATED_GPUS}"
if (( ${#GPU_ARR[@]} != 1 )); then
  echo "[ERROR] This launcher expects exactly 1 GPU; got ${ALLOCATED_GPUS}" >&2
  exit 2
fi

echo "[INFO] HF_MODEL_DIR=${HF_MODEL_DIR}"
echo "[INFO] MCORE_OUTPUT_DIR=${MCORE_OUTPUT_DIR}"
echo "[INFO] HOST_GPU_DEVICES=${ALLOCATED_GPUS}"

unset CUDA_VISIBLE_DEVICES || true
unset NVIDIA_VISIBLE_DEVICES || true

HOST_GPU_DEVICES="${ALLOCATED_GPUS}" \
INNER_SCRIPT_REL="${INNER_SCRIPT_REL}" \
HF_MODEL_DIR="${HF_MODEL_DIR}" \
MCORE_OUTPUT_DIR="${MCORE_OUTPUT_DIR}" \
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}" \
CONVERT_OVERWRITE="${CONVERT_OVERWRITE:-0}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
bash "${WRAPPER}"
