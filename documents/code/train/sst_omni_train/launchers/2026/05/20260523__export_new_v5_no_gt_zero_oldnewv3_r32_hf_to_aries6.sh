#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"
MCORE_ADAPTERS="${MCORE_ADAPTERS_OVERRIDE:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_r32a64_tp2_aries2/keep1.0_r32/v0-20260523-050346}"
HF_OUTPUT_DIR="${HF_OUTPUT_DIR_OVERRIDE:-/mnt/aries/data6/jiaxuanluo/slm/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_r32a64_tp2_aries2/keep1.0_r32/v0-20260523-050346-hf}"
GPU_DEVICES="${GPU_DEVICES_OVERRIDE:-4,5}"

for p in "${WRAPPER}" "${MCORE_ADAPTERS}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

if [[ -e "${HF_OUTPUT_DIR}" && "${EXPORT_OVERWRITE:-0}" != "1" ]]; then
  echo "[ERROR] HF output already exists. Set EXPORT_OVERWRITE=1 to replace: ${HF_OUTPUT_DIR}" >&2
  exit 2
fi
if [[ -e "${HF_OUTPUT_DIR}" ]]; then
  rm -rf "${HF_OUTPUT_DIR}"
fi

echo "[INFO] MCORE_ADAPTERS=${MCORE_ADAPTERS}"
echo "[INFO] HF_OUTPUT_DIR=${HF_OUTPUT_DIR}"
echo "[INFO] GPU_DEVICES=${GPU_DEVICES}"

unset CUDA_VISIBLE_DEVICES || true
unset NVIDIA_VISIBLE_DEVICES || true

MOUNT_ROOTS="/mnt/gemini /mnt/taurus /mnt/aries /mnt/data7 /mnt/data6" \
HOST_GPU_DEVICES="${GPU_DEVICES}" \
INNER_SCRIPT_REL="documents/code/train/sst_omni_train/export_mcore_checkpoint_to_hf.sh" \
MCORE_ADAPTERS="${MCORE_ADAPTERS}" \
HF_OUTPUT_DIR="${HF_OUTPUT_DIR}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
bash "${WRAPPER}"
