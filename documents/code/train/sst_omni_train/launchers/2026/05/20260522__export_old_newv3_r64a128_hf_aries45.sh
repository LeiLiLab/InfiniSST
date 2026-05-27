#!/usr/bin/env bash
set -euo pipefail

# Export old new_v3 r64/a128 MCore speech-LLM checkpoints to HF for vLLM eval.
# Runs serially on the provided GPU pair by default.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/data7/jiaxuanluo/slm/old_newv3_rank_ablation}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/old_newv3_rank_ablation_export}"
GPU_DEVICES="${GPU_DEVICES_OVERRIDE:-4,5}"

declare -a NAMES=(
  "q159wce4_newv3_r64a128_full"
  "rj1v1p7r_newv3_random_r64a128"
)

declare -a MCORE_DIRS=(
  "/mnt/gemini/data2/jiaxuanluo/speech_llm_tcmw100kgt_tau075_new_v3_r64a128_taurus4/keep1.0_r64/v1-20260508-135111"
  "/mnt/gemini/data2/jiaxuanluo/speech_llm_tcmw100kgt_tau075_new_v3_random_r64a128_aries8/keep1.0_r64/v1-20260508-143031"
)

if [[ ! -f "${WRAPPER}" ]]; then
  echo "[ERROR] Missing wrapper: ${WRAPPER}" >&2
  exit 3
fi

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"
  mcore="${MCORE_DIRS[$idx]}"
  out="${OUT_ROOT}/${name}-hf"
  log="${LOG_ROOT}/export_${name}_$(date -u +%Y%m%dT%H%M%S)"

  if [[ ! -d "${mcore}" ]]; then
    echo "[ERROR] Missing MCore checkpoint for ${name}: ${mcore}" >&2
    exit 3
  fi
  if [[ -e "${out}" && "${EXPORT_OVERWRITE:-0}" != "1" ]]; then
    echo "[ERROR] HF output already exists. Set EXPORT_OVERWRITE=1 to replace: ${out}" >&2
    exit 2
  fi
  if [[ -e "${out}" ]]; then
    rm -rf "${out}"
  fi

  echo "[EXPORT] name=${name}"
  echo "[EXPORT] mcore=${mcore}"
  echo "[EXPORT] out=${out}"
  echo "[EXPORT] gpus=${GPU_DEVICES}"
  echo "[EXPORT] log=${log}.out"

  (
    unset CUDA_VISIBLE_DEVICES || true
    unset NVIDIA_VISIBLE_DEVICES || true
    MOUNT_ROOTS="/mnt/gemini /mnt/taurus /mnt/data7 /mnt/data6" \
    HOST_GPU_DEVICES="${GPU_DEVICES}" \
    INNER_SCRIPT_REL="documents/code/train/sst_omni_train/export_mcore_checkpoint_to_hf.sh" \
    MCORE_ADAPTERS="${mcore}" \
    HF_OUTPUT_DIR="${out}" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    bash "${WRAPPER}"
  ) > "${log}.out" 2> "${log}.err"

  echo "[DONE] ${name} -> ${out}"
done

echo "[ALL DONE] HF exports under ${OUT_ROOT}"
