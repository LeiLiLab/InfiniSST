#!/usr/bin/env bash
set -euo pipefail

: "${MCORE_ADAPTERS:?Set MCORE_ADAPTERS to the Megatron checkpoint directory}"
: "${HF_OUTPUT_DIR:?Set HF_OUTPUT_DIR to the desired HF export directory}"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../../../../" && pwd)}"
source "${ROOT_DIR}/documents/code/train/sst_omni_train/common/hf_export_staging.sh"

if [[ ! -d "${MCORE_ADAPTERS}" ]]; then
  echo "[ERROR] Missing MCORE_ADAPTERS: ${MCORE_ADAPTERS}" >&2
  exit 3
fi

mkdir -p "$(dirname "${HF_OUTPUT_DIR}")"

echo "[INFO] MCORE_ADAPTERS=${MCORE_ADAPTERS}"
echo "[INFO] HF_OUTPUT_DIR=${HF_OUTPUT_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] HF_EXPORT_STAGE_ROOT=${HF_EXPORT_STAGE_ROOT:-<direct>}"

export_mcore_checkpoint_to_hf_staged "${MCORE_ADAPTERS}" "${HF_OUTPUT_DIR}"
