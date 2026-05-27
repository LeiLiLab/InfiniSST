#!/usr/bin/env bash
# Export V3 speech-LLM MCore checkpoints to HF format on /mnt/aries/data7.
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/aries/data7/jiaxuanluo/slm/v3_speech_llm}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs}"
MAX_PARALLEL="${MAX_PARALLEL_EXPORTS_OVERRIDE:-1}"

declare -a NAMES=(
  "real_r8a32_iter793"
  "tagged_r8a32_iter923"
  "adv_r8a32_iter803"
)

declare -a MCORE_DIRS=(
  "/mnt/gemini/data2/jiaxuanluo/speech_llm_v3_real_termmap_zh_lh1b88kw_tau073_srcmatch100k_r8a32_taurus2/keep1.0_r8/v1-20260521-070000"
  "/mnt/gemini/data2/jiaxuanluo/speech_llm_v3_tagged_termmap_zh_lh1b88kw_tau073_srcmatch100k_r8a32_taurus2/keep1.0_r8/v0-20260521-070000"
  "/mnt/gemini/data2/jiaxuanluo/speech_llm_v3_adv_termmap_zh_lh1b88kw_tau073_srcmatch100k_r8a32_taurus2/keep1.0_r8/v0-20260521-070000"
)

declare -a GPU_PAIRS=(
  "0,1"
  "2,3"
  "4,5"
)

if [[ ! -f "${WRAPPER}" ]]; then
  echo "[ERROR] Missing wrapper: ${WRAPPER}" >&2
  exit 3
fi
if (( MAX_PARALLEL < 1 )); then
  echo "[ERROR] MAX_PARALLEL_EXPORTS_OVERRIDE must be >= 1" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

run_one() {
  local idx="$1"
  local name="${NAMES[$idx]}"
  local mcore="${MCORE_DIRS[$idx]}"
  local gpus="${GPU_PAIRS[$idx]}"
  local out="${OUT_ROOT}/${name}-hf"
  local log="${LOG_ROOT}/export_v3_speech_llm_${name}_$(date -u +%Y%m%dT%H%M%S)"

  if [[ ! -d "${mcore}" ]]; then
    echo "[ERROR] Missing MCore checkpoint for ${name}: ${mcore}" >&2
    return 3
  fi
  if [[ -e "${out}" && "${EXPORT_OVERWRITE:-0}" != "1" ]]; then
    echo "[ERROR] HF output already exists. Set EXPORT_OVERWRITE=1 to replace: ${out}" >&2
    return 2
  fi
  if [[ -e "${out}" ]]; then
    rm -rf "${out}"
  fi

  echo "[EXPORT] name=${name}"
  echo "[EXPORT] mcore=${mcore}"
  echo "[EXPORT] out=${out}"
  echo "[EXPORT] gpus=${gpus}"
  echo "[EXPORT] log=${log}.out"

  (
    unset CUDA_VISIBLE_DEVICES || true
    unset NVIDIA_VISIBLE_DEVICES || true
    MOUNT_ROOTS="/mnt/gemini /mnt/taurus /mnt/aries /mnt/data6" \
    HOST_GPU_DEVICES="${gpus}" \
    INNER_SCRIPT_REL="documents/code/train/sst_omni_train/export_mcore_checkpoint_to_hf.sh" \
    MCORE_ADAPTERS="${mcore}" \
    HF_OUTPUT_DIR="${out}" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    bash "${WRAPPER}"
  ) > "${log}.out" 2> "${log}.err"

  echo "[DONE] ${name} -> ${out}"
}

active=0
for i in "${!NAMES[@]}"; do
  run_one "${i}" &
  active=$((active + 1))
  if (( active >= MAX_PARALLEL )); then
    wait -n
    active=$((active - 1))
  fi
done
wait

echo "[ALL DONE] HF exports under ${OUT_ROOT}"
