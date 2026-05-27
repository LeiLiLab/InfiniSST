#!/usr/bin/env bash
# Wait for the requested Aries GPU set, then start ja New V9 6-GPU Speech LLM SFT.
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
TRAIN_LAUNCHER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260524__speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_aries6_r32a64_tp2.sh"
TARGET_GPUS="${TARGET_GPUS_OVERRIDE:-0,1,2,5,6,7}"
MAX_USED_MB="${MAX_USED_MB_OVERRIDE:-1000}"
SLEEP_SEC="${SLEEP_SEC_OVERRIDE:-60}"
MASTER_PORT="${MASTER_PORT_OVERRIDE:-29646}"

if [[ ! -f "${TRAIN_LAUNCHER}" ]]; then
  echo "[ERROR] Missing train launcher: ${TRAIN_LAUNCHER}" >&2
  exit 3
fi

IFS=',' read -r -a TARGET_ARR <<< "${TARGET_GPUS}"
if (( ${#TARGET_ARR[@]} != 6 )); then
  echo "[ERROR] Expected 6 target GPUs, got ${TARGET_GPUS}" >&2
  exit 2
fi

gpu_used_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' '
}

while true; do
  busy=()
  for gpu in "${TARGET_ARR[@]}"; do
    used="$(gpu_used_mb "${gpu}")"
    if (( used > MAX_USED_MB )); then
      busy+=("${gpu}:${used}MiB")
    fi
  done
  if (( ${#busy[@]} == 0 )); then
    echo "[WATCH] $(date -Is) target GPUs free: ${TARGET_GPUS}"
    break
  fi
  echo "[WATCH] $(date -Is) waiting for GPUs ${TARGET_GPUS}; busy=${busy[*]}"
  sleep "${SLEEP_SEC}"
done

echo "[WATCH] launching ja New V9 SFT on ${TARGET_GPUS}"
exec env \
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  HOST_GPU_DEVICES_OVERRIDE="${TARGET_GPUS}" \
  MASTER_PORT_OVERRIDE="${MASTER_PORT}" \
  bash "${TRAIN_LAUNCHER}"
