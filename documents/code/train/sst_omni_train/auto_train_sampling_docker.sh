#!/usr/bin/env bash
set -euo pipefail

###### ======Configuration=====
# This wrapper runs the sampling training script inside a docker image that provides:
# - megatron (CLI: megatron)
# - swift (CLI: swift)
#
# It is intended for clusters where the host environment does NOT have megatron/swift installed.

# Docker image containing megatron + swift
DEFAULT_DOCKER_IMAGE="modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1"

# Docker runtime settings
DEFAULT_DOCKER_SHM_SIZE="16g"
DEFAULT_DOCKER_IPC_MODE="host"
DEFAULT_DOCKER_GPU_FLAG="all"

# GPU selection:
# - If HOST_GPU_DEVICES is set (e.g., "4,5,6,7"), the container will ONLY see those physical GPUs.
# - In that case, the wrapper will also set CUDA_VISIBLE_DEVICES inside the container to "0,1,2,3"
#   (or generally "0..N-1"), so torchrun ranks map cleanly.
DEFAULT_HOST_GPU_DEVICES=""

# Project layout
DEFAULT_ROOT_DIR_REL_FROM_SCRIPT="../../../../"
DEFAULT_CONTAINER_WORKDIR="/workspace/InfiniSST"

# Script to run inside the container (path is relative to project root)
DEFAULT_INNER_SCRIPT_REL="documents/code/train/sst_omni_train/auto_train_sampling_rank32_try.sh"

# Common mount roots (mounted only if they exist on host)
DEFAULT_MOUNT_ROOTS="/mnt/gemini /mnt/taurus /mnt/aries /mnt/data /mnt/data1 /mnt/data2 /mnt/data6 /mnt/data7"

###### ======Configuration=====

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/${DEFAULT_ROOT_DIR_REL_FROM_SCRIPT}" && pwd)"

DOCKER_IMAGE="${DOCKER_IMAGE:-${DEFAULT_DOCKER_IMAGE}}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-${DEFAULT_DOCKER_SHM_SIZE}}"
DOCKER_IPC_MODE="${DOCKER_IPC_MODE:-${DEFAULT_DOCKER_IPC_MODE}}"
DOCKER_GPU_FLAG="${DOCKER_GPU_FLAG:-${DEFAULT_DOCKER_GPU_FLAG}}"
HOST_GPU_DEVICES="${HOST_GPU_DEVICES:-${DEFAULT_HOST_GPU_DEVICES}}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-${DEFAULT_CONTAINER_WORKDIR}}"
INNER_SCRIPT_REL="${INNER_SCRIPT_REL:-${DEFAULT_INNER_SCRIPT_REL}}"

INNER_SCRIPT_HOST="${ROOT_DIR}/${INNER_SCRIPT_REL}"

die() {
  local msg="$1"
  echo "Error: ${msg}" >&2
  exit 1
}

if ! command -v docker >/dev/null 2>&1; then
  die "docker is not available on this machine. Please install docker or use a compatible container runtime (e.g., apptainer) and update this wrapper."
fi

if [[ ! -f "${INNER_SCRIPT_HOST}" ]]; then
  die "Inner script not found: ${INNER_SCRIPT_HOST}"
fi

count_csv_items() {
  local csv="$1"
  local -a items=()
  IFS=',' read -r -a items <<< "${csv}"
  echo "${#items[@]}"
}

make_cuda_visible_devices_0_to_n_minus_1() {
  local n="$1"
  if [[ "${n}" -le 0 ]]; then
    echo ""
    return 0
  fi
  local out="0"
  local i="1"
  while [[ "${i}" -lt "${n}" ]]; do
    out+=",${i}"
    i="$((i + 1))"
  done
  echo "${out}"
}

mount_args=()
mount_args+=("-v" "${ROOT_DIR}:${CONTAINER_WORKDIR}")

for root in ${MOUNT_ROOTS:-${DEFAULT_MOUNT_ROOTS}}; do
  if [[ -d "${root}" ]]; then
    mount_args+=("-v" "${root}:${root}")
  fi
done

# Megatron checkpoints exported via `swift export` in this repo often embed the
# original HF snapshot path as `/workspace/Qwen3-Omni-30B-A3B-Instruct` inside docker.
# Mount the host HF tree there so `swift megatron` can resolve tokenizer/config.
BASE_MODEL_HOST="${BASE_MODEL_HOST:-/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct}"
BASE_MODEL_DOCKER="${BASE_MODEL_DOCKER:-/workspace/Qwen3-Omni-30B-A3B-Instruct}"
if [[ -d "${BASE_MODEL_HOST}" ]]; then
  stage_base_model_if_requested() {
    local src_dir="$1"
    local stage_root="${BASE_MODEL_STAGE_ROOT:-}"
    local min_free_gb="${BASE_MODEL_STAGE_MIN_FREE_GB:-90}"
    if [[ -z "${stage_root}" || "${stage_root}" == "0" || "${stage_root}" == "false" || "${stage_root}" == "none" ]]; then
      echo "${src_dir}"
      return 0
    fi
    if ! command -v rsync >/dev/null 2>&1; then
      die "BASE_MODEL_STAGE_ROOT requires rsync on the host"
    fi
    mkdir -p "${stage_root}"
    python3 - "${stage_root}" "${min_free_gb}" 1>&2 <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_free_gb = float(sys.argv[2])
free_gb = shutil.disk_usage(path).free / (1024 ** 3)
print(f"[INFO] base_model_stage free_space path={path} free_gb={free_gb:.1f} required_gb={min_free_gb:.1f}", flush=True)
if free_gb < min_free_gb:
    raise SystemExit(f"Not enough free space for BASE_MODEL_STAGE_ROOT: {path} has {free_gb:.1f} GB")
PY
    local base_name
    base_name="$(basename "${src_dir%/}")"
    local dst_dir="${stage_root%/}/${base_name}"
    mkdir -p "${dst_dir}"
    echo "[INFO] Staging base model for faster export:" >&2
    echo "[INFO]   src=${src_dir}" >&2
    echo "[INFO]   dst=${dst_dir}" >&2
    rsync -a --delete --info=progress2 "${src_dir%/}/" "${dst_dir%/}/" >&2
    [[ -f "${dst_dir}/config.json" ]] || die "Staged base model missing config.json: ${dst_dir}"
    echo "${dst_dir}"
  }
  BASE_MODEL_MOUNT_HOST="$(stage_base_model_if_requested "${BASE_MODEL_HOST}")"
  mount_args+=("-v" "${BASE_MODEL_MOUNT_HOST}:${BASE_MODEL_DOCKER}:ro")
fi

env_args=()
pass_envs=(
  CUDA_VISIBLE_DEVICES
  NPROC_PER_NODE
  MCORE_MODEL
  VAL_DATASET
  DATASET_PREFIX
  DATASET_SUFFIX
  KEEP_RATIO
  LORA_RANK
  LORA_ALPHA
  MAX_EPOCHS
  MICRO_BATCH_SIZE
  GLOBAL_BATCH_SIZE
  ITERATIONS_PER_EPOCH
  SAVE_INTERVAL
  SAVE_BASE
  TRAIN_LOG_DIR
  MASTER_ADDR
  MASTER_PORT
  NCCL_P2P_DISABLE
  NCCL_IB_DISABLE
  NCCL_DEBUG
  TORCH_NCCL_ENABLE_MONITORING
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
  CUDA_DEVICE_MAX_CONNECTIONS
  PYTORCH_CUDA_ALLOC_CONF
  WANDB_API_KEY
  WANDB_PROJECT
  WANDB_EXP_PREFIX
  WANDB_TAGS
  WANDB_NOTES
  ENABLE_MD_UPDATE
  EXPERT_MODEL_PARALLEL_SIZE
  TENSOR_MODEL_PARALLEL_SIZE
  SEQUENCE_PARALLEL
  MAX_LENGTH
  DATASET_PATH
  MCORE_ADAPTERS
  HF_OUTPUT_DIR
  HF_MODEL_DIR
  MCORE_OUTPUT_DIR
  TORCH_DTYPE
  CONVERT_OVERWRITE
  SWIFT_CMD
  SWIFT_TORCH_DTYPE
  HF_EXPORT_STAGE_ROOT
  HF_EXPORT_MIN_FREE_GB
  HF_EXPORT_KEEP_STAGE
  HF_EXPORT_OVERWRITE
  HF_EXPORT_RSYNC_FLAGS
  HF_EXPORT_SWIFT_EXTRA_ARGS
  HF_EXPORT_LOCAL_CACHE_ROOT
  HF_EXPORT_LOCAL_CACHE_MIN_FREE_GB
  HF_EXPORT_LOCAL_LATEST_LINK
)

if [[ -n "${HOST_GPU_DEVICES}" ]]; then
  gpu_count="$(count_csv_items "${HOST_GPU_DEVICES}")"
  inner_cuda_visible_devices="$(make_cuda_visible_devices_0_to_n_minus_1 "${gpu_count}")"
  env_args+=("-e" "CUDA_VISIBLE_DEVICES=${inner_cuda_visible_devices}")
fi

for name in "${pass_envs[@]}"; do
  if [[ -n "${!name-}" ]]; then
    env_args+=("-e" "${name}=${!name}")
  fi
done

echo "Starting docker wrapper..."
echo "DOCKER_IMAGE=${DOCKER_IMAGE}"
echo "ROOT_DIR=${ROOT_DIR}"
echo "CONTAINER_WORKDIR=${CONTAINER_WORKDIR}"
echo "INNER_SCRIPT_REL=${INNER_SCRIPT_REL}"
echo "HOST_GPU_DEVICES=${HOST_GPU_DEVICES:-<empty>}"

docker_gpu_arg=("--gpus" "${DOCKER_GPU_FLAG}")
if [[ -n "${HOST_GPU_DEVICES}" ]]; then
  # Docker's --gpus parser needs quotes around comma-separated device lists.
  # Without the embedded quotes, some NVIDIA runtime versions interpret the
  # request as both Count and DeviceIDs.
  docker_gpu_arg=("--gpus" "\"device=${HOST_GPU_DEVICES}\"")
  unset CUDA_VISIBLE_DEVICES || true
  unset NVIDIA_VISIBLE_DEVICES || true
fi

docker run --rm \
  "${docker_gpu_arg[@]}" \
  --ipc="${DOCKER_IPC_MODE}" \
  --shm-size="${DOCKER_SHM_SIZE}" \
  "${env_args[@]}" \
  "${mount_args[@]}" \
  -w "${CONTAINER_WORKDIR}" \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    bash '${INNER_SCRIPT_REL}'
  "
