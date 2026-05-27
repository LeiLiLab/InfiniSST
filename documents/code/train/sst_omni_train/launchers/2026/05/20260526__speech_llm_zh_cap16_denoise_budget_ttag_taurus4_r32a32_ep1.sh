#!/usr/bin/env bash
# 4-GPU Speech LLM SFT launcher for Chinese cap16 denoise-budget short-tag data.
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"

LANG_CODE="zh"
DATA_DIR="${DATA_DIR_OVERRIDE:-/mnt/data1/jiaxuanluo/speech_llm_zh_cap16_denoise_budget_20260526/zh/hn1024_tau078_cap16_denoise_budget_ttag_v1}"
DATASET_PATH="${DATASET_PATH_OVERRIDE:-${DATA_DIR}/train_s_zh_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary.jsonl}"
VAL_DATASET="${VAL_DATASET_OVERRIDE:-${DATA_DIR}/dev_s_zh_retriever_hn1024_tau078_cap16_denoise_budget_ttag_exactboundary_first355.jsonl}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/train/sst_omni_train/notes/2026/05/20260526__speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4.md}"

LORA_RANK="${LORA_RANK_OVERRIDE:-32}"
LORA_ALPHA="${LORA_ALPHA_OVERRIDE:-32}"
RANK_TAG="r${LORA_RANK}a${LORA_ALPHA}"
SAVE_BASE="${SAVE_BASE_OVERRIDE:-/mnt/data1/jiaxuanluo/slm/speech_llm_zh_cap16_denoise_budget_ttag_${RANK_TAG}_ep1_taurus4}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR_OVERRIDE:-/mnt/data1/jiaxuanluo/logs/speech_llm_zh_cap16_denoise_budget_ttag_${RANK_TAG}_ep1_taurus4}"
HF_EXPORT_STAGE_ROOT="${HF_EXPORT_STAGE_ROOT_OVERRIDE:-/mnt/data1/jiaxuanluo/slm_local_cache/hf_export_stage/speech_llm_zh_cap16_denoise_budget_ttag_${RANK_TAG}_ep1_taurus4}"
HF_EXPORT_MIN_FREE_GB="${HF_EXPORT_MIN_FREE_GB_OVERRIDE:-90}"
HF_EXPORT_LOCAL_CACHE_ROOT="${HF_EXPORT_LOCAL_CACHE_ROOT_OVERRIDE:-/mnt/data1/jiaxuanluo/slm_local_cache/speech_llm_zh_cap16_denoise_budget_ttag_${RANK_TAG}_ep1_taurus4/keep1.0_r32}"
HF_EXPORT_LOCAL_LATEST_LINK="${HF_EXPORT_LOCAL_LATEST_LINK_OVERRIDE:-/mnt/data1/jiaxuanluo/slm_local_cache/speech_llm_zh_cap16_denoise_budget_ttag_${RANK_TAG}_ep1_taurus4/latest-hf}"
HF_EXPORT_SWIFT_EXTRA_ARGS="${HF_EXPORT_SWIFT_EXTRA_ARGS_OVERRIDE:---device_map auto}"

MCORE_MODEL="${MCORE_MODEL_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-v2/}"
BASE_MODEL_HOST="${BASE_MODEL_HOST_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct}"
BASE_MODEL_STAGE_ROOT="${BASE_MODEL_STAGE_ROOT_OVERRIDE:-}"

MAX_EPOCHS="${MAX_EPOCHS_OVERRIDE:-1}"
MAX_LENGTH="${MAX_LENGTH_OVERRIDE:-3072}"
ITERATIONS_PER_EPOCH="${ITERATIONS_PER_EPOCH_OVERRIDE:-452}"
SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE:-${ITERATIONS_PER_EPOCH}}"
MASTER_PORT="${MASTER_PORT_OVERRIDE:-29726}"

for p in "${WRAPPER}" "${DATASET_PATH}" "${VAL_DATASET}" "${NOTES_FILE}" "${MCORE_MODEL}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

if [[ -n "${HOST_GPU_DEVICES_OVERRIDE_CSV:-}" ]]; then
  HOST_GPU_DEVICES_OVERRIDE="${HOST_GPU_DEVICES_OVERRIDE_CSV//:/,}"
fi
ALLOCATED_GPUS="${HOST_GPU_DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -r -a GPU_ARR <<< "${ALLOCATED_GPUS}"
if (( ${#GPU_ARR[@]} != 4 )); then
  echo "[ERROR] This launcher expects exactly 4 GPUs; got ${ALLOCATED_GPUS}" >&2
  exit 2
fi

TAGS=(
  "family:speech_llm_tcm_termmap"
  "task:train"
  "data:zh_c16_den_t"
  "variant:den_t_${RANK_TAG}_ep1_4g"
  "status:running"
  "compute:taurus4"
)
for tag in "${TAGS[@]}"; do
  if (( ${#tag} < 1 || ${#tag} > 64 )); then
    echo "[ERROR] WandB tag length out of range (${#tag}): ${tag}" >&2
    exit 2
  fi
done
WANDB_TAGS="$(IFS=,; echo "${TAGS[*]}")"
WANDB_NOTES="$(python3 - "${NOTES_FILE}" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).read_text(encoding="utf-8"))
PY
)"

echo "[INFO] LANG_CODE=${LANG_CODE}"
echo "[INFO] DATASET_PATH=${DATASET_PATH}"
echo "[INFO] VAL_DATASET=${VAL_DATASET}"
echo "[INFO] SAVE_BASE=${SAVE_BASE}"
echo "[INFO] TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
echo "[INFO] HOST_GPU_DEVICES=${ALLOCATED_GPUS}"
echo "[INFO] LORA_RANK=${LORA_RANK} LORA_ALPHA=${LORA_ALPHA}"
echo "[INFO] MAX_LENGTH=${MAX_LENGTH}"
echo "[INFO] ITERATIONS_PER_EPOCH=${ITERATIONS_PER_EPOCH}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
echo "[INFO] WANDB_TAGS=${WANDB_TAGS}"
echo "[INFO] HF_EXPORT_STAGE_ROOT=${HF_EXPORT_STAGE_ROOT}"
echo "[INFO] HF_EXPORT_LOCAL_CACHE_ROOT=${HF_EXPORT_LOCAL_CACHE_ROOT}"
echo "[INFO] HF_EXPORT_LOCAL_LATEST_LINK=${HF_EXPORT_LOCAL_LATEST_LINK}"
echo "[INFO] HF_EXPORT_SWIFT_EXTRA_ARGS=${HF_EXPORT_SWIFT_EXTRA_ARGS}"
echo "[INFO] BASE_MODEL_STAGE_ROOT=${BASE_MODEL_STAGE_ROOT:-<disabled>}"
df -h /mnt/data1 /mnt/gemini/data1 /mnt/gemini/data2 || true

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] Preflight complete; training not launched."
  exit 0
fi

if [[ "${SKIP_GPU_IDLE_PREFLIGHT:-0}" != "1" ]]; then
  python3 - "${ALLOCATED_GPUS}" "${MAX_GPU_MEM_MIB_OVERRIDE:-1000}" <<'PY'
import subprocess
import sys

expected = {int(x) for x in sys.argv[1].split(",") if x.strip()}
limit = int(sys.argv[2])
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
busy = []
seen = set()
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        continue
    idx, mem, util = map(int, parts)
    if idx not in expected:
        continue
    seen.add(idx)
    if mem > limit:
        busy.append((idx, mem, util))
missing = sorted(expected - seen)
if missing:
    raise SystemExit(f"[ERROR] Missing GPU ids from nvidia-smi: {missing}")
if busy:
    detail = ", ".join(f"{i}:mem={m}MiB util={u}%" for i, m, u in busy)
    raise SystemExit(f"[ERROR] Refusing to launch on busy GPUs: {detail}")
print(f"[INFO] GPU idle preflight passed for {sorted(expected)} under {limit}MiB")
PY
fi

mkdir -p "${SAVE_BASE}" "${TRAIN_LOG_DIR}"
unset CUDA_VISIBLE_DEVICES || true
unset NVIDIA_VISIBLE_DEVICES || true

BASE_MODEL_HOST="${BASE_MODEL_HOST}" \
BASE_MODEL_STAGE_ROOT="${BASE_MODEL_STAGE_ROOT}" \
HOST_GPU_DEVICES="${ALLOCATED_GPUS}" \
MOUNT_ROOTS="${MOUNT_ROOTS_OVERRIDE:-/mnt/gemini /mnt/taurus /mnt/aries /mnt/data /mnt/data1 /mnt/data2 /mnt/data7 /mnt/data6}" \
NPROC_PER_NODE=4 \
EXPERT_MODEL_PARALLEL_SIZE=2 \
TENSOR_MODEL_PARALLEL_SIZE=2 \
SEQUENCE_PARALLEL=true \
KEEP_RATIO="" \
DATASET_PATH="${DATASET_PATH}" \
VAL_DATASET="${VAL_DATASET}" \
SAVE_BASE="${SAVE_BASE}" \
TRAIN_LOG_DIR="${TRAIN_LOG_DIR}" \
MCORE_MODEL="${MCORE_MODEL}" \
LORA_RANK="${LORA_RANK}" \
LORA_ALPHA="${LORA_ALPHA}" \
MAX_EPOCHS="${MAX_EPOCHS}" \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=4 \
MAX_LENGTH="${MAX_LENGTH}" \
HF_EXPORT_STAGE_ROOT="${HF_EXPORT_STAGE_ROOT}" \
HF_EXPORT_MIN_FREE_GB="${HF_EXPORT_MIN_FREE_GB}" \
HF_EXPORT_SWIFT_EXTRA_ARGS="${HF_EXPORT_SWIFT_EXTRA_ARGS}" \
HF_EXPORT_LOCAL_CACHE_ROOT="${HF_EXPORT_LOCAL_CACHE_ROOT}" \
HF_EXPORT_LOCAL_LATEST_LINK="${HF_EXPORT_LOCAL_LATEST_LINK}" \
ITERATIONS_PER_EPOCH="${ITERATIONS_PER_EPOCH}" \
SAVE_INTERVAL="${SAVE_INTERVAL}" \
MASTER_PORT="${MASTER_PORT}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=INFO \
TORCH_NCCL_ENABLE_MONITORING=0 \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
WANDB_PROJECT="sst_omni" \
WANDB_EXP_PREFIX="${WANDB_EXP_PREFIX_OVERRIDE:-speech-llm-zh-c16-denoise-ttag-${RANK_TAG}-ep1-taurus4}" \
WANDB_TAGS="${WANDB_TAGS}" \
WANDB_NOTES="${WANDB_NOTES}" \
ENABLE_MD_UPDATE=0 \
bash "${WRAPPER}"
