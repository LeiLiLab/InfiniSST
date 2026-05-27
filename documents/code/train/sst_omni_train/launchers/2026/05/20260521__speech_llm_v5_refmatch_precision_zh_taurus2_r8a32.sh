#!/usr/bin/env bash
# 2-GPU Speech LLM SFT launcher for V5 refmatch precision term_map data.
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"

DATASET_PATH="${DATASET_PATH_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v5_refmatch_precision_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/train_s_zh_v5_refmatch_precision_termmap_lh1b88kw_tau073_srcmatch100k.jsonl}"
VAL_DATASET="${VAL_DATASET_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_v5_refmatch_precision_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/dev_s_zh_v5_refmatch_precision_termmap_lh1b88kw_tau073_srcmatch100k.jsonl}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/train/sst_omni_train/notes/2026/05/20260521__speech_llm_v5_refmatch_precision_zh_r8a32.md}"
SAVE_BASE="${SAVE_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/speech_llm_v5_refmatch_precision_termmap_zh_lh1b88kw_tau073_srcmatch100k_r8a32_taurus2}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR_OVERRIDE:-${ROOT_DIR}/documents/logs/speech_llm_v5_refmatch_precision_zh_r8a32_taurus2}"

MCORE_MODEL="${MCORE_MODEL_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-v2/}"
BASE_MODEL_HOST="${BASE_MODEL_HOST_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct}"

LORA_RANK="${LORA_RANK_OVERRIDE:-8}"
LORA_ALPHA="${LORA_ALPHA_OVERRIDE:-32}"
MAX_EPOCHS="${MAX_EPOCHS_OVERRIDE:-1}"
MAX_LENGTH="${MAX_LENGTH_OVERRIDE:-4096}"
ITERATIONS_PER_EPOCH="${ITERATIONS_PER_EPOCH_OVERRIDE:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE:-${ITERATIONS_PER_EPOCH}}"

for p in "${WRAPPER}" "${DATASET_PATH}" "${VAL_DATASET}" "${NOTES_FILE}" "${MCORE_MODEL}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

if [[ -n "${HOST_GPU_DEVICES_OVERRIDE_CSV:-}" ]]; then
  HOST_GPU_DEVICES_OVERRIDE="${HOST_GPU_DEVICES_OVERRIDE_CSV//:/,}"
fi
ALLOCATED_GPUS="${HOST_GPU_DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0,1}}"
IFS=',' read -r -a GPU_ARR <<< "${ALLOCATED_GPUS}"
if (( ${#GPU_ARR[@]} != 2 )); then
  echo "[ERROR] This launcher expects exactly 2 GPUs; got ${ALLOCATED_GPUS}" >&2
  exit 2
fi

TAGS=(
  "family:speech_llm_termmap_retriever"
  "task:train"
  "data:v5_refm_zh"
  "variant:${WANDB_VARIANT_TAG_OVERRIDE:-v5_refm_r8a32}"
  "status:running"
  "compute:taurus2"
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

echo "[INFO] DATASET_PATH=${DATASET_PATH}"
echo "[INFO] VAL_DATASET=${VAL_DATASET}"
echo "[INFO] SAVE_BASE=${SAVE_BASE}"
echo "[INFO] TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
echo "[INFO] MCORE_MODEL=${MCORE_MODEL}"
echo "[INFO] HOST_GPU_DEVICES=${ALLOCATED_GPUS}"
echo "[INFO] LORA_RANK=${LORA_RANK} LORA_ALPHA=${LORA_ALPHA}"
echo "[INFO] MAX_LENGTH=${MAX_LENGTH}"
echo "[INFO] ITERATIONS_PER_EPOCH=${ITERATIONS_PER_EPOCH} SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "[INFO] WANDB_TAGS=${WANDB_TAGS}"

unset CUDA_VISIBLE_DEVICES || true
unset NVIDIA_VISIBLE_DEVICES || true

BASE_MODEL_HOST="${BASE_MODEL_HOST}" \
HOST_GPU_DEVICES="${ALLOCATED_GPUS}" \
NPROC_PER_NODE=2 \
EXPERT_MODEL_PARALLEL_SIZE=2 \
TENSOR_MODEL_PARALLEL_SIZE=1 \
SEQUENCE_PARALLEL=false \
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
GLOBAL_BATCH_SIZE=2 \
MAX_LENGTH="${MAX_LENGTH}" \
ITERATIONS_PER_EPOCH="${ITERATIONS_PER_EPOCH}" \
SAVE_INTERVAL="${SAVE_INTERVAL}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=INFO \
TORCH_NCCL_ENABLE_MONITORING=0 \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
WANDB_PROJECT="sst_omni" \
WANDB_EXP_PREFIX="${WANDB_EXP_PREFIX_OVERRIDE:-speech-llm-v5-refm-precision-zh-r8a32-taurus2}" \
WANDB_TAGS="${WANDB_TAGS}" \
WANDB_NOTES="${WANDB_NOTES}" \
ENABLE_MD_UPDATE=0 \
bash "${WRAPPER}"
