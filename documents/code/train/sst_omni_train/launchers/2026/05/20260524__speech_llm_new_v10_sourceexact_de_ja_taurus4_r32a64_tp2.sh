#!/usr/bin/env bash
# CANCELED / DO NOT USE:
# This v13/sourceexact-derived de/ja branch is the wrong lineage for the final
# Speech LLM line.  The mainline zh model comes from old new_v3 r32 -> new_v4 ->
# new_v5 -> new_v9, not v13/sourceexact.  Kept only as a canceled provenance
# record; do not launch.
echo "[CANCELED] Wrong lineage: use old new_v3 r32 -> new_v4 -> new_v5 -> new_v9 instead." >&2
exit 2
# 4-GPU Speech LLM SFT launcher for clean source-exact New V10 de/ja.
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"

LANG_CODE="${LANG_CODE_OVERRIDE:-de}"
if [[ "${LANG_CODE}" != "de" && "${LANG_CODE}" != "ja" ]]; then
  echo "[ERROR] LANG_CODE_OVERRIDE must be de or ja; got ${LANG_CODE}" >&2
  exit 2
fi

DATA_DIR="${DATA_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_sourceexact_${LANG_CODE}_20260524}"
DATASET_PATH="${DATASET_PATH_OVERRIDE:-${DATA_DIR}/train_s_${LANG_CODE}_new_v10_sourceexact_llmvariant_no_gt_zero_termtag_boundary.jsonl}"
VAL_DATASET="${VAL_DATASET_OVERRIDE:-${DATA_DIR}/dev_s_${LANG_CODE}_new_v10_sourceexact_llmvariant_no_gt_zero_termtag_boundary_first355.jsonl}"

LORA_RANK="${LORA_RANK_OVERRIDE:-32}"
LORA_ALPHA="${LORA_ALPHA_OVERRIDE:-64}"
RANK_TAG="r${LORA_RANK}a${LORA_ALPHA}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/train/sst_omni_train/notes/2026/05/20260524__speech_llm_new_v10_sourceexact_${LANG_CODE}_r32a64_tp2.md}"
SAVE_BASE="${SAVE_BASE_OVERRIDE:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_new_v10_sourceexact_${LANG_CODE}_${RANK_TAG}_tp2_taurus4}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/speech_llm_new_v10_sourceexact_${LANG_CODE}_${RANK_TAG}_tp2_taurus4}"

MCORE_MODEL="${MCORE_MODEL_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-v2/}"
BASE_MODEL_HOST="${BASE_MODEL_HOST_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct}"

MAX_EPOCHS="${MAX_EPOCHS_OVERRIDE:-1}"
MAX_LENGTH="${MAX_LENGTH_OVERRIDE:-3072}"
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
ALLOCATED_GPUS="${HOST_GPU_DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0,1,6,7}}"
IFS=',' read -r -a GPU_ARR <<< "${ALLOCATED_GPUS}"
if (( ${#GPU_ARR[@]} != 4 )); then
  echo "[ERROR] This launcher expects exactly 4 GPUs; got ${ALLOCATED_GPUS}" >&2
  exit 2
fi

TAGS=(
  "family:speech_llm_tcm_termmap"
  "task:train"
  "data:new_v10_${LANG_CODE}_sourceexact"
  "variant:new_v10_${LANG_CODE}_${RANK_TAG}_tp2"
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
echo "[INFO] WANDB_TAGS=${WANDB_TAGS}"

mkdir -p "${SAVE_BASE}" "${TRAIN_LOG_DIR}"
unset CUDA_VISIBLE_DEVICES || true
unset NVIDIA_VISIBLE_DEVICES || true

BASE_MODEL_HOST="${BASE_MODEL_HOST}" \
HOST_GPU_DEVICES="${ALLOCATED_GPUS}" \
MOUNT_ROOTS="${MOUNT_ROOTS_OVERRIDE:-/mnt/gemini /mnt/taurus /mnt/aries /mnt/data7 /mnt/data6}" \
NPROC_PER_NODE=4 \
EXPERT_MODEL_PARALLEL_SIZE=4 \
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
WANDB_EXP_PREFIX="${WANDB_EXP_PREFIX_OVERRIDE:-speech-llm-new_v10-sourceexact-${LANG_CODE}-${RANK_TAG}-tp2-taurus4}" \
WANDB_TAGS="${WANDB_TAGS}" \
WANDB_NOTES="${WANDB_NOTES}" \
ENABLE_MD_UPDATE=0 \
bash "${WRAPPER}"
