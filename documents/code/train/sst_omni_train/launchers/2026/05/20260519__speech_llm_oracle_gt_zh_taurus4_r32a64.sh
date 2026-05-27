#!/usr/bin/env bash
#SBATCH --job-name=slm_oraclegt_zh
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_slm_oraclegt_zh_taurus4.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_slm_oraclegt_zh_taurus4.err

set -euo pipefail

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"
NOTES_FILE="${ROOT_DIR}/documents/code/train/sst_omni_train/notes/2026/05/20260519__speech_llm_oracle_gt_zh_r32a64.md"

DATASET_PATH="/mnt/gemini/data1/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_20260519/train_s_zh_v4_ner_baseline_aligned_rate1p0_k20_oracle_gt_termmap_none.jsonl"
VAL_DATASET="/mnt/gemini/data1/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_20260519/dev_s_zh_v4_ner_baseline_aligned_freq_k20_oracle_gt_termmap_none.jsonl"
SAVE_BASE="/mnt/gemini/data2/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_r32a64_taurus4"
TRAIN_LOG_DIR="${ROOT_DIR}/documents/logs/speech_llm_oracle_gt_zh_r32a64_taurus4"

# Default is the initial Qwen3-Omni mcore base used by the original SFT recipe.
# The pure-streaming HF baseline is evaluated zero-shot instead of being
# converted to mcore for additional SFT.
MCORE_MODEL="${MCORE_MODEL_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-v2/}"
BASE_MODEL_HOST="${BASE_MODEL_HOST_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct}"

LORA_RANK="${LORA_RANK_OVERRIDE:-32}"
LORA_ALPHA="${LORA_ALPHA_OVERRIDE:-64}"
MAX_EPOCHS="${MAX_EPOCHS_OVERRIDE:-1}"
ITERATIONS_PER_EPOCH="${ITERATIONS_PER_EPOCH_OVERRIDE:-452}"
SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE:-${ITERATIONS_PER_EPOCH}}"

for p in "${WRAPPER}" "${NOTES_FILE}" "${DATASET_PATH}" "${VAL_DATASET}" "${MCORE_MODEL}"; do
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
  "family:speech_llm_termmap_oracle"
  "task:train"
  "data:oracle_gt_zh"
  "variant:oraclegt_r32a64"
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

echo "[INFO] DATASET_PATH=${DATASET_PATH}"
echo "[INFO] VAL_DATASET=${VAL_DATASET}"
echo "[INFO] SAVE_BASE=${SAVE_BASE}"
echo "[INFO] TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"
echo "[INFO] MCORE_MODEL=${MCORE_MODEL}"
echo "[INFO] HOST_GPU_DEVICES=${ALLOCATED_GPUS}"
echo "[INFO] LORA_RANK=${LORA_RANK} LORA_ALPHA=${LORA_ALPHA}"
echo "[INFO] ITERATIONS_PER_EPOCH=${ITERATIONS_PER_EPOCH} SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "[INFO] WANDB_TAGS=${WANDB_TAGS}"

# SLURM + docker: host CUDA_VISIBLE_DEVICES with `docker --gpus device=...` can error
# "cannot set both Count and DeviceIDs". Wrapper sets container CUDA to 0..N-1.
unset CUDA_VISIBLE_DEVICES || true
unset NVIDIA_VISIBLE_DEVICES || true

BASE_MODEL_HOST="${BASE_MODEL_HOST}" \
HOST_GPU_DEVICES="${ALLOCATED_GPUS}" \
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
MAX_LENGTH=3072 \
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
WANDB_EXP_PREFIX="speech-llm-oraclegt-zh-r32a64-taurus4" \
WANDB_TAGS="${WANDB_TAGS}" \
WANDB_NOTES="${WANDB_NOTES}" \
ENABLE_MD_UPDATE=0 \
bash "${WRAPPER}"
