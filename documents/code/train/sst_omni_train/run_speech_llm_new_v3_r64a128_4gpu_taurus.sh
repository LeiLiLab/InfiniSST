#!/usr/bin/env bash
#SBATCH --job-name=slm_v3_r64_4g
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_slm_v3_r64_4g.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_slm_v3_r64_4g.err

set -euo pipefail

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"
NOTES_FILE="${ROOT_DIR}/documents/code/train/sst_omni_train/notes_speech_llm_new_v3_r64a128.md"

DATASET_PATH="/mnt/gemini/data1/jiaxuanluo/tcm_wiki100k_gt_zh_tau075_d9_k20_postfiltercap_termmap_v2_sourcefinal_gtzh/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_sourcefinal_tcmwiki100kgt_tau075_d9_k20_postfiltercap_gtzhoverride.jsonl"
VAL_DATASET="/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20_final.jsonl"
SAVE_BASE="/mnt/gemini/data2/jiaxuanluo/speech_llm_tcmw100kgt_tau075_new_v3_r64a128_taurus4"
TRAIN_LOG_DIR="${ROOT_DIR}/documents/logs/speech_llm_new_v3_r64a128_taurus4"

for p in "${WRAPPER}" "${NOTES_FILE}" "${DATASET_PATH}" "${VAL_DATASET}"; do
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
  echo "[ERROR] This launcher expects exactly 4 allocated GPUs; got ${ALLOCATED_GPUS}" >&2
  exit 2
fi

TAGS=(
  "family:speech_llm_tcm_termmap"
  "task:train"
  "data:tcmw100kgt075v3"
  "variant:v3_r64a128"
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
echo "[INFO] HOST_GPU_DEVICES=${ALLOCATED_GPUS}"
echo "[INFO] WANDB_TAGS=${WANDB_TAGS}"

unset CUDA_VISIBLE_DEVICES || true
unset NVIDIA_VISIBLE_DEVICES || true

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
LORA_RANK=64 \
LORA_ALPHA=128 \
MAX_EPOCHS=1 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=4 \
MAX_LENGTH=3072 \
ITERATIONS_PER_EPOCH=452 \
SAVE_INTERVAL=452 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=INFO \
TORCH_NCCL_ENABLE_MONITORING=0 \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
WANDB_PROJECT="sst_omni" \
WANDB_EXP_PREFIX="speech-llm-new_v3-r64a128-taurus4" \
WANDB_TAGS="${WANDB_TAGS}" \
WANDB_NOTES="${WANDB_NOTES}" \
ENABLE_MD_UPDATE=0 \
bash "${WRAPPER}"
