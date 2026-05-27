#!/usr/bin/env bash
#SBATCH --job-name=slm_v2_rank
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --time=06:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_slm_v2_rank_taurus.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_slm_v2_rank_taurus.err

set -euo pipefail

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
WRAPPER="${ROOT_DIR}/documents/code/train/sst_omni_train/auto_train_sampling_docker.sh"

LORA_RANK_OVERRIDE="${LORA_RANK_OVERRIDE:?Set LORA_RANK_OVERRIDE to 64 or 128}"
case "${LORA_RANK_OVERRIDE}" in
  64|128) ;;
  *) echo "[ERROR] Expected LORA_RANK_OVERRIDE=64 or 128, got ${LORA_RANK_OVERRIDE}" >&2; exit 2 ;;
esac

NOTES_FILE="${ROOT_DIR}/documents/code/train/sst_omni_train/notes_speech_llm_tcmwiki100k_tau075_sourcefinal_gtzh_v2_rank${LORA_RANK_OVERRIDE}.md"
DATASET_PATH="/mnt/gemini/data1/jiaxuanluo/tcm_wiki100k_gt_zh_tau075_termmap_v2_sourcefinal_gtzh/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_sourcefinal_tcmwiki100kgt_tau075_gtzhoverride.jsonl"
VAL_DATASET="/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20_final.jsonl"
SAVE_BASE="/mnt/gemini/data2/jiaxuanluo/speech_llm_tcm_filtered_wiki100kgt_tau075_v2_sourcefinal_gtzh_rank_sweep"
TRAIN_LOG_DIR="${ROOT_DIR}/documents/logs/speech_llm_tcmwiki100kgt_tau075_v2_rank_sweep_taurus"

for p in "${WRAPPER}" "${NOTES_FILE}" "${DATASET_PATH}" "${VAL_DATASET}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

if [[ -n "${HOST_GPU_DEVICES_OVERRIDE_CSV:-}" ]]; then
  HOST_GPU_DEVICES_OVERRIDE="${HOST_GPU_DEVICES_OVERRIDE_CSV//:/,}"
fi
HOST_GPU_DEVICES_OVERRIDE="${HOST_GPU_DEVICES_OVERRIDE:-0,1,2,3,4,5,6,7}"
ALLOCATED_GPUS="${HOST_GPU_DEVICES_OVERRIDE:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
IFS=',' read -r -a GPU_ARR <<< "${ALLOCATED_GPUS}"
if (( ${#GPU_ARR[@]} != 8 )); then
  echo "[ERROR] This launcher expects exactly 8 allocated GPUs; got ${ALLOCATED_GPUS}" >&2
  exit 2
fi

TAGS=(
  "family:speech_llm_tcm_termmap"
  "task:train"
  "data:tcmw100kgt075v2"
  "variant:srcfinal_r${LORA_RANK_OVERRIDE}"
  "status:running"
  "compute:taurus8"
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

echo "[INFO] LORA_RANK=${LORA_RANK_OVERRIDE}"
echo "[INFO] DATASET_PATH=${DATASET_PATH}"
echo "[INFO] VAL_DATASET=${VAL_DATASET}"
echo "[INFO] SAVE_BASE=${SAVE_BASE}"
echo "[INFO] HOST_GPU_DEVICES=${ALLOCATED_GPUS}"
echo "[INFO] WANDB_TAGS=${WANDB_TAGS}"

unset CUDA_VISIBLE_DEVICES || true
unset NVIDIA_VISIBLE_DEVICES || true

HOST_GPU_DEVICES="${ALLOCATED_GPUS}" \
NPROC_PER_NODE=8 \
EXPERT_MODEL_PARALLEL_SIZE=4 \
TENSOR_MODEL_PARALLEL_SIZE=2 \
SEQUENCE_PARALLEL=true \
KEEP_RATIO="" \
DATASET_PATH="${DATASET_PATH}" \
VAL_DATASET="${VAL_DATASET}" \
SAVE_BASE="${SAVE_BASE}" \
TRAIN_LOG_DIR="${TRAIN_LOG_DIR}" \
LORA_RANK="${LORA_RANK_OVERRIDE}" \
LORA_ALPHA="${LORA_RANK_OVERRIDE}" \
MAX_EPOCHS=1 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=8 \
MAX_LENGTH=3072 \
ITERATIONS_PER_EPOCH=226 \
SAVE_INTERVAL=226 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=INFO \
TORCH_NCCL_ENABLE_MONITORING=0 \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
WANDB_PROJECT="sst_omni" \
WANDB_EXP_PREFIX="speech-llm-tcmw100kgt-v2-srcfinal-r${LORA_RANK_OVERRIDE}-taurus8" \
WANDB_TAGS="${WANDB_TAGS}" \
WANDB_NOTES="${WANDB_NOTES}" \
ENABLE_MD_UPDATE=0 \
bash "${WRAPPER}"
