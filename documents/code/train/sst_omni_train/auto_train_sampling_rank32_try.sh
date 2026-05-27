#!/usr/bin/env bash
set -euo pipefail


# docker run -it --rm \
#   --gpus all \
#   --shm-size=32g \
#   -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   -e NCCL_P2P_DISABLE=1 \
#   -e NCCL_IB_DISABLE=1 \
#   -v /home/jiaxuanluo/InfiniSST:/workspace/InfiniSST \
#   -v /mnt/gemini/data:/mnt/gemini/data \
#   -v /mnt/gemini/data1:/mnt/gemini/data1 \
#   -v /mnt/gemini/data2:/mnt/gemini/data2 \
#   modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1 \
#   /bin/bash

# Automated training + HF export for multiple sampling keep ratios.
#
# This script is designed to be run INSIDE the docker container environment
# described in documents/data/sst_omni_train_transcript.md (megatron + swift installed).
#
# It will:
# - run ONE job only (sampling keep=1.0)
# - use multiple GPUs (default: 8 GPUs: 0-7)
# - use LoRA rank=32
# - run `megatron sft` + `swift export`
# - write the HF output path back into documents/data/sst_omni_train_dataset.md table (keep_ratio=1.0)
#
# All logs are in English.

# ======Configuration=====
DEFAULT_ROOT_DIR_DOCKER="/workspace/InfiniSST"
DEFAULT_ROOT_DIR_LOCAL="/home/jiaxuanluo/InfiniSST"
DEFAULT_TRAIN_LOG_DIR_NAME="auto_train_sampling_rank32"
# ======Configuration=====

if [[ -d "${DEFAULT_ROOT_DIR_DOCKER}" ]]; then
  ROOT_DIR="${ROOT_DIR-${DEFAULT_ROOT_DIR_DOCKER}}"
else
  ROOT_DIR="${ROOT_DIR-${DEFAULT_ROOT_DIR_LOCAL}}"
fi
source "${ROOT_DIR}/documents/code/train/sst_omni_train/common/hf_export_staging.sh"
MD_PATH="${ROOT_DIR}/documents/data/sst_omni_train_dataset.md"
MD_UPDATER="${ROOT_DIR}/documents/code/update_sampling_models_md.py"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---- Required / recommended env vars (with safe defaults) ----
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
# Auto-compute NPROC_PER_NODE from CUDA_VISIBLE_DEVICES if not explicitly set.
if [[ -z "${NPROC_PER_NODE-}" ]]; then
  IFS=',' read -r -a _gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_gpu_ids[@]}"
fi
: "${ENABLE_AUDIO_OUTPUT:=False}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
: "${WANDB_PROJECT:=gigaspeech_zh}"
: "${WANDB_EXP_PREFIX:=omni-sampling}"

# Megatron-core model (produced by: swift export --model <HF_dir> --to_mcore true --output_dir <mcore_dir>).
: "${MCORE_MODEL:=/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-v2/}"
: "${VAL_DATASET:=/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20_final.jsonl}"

# Distributed rendezvous (must be unique per parallel job on the same host)
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29519}"

# Training output root (each ratio goes into SAVE_BASE/keep{ratio})
: "${SAVE_BASE:=/mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107}"

# Dataset path template
# Example:
# /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep0.5.seed1.jsonl
DATASET_PREFIX="${DATASET_PREFIX-/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep}"
DATASET_SUFFIX="${DATASET_SUFFIX-.seed1.jsonl}"
# Optional: direct dataset path. If set (non-empty), this has highest priority.
DATASET_PATH="${DATASET_PATH-}"

# Run configuration (single job)
KEEP_RATIO="${KEEP_RATIO-1.0}"
: "${LORA_RANK:=32}"
: "${LORA_ALPHA:=${LORA_RANK}}"
: "${MAX_EPOCHS:=1}"
# expert_model_parallel_size must divide NPROC_PER_NODE.
# Default: min(4, NPROC_PER_NODE) — use 4-way EP when enough GPUs, otherwise fall back.
if [[ -z "${EXPERT_MODEL_PARALLEL_SIZE-}" ]]; then
  if (( NPROC_PER_NODE >= 4 )); then
    EXPERT_MODEL_PARALLEL_SIZE=4
  else
    EXPERT_MODEL_PARALLEL_SIZE="${NPROC_PER_NODE}"
  fi
fi
: "${TENSOR_MODEL_PARALLEL_SIZE:=1}"
if [[ "${SEQUENCE_PARALLEL:-}" == "" ]]; then
  if (( EXPERT_MODEL_PARALLEL_SIZE > 1 && TENSOR_MODEL_PARALLEL_SIZE > 1 )); then
    SEQUENCE_PARALLEL="true"
  else
    SEQUENCE_PARALLEL="false"
  fi
fi
: "${MAX_LENGTH:=4096}"

# Batch size (must satisfy: global_batch_size % (micro_batch_size * data_parallel_size) == 0)
# In this script we approximate data_parallel_size == NPROC_PER_NODE for single-node runs.
: "${MICRO_BATCH_SIZE:=1}"
# If GLOBAL_BATCH_SIZE is not set, default to MICRO_BATCH_SIZE * NPROC_PER_NODE (so num_microbatches==1).
if [[ -z "${GLOBAL_BATCH_SIZE-}" ]]; then
  GLOBAL_BATCH_SIZE="$((MICRO_BATCH_SIZE * NPROC_PER_NODE))"
fi

# Save policy:
# - Megatron saves by iterations. To "save every epoch", set SAVE_INTERVAL to iterations_per_epoch.
# - For this dataset/config, one epoch was observed as ~452 iterations (global_batch_size=4).
#   Override if your setup differs.
: "${ITERATIONS_PER_EPOCH:=452}"
: "${SAVE_INTERVAL:=${ITERATIONS_PER_EPOCH}}"

# Logs
TRAIN_LOG_DIR="${TRAIN_LOG_DIR-${ROOT_DIR}/documents/logs/${DEFAULT_TRAIN_LOG_DIR_NAME}}"

# Preflight: NPROC_PER_NODE must not exceed the number of GPUs in CUDA_VISIBLE_DEVICES.
IFS=',' read -r -a _cvd_arr <<< "${CUDA_VISIBLE_DEVICES}"
_num_visible_gpus="${#_cvd_arr[@]}"
if (( NPROC_PER_NODE > _num_visible_gpus )); then
  echo "Error: NPROC_PER_NODE (${NPROC_PER_NODE}) > number of GPUs in CUDA_VISIBLE_DEVICES (${_num_visible_gpus}: ${CUDA_VISIBLE_DEVICES})." >&2
  echo "Either set CUDA_VISIBLE_DEVICES to ${NPROC_PER_NODE} GPUs, or reduce NPROC_PER_NODE." >&2
  exit 2
fi

if (( NPROC_PER_NODE % EXPERT_MODEL_PARALLEL_SIZE != 0 )); then
  echo "Error: NPROC_PER_NODE (${NPROC_PER_NODE}) must be divisible by EXPERT_MODEL_PARALLEL_SIZE (${EXPERT_MODEL_PARALLEL_SIZE})." >&2
  exit 2
fi
if (( NPROC_PER_NODE % TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
  echo "Error: NPROC_PER_NODE (${NPROC_PER_NODE}) must be divisible by TENSOR_MODEL_PARALLEL_SIZE (${TENSOR_MODEL_PARALLEL_SIZE})." >&2
  exit 2
fi

echo "Starting automated sampling training..."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE}"
echo "TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}"
echo "SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL}"
echo "MCORE_MODEL=${MCORE_MODEL}"
echo "VAL_DATASET=${VAL_DATASET}"
echo "SAVE_BASE=${SAVE_BASE}"
echo "KEEP_RATIO=${KEEP_RATIO}"
echo "LORA_RANK=${LORA_RANK}"
echo "LORA_ALPHA=${LORA_ALPHA}"
echo "MAX_EPOCHS=${MAX_EPOCHS}"
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "ITERATIONS_PER_EPOCH=${ITERATIONS_PER_EPOCH}  SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Warning: WANDB_API_KEY is not set. Training will still run, but wandb logging may fail." >&2
fi

if [[ ! -d "${MCORE_MODEL}" ]]; then
  echo "Error: MCORE_MODEL directory does not exist: ${MCORE_MODEL}" >&2
  echo "Create it first: swift export --model <HF_model_dir> --to_mcore true --torch_dtype bfloat16 --output_dir ${MCORE_MODEL}" >&2
  exit 2
fi

pick_latest_run_dir() {
  local save_root="$1"
  # If there are subdirs, pick the newest; otherwise use save_root itself.
  local latest
  latest="$(ls -1dt "${save_root}"/*/ 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    # strip trailing slash
    echo "${latest%/}"
  else
    echo "${save_root}"
  fi
}

update_md_locked() {
  local ratio="$1"
  local hf_dir="$2"
  local lock_file="${MD_PATH}.lock"
  # Use file lock to avoid concurrent writes from parallel jobs.
  flock "${lock_file}" python "${MD_UPDATER}" --md "${MD_PATH}" --keep-ratio "${ratio}" --hf-path "${hf_dir}"
}

ratio="${KEEP_RATIO}"
if [[ -n "${DATASET_PATH}" ]]; then
  dataset_path="${DATASET_PATH}"
  ratio_label="${ratio:-direct}"
else
  dataset_path="${DATASET_PREFIX}${ratio}${DATASET_SUFFIX}"
  ratio_label="${ratio}"
fi
save_root="${SAVE_BASE}/keep${ratio_label}_r${LORA_RANK}"
wandb_exp_name="${WANDB_EXP_PREFIX}_keep${ratio_label}_r${LORA_RANK}"

if ! mkdir -p "${TRAIN_LOG_DIR}" 2>/dev/null; then
  TRAIN_LOG_DIR="/tmp/${DEFAULT_TRAIN_LOG_DIR_NAME}"
  mkdir -p "${TRAIN_LOG_DIR}"
  echo "Warning: cannot write TRAIN_LOG_DIR under ROOT_DIR, fallback to ${TRAIN_LOG_DIR}" >&2
fi
mkdir -p "${save_root}"
ts="$(date +%Y%m%d_%H%M%S)"
log_file="${TRAIN_LOG_DIR}/train_keep${ratio_label}_r${LORA_RANK}_${ts}.log"
export TORCHELASTIC_ERROR_FILE="${TRAIN_LOG_DIR}/torchelastic_error_$$_${ts}.json"

{
  echo ""
  echo "========================================"
  echo "keep_ratio=${ratio}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "master_addr=${MASTER_ADDR}"
  echo "master_port=${MASTER_PORT}"
  echo "micro_batch_size=${MICRO_BATCH_SIZE}"
  echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
  echo "dataset=${dataset_path}"
  echo "save_root=${save_root}"
  echo "log_file=${log_file}"
  echo "wandb_exp_name=${wandb_exp_name}"

  # Preflight check: avoid Megatron assertion
  if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * NPROC_PER_NODE) != 0 )); then
    echo "Error: GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by MICRO_BATCH_SIZE (${MICRO_BATCH_SIZE}) * NPROC_PER_NODE (${NPROC_PER_NODE})." >&2
    exit 2
  fi
  if [[ ! -f "${dataset_path}" ]]; then
    echo "Error: dataset file not found: ${dataset_path}" >&2
    exit 2
  fi
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" \
  ENABLE_AUDIO_OUTPUT="${ENABLE_AUDIO_OUTPUT}" \
  WANDB_API_KEY="${WANDB_API_KEY:-}" \
  TORCHELASTIC_ERROR_FILE="${TORCHELASTIC_ERROR_FILE}" \
  megatron sft \
      --load "${MCORE_MODEL}" \
      --dataset "${dataset_path}" \
      --val_dataset "${VAL_DATASET}" \
      --load_from_cache_file true \
      --train_type lora \
      --lora_rank "${LORA_RANK}" \
      --lora_alpha "${LORA_ALPHA}" \
      --target_modules all-linear \
      --freeze_llm false \
      --freeze_vit true \
      --freeze_aligner true \
      --vit_gradient_checkpointing false \
      --packing true \
      --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE}" \
      --expert_model_parallel_size "${EXPERT_MODEL_PARALLEL_SIZE}" \
      --sequence_parallel "${SEQUENCE_PARALLEL}" \
      --moe_permute_fusion true \
      --moe_grouped_gemm true \
      --moe_shared_expert_overlap true \
      --moe_aux_loss_coeff 1e-3 \
      --micro_batch_size "${MICRO_BATCH_SIZE}" \
      --global_batch_size "${GLOBAL_BATCH_SIZE}" \
      --recompute_granularity full \
      --recompute_method uniform \
      --recompute_num_layers 1 \
      --finetune true \
      --cross_entropy_loss_fusion true \
      --lr 1e-4 \
      --lr_warmup_fraction 0.05 \
      --min_lr 1e-5 \
      --weight_decay 0.01 \
      --clip_grad 1.0 \
      --max_epochs "${MAX_EPOCHS}" \
      --save "${save_root}" \
      --log_interval 100 \
      --eval_interval 1000 \
      --save_interval "${SAVE_INTERVAL}" \
      --max_length "${MAX_LENGTH}" \
      --num_workers 8 \
      --dataset_num_proc 8 \
      --no_save_optim true \
      --no_save_rng true \
      --attention_backend flash \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_exp_name "${wandb_exp_name}" \
      --strict True

  run_dir="$(pick_latest_run_dir "${save_root}")"
  hf_dir="${run_dir}-hf"

  echo "Training done."
  echo "run_dir=${run_dir}"
  echo "hf_dir=${hf_dir}"

  export_mcore_checkpoint_to_hf_staged "${run_dir}" "${hf_dir}"

  echo "Export done."

  if [[ -n "${ratio}" ]]; then
    update_md_locked "${ratio}" "${hf_dir}"
    echo "Markdown updated: keep_ratio=${ratio} -> ${hf_dir}"
  else
    echo "Markdown update skipped: KEEP_RATIO is empty (direct DATASET_PATH mode)."
  fi
} 2>&1 | tee "${log_file}"
train_exit=${PIPESTATUS[0]}

if [[ ${train_exit} -ne 0 ]]; then
  echo ""
  echo "Training failed (exit code ${train_exit}). To see the actual Python traceback, re-run with single GPU:"
  echo "  NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 GLOBAL_BATCH_SIZE=1 BASE_MODEL=<path> DATASET_PATH=<path> bash $0"
  if [[ -n "${TORCHELASTIC_ERROR_FILE:-}" ]] && [[ -f "${TORCHELASTIC_ERROR_FILE}" ]]; then
    echo ""
    echo "--- First rank error (TORCHELASTIC_ERROR_FILE) ---"
    cat "${TORCHELASTIC_ERROR_FILE}"
  fi
  exit ${train_exit}
fi

echo ""
echo "All done."

