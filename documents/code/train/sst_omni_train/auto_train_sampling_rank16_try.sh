#!/usr/bin/env bash
set -euo pipefail

# Automated training + HF export for multiple sampling keep ratios.
#
# This script is designed to be run INSIDE the docker container environment
# described in documents/data/sst_omni_train_transcript.md (megatron + swift installed).
#
# It will:
# - run ONE job only (sampling keep=1.0)
# - use multiple GPUs (default: 4 GPUs: 4,5,6,7)
# - use LoRA rank=16
# - run `megatron sft` + `swift export`
# - write the HF output path back into documents/data/sst_omni_train_dataset.md table (keep_ratio=1.0)
#
# All logs are in English.

###### ======Configuration=====
# Notes:
# - Avoid hard-coded absolute paths. Prefer environment variables when possible.
# - All strings/logs are in English by design.

# Project layout
DEFAULT_ROOT_DIR_REL_FROM_SCRIPT="../../../"

# Hardware / distributed
DEFAULT_CUDA_VISIBLE_DEVICES="4,5,6,7"
DEFAULT_NPROC_PER_NODE="4"
DEFAULT_MASTER_ADDR="127.0.0.1"
DEFAULT_MASTER_PORT="29519"

# Training run (single job)
DEFAULT_KEEP_RATIO="1.0"
DEFAULT_LORA_RANK="16"
DEFAULT_MAX_EPOCHS="1"

# Batch size
DEFAULT_MICRO_BATCH_SIZE="1"

# Save policy
DEFAULT_ITERATIONS_PER_EPOCH="452"

# Logging
DEFAULT_TRAIN_LOG_SUBDIR="documents/logs/auto_train_sampling_rank16"

# GPU diagnostics / preflight
DEFAULT_ENABLE_GPU_PRECHECK="True"
DEFAULT_STRICT_GPU_IDLE="True"

# Model / datasets (read paths; customize per cluster)
DEFAULT_BASE_MODEL="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-v2/"
DEFAULT_VAL_DATASET="/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20_final.jsonl"
DEFAULT_DATASET_PREFIX="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep"
DEFAULT_DATASET_SUFFIX=".seed1.jsonl"

# Output dirs (write paths; must be writable on the current machine)
DEFAULT_SAVE_BASE_PRIMARY="/mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107"
DEFAULT_SAVE_BASE_FALLBACK="${HOME}/InfiniSST_outputs/Omni-30B-sampling-0107"

# W&B
DEFAULT_WANDB_PROJECT="gigaspeech_zh"
DEFAULT_WANDB_EXP_PREFIX="omni-sampling"
WANDB_API_KEY=${WANDB_API_KEY:-}

# Megatron/Swift misc
DEFAULT_ENABLE_AUDIO_OUTPUT="False"
DEFAULT_PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Megatron hyperparameters
DEFAULT_LORA_ALPHA="32"
DEFAULT_TARGET_MODULES="all-linear"
DEFAULT_FREEZE_LLM="false"
DEFAULT_FREEZE_VIT="true"
DEFAULT_FREEZE_ALIGNER="true"
DEFAULT_VIT_GRADIENT_CHECKPOINTING="false"
DEFAULT_PACKING="true"
DEFAULT_EXPERT_MODEL_PARALLEL_SIZE="4"
DEFAULT_MOE_PERMUTE_FUSION="true"
DEFAULT_MOE_GROUPED_GEMM="true"
DEFAULT_MOE_SHARED_EXPERT_OVERLAP="true"
DEFAULT_MOE_AUX_LOSS_COEFF="1e-3"
DEFAULT_RECOMPUTE_GRANULARITY="full"
DEFAULT_RECOMPUTE_METHOD="uniform"
DEFAULT_RECOMPUTE_NUM_LAYERS="1"
DEFAULT_FINETUNE="true"
DEFAULT_CROSS_ENTROPY_LOSS_FUSION="true"
DEFAULT_LR="1e-4"
DEFAULT_LR_WARMUP_FRACTION="0.05"
DEFAULT_MIN_LR="1e-5"
DEFAULT_WEIGHT_DECAY="0.01"
DEFAULT_CLIP_GRAD="1.0"
DEFAULT_LOG_INTERVAL="100"
DEFAULT_EVAL_INTERVAL="1000"
DEFAULT_MAX_LENGTH="4096"
DEFAULT_NUM_WORKERS="8"
DEFAULT_DATASET_NUM_PROC="8"
DEFAULT_NO_SAVE_OPTIM="true"
DEFAULT_NO_SAVE_RNG="true"
DEFAULT_ATTENTION_BACKEND="flash"
DEFAULT_STRICT="True"

# Swift export
DEFAULT_SWIFT_TORCH_DTYPE="bfloat16"

# Markdown update
DEFAULT_ENABLE_MD_UPDATE="True"
###### ======Configuration=====

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

if [[ -z "${ROOT_DIR-}" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/${DEFAULT_ROOT_DIR_REL_FROM_SCRIPT}" && pwd)"
fi

source "${ROOT_DIR}/documents/code/train/sst_omni_train/common/hf_export_staging.sh"
MD_PATH="${ROOT_DIR}/documents/data/sst_omni_train_dataset.md"
MD_UPDATER="${ROOT_DIR}/documents/code/update_sampling_models_md.py"

# ---- Required / recommended env vars (with safe defaults) ----
: "${NPROC_PER_NODE:=${DEFAULT_NPROC_PER_NODE}}"
: "${ENABLE_AUDIO_OUTPUT:=${DEFAULT_ENABLE_AUDIO_OUTPUT}}"
: "${PYTORCH_CUDA_ALLOC_CONF:=${DEFAULT_PYTORCH_CUDA_ALLOC_CONF}}"
: "${WANDB_PROJECT:=${DEFAULT_WANDB_PROJECT}}"
: "${WANDB_EXP_PREFIX:=${DEFAULT_WANDB_EXP_PREFIX}}"
: "${MEGATRON_CMD:=megatron}"
: "${SWIFT_CMD:=swift}"
: "${WANDB_API_KEY:=${WANDB_API_KEY}}"

: "${ENABLE_GPU_PRECHECK:=${DEFAULT_ENABLE_GPU_PRECHECK}}"
: "${STRICT_GPU_IDLE:=${DEFAULT_STRICT_GPU_IDLE}}"

: "${LORA_ALPHA:=${DEFAULT_LORA_ALPHA}}"
: "${TARGET_MODULES:=${DEFAULT_TARGET_MODULES}}"
: "${FREEZE_LLM:=${DEFAULT_FREEZE_LLM}}"
: "${FREEZE_VIT:=${DEFAULT_FREEZE_VIT}}"
: "${FREEZE_ALIGNER:=${DEFAULT_FREEZE_ALIGNER}}"
: "${VIT_GRADIENT_CHECKPOINTING:=${DEFAULT_VIT_GRADIENT_CHECKPOINTING}}"
: "${PACKING:=${DEFAULT_PACKING}}"
: "${EXPERT_MODEL_PARALLEL_SIZE:=${DEFAULT_EXPERT_MODEL_PARALLEL_SIZE}}"
: "${MOE_PERMUTE_FUSION:=${DEFAULT_MOE_PERMUTE_FUSION}}"
: "${MOE_GROUPED_GEMM:=${DEFAULT_MOE_GROUPED_GEMM}}"
: "${MOE_SHARED_EXPERT_OVERLAP:=${DEFAULT_MOE_SHARED_EXPERT_OVERLAP}}"
: "${MOE_AUX_LOSS_COEFF:=${DEFAULT_MOE_AUX_LOSS_COEFF}}"
: "${RECOMPUTE_GRANULARITY:=${DEFAULT_RECOMPUTE_GRANULARITY}}"
: "${RECOMPUTE_METHOD:=${DEFAULT_RECOMPUTE_METHOD}}"
: "${RECOMPUTE_NUM_LAYERS:=${DEFAULT_RECOMPUTE_NUM_LAYERS}}"
: "${FINETUNE:=${DEFAULT_FINETUNE}}"
: "${CROSS_ENTROPY_LOSS_FUSION:=${DEFAULT_CROSS_ENTROPY_LOSS_FUSION}}"
: "${LR:=${DEFAULT_LR}}"
: "${LR_WARMUP_FRACTION:=${DEFAULT_LR_WARMUP_FRACTION}}"
: "${MIN_LR:=${DEFAULT_MIN_LR}}"
: "${WEIGHT_DECAY:=${DEFAULT_WEIGHT_DECAY}}"
: "${CLIP_GRAD:=${DEFAULT_CLIP_GRAD}}"
: "${LOG_INTERVAL:=${DEFAULT_LOG_INTERVAL}}"
: "${EVAL_INTERVAL:=${DEFAULT_EVAL_INTERVAL}}"
: "${MAX_LENGTH:=${DEFAULT_MAX_LENGTH}}"
: "${NUM_WORKERS:=${DEFAULT_NUM_WORKERS}}"
: "${DATASET_NUM_PROC:=${DEFAULT_DATASET_NUM_PROC}}"
: "${NO_SAVE_OPTIM:=${DEFAULT_NO_SAVE_OPTIM}}"
: "${NO_SAVE_RNG:=${DEFAULT_NO_SAVE_RNG}}"
: "${ATTENTION_BACKEND:=${DEFAULT_ATTENTION_BACKEND}}"
: "${STRICT:=${DEFAULT_STRICT}}"

: "${SWIFT_TORCH_DTYPE:=${DEFAULT_SWIFT_TORCH_DTYPE}}"

is_readonly_var() {
  local var_name="$1"
  declare -p "${var_name}" 2>/dev/null | grep -q "declare -r"
}

if [[ -z "${CUDA_VISIBLE_DEVICES-}" ]]; then
  if is_readonly_var "CUDA_VISIBLE_DEVICES"; then
    die "CUDA_VISIBLE_DEVICES is readonly but empty. Please set it via scheduler (e.g., srun/sbatch) or export it before running."
  fi
  CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES}"
fi

gpu_precheck() {
  if [[ "${ENABLE_GPU_PRECHECK}" != "True" ]]; then
    echo "Info: ENABLE_GPU_PRECHECK is not True; skip GPU precheck."
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: nvidia-smi not found; skip GPU precheck." >&2
    return 0
  fi

  echo "GPU precheck: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  local -a gpu_ids=()
  IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"

  local any_busy="False"
  local id
  for id in "${gpu_ids[@]}"; do
    if [[ -z "${id}" ]]; then
      continue
    fi
    if ! nvidia-smi -i "${id}" >/dev/null 2>&1; then
      echo "Warning: nvidia-smi cannot query GPU id '${id}'. It may be a UUID or remapped index; skip per-GPU check." >&2
      continue
    fi

    local procs
    procs="$(nvidia-smi -i "${id}" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "${procs}" ]]; then
      any_busy="True"
      echo "Warning: GPU ${id} has existing compute processes:" >&2
      echo "${procs}" >&2
    fi
  done

  if [[ "${any_busy}" == "True" && "${STRICT_GPU_IDLE}" == "True" ]]; then
    die "Selected GPUs are not idle. Choose different CUDA_VISIBLE_DEVICES or request exclusive GPUs. Tip: on the allocated node, run: nvidia-smi -i <gpu_id>."
  fi
}

# Base model / datasets
: "${BASE_MODEL:=${DEFAULT_BASE_MODEL}}"
: "${VAL_DATASET:=${DEFAULT_VAL_DATASET}}"

# Distributed rendezvous (must be unique per parallel job on the same host)
: "${MASTER_ADDR:=${DEFAULT_MASTER_ADDR}}"
: "${MASTER_PORT:=${DEFAULT_MASTER_PORT}}"

# Training output root (each ratio goes into SAVE_BASE/keep{ratio})
if [[ -z "${SAVE_BASE-}" ]]; then
  if [[ -d "${DEFAULT_SAVE_BASE_PRIMARY%/*}" && -w "${DEFAULT_SAVE_BASE_PRIMARY%/*}" ]]; then
    SAVE_BASE="${DEFAULT_SAVE_BASE_PRIMARY}"
  else
    SAVE_BASE="${DEFAULT_SAVE_BASE_FALLBACK}"
  fi
fi

# Dataset path template
# Example:
# /mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_enriched_with_negatives.sample_keep0.5.seed1.jsonl
: "${DATASET_PREFIX:=${DEFAULT_DATASET_PREFIX}}"
: "${DATASET_SUFFIX:=${DEFAULT_DATASET_SUFFIX}}"

# Run configuration (single job)
: "${KEEP_RATIO:=${DEFAULT_KEEP_RATIO}}"
: "${LORA_RANK:=${DEFAULT_LORA_RANK}}"
: "${MAX_EPOCHS:=${DEFAULT_MAX_EPOCHS}}"

# Batch size (must satisfy: global_batch_size % (micro_batch_size * data_parallel_size) == 0)
# In this script we approximate data_parallel_size == NPROC_PER_NODE for single-node runs.
: "${MICRO_BATCH_SIZE:=${DEFAULT_MICRO_BATCH_SIZE}}"
# If GLOBAL_BATCH_SIZE is not set, default to MICRO_BATCH_SIZE * NPROC_PER_NODE (so num_microbatches==1).
if [[ -z "${GLOBAL_BATCH_SIZE-}" ]]; then
  GLOBAL_BATCH_SIZE="$((MICRO_BATCH_SIZE * NPROC_PER_NODE))"
fi

# Save policy:
# - Megatron saves by iterations. To "save every epoch", set SAVE_INTERVAL to iterations_per_epoch.
# - For this dataset/config, one epoch was observed as ~452 iterations (global_batch_size=4).
#   Override if your setup differs.
: "${ITERATIONS_PER_EPOCH:=${DEFAULT_ITERATIONS_PER_EPOCH}}"
: "${SAVE_INTERVAL:=${ITERATIONS_PER_EPOCH}}"

# Logs
if [[ -z "${TRAIN_LOG_DIR-}" ]]; then
  if [[ -w "${ROOT_DIR}" ]]; then
    TRAIN_LOG_DIR="${ROOT_DIR}/${DEFAULT_TRAIN_LOG_SUBDIR}"
  else
    TRAIN_LOG_DIR="${HOME}/InfiniSST_logs/auto_train_sampling_rank16"
  fi
fi

: "${ENABLE_MD_UPDATE:=${DEFAULT_ENABLE_MD_UPDATE}}"

die() {
  local msg="$1"
  echo "Error: ${msg}" >&2
  exit 1
}

ensure_dir() {
  local dir_path="$1"
  mkdir -p "${dir_path}" 2>/dev/null || return 1
  [[ -d "${dir_path}" && -w "${dir_path}" ]]
}

echo "Starting automated sampling training..."
echo "ROOT_DIR=${ROOT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MEGATRON_CMD=${MEGATRON_CMD}"
echo "SWIFT_CMD=${SWIFT_CMD}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "VAL_DATASET=${VAL_DATASET}"
echo "SAVE_BASE=${SAVE_BASE}"
echo "KEEP_RATIO=${KEEP_RATIO}"
echo "LORA_RANK=${LORA_RANK}"
echo "MAX_EPOCHS=${MAX_EPOCHS}"
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "ITERATIONS_PER_EPOCH=${ITERATIONS_PER_EPOCH}  SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "TRAIN_LOG_DIR=${TRAIN_LOG_DIR}"

gpu_precheck

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Warning: WANDB_API_KEY is not set. Training will still run, but wandb logging may fail." >&2
fi

if [[ ! -d "${BASE_MODEL}" ]]; then
  die "BASE_MODEL does not exist or is not a directory: ${BASE_MODEL}"
fi

if [[ ! -f "${VAL_DATASET}" ]]; then
  die "VAL_DATASET does not exist or is not a file: ${VAL_DATASET}"
fi

if ! command -v "${MEGATRON_CMD}" >/dev/null 2>&1; then
  die "MEGATRON_CMD is not available in PATH: ${MEGATRON_CMD}. This workflow expects the docker image described in documents/data/sst_omni_train_transcript.md. On a cluster, use the docker wrapper script: documents/code/sst_omni_train/auto_train_sampling_rank16_docker.sh"
fi

if ! command -v "${SWIFT_CMD}" >/dev/null 2>&1; then
  die "SWIFT_CMD is not available in PATH: ${SWIFT_CMD}. This workflow expects the docker image described in documents/data/sst_omni_train_transcript.md. On a cluster, use the docker wrapper script: documents/code/sst_omni_train/auto_train_sampling_rank16_docker.sh"
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
  if [[ "${ENABLE_MD_UPDATE}" != "True" ]]; then
    echo "Info: ENABLE_MD_UPDATE is not True; skip markdown update."
    return 0
  fi
  if [[ ! -f "${MD_PATH}" ]]; then
    echo "Warning: Markdown file not found; skip update: ${MD_PATH}" >&2
    return 0
  fi
  if [[ ! -f "${MD_UPDATER}" ]]; then
    echo "Warning: Markdown updater script not found; skip update: ${MD_UPDATER}" >&2
    return 0
  fi
  flock "${lock_file}" python "${MD_UPDATER}" --md "${MD_PATH}" --keep-ratio "${ratio}" --hf-path "${hf_dir}"
}

ratio="${KEEP_RATIO}"
dataset_path="${DATASET_PREFIX}${ratio}${DATASET_SUFFIX}"
save_root="${SAVE_BASE}/keep${ratio}_r${LORA_RANK}"
wandb_exp_name="${WANDB_EXP_PREFIX}_keep${ratio}_r${LORA_RANK}"

if [[ ! -f "${dataset_path}" ]]; then
  die "Dataset does not exist: ${dataset_path}"
fi

if ! ensure_dir "${save_root}"; then
  die "Cannot create or write to save_root: ${save_root}. Please set SAVE_BASE to a writable directory."
fi

if ! ensure_dir "${TRAIN_LOG_DIR}"; then
  die "Cannot create or write to TRAIN_LOG_DIR: ${TRAIN_LOG_DIR}. Please set TRAIN_LOG_DIR to a writable directory."
fi

ts="$(date +%Y%m%d_%H%M%S)"
log_file="${TRAIN_LOG_DIR}/train_keep${ratio}_r${LORA_RANK}_${ts}.log"

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

  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" \
  ENABLE_AUDIO_OUTPUT="${ENABLE_AUDIO_OUTPUT}" \
  WANDB_API_KEY="${WANDB_API_KEY:-}" \
  "${MEGATRON_CMD}" sft \
      --load "${BASE_MODEL}" \
      --dataset "${dataset_path}" \
      --val_dataset "${VAL_DATASET}" \
      --load_from_cache_file true \
      --train_type lora \
      --lora_rank "${LORA_RANK}" \
      --lora_alpha "${LORA_ALPHA}" \
      --target_modules "${TARGET_MODULES}" \
      --freeze_llm "${FREEZE_LLM}" \
      --freeze_vit "${FREEZE_VIT}" \
      --freeze_aligner "${FREEZE_ALIGNER}" \
      --vit_gradient_checkpointing "${VIT_GRADIENT_CHECKPOINTING}" \
      --packing "${PACKING}" \
      --expert_model_parallel_size "${EXPERT_MODEL_PARALLEL_SIZE}" \
      --moe_permute_fusion "${MOE_PERMUTE_FUSION}" \
      --moe_grouped_gemm "${MOE_GROUPED_GEMM}" \
      --moe_shared_expert_overlap "${MOE_SHARED_EXPERT_OVERLAP}" \
      --moe_aux_loss_coeff "${MOE_AUX_LOSS_COEFF}" \
      --micro_batch_size "${MICRO_BATCH_SIZE}" \
      --global_batch_size "${GLOBAL_BATCH_SIZE}" \
      --recompute_granularity "${RECOMPUTE_GRANULARITY}" \
      --recompute_method "${RECOMPUTE_METHOD}" \
      --recompute_num_layers "${RECOMPUTE_NUM_LAYERS}" \
      --finetune "${FINETUNE}" \
      --cross_entropy_loss_fusion "${CROSS_ENTROPY_LOSS_FUSION}" \
      --lr "${LR}" \
      --lr_warmup_fraction "${LR_WARMUP_FRACTION}" \
      --min_lr "${MIN_LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --clip_grad "${CLIP_GRAD}" \
      --max_epochs "${MAX_EPOCHS}" \
      --save "${save_root}" \
      --log_interval "${LOG_INTERVAL}" \
      --eval_interval "${EVAL_INTERVAL}" \
      --save_interval "${SAVE_INTERVAL}" \
      --max_length "${MAX_LENGTH}" \
      --num_workers "${NUM_WORKERS}" \
      --dataset_num_proc "${DATASET_NUM_PROC}" \
      --no_save_optim "${NO_SAVE_OPTIM}" \
      --no_save_rng "${NO_SAVE_RNG}" \
      --attention_backend "${ATTENTION_BACKEND}" \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_exp_name "${wandb_exp_name}" \
      --strict "${STRICT}"

  run_dir="$(pick_latest_run_dir "${save_root}")"
  hf_dir="${run_dir}-hf"

  echo "Training done."
  echo "run_dir=${run_dir}"
  echo "hf_dir=${hf_dir}"

  export_mcore_checkpoint_to_hf_staged "${run_dir}" "${hf_dir}"

  echo "Export done."

  update_md_locked "${ratio}" "${hf_dir}"
  echo "Markdown updated: keep_ratio=${ratio} -> ${hf_dir}"
} 2>&1 | tee "${log_file}"

echo ""
echo "All done."

