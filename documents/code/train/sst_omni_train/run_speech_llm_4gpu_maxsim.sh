#!/usr/bin/env bash
set -euo pipefail

# Train Speech LLM (Qwen3-Omni-30B-A3B) with density-parameterized term_map.
# Default: 2x A6000, EP=2, TP=1. Runs INSIDE the megatron docker container.
#
# For higher LoRA ranks (r>=32) that OOM on 2x48GB, use the 4-GPU recipe by
# setting:
#   NPROC_PER_NODE_OVERRIDE=4
#   EP_OVERRIDE=4                 # shard experts across all 4 ranks
#   TP_OVERRIDE=2                 # tensor-parallel shards the fused-CE logits
#
# Usage (inside docker):
#   bash run_speech_llm_4gpu_maxsim.sh <DENSITY> [MASTER_PORT]
#
# Or via sbatch wrapper (see run_density_train_sbatch.sh)

# ======Configuration=====
DENSITY="${1:-5}"
MASTER_PORT_ARG="${2:-29519}"

# Parallelism knobs. Defaults preserve the historical 2-GPU EP=2 TP=1 recipe
# so existing d5/d5_cap/d5_cap_adv runs remain byte-identical. The 4-GPU
# recipe is opt-in via *_OVERRIDE env vars.
NPROC_PER_NODE="${NPROC_PER_NODE_OVERRIDE:-2}"
EXPERT_MODEL_PARALLEL_SIZE="${EP_OVERRIDE:-2}"
TENSOR_MODEL_PARALLEL_SIZE="${TP_OVERRIDE:-1}"
# Megatron requires sequence_parallel=true whenever both EP>1 and TP>1 are
# active. Default to auto-enable so the 4-GPU EP+TP recipe works; the legacy
# 2-GPU EP=2 TP=1 path keeps SP=false (its previous behaviour).
if [[ "${SEQUENCE_PARALLEL_OVERRIDE:-}" != "" ]]; then
  SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL_OVERRIDE}"
elif (( EXPERT_MODEL_PARALLEL_SIZE > 1 && TENSOR_MODEL_PARALLEL_SIZE > 1 )); then
  SEQUENCE_PARALLEL="true"
else
  SEQUENCE_PARALLEL="false"
fi

# Derive the default CUDA_VISIBLE_DEVICES as 0..NPROC-1 if the caller did not
# set it explicitly. Slurm provides CUDA_VISIBLE_DEVICES automatically for
# sbatch-dispatched runs, so this default only applies to local bare-metal
# invocations inside docker.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _default_cvd=""
  for i in $(seq 0 $((NPROC_PER_NODE - 1))); do
    _default_cvd+="${_default_cvd:+,}${i}"
  done
  export CUDA_VISIBLE_DEVICES="${_default_cvd}"
fi

# Fail loudly on invalid parallelism combos.
if (( NPROC_PER_NODE <= 0 )); then
  echo "[FATAL] NPROC_PER_NODE must be > 0 (got ${NPROC_PER_NODE})"; exit 2
fi
if (( TENSOR_MODEL_PARALLEL_SIZE <= 0 )); then
  echo "[FATAL] TP must be > 0 (got ${TENSOR_MODEL_PARALLEL_SIZE})"; exit 2
fi
if (( NPROC_PER_NODE % TENSOR_MODEL_PARALLEL_SIZE != 0 )); then
  echo "[FATAL] NPROC_PER_NODE (${NPROC_PER_NODE}) must be divisible by TP (${TENSOR_MODEL_PARALLEL_SIZE})"; exit 2
fi

LORA_RANK="${LORA_RANK_OVERRIDE:-16}"
LORA_ALPHA="${LORA_ALPHA_OVERRIDE:-$((LORA_RANK * 2))}"
MAX_EPOCHS=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=4
# MAX_LENGTH controls the packed-sequence length cap. Larger values give
# higher packing efficiency but increase peak activation memory (especially
# for audio_tower chunks). For higher LoRA ranks (r>=32) on 2x48GB GPUs,
# reduce this to 3072 or 2048 to leave headroom for optimizer states.
MAX_LENGTH="${MAX_LENGTH_OVERRIDE:-4096}"

MCORE_MODEL="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-v2/"
DATASET_PATH="${DATASET_PATH_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/density_ablation/train_maxsim_varlen_d${DENSITY}.jsonl}"
VAL_DATASET="/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20_final.jsonl"

SAVE_BASE="${SAVE_BASE_OVERRIDE:-/mnt/aries/data4/jiaxuanluo/speech_llm_density_ablation/d${DENSITY}}"
WANDB_PROJECT="gigaspeech_zh"
WANDB_EXP_NAME="omni-maxsim-varlen-d${DENSITY}-r${LORA_RANK}-${NPROC_PER_NODE}gpu-ep${EXPERT_MODEL_PARALLEL_SIZE}-tp${TENSOR_MODEL_PARALLEL_SIZE}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

MASTER_ADDR="127.0.0.1"
MASTER_PORT="${MASTER_PORT_ARG}"

ENABLE_AUDIO_OUTPUT="False"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# ======Configuration=====

if [ ! -f "${DATASET_PATH}" ]; then
    echo "[FATAL] Training data not found: ${DATASET_PATH}"
    echo "[HINT] Run run_density_ablation.sh first to generate density variants."
    exit 1
fi

save_root="${SAVE_BASE}/r${LORA_RANK}"
mkdir -p "${save_root}"

LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs"
mkdir -p "${LOG_DIR}"
ts="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/speech_llm_train_d${DENSITY}_r${LORA_RANK}_${ts}.log"

echo "========================================"
echo "Speech LLM Training (Density Ablation d=${DENSITY})"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE}"
echo "TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}"
echo "SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL}"
echo "LORA_RANK=${LORA_RANK}, LORA_ALPHA=${LORA_ALPHA}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "SAVE_BASE=${save_root}"
echo "========================================"

{
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" \
  ENABLE_AUDIO_OUTPUT="${ENABLE_AUDIO_OUTPUT}" \
  WANDB_API_KEY="${WANDB_API_KEY}" \
  megatron sft \
      --load "${MCORE_MODEL}" \
      --dataset "${DATASET_PATH}" \
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
      --log_interval 50 \
      --eval_interval 500 \
      --save_interval 500 \
      --max_length "${MAX_LENGTH}" \
      --num_workers 8 \
      --dataset_num_proc 8 \
      --no_save_optim true \
      --no_save_rng true \
      --attention_backend flash \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_exp_name "${WANDB_EXP_NAME}" \
      --strict True

  echo "Training done."

  run_dir="$(ls -1dt "${save_root}"/*/ 2>/dev/null | head -n 1)"
  run_dir="${run_dir%/}"
  hf_dir="${run_dir}-hf"

  echo "run_dir=${run_dir}"
  echo "Exporting to HF format..."

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  swift export \
      --mcore_adapters "${run_dir}" \
      --to_hf true \
      --torch_dtype bfloat16 \
      --output_dir "${hf_dir}"

  echo "Export done: ${hf_dir}"
} 2>&1 | tee "${log_file}"

echo ""
echo "All done. Log: ${log_file}"
