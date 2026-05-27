#!/bin/bash
# Eval one checkpoint for the BGE-large vs BGE-M3 epoch-0 text-encoder compare.
#
# Usage:
#   EVAL_VARIANT=bgel_epoch0 CUDA_VISIBLE_DEVICES=0 bash this_script.sh
#   EVAL_VARIANT=bgem3_epoch0 CUDA_VISIBLE_DEVICES=1 bash this_script.sh

set -euo pipefail

: "${EVAL_VARIANT:?set EVAL_VARIANT=bgel_epoch0 or bgem3_epoch0}"

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
COMMON_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-taurus}-${NUM_GPUS}gpu"

export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl"
export DEV_JSONL="/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl"
export ACL_DEV_JSONL="/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl"
export MEDICINE_DEV_JSONL="/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_dev_dataset.jsonl"

case "${EVAL_VARIANT}" in
  bgel_epoch0)
    export VARIANT_TAG="eval_bgel_epoch0"
    export VERSION="3var_gsdedup_vctx576_eval_bgel_epoch0_${COMPUTE_TAG}"
    export WANDB_EXP_NAME="eval_bgel_epoch0_vs_bgem3_${COMPUTE_TAG}_20260517"
    export NOTES_FILE="${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260517__eval_bgel_epoch0_vs_bgem3_compare.md"
    export DATA_TAG="vctx576_epoch0_compare_bgel"
    export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG} comparison:epoch0_text_encoder readout:dev_acl_med"
    export BASELINE_RUN_IDS="mhukv2bi lh1b88kw"
    export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_txt_bgel_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval240_taurus8_smoke2000_epoch_0.pt"
    export TEXT_ENCODER_PRESET="bge-large-en-v1.5"
    export TEXT_MODEL_ID="BAAI/bge-large-en-v1.5"
    export TEXT_INPUT_PREFIX=""
    export TEXT_POOLING="cls"
    ;;
  bgem3_epoch0)
    export VARIANT_TAG="eval_bgem3_epoch0"
    export VERSION="3var_gsdedup_vctx576_eval_bgem3_epoch0_${COMPUTE_TAG}"
    export WANDB_EXP_NAME="eval_bgem3_epoch0_vs_bgel_${COMPUTE_TAG}_20260517"
    export NOTES_FILE="${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260517__eval_bgem3_epoch0_vs_bgel_compare.md"
    export DATA_TAG="vctx576_epoch0_compare_bgem3"
    export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG} comparison:epoch0_text_encoder readout:dev_acl_med"
    export BASELINE_RUN_IDS="lh1b88kw mhukv2bi"
    export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_epoch_0.pt"
    export TEXT_ENCODER_PRESET="bge-m3"
    export TEXT_MODEL_ID="BAAI/bge-m3"
    export TEXT_INPUT_PREFIX=""
    export TEXT_POOLING="cls"
    ;;
  *)
    echo "[FATAL] unknown EVAL_VARIANT=${EVAL_VARIANT}" >&2
    exit 2
    ;;
esac

export EVAL_ONLY=true
export TASK_TAG="eval"
export EXPERIMENT_FAMILY="sst_ood_hardneg"
export SELECT_CLEAN_GPUS=true

export FIXED_AUDIO_SECONDS=5.76
export EVAL_FIXED_AUDIO_SECONDS=5.76
export AUDIO_ENCODER_PRESET="qwen3-omni"
export AUDIO_ENCODER_TYPE="qwen3_omni"
export AUDIO_MODEL_ID="Atotti/Qwen3-Omni-AudioTransformer"
export AUDIO_FEATURE_EXTRACTOR_ID="openai/whisper-large-v3"
export AUDIO_INPUT_DTYPE="auto"
export TEXT_LORA_RANK=128
export TEXT_LORA_ALPHA=256
export LORA_RANK=128
export LORA_ALPHA=256
export TARGET_DIM=1024
export POOLING_TYPE="transformer"
export USE_MAXSIM=true
export MFA_SUPERVISED=true
export MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"
export MAXSIM_STRIDE=2
export MFA_WINDOW_SELECTION="smallest"
export MFA_POSITIVE_SCOPE="auto"
export TERM_ID_NORMALIZE="aggressive"

# Eval-only: disable all train-time negatives and losses.
export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=0
export NEG_BANK_SIZE=0
export GLOSSARY_NEG_PATH=""
export GLOSSARY_NEG_REFRESH_STEPS=0
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_POS_THRESHOLD=0.85
export TCM_NEG_THRESHOLD=0.60
export TCM_LOSS_FORM="hinge"
export TCM_REDUCTION="mean_viol"
export TCM_NEG_SCOPE="all"
export TCM_NEG_TOPK=0
export TCM_WARMUP_STEPS=0

export GRAD_CACHE_CHUNK_SIZE=256
export EPOCHS=1
export SCHEDULER_EPOCHS=1
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESET_BEST_ON_RESUME=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false
export MASTER_PORT="${MASTER_PORT:-30420}"

export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
export EVAL_GLOSSARY_SIZES="1000 10000"
export ACL_EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json"
export ACL_EVAL_GLOSSARY_SIZES="1000 10000"
export MEDICINE_EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_glossary_gt_plus_medicine_wiki_gs10000.json"
export MEDICINE_EVAL_GLOSSARY_SIZES="1000 10000"
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/recall@10_gs10000"
export BEST_METRIC_SECONDARY="eval_acl6060/recall@10_gs10000"
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export EVAL_SAMPLE_LIMIT=0
export ACL_EVAL_SAMPLE_LIMIT=0
export MEDICINE_EVAL_SAMPLE_LIMIT=0
export EVAL_SAMPLE_SEED=17
export EVAL_SCORE_DEVICE="cuda"
export EVAL_SCORE_QUERY_CHUNK=256
export EVAL_SCORE_TEXT_CHUNK=1024
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS=2
export TCM_SWEEP_THRESHOLDS="0.75"
export TCM_SWEEP_FBETA=3.0
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="${RUN_VERDICT:-Standalone full eval-only readout for the epoch-0 text-encoder checkpoint comparison.}"

mkdir -p /mnt/gemini/data1/jiaxuanluo/logs

for tag in \
  "family:${EXPERIMENT_FAMILY}" \
  "task:${TASK_TAG}" \
  "data:${DATA_TAG}" \
  "status:running" \
  ${EXTRA_WANDB_TAGS}; do
  if [ "${#tag}" -lt 1 ] || [ "${#tag}" -gt 64 ]; then
    echo "[ERROR] invalid WandB tag length (${#tag}): ${tag}" >&2
    exit 2
  fi
done

REQUIRED_PATHS=(
  "${COMMON_LAUNCHER}"
  "${RESUME}"
  "${NOTES_FILE}"
  "${TRAIN_JSONL}"
  "${DEV_JSONL}"
  "${ACL_DEV_JSONL}"
  "${MEDICINE_DEV_JSONL}"
  "${EVAL_WIKI_GLOSSARY}"
  "${ACL_EVAL_WIKI_GLOSSARY}"
  "${MEDICINE_EVAL_WIKI_GLOSSARY}"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
  if [ ! -f "${required_path}" ]; then
    echo "[ERROR] required file missing: ${required_path}" >&2
    exit 2
  fi
done

echo "[EPOCH0_COMPARE] variant=${EVAL_VARIANT} ckpt=${RESUME}"
echo "[EPOCH0_COMPARE] compute=${COMPUTE_TAG} cuda=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[EPOCH0_COMPARE] eval dev/ACL/medicine full readout, glossary sizes base/gs1k/gs10k, tau=0.75"

source "${COMMON_LAUNCHER}"
