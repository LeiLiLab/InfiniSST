#!/bin/bash
# Eval-only readout for the fixed 3.84s context ablation checkpoint.
#
# Intended direct Aries usage:
#   cd /mnt/taurus/home/jiaxuanluo/InfiniSST
#   LOG=/mnt/gemini/data1/jiaxuanluo/logs/direct_ctx384_s1040_eval_$(date -u +%Y%m%dT%H%M%S)
#   nohup env CUDA_VISIBLE_DEVICES=7 SLURM_JOB_PARTITION=aries SLURM_JOB_ID=direct_ctx384_s1040_eval \
#     MASTER_PORT=30484 \
#     bash documents/code/train/term_train/launchers/2026/05/20260518__ctx384_s1040_dev_acl_med_aries1_eval.sh \
#     > ${LOG}.out 2> ${LOG}.err &
#   echo $!
#   echo ${LOG}.out
#   echo ${LOG}.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
COMMON_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="ctx384_s1040_eval"
export VERSION="3var_gsdedup_ctx384_s1040_dev_acl_med_${COMPUTE_TAG}"
export WANDB_EXP_NAME="ctx384_s1040_dev_acl_med_${COMPUTE_TAG}_20260518"
export NOTES_FILE="${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260518__ctx384_s1040_dev_acl_med_eval.md"

export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_gsctx3p84.jsonl"
export DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_m4.jsonl"
export ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_ctx3p84/acl6060_dev_dataset.jsonl"
export MEDICINE_DEV_JSONL="/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_dev_dataset_ctx3p84.jsonl"

export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_ctx384_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt"

export EVAL_ONLY=true
export TASK_TAG="eval"
export DATA_TAG="3variant_gsv2full_gsfix_mfa_gsdedup_ctx384_s1040_readout"
export EXPERIMENT_FAMILY="sst_ood_hardneg"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG} ablation:ctx384 readout:dev_acl_med"
export BASELINE_RUN_IDS="dxwrgbln lh1b88kw"
export SELECT_CLEAN_GPUS=true

# Match the fixed-context checkpoint recipe from W&B run dxwrgbln.
export FIXED_AUDIO_SECONDS=3.84
export EVAL_FIXED_AUDIO_SECONDS=3.84
export AUDIO_ENCODER_PRESET="qwen3-omni"
export AUDIO_ENCODER_TYPE="qwen3_omni"
export AUDIO_MODEL_ID="Atotti/Qwen3-Omni-AudioTransformer"
export AUDIO_FEATURE_EXTRACTOR_ID="openai/whisper-large-v3"
export AUDIO_INPUT_DTYPE="auto"
export TEXT_ENCODER_PRESET="bge-m3"
export TEXT_MODEL_ID="BAAI/bge-m3"
export TEXT_INPUT_PREFIX=""
export TEXT_POOLING="cls"
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

# Eval-only: disable train-time negative-bank initialization and all auxiliary losses.
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
export MASTER_PORT="${MASTER_PORT:-30484}"

# Recall readout shape: base / 1k / 10k for each domain.
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
export RUN_VERDICT="${RUN_VERDICT:-Eval-only fixed 3.84s context readout for dxwrgbln step-1040 dev-best checkpoint across dev, ACL6060, and medicine base/gs1k/gs10k.}"

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

echo "[CTX384_EVAL] ckpt=${RESUME}"
echo "[CTX384_EVAL] dev=${DEV_JSONL} glossary=${EVAL_WIKI_GLOSSARY} sizes=base ${EVAL_GLOSSARY_SIZES}"
echo "[CTX384_EVAL] acl=${ACL_DEV_JSONL} glossary=${ACL_EVAL_WIKI_GLOSSARY} sizes=base ${ACL_EVAL_GLOSSARY_SIZES}"
echo "[CTX384_EVAL] medicine=${MEDICINE_DEV_JSONL} glossary=${MEDICINE_EVAL_WIKI_GLOSSARY} sizes=base ${MEDICINE_EVAL_GLOSSARY_SIZES}"
echo "[CTX384_EVAL] compute=${COMPUTE_TAG} cuda=${CUDA_VISIBLE_DEVICES:-<unset>}"

source "${COMMON_LAUNCHER}"
