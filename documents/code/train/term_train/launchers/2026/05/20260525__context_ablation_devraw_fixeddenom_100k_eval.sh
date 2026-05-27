#!/bin/bash
set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"

: "${CONTEXT_TAG:?CONTEXT_TAG is required}"
: "${SOURCE_RUN_ID:?SOURCE_RUN_ID is required}"
: "${RESUME:?RESUME checkpoint path is required}"
: "${TRAIN_JSONL:?TRAIN_JSONL is required}"
: "${DEV_JSONL:?DEV_JSONL is required}"
: "${FIXED_SECONDS:?FIXED_SECONDS is required}"
: "${NOTES_FILE:?NOTES_FILE is required}"

if [ ! -f "${RESUME}" ]; then
    echo "[FATAL] checkpoint not found: ${RESUME}" >&2
    exit 1
fi
if [ ! -f "${TRAIN_JSONL}" ]; then
    echo "[FATAL] train jsonl not found: ${TRAIN_JSONL}" >&2
    exit 1
fi
if [ ! -f "${DEV_JSONL}" ]; then
    echo "[FATAL] dev jsonl not found: ${DEV_JSONL}" >&2
    exit 1
fi
if [ ! -f "${NOTES_FILE}" ]; then
    echo "[FATAL] notes file not found: ${NOTES_FILE}" >&2
    exit 1
fi

RUN_STAMP="${RUN_STAMP:-ctxab_${CONTEXT_TAG}_devraw100k_$(date -u +%Y%m%dT%H%M%SZ)}"

export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-aries}"
export MODEL_TAG="${MODEL_TAG:-${CONTEXT_TAG}}"
export VARIANT_TAG="${VARIANT_TAG:-ctxab_${CONTEXT_TAG}_devraw100k}"
export VERSION="${VERSION:-3var_context_ablation_${CONTEXT_TAG}_devraw_fixeddenom_100k_${RUN_STAMP}}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-context_ablation_${CONTEXT_TAG}_devraw_fixeddenom_100k_${RUN_STAMP}}"
export DATA_TAG="${DATA_TAG:-devraw_ctxab}"
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_context_ablation}"
export TASK_TAG="eval"
export MAXSIM_WINDOWS="${MAXSIM_WINDOWS:-2 3 4 5 6 7 8 10 12 16 20 24}"
MAXSIM_WINDOWS_TAG="${MAXSIM_WINDOWS// /_}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:${VARIANT_TAG} compute:aries-gpu0-shared ablation:context protocol:devraw-fixeddenom readout:dev-only source:${SOURCE_RUN_ID} maxsim:${MAXSIM_WINDOWS_TAG}}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-${SOURCE_RUN_ID} lh1b88kw}"

export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0}"
export NUM_GPUS=1
export PER_GPU_BATCH=1
export BATCH_SIZE=1
export MASTER_PORT="${MASTER_PORT:-20049}"
export SELECT_CLEAN_GPUS=false
export WAIT_FOR_CLEAN_GPUS=false
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/tmp/jiaxuanluo_ctxab_${CONTEXT_TAG}_${RUN_STAMP}}"

export WANDB_DIR="/mnt/gemini/data1/jiaxuanluo/wandb"
export WANDB_CACHE_DIR="/mnt/aries/data4/jiaxuanluo/cache/wandb"
export XDG_CACHE_HOME="/mnt/aries/data4/jiaxuanluo/cache"
export XDG_CONFIG_HOME="/mnt/aries/data4/jiaxuanluo/config"

export EVAL_WIKI_GLOSSARY="${EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json}"
export EVAL_METRICS_GLOSSARY="${EVAL_METRICS_GLOSSARY:-${REPO_ROOT}/documents/code/train/term_train/reports/figures/20260524_dev_raw_glossary_from_term_dev_varctx576.json}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-1000 10000 100000}"
export EVAL_METRIC_DENOMINATOR="fixed_raw"
export BEST_METRIC="eval_dev/recall@10_gs100000"
export BEST_METRIC_SECONDARY=""

if [ ! -f "${EVAL_WIKI_GLOSSARY}" ]; then
    echo "[FATAL] eval wiki glossary not found: ${EVAL_WIKI_GLOSSARY}" >&2
    exit 1
fi
if [ ! -f "${EVAL_METRICS_GLOSSARY}" ]; then
    echo "[FATAL] eval metrics glossary not found: ${EVAL_METRICS_GLOSSARY}" >&2
    exit 1
fi

export FIXED_AUDIO_SECONDS="${FIXED_SECONDS}"
export EVAL_FIXED_AUDIO_SECONDS="${FIXED_SECONDS}"
export AUDIO_ENCODER_PRESET="qwen3-omni"
export AUDIO_ENCODER_TYPE="qwen3_omni"
export AUDIO_MODEL_ID="Atotti/Qwen3-Omni-AudioTransformer"
export AUDIO_FEATURE_EXTRACTOR_ID="openai/whisper-large-v3"
export AUDIO_INPUT_DTYPE="auto"
export AUDIO_HIDDEN_DIM=0
export TEXT_ENCODER_PRESET="bge-m3"
export TEXT_MODEL_ID="BAAI/bge-m3"
export TEXT_INPUT_PREFIX=""
export TEXT_POOLING="${TEXT_POOLING:-cls}"
export TEXT_TARGET_MODULES="query key value dense"
export TEXT_LORA_RANK=128
export TEXT_LORA_ALPHA=256
export LORA_RANK=128
export LORA_ALPHA=256
export TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
export TARGET_DIM=1024
export POOLING_TYPE="${POOLING_TYPE:-transformer}"
export USE_MAXSIM="${USE_MAXSIM:-true}"
export MFA_SUPERVISED="${MFA_SUPERVISED:-true}"
export MFA_WINDOW_SELECTION="${MFA_WINDOW_SELECTION:-smallest}"
export MFA_POSITIVE_SCOPE="${MFA_POSITIVE_SCOPE:-auto}"
export MAXSIM_STRIDE=2
export TERM_ID_NORMALIZE="aggressive"

export EVAL_ONLY=true
export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=0
export NEG_BANK_SIZE=0
export GLOSSARY_NEG_PATH=""
export GLOSSARY_NEG_REFRESH_STEPS=0
export NEG_BANK_REFRESH_STEPS=50
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
export TCM_SWEEP_THRESHOLDS="0.75"
export TCM_SWEEP_FBETA=3.0

export GRAD_CACHE_CHUNK_SIZE=128
export NUM_WORKERS=0
export EPOCHS=1
export SCHEDULER_EPOCHS=1
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESET_BEST_ON_RESUME=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false
export SAVE_STEPS=999999
export SAVE_LATEST_ON_EVAL=false
export KEEP_CHECKPOINTS=0
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export EVAL_SAMPLE_LIMIT=0
export EVAL_SAMPLE_SEED=17
export EVAL_SCORE_DEVICE="cuda"
export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-128}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-512}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS=2

export ACL_DEV_JSONL=""
export TAGGED_ACL_DEV_JSONL=""
export MEDICINE_DEV_JSONL=""
export ACL_EVAL_WIKI_GLOSSARY=""
export ACL_EVAL_GLOSSARY_SIZES=""
export ACL_EVAL_METRICS_GLOSSARY=""
export TAGGED_ACL_EVAL_WIKI_GLOSSARY=""
export TAGGED_ACL_EVAL_GLOSSARY_SIZES=""
export TAGGED_ACL_EVAL_METRICS_GLOSSARY=""
export MEDICINE_EVAL_WIKI_GLOSSARY=""
export MEDICINE_EVAL_GLOSSARY_SIZES=""
export MEDICINE_EVAL_METRICS_GLOSSARY=""
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export AUTO_FULL_EVAL_ON_BEST=false

export RUN_VERDICT="Context ablation dev-only eval. Denominator is fixed to the full dev raw glossary via EVAL_METRICS_GLOSSARY; runtime retrieval bank changes raw -> gs1k -> gs10k -> gs100k. ACL/tagged/medicine disabled."

echo "[CTXAB_EVAL] context_tag=${CONTEXT_TAG}"
echo "[CTXAB_EVAL] source_run=${SOURCE_RUN_ID}"
echo "[CTXAB_EVAL] checkpoint=${RESUME}"
echo "[CTXAB_EVAL] train_jsonl=${TRAIN_JSONL}"
echo "[CTXAB_EVAL] dev_jsonl=${DEV_JSONL}"
echo "[CTXAB_EVAL] fixed_seconds=${FIXED_SECONDS}"
echo "[CTXAB_EVAL] metrics_glossary=${EVAL_METRICS_GLOSSARY}"
echo "[CTXAB_EVAL] retriever_glossary=${EVAL_WIKI_GLOSSARY} sizes=${EVAL_GLOSSARY_SIZES}"
echo "[CTXAB_EVAL] maxsim_windows=${MAXSIM_WINDOWS}"

cd "${REPO_ROOT}"
source "${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
