#!/bin/bash
#SBATCH --job-name=q3_encab_devraw
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=260G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_encab_devraw_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_encab_devraw_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"

: "${MODEL_TAG:?MODEL_TAG is required, e.g. main_q3o_bgem3}"
: "${ENCODER_CONFIG:?ENCODER_CONFIG is required: main_q3o_bgem3 | text_e5 | audio_wavlm}"
: "${RESUME:?RESUME checkpoint path is required}"
: "${NOTES_FILE:?NOTES_FILE is required}"

if [ ! -f "${RESUME}" ]; then
    echo "[FATAL] checkpoint not found: ${RESUME}" >&2
    exit 1
fi
if [ ! -f "${NOTES_FILE}" ]; then
    echo "[FATAL] notes file not found: ${NOTES_FILE}" >&2
    exit 1
fi

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export MASTER_PORT="${MASTER_PORT:-30524}"

export VARIANT_TAG="${VARIANT_TAG:-encab_${MODEL_TAG}_devraw_gs100k}"
export VERSION="${VERSION:-3var_gsdedup_vctx576_${MODEL_TAG}_devraw_fixeddenom_gs100k_${RUN_STAMP}}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-encoder_ablation_${MODEL_TAG}_devraw_fixeddenom_gs100k_${RUN_STAMP}}"
export TASK_TAG="eval"
export DATA_TAG="${DATA_TAG:-devraw_encoder_ablation}"
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_encoder_ablation}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:${VARIANT_TAG} compute:taurus-1gpu ablation:encoder protocol:devraw-fixeddenom readout:dev-only}"

export TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl}"
export DEV_JSONL="${DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl}"
export EVAL_WIKI_GLOSSARY="${EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json}"
export EVAL_METRICS_GLOSSARY="${EVAL_METRICS_GLOSSARY:-${REPO_ROOT}/documents/code/train/term_train/reports/figures/20260524_dev_raw_glossary_from_term_dev_varctx576.json}"

for required_path in "${TRAIN_JSONL}" "${DEV_JSONL}" "${EVAL_WIKI_GLOSSARY}" "${EVAL_METRICS_GLOSSARY}"; do
    if [ ! -f "${required_path}" ]; then
        echo "[FATAL] required input not found: ${required_path}" >&2
        exit 1
    fi
done

case "${ENCODER_CONFIG}" in
    main_q3o_bgem3)
        export AUDIO_ENCODER_PRESET="qwen3-omni"
        export AUDIO_ENCODER_TYPE="qwen3_omni"
        export AUDIO_MODEL_ID="Atotti/Qwen3-Omni-AudioTransformer"
        export AUDIO_FEATURE_EXTRACTOR_ID="openai/whisper-large-v3"
        export AUDIO_INPUT_DTYPE="auto"
        export TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
        export MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"
        export MAXSIM_STRIDE=2
        export TEXT_ENCODER_PRESET="bge-m3"
        export TEXT_MODEL_ID="BAAI/bge-m3"
        export TEXT_INPUT_PREFIX=""
        export TEXT_POOLING="cls"
        export TEXT_TARGET_MODULES="query key value dense"
        ;;
    text_e5)
        export AUDIO_ENCODER_PRESET="qwen3-omni"
        export AUDIO_ENCODER_TYPE="qwen3_omni"
        export AUDIO_MODEL_ID="Atotti/Qwen3-Omni-AudioTransformer"
        export AUDIO_FEATURE_EXTRACTOR_ID="openai/whisper-large-v3"
        export AUDIO_INPUT_DTYPE="auto"
        export TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
        export MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"
        export MAXSIM_STRIDE=2
        export TEXT_ENCODER_PRESET="multilingual-e5-large"
        export TEXT_MODEL_ID="intfloat/multilingual-e5-large"
        export TEXT_INPUT_PREFIX="query: "
        export TEXT_POOLING="mean"
        export TEXT_TARGET_MODULES="query key value dense"
        ;;
    audio_wavlm)
        export AUDIO_ENCODER_PRESET="wavlm-large"
        export AUDIO_ENCODER_TYPE="wavlm"
        export AUDIO_MODEL_ID="microsoft/wavlm-large"
        export AUDIO_FEATURE_EXTRACTOR_ID="microsoft/wavlm-large"
        export AUDIO_INPUT_DTYPE="fp32"
        export TARGET_MODULES="q_proj k_proj v_proj out_proj intermediate_dense output_dense"
        export MAXSIM_WINDOWS="8 12 16 20 24 28 32 40 48 64 80 96"
        export MAXSIM_STRIDE=8
        export TEXT_ENCODER_PRESET="bge-m3"
        export TEXT_MODEL_ID="BAAI/bge-m3"
        export TEXT_INPUT_PREFIX=""
        export TEXT_POOLING="cls"
        export TEXT_TARGET_MODULES="query key value dense"
        ;;
    *)
        echo "[FATAL] unknown ENCODER_CONFIG=${ENCODER_CONFIG}" >&2
        exit 2
        ;;
esac

export FIXED_AUDIO_SECONDS=5.76
export EVAL_FIXED_AUDIO_SECONDS=5.76
export AUDIO_HIDDEN_DIM=0
export TEXT_LORA_RANK=128
export TEXT_LORA_ALPHA=256
export LORA_RANK=128
export LORA_ALPHA=256
export TARGET_DIM=1024
export POOLING_TYPE="transformer"
export USE_MAXSIM=true
export MFA_SUPERVISED=true
export MFA_WINDOW_SELECTION="smallest"
export MFA_POSITIVE_SCOPE="auto"
export TERM_ID_NORMALIZE="aggressive"

export EVAL_ONLY=true
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

export GRAD_CACHE_CHUNK_SIZE=128
export NUM_WORKERS=0
export EPOCHS=1
export SCHEDULER_EPOCHS=1
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESET_BEST_ON_RESUME=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false

export EVAL_METRIC_DENOMINATOR="fixed_raw"
export EVAL_GLOSSARY_SIZES="10000 100000"
export BEST_METRIC="eval_dev/recall@10_gs100000"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export EVAL_SAMPLE_LIMIT=0
export EVAL_SAMPLE_SEED=17
export EVAL_SCORE_DEVICE="${EVAL_SCORE_DEVICE:-cuda}"
export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-256}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-4096}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS=2
export TCM_SWEEP_THRESHOLDS="0.75"
export TCM_SWEEP_FBETA=3.0
export AUTO_FULL_EVAL_ON_BEST=false

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

export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw xw53jzn0 cyzz2lw0}"
export RUN_VERDICT="${RUN_VERDICT:-Dev-only matched encoder ablation. Denominator is fixed to the full dev raw glossary via EVAL_METRICS_GLOSSARY; runtime retrieval bank changes raw -> gs10k -> gs100k. No held-out ACL/tagged/medicine readout.}"

echo "[ENCODER_ABLATION] model_tag=${MODEL_TAG}"
echo "[ENCODER_ABLATION] encoder_config=${ENCODER_CONFIG}"
echo "[ENCODER_ABLATION] checkpoint=${RESUME}"
echo "[ENCODER_ABLATION] dev_jsonl=${DEV_JSONL}"
echo "[ENCODER_ABLATION] metrics_glossary=${EVAL_METRICS_GLOSSARY}"
echo "[ENCODER_ABLATION] retriever_glossary=${EVAL_WIKI_GLOSSARY} sizes=${EVAL_GLOSSARY_SIZES}"
echo "[ENCODER_ABLATION] denominator=${EVAL_METRIC_DENOMINATOR}"
echo "[ENCODER_ABLATION] wandb_name=${WANDB_EXP_NAME}"

source "${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
