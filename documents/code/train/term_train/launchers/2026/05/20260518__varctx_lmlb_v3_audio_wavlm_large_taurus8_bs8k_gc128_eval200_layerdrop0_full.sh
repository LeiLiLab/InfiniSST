#!/bin/bash
#SBATCH --job-name=q3_vctx_wavl_l0_b8g128
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_wavl_l0_b8g128_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_wavl_l0_b8g128_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh"

export AUDIO_ENCODER_PRESET="${AUDIO_ENCODER_PRESET:-wavlm-large}"
export AUDIO_ENCODER_TYPE="${AUDIO_ENCODER_TYPE:-wavlm}"
export AUDIO_MODEL_ID="${AUDIO_MODEL_ID:-microsoft/wavlm-large}"
export AUDIO_FEATURE_EXTRACTOR_ID="${AUDIO_FEATURE_EXTRACTOR_ID:-microsoft/wavlm-large}"
export AUDIO_INPUT_DTYPE="${AUDIO_INPUT_DTYPE:-fp32}"
export TARGET_MODULES="${TARGET_MODULES:-q_proj k_proj v_proj out_proj intermediate_dense output_dense}"
export MAXSIM_WINDOWS="${MAXSIM_WINDOWS:-8 12 16 20 24 28 32 40 48 64 80 96}"
export MAXSIM_STRIDE="${MAXSIM_STRIDE:-8}"

export TEXT_ENCODER_PRESET="${TEXT_ENCODER_PRESET:-bge-m3}"
export TEXT_MODEL_ID="${TEXT_MODEL_ID:-BAAI/bge-m3}"
export TEXT_POOLING="${TEXT_POOLING:-cls}"

export NUM_GPUS="${NUM_GPUS:-8}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1024}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-128}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_STEPS="${MAX_STEPS:-0}"
export MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-0}"

export EVAL_STEPS_SAMPLE="${EVAL_STEPS_SAMPLE:-200}"
export EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-0}"
export EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-100}"
export ACL_EVAL_SAMPLE_LIMIT="${ACL_EVAL_SAMPLE_LIMIT:-0}"
export MEDICINE_EVAL_SAMPLE_LIMIT="${MEDICINE_EVAL_SAMPLE_LIMIT:-0}"
export EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-17}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-1000 10000}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json}"
export ACL_EVAL_GLOSSARY_SIZES="${ACL_EVAL_GLOSSARY_SIZES:-10000}"
export MEDICINE_EVAL_GLOSSARY_SIZES="${MEDICINE_EVAL_GLOSSARY_SIZES:-10000}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS:-2}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.75}"
export EVAL_SCORE_DEVICE="${EVAL_SCORE_DEVICE:-cuda}"
export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-256}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-1024}"

export VARIANT_TAG="${VARIANT_TAG:-vctx576_aud_wavl_l0_t8_b8k_g128}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_aud_wavl_l0_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep${EPOCHS}_v3_dev100Tau1_eval200_taurus8}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_varctx576_aud_wavl_l0_taurus8_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_dev100Tau1_eval200_full}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260518__varctx_lmlb_v3_audio_wavlm_large_taurus8_bs8k_gc128_eval200_layerdrop0_full.md}"
export DATA_TAG="${DATA_TAG:-3variant_gsv2full_gsdedup_vctx576_wavl}"
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_hardneg}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:vctx576_aud_wavl_l0_t8_b8k_g128 compute:taurus-8gpu}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw r1pxeaxj 9034wae5 hikwfmaa 34sbpz92}"
export MASTER_PORT="${MASTER_PORT:-30485}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export RUN_VERDICT="${RUN_VERDICT:-WavLM-large speech-encoder ablation with BGE-M3 text encoder, WavLM checkpointing shim, WavLM layerdrop disabled for DDP stability, dynamic MFA frame-time mapping, global batch 8192, GradCache chunk 128, and eval200 aligned to refresh50.}"

mkdir -p /mnt/gemini/data1/jiaxuanluo/logs

REQUIRED_PATHS=(
  "${BASE_LAUNCHER}"
  "${NOTES_FILE}"
  "${ACL_EVAL_WIKI_GLOSSARY}"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
  if [ ! -f "${required_path}" ]; then
    echo "[ERROR] required file missing: ${required_path}" >&2
    exit 2
  fi
done

exec bash "${BASE_LAUNCHER}"
