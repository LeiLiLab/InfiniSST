#!/bin/bash
#SBATCH --job-name=q3_vctx_bgel_rs
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_bgel_rs_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_bgel_rs_%x.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh}"

# Resume the BGE-large text-encoder Taurus run from the epoch-0 checkpoint.
# This continuation removes the 2000-step smoke cap, raises GradCache chunk
# from 128 to 256, and evaluates every 100 steps.
export RESUME="${RESUME:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_txt_bgel_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_dev100Tau1_eval240_taurus8_smoke2000_epoch_0.pt}"
export RESET_SCHEDULER="${RESET_SCHEDULER:-false}"
export RESET_BEST_ON_RESUME="${RESET_BEST_ON_RESUME:-false}"
export RESUME_COSINE_DECAY_TO_MAX_STEPS="${RESUME_COSINE_DECAY_TO_MAX_STEPS:-false}"

export TEXT_ENCODER_PRESET="${TEXT_ENCODER_PRESET:-bge-large-en-v1.5}"
export TEXT_MODEL_ID="${TEXT_MODEL_ID:-BAAI/bge-large-en-v1.5}"

export NUM_GPUS="${NUM_GPUS:-8}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1024}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-256}"
export MAX_STEPS="${MAX_STEPS:-0}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-0}"

# Inline eval stays dev+ACL+medicine for W&B continuity.  Dev is a deterministic
# 100-sample smoke readout; ACL and medicine remain held-out/full readouts.
export EVAL_STEPS_SAMPLE="${EVAL_STEPS_SAMPLE:-100}"
export EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-0}"
export EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-100}"
export ACL_EVAL_SAMPLE_LIMIT="${ACL_EVAL_SAMPLE_LIMIT:-0}"
export MEDICINE_EVAL_SAMPLE_LIMIT="${MEDICINE_EVAL_SAMPLE_LIMIT:-0}"
export EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-17}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-1000 10000}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS:-2}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json}"
export ACL_EVAL_GLOSSARY_SIZES="${ACL_EVAL_GLOSSARY_SIZES:-10000}"
export MEDICINE_EVAL_GLOSSARY_SIZES="${MEDICINE_EVAL_GLOSSARY_SIZES:-10000}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.75}"
export EVAL_SCORE_DEVICE="${EVAL_SCORE_DEVICE:-cuda}"
export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-256}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-1024}"

export VARIANT_TAG="${VARIANT_TAG:-vctx576_txt_bgel_resume_g256}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_txt_bgel_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep${EPOCHS}_v3_dev100Tau1_eval100_taurus8_resumeE0s882}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_varctx576_txt_bgel_taurus8_resume_e0s882_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_dev100Tau1_eval100}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260517__varctx_lmlb_v3_text_bge_large_en_resume_e0s882_gc256_eval100.md}"
export DATA_TAG="${DATA_TAG:-3variant_gsv2full_gsdedup_vctx576_bgel}"
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_hardneg}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:vctx576_bgel_resume_g256 compute:taurus-8gpu source:ggeqpwie}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-ggeqpwie lh1b88kw}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export MASTER_PORT="${MASTER_PORT:-30033}"
export RUN_VERDICT="${RUN_VERDICT:-Resume BGE-large-en-v1.5 varctx576 from epoch0/global_step882 checkpoint, GradCache chunk 256, eval every 100 steps, no smoke max-step cap.}"

mkdir -p /mnt/gemini/data1/jiaxuanluo/logs

REQUIRED_PATHS=(
  "${BASE_LAUNCHER}"
  "${NOTES_FILE}"
  "${RESUME}"
  "${ACL_EVAL_WIKI_GLOSSARY}"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
  if [ ! -f "${required_path}" ]; then
    echo "[ERROR] required file missing: ${required_path}" >&2
    exit 2
  fi
done

exec bash "${BASE_LAUNCHER}"
