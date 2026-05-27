#!/bin/bash
#SBATCH --job-name=q3_vctx_q3o_rs_e2
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_q3o_rs_e2_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_q3o_rs_e2_%x.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh}"

# Continue the Qwen3-Omni varctx576 source run from the epoch-2 checkpoint.
# ACL and medicine are inline readouts only; primary checkpoint selection stays
# on dev gs10k so held-out ACL does not choose a model checkpoint.
export RESUME="${RESUME:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_epoch_2.pt}"
export RESET_SCHEDULER="${RESET_SCHEDULER:-false}"
export RESET_BEST_ON_RESUME="${RESET_BEST_ON_RESUME:-false}"
export RESUME_COSINE_DECAY_TO_MAX_STEPS="${RESUME_COSINE_DECAY_TO_MAX_STEPS:-false}"

export AUDIO_ENCODER_PRESET="${AUDIO_ENCODER_PRESET:-qwen3-omni}"
export AUDIO_ENCODER_TYPE="${AUDIO_ENCODER_TYPE:-qwen3_omni}"
export AUDIO_MODEL_ID="${AUDIO_MODEL_ID:-Atotti/Qwen3-Omni-AudioTransformer}"
export AUDIO_FEATURE_EXTRACTOR_ID="${AUDIO_FEATURE_EXTRACTOR_ID:-openai/whisper-large-v3}"
export AUDIO_INPUT_DTYPE="${AUDIO_INPUT_DTYPE:-auto}"

export TEXT_ENCODER_PRESET="${TEXT_ENCODER_PRESET:-bge-m3}"
export TEXT_MODEL_ID="${TEXT_MODEL_ID:-BAAI/bge-m3}"
export TEXT_POOLING="${TEXT_POOLING:-cls}"

export NUM_GPUS="${NUM_GPUS:-8}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1024}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-128}"
export MAX_STEPS="${MAX_STEPS:-0}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-0}"

export EVAL_STEPS_SAMPLE="${EVAL_STEPS_SAMPLE:-100}"
export EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-0}"
export EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-0}"
export ACL_EVAL_SAMPLE_LIMIT="${ACL_EVAL_SAMPLE_LIMIT:-0}"
export MEDICINE_EVAL_SAMPLE_LIMIT="${MEDICINE_EVAL_SAMPLE_LIMIT:-0}"
export EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-17}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-10000}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json}"
export ACL_EVAL_GLOSSARY_SIZES="${ACL_EVAL_GLOSSARY_SIZES:-10000}"
export MEDICINE_DEV_JSONL="${MEDICINE_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_dev_dataset.jsonl}"
export MEDICINE_EVAL_WIKI_GLOSSARY="${MEDICINE_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/medicine_glossary_gt_plus_medicine_wiki_gs10000.json}"
export MEDICINE_EVAL_GLOSSARY_SIZES="${MEDICINE_EVAL_GLOSSARY_SIZES:-10000}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS:-2}"
export EVAL_SCORE_DEVICE="${EVAL_SCORE_DEVICE:-cuda}"
export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-256}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-1024}"

export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs10000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-eval_dev/recall@10}"

export VARIANT_TAG="${VARIANT_TAG:-hn1024_vctx576_q3o_resume_e2s2646}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_resume_e2s2646_eval100_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep${EPOCHS}_v3_q3o_taurus8}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn1024_gsv2full_gsdedup_varctx576_resume_e2s2646_taurus8_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_eval100_med}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260518__varctx_lmlb_v3_qwen3omni_resume_e2s2646_taurus8_bs8k_gc128_eval100_med.md}"
export DATA_TAG="${DATA_TAG:-3variant_gsv2full_gsdedup_varctx576}"
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_hardneg}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn1024_vctx576_q3o_resume_e2s2646 compute:taurus-8gpu source:lh1b88kw}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw ah9u1bao dxwrgbln}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export MASTER_PORT="${MASTER_PORT:-30527}"

mkdir -p /mnt/gemini/data1/jiaxuanluo/logs

REQUIRED_PATHS=(
  "${BASE_LAUNCHER}"
  "${NOTES_FILE}"
  "${RESUME}"
  "${ACL_EVAL_WIKI_GLOSSARY}"
  "${MEDICINE_DEV_JSONL}"
  "${MEDICINE_EVAL_WIKI_GLOSSARY}"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
  if [ ! -f "${required_path}" ]; then
    echo "[ERROR] required file missing: ${required_path}" >&2
    exit 2
  fi
done

exec bash "${BASE_LAUNCHER}"
