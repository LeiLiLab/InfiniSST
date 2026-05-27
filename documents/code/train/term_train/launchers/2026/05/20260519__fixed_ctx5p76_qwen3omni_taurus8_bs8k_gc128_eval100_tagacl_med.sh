#!/bin/bash
#SBATCH --job-name=q3_ctx576_q3o_eval100
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ctx576_q3o_eval100_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ctx576_q3o_eval100_%x.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jiaxuanluo/InfiniSST}"
COMMON_LAUNCHER="${COMMON_LAUNCHER:-${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh}"

# Fixed-long-context control for the `lh1b88kw` variable-context run.
# ACL is kept as a held-out readout: checkpoint selection uses dev metrics only.
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
export BATCH_SIZE="${BATCH_SIZE:-8192}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_STEPS="${MAX_STEPS:-0}"
export MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-0}"
export MASTER_PORT="${MASTER_PORT:-30576}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"

export FIXED_AUDIO_SECONDS="${FIXED_AUDIO_SECONDS:-5.76}"
export EVAL_FIXED_AUDIO_SECONDS="${EVAL_FIXED_AUDIO_SECONDS:-5.76}"
export MAXSIM_WINDOWS="${MAXSIM_WINDOWS:-2 3 4 5 6 7 8 10 12 16 20 24}"
export MAXSIM_STRIDE="${MAXSIM_STRIDE:-2}"
export MFA_WINDOW_SELECTION="${MFA_WINDOW_SELECTION:-smallest}"
export MFA_POSITIVE_SCOPE="${MFA_POSITIVE_SCOPE:-auto}"

export HARD_NEG_K="${HARD_NEG_K:-0}"
export HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-1024}"
export NEG_BANK_REFRESH_STEPS="${NEG_BANK_REFRESH_STEPS:-50}"
export TCM_LOSS_WEIGHT="${TCM_LOSS_WEIGHT:-0.0}"
export TCM_POS_LOSS_WEIGHT="${TCM_POS_LOSS_WEIGHT:-0.0}"
export TCM_NEG_LOSS_WEIGHT="${TCM_NEG_LOSS_WEIGHT:-0.0}"
export TCM_POS_THRESHOLD="${TCM_POS_THRESHOLD:-0.80}"
export TCM_NEG_THRESHOLD="${TCM_NEG_THRESHOLD:-0.60}"
export TCM_WARMUP_STEPS="${TCM_WARMUP_STEPS:-0}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.85 0.80 0.75 0.70}"
export TERM_ID_NORMALIZE="${TERM_ID_NORMALIZE:-aggressive}"

export TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_ctx5p76.jsonl}"
export DEV_JSONL="${DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_ctx5p76_new_version.jsonl}"
export ACL_DEV_JSONL="${ACL_DEV_JSONL:-}"
export TAGGED_ACL_DEV_JSONL="${TAGGED_ACL_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_tagged_glossary_ctx5p76/acl6060_tagged_dev_dataset.jsonl}"
export MEDICINE_DEV_JSONL="${MEDICINE_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/medicine_eval_ctx5p76/medicine_dev_dataset.jsonl}"

export EVAL_WIKI_GLOSSARY="${EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-1000 10000}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-}"
export ACL_EVAL_GLOSSARY_SIZES="${ACL_EVAL_GLOSSARY_SIZES:-}"
export TAGGED_ACL_EVAL_WIKI_GLOSSARY="${TAGGED_ACL_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill_ctx5p76.json}"
export TAGGED_ACL_EVAL_GLOSSARY_SIZES="${TAGGED_ACL_EVAL_GLOSSARY_SIZES:-1000 10000}"
export MEDICINE_EVAL_WIKI_GLOSSARY="${MEDICINE_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/medicine_eval_ctx5p76/medicine_glossary_gt_plus_medicine_wiki_gs10000.json}"
export MEDICINE_EVAL_GLOSSARY_SIZES="${MEDICINE_EVAL_GLOSSARY_SIZES:-1000 10000}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS:-2}"
export EVAL_SCORE_DEVICE="${EVAL_SCORE_DEVICE:-cuda}"
export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-256}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-1024}"
export EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-0}"
export ACL_EVAL_SAMPLE_LIMIT="${ACL_EVAL_SAMPLE_LIMIT:-0}"
export TAGGED_ACL_EVAL_SAMPLE_LIMIT="${TAGGED_ACL_EVAL_SAMPLE_LIMIT:-0}"
export MEDICINE_EVAL_SAMPLE_LIMIT="${MEDICINE_EVAL_SAMPLE_LIMIT:-0}"
export EVAL_STEPS_SAMPLE="${EVAL_STEPS_SAMPLE:-100}"
export EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-0}"
export EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-17}"

export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs10000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-eval_dev/recall@10}"

export VARIANT_TAG="${VARIANT_TAG:-hn1024_ctx5p76_q3o_t8_b8k_g128}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_ctx5p76_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep${EPOCHS}_v3_q3o_taurus8}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn1024_gsv2full_gsdedup_ctx5p76_q3o_taurus8_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_eval100_tagacl_med}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260519__fixed_ctx5p76_qwen3omni_taurus8_bs8k_gc128_eval100_tagacl_med.md}"
export DATA_TAG="${DATA_TAG:-3variant_gsv2full_gsdedup_ctx5p76}"
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_hardneg}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn1024_ctx5p76_q3o_t8_b8k_g128 compute:taurus-8gpu source:lh1b88kw}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw ah9u1bao dxwrgbln}"
export RUN_VERDICT="${RUN_VERDICT:-Fixed 5.76s Qwen3-Omni retriever context-ablation control for variable-context source run lh1b88kw; dev-only checkpoint selection, tagged ACL and medicine readouts, eval every 100 steps.}"

NLP_AI_CS_EVAL_GLOSSARY="${NLP_AI_CS_EVAL_GLOSSARY:-${REPO_ROOT}/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs_enriched.json}"
export TRAIN_EXCLUDE_TERM_GLOSSARIES="${TRAIN_EXCLUDE_TERM_GLOSSARIES:-${NLP_AI_CS_EVAL_GLOSSARY} ${MEDICINE_EVAL_WIKI_GLOSSARY}}"
export STRICT_TRAIN_EVAL_TERM_FILTER="${STRICT_TRAIN_EVAL_TERM_FILTER:-false}"

mkdir -p /mnt/gemini/data1/jiaxuanluo/logs

for tag in \
  "family:${EXPERIMENT_FAMILY}" \
  "task:${TASK_TAG:-train}" \
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
  "${REPO_ROOT}/documents/code/train/term_train/qwen3_glossary_neg_train.py"
  "${NOTES_FILE}"
  "${TRAIN_JSONL}"
  "${DEV_JSONL}"
  "${TAGGED_ACL_DEV_JSONL}"
  "${MEDICINE_DEV_JSONL}"
  "${EVAL_WIKI_GLOSSARY}"
  "${TAGGED_ACL_EVAL_WIKI_GLOSSARY}"
  "${MEDICINE_EVAL_WIKI_GLOSSARY}"
)
if [ "${STRICT_TRAIN_EVAL_TERM_FILTER}" = "true" ]; then
  REQUIRED_PATHS+=( ${TRAIN_EXCLUDE_TERM_GLOSSARIES} )
fi

for required_path in "${REQUIRED_PATHS[@]}"; do
  if [ ! -f "${required_path}" ]; then
    echo "[ERROR] required file missing: ${required_path}" >&2
    exit 2
  fi
done

source "${COMMON_LAUNCHER}"
