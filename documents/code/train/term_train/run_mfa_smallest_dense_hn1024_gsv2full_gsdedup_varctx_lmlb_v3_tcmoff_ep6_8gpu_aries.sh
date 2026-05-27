#!/bin/bash
#SBATCH --job-name=q3_hn1024_varctx_v3
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/gemini/home/jiaxuanluo/logs/%j_q3_hn1024_varctx_v3_%x.out
#SBATCH --error=/mnt/gemini/home/jiaxuanluo/logs/%j_q3_hn1024_varctx_v3_%x.err

set -euo pipefail

# Full GSV2 k=1024 TCM-off run with balanced variable 2.88/3.84/4.80/5.76s
# train/dev/ACL contexts. Env defaults are intentionally overrideable so the
# delayed launcher can smoke-test GradCache chunk sizes before the full run.

export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-512}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_STEPS="${MAX_STEPS:-0}"

export VARIANT_TAG="${VARIANT_TAG:-hn1024_varctx576_v3_tcmoff_ep6}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_gc${GRAD_CACHE_CHUNK_SIZE}_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep${EPOCHS}_v3_bs12k_smallest_dense_normAGGR_8gpu_aries}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn1024_gsv2full_gsdedup_varctx576_v3_gc${GRAD_CACHE_CHUNK_SIZE}_tcmoff_ep${EPOCHS}_8gpu_aries}"
export NOTES_FILE="${NOTES_FILE:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_aries.md}"
export TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl}"

export HARD_NEG_K="${HARD_NEG_K:-0}"
export HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-1024}"
export TCM_LOSS_WEIGHT="${TCM_LOSS_WEIGHT:-0.0}"
export TCM_POS_LOSS_WEIGHT="${TCM_POS_LOSS_WEIGHT:-0.0}"
export TCM_NEG_LOSS_WEIGHT="${TCM_NEG_LOSS_WEIGHT:-0.0}"
export TCM_POS_THRESHOLD="${TCM_POS_THRESHOLD:-0.80}"
export TCM_NEG_THRESHOLD="${TCM_NEG_THRESHOLD:-0.60}"
export TCM_WARMUP_STEPS="${TCM_WARMUP_STEPS:-0}"

export NUM_GPUS="${NUM_GPUS:-8}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1536}"
export FIXED_AUDIO_SECONDS="${FIXED_AUDIO_SECONDS:-5.76}"
export EVAL_FIXED_AUDIO_SECONDS="${EVAL_FIXED_AUDIO_SECONDS:-5.76}"
export MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-0}"
export MASTER_PORT="${MASTER_PORT:-29991}"
export DATA_TAG="${DATA_TAG:-3variant_gsv2full_gsdedup_varctx576}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn1024_varctx576_v3 compute:aries-8gpu}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-ah9u1bao dxwrgbln}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"

export DEV_JSONL="${DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl}"
export ACL_DEV_JSONL="${ACL_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl}"
export MEDICINE_DEV_JSONL="${MEDICINE_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl}"
export EVAL_WIKI_GLOSSARY="${EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-10000}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json}"
export ACL_EVAL_GLOSSARY_SIZES="${ACL_EVAL_GLOSSARY_SIZES:-10000}"
export MEDICINE_EVAL_WIKI_GLOSSARY="${MEDICINE_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json}"
export MEDICINE_EVAL_GLOSSARY_SIZES="${MEDICINE_EVAL_GLOSSARY_SIZES:-10000}"
export NLP_AI_CS_EVAL_GLOSSARY="${NLP_AI_CS_EVAL_GLOSSARY:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs_enriched.json}"
export TRAIN_EXCLUDE_TERM_GLOSSARIES="${TRAIN_EXCLUDE_TERM_GLOSSARIES:-${NLP_AI_CS_EVAL_GLOSSARY} ${MEDICINE_EVAL_WIKI_GLOSSARY}}"
export STRICT_TRAIN_EVAL_TERM_FILTER="${STRICT_TRAIN_EVAL_TERM_FILTER:-false}"
export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs10000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-eval_acl6060/recall@10}"
export EVAL_STEPS_SAMPLE="${EVAL_STEPS_SAMPLE:-80}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.75}"

mkdir -p /mnt/gemini/home/jiaxuanluo/logs

for tag in \
  "family:${EXPERIMENT_FAMILY:-sst_ood_hardneg}" \
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
  "${TRAIN_JSONL}"
  "${DEV_JSONL}"
  "${ACL_DEV_JSONL}"
  "${MEDICINE_DEV_JSONL}"
  "${MEDICINE_EVAL_WIKI_GLOSSARY}"
  "${NOTES_FILE}"
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

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
