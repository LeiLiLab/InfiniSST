#!/bin/bash
#SBATCH --job-name=q3_vctx_bgel_g256
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_bgel_g256_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vctx_bgel_g256_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh"

# Next-version text-encoder ablation launcher.  Keep the same BGE-large
# varctx576 setup as the gc128 run, but default GradCache to 256 to reduce
# per-step chunking overhead while leaving enough A6000 memory headroom.
export TEXT_ENCODER_PRESET="${TEXT_ENCODER_PRESET:-bge-large-en-v1.5}"
export TEXT_MODEL_ID="${TEXT_MODEL_ID:-BAAI/bge-large-en-v1.5}"

export NUM_GPUS="${NUM_GPUS:-8}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1024}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-256}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-0}"

# Inline eval stays dev+ACL+medicine for W&B selection/readout continuity.
# Dev is a deterministic 100-sample smoke readout; threshold diagnostics use
# only tau=0.75 because tau=0.0 is already represented by raw recall@10.
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

export VARIANT_TAG="${VARIANT_TAG:-vctx576_txt_bgel_t8_d100_tau1_g256}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_txt_bgel_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep${EPOCHS}_v3_dev100Tau1_eval100_taurus8}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_varctx576_txt_bgel_taurus8_bs8k_gc${GRAD_CACHE_CHUNK_SIZE}_dev100Tau1_eval100}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260517__varctx_lmlb_v3_text_bge_large_en_taurus8_gc256_dev100_tau1_eval.md}"
export DATA_TAG="${DATA_TAG:-3variant_gsv2full_gsdedup_vctx576_bgel}"
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_hardneg}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:vctx576_txt_bgel_t8_d100_tau1_g256 compute:taurus-8gpu}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export MASTER_PORT="${MASTER_PORT:-30032}"
export RUN_VERDICT="${RUN_VERDICT:-BGE-large-en-v1.5 text-encoder ablation on varctx576, Taurus 8GPU, GradCache chunk 256, GPU-chunked eval scoring, dev100 base/gs1k/gs10k inline eval plus full ACL/medicine readouts, tau sweep 0.75 only.}"

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
