#!/bin/bash
#SBATCH --job-name=q3_lh1b88kw_tagacl_tau3
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=260G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_lh1b88kw_tagacl_tau3_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_lh1b88kw_tagacl_tau3_%x.err

set -euo pipefail

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"

export VARIANT_TAG="${VARIANT_TAG:-lh1b88kw_s2640_tagged_acl_tau073_075_078}"
export VERSION="${VERSION:-3var_gsdedup_vctx576_lh1b88kw_s2640_tagged_acl_tau073_075_078_${COMPUTE_TAG}}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-lh1b88kw_s2640_tagged_acl_tau073_075_078_${COMPUTE_TAG}_${RUN_STAMP}}"
export NOTES_FILE="${NOTES_FILE:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes/2026/05/20260518__lh1b88kw_s2640_tagged_acl_tau073_075_078_eval.md}"

export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl"
export DEV_JSONL=""
export ACL_DEV_JSONL=""
export TAGGED_ACL_DEV_JSONL="${TAGGED_ACL_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_tagged_glossary_varctx2p88_3p84_4p80_5p76/acl6060_tagged_dev_dataset.jsonl}"
export MEDICINE_DEV_JSONL=""

export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt"
if [ ! -f "${RESUME}" ]; then
    echo "[FATAL] checkpoint not found: ${RESUME}" >&2
    exit 1
fi

export EVAL_ONLY=true
export TASK_TAG="eval"
export DATA_TAG="${DATA_TAG:-vctx576_lh1b88kw_tagged_acl_tau3}"
export EXPERIMENT_FAMILY="sst_ood_hardneg"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:${VARIANT_TAG} compute:${COMPUTE_TAG} readout:tagged_acl calibration:dev_tau_reuse}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw 4g108a3w qjy4m1x9}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"

# Match the checkpoint recipe from W&B run lh1b88kw.
export FIXED_AUDIO_SECONDS=5.76
export EVAL_FIXED_AUDIO_SECONDS=5.76
export AUDIO_ENCODER_PRESET="qwen3-omni"
export AUDIO_ENCODER_TYPE="qwen3_omni"
export AUDIO_MODEL_ID="Atotti/Qwen3-Omni-AudioTransformer"
export AUDIO_FEATURE_EXTRACTOR_ID="openai/whisper-large-v3"
export AUDIO_INPUT_DTYPE="auto"
export TEXT_ENCODER_PRESET="bge-m3"
export TEXT_MODEL_ID="BAAI/bge-m3"
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
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.73 0.75 0.78}"
export TCM_SWEEP_FBETA=3.0

export GRAD_CACHE_CHUNK_SIZE=256
export NUM_WORKERS=0
export EPOCHS=1
export SCHEDULER_EPOCHS=1
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESET_BEST_ON_RESUME=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false
export MASTER_PORT="${MASTER_PORT:-30385}"

export EVAL_WIKI_GLOSSARY="${EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-1000 10000}"
export ACL_EVAL_WIKI_GLOSSARY=""
export ACL_EVAL_GLOSSARY_SIZES=""
export TAGGED_ACL_EVAL_WIKI_GLOSSARY="${TAGGED_ACL_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
export TAGGED_ACL_EVAL_GLOSSARY_SIZES="${TAGGED_ACL_EVAL_GLOSSARY_SIZES:-1000 10000}"
export MEDICINE_EVAL_WIKI_GLOSSARY=""
export MEDICINE_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export BEST_METRIC="eval_tagged_acl/recall@10_gs10000"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export EVAL_SAMPLE_LIMIT=0
export ACL_EVAL_SAMPLE_LIMIT=0
export TAGGED_ACL_EVAL_SAMPLE_LIMIT=0
export MEDICINE_EVAL_SAMPLE_LIMIT=0
export EVAL_SAMPLE_SEED=17
export EVAL_SCORE_DEVICE="${EVAL_SCORE_DEVICE:-cuda}"
export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-256}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-1024}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS=2

export DUMP_EVAL_MISSES_DIR="${DUMP_EVAL_MISSES_DIR:-}"
export DUMP_EVAL_MISSES_EVAL_NAMES="${DUMP_EVAL_MISSES_EVAL_NAMES:-tagged_acl}"
export DUMP_EVAL_MISSES_BANKS="${DUMP_EVAL_MISSES_BANKS:-base gs1000 gs10000}"
export DUMP_EVAL_MISSES_TOPN="${DUMP_EVAL_MISSES_TOPN:-120}"
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="${RUN_VERDICT:-Tagged ACL-only readout for lh1b88kw step-2640 at tau 0.73/0.75/0.78. Tau choices reuse the prior dev calibration; no dev/ACL-paper/medicine scoring is run here.}"

echo "[TAGGED_ACL_TAU3_EVAL] ckpt=${RESUME}"
echo "[TAGGED_ACL_TAU3_EVAL] tagged_acl=${TAGGED_ACL_DEV_JSONL}"
echo "[TAGGED_ACL_TAU3_EVAL] glossary=${TAGGED_ACL_EVAL_WIKI_GLOSSARY} sizes=${TAGGED_ACL_EVAL_GLOSSARY_SIZES}"
echo "[TAGGED_ACL_TAU3_EVAL] tau_grid=${TCM_SWEEP_THRESHOLDS}"
echo "[TAGGED_ACL_TAU3_EVAL] compute=${COMPUTE_TAG}"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
