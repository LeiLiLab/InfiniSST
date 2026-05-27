#!/bin/bash
#SBATCH --job-name=q3_lh1b88kw_tau_tagacl
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=260G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_lh1b88kw_tau_tagacl_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_lh1b88kw_tau_tagacl_%x.err

set -euo pipefail

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="lh1b88kw_s2640_tau_delta_tagacl_medstrict"
export VERSION="3var_gsdedup_vctx576_lh1b88kw_s2640_tau_delta_tagacl_medstrict_${COMPUTE_TAG}"
export WANDB_EXP_NAME="lh1b88kw_s2640_tau_delta_dev_acl_tagacl_medstrict_${COMPUTE_TAG}_20260518"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes/2026/05/20260518__lh1b88kw_s2640_tau_delta_dev_acl_tagacl_medstrict_eval.md"

export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl"
export DEV_JSONL="/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl"
export ACL_DEV_JSONL="/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl"
export TAGGED_ACL_DEV_JSONL="/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_tagged_glossary_varctx2p88_3p84_4p80_5p76/acl6060_tagged_dev_dataset.jsonl"
export MEDICINE_DEV_JSONL="${MEDICINE_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl}"

export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt"
if [ ! -f "${RESUME}" ]; then
    echo "[FATAL] checkpoint not found: ${RESUME}" >&2
    exit 1
fi

export EVAL_ONLY=true
export TASK_TAG="eval"
export DATA_TAG="vctx576_lh1b88kw_tau_delta_tagacl_medstrict"
export EXPERIMENT_FAMILY="sst_ood_hardneg"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG} calibration:dev_delta readout:acl_taggedacl_medstrict"
export BASELINE_RUN_IDS="lh1b88kw 4g108a3w 614b3nbi"
export SELECT_CLEAN_GPUS=true

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

export GRAD_CACHE_CHUNK_SIZE=256
export EPOCHS=1
export SCHEDULER_EPOCHS=1
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESET_BEST_ON_RESUME=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false
export MASTER_PORT="${MASTER_PORT:-30368}"

# Calibration rule:
# - tau=0.0 is the raw recall@10 baseline.
# - Select the largest tau whose dev max recall drop across base/gs10k/gs100k
#   is within delta = 0.5pp, 1.0pp, or 1.5pp.
# - Report precision alongside the selected tau values, but do not use it for
#   tau selection.
# - ACL6060, tagged ACL6060, and strict MFA-only medicine are readouts only.
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json"
export EVAL_GLOSSARY_SIZES="10000 100000"
export ACL_EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json"
export ACL_EVAL_GLOSSARY_SIZES="1000 10000"
export TAGGED_ACL_EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json"
export TAGGED_ACL_EVAL_GLOSSARY_SIZES="1000 10000"
export MEDICINE_EVAL_WIKI_GLOSSARY="${MEDICINE_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json}"
export MEDICINE_EVAL_GLOSSARY_SIZES="${MEDICINE_EVAL_GLOSSARY_SIZES:-1000 10000}"
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/recall@10_gs100000"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export EVAL_SAMPLE_LIMIT=0
export ACL_EVAL_SAMPLE_LIMIT=0
export TAGGED_ACL_EVAL_SAMPLE_LIMIT=0
export MEDICINE_EVAL_SAMPLE_LIMIT=0
export EVAL_SAMPLE_SEED=17
export EVAL_SCORE_DEVICE="cuda"
export EVAL_SCORE_QUERY_CHUNK=256
export EVAL_SCORE_TEXT_CHUNK=1024
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS=2
export TCM_SWEEP_THRESHOLDS="0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90"
export TCM_SWEEP_FBETA=3.0
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="Eval-only tau-delta sweep for lh1b88kw secondary checkpoint. Select tau from dev base/gs10k/gs100k recall retention; ACL6060, tagged ACL6060, and strict MFA-only medicine are readout only."

echo "[TAU_DELTA_TAGACL_EVAL] ckpt=${RESUME}"
echo "[TAU_DELTA_TAGACL_EVAL] dev=${DEV_JSONL} glossary=${EVAL_WIKI_GLOSSARY} sizes=${EVAL_GLOSSARY_SIZES}"
echo "[TAU_DELTA_TAGACL_EVAL] acl=${ACL_DEV_JSONL} glossary=${ACL_EVAL_WIKI_GLOSSARY} sizes=${ACL_EVAL_GLOSSARY_SIZES}"
echo "[TAU_DELTA_TAGACL_EVAL] tagged_acl=${TAGGED_ACL_DEV_JSONL} glossary=${TAGGED_ACL_EVAL_WIKI_GLOSSARY} sizes=${TAGGED_ACL_EVAL_GLOSSARY_SIZES}"
echo "[TAU_DELTA_TAGACL_EVAL] medicine=${MEDICINE_DEV_JSONL} glossary=${MEDICINE_EVAL_WIKI_GLOSSARY} sizes=${MEDICINE_EVAL_GLOSSARY_SIZES}"
echo "[TAU_DELTA_TAGACL_EVAL] tau_grid=${TCM_SWEEP_THRESHOLDS}"
echo "[TAU_DELTA_TAGACL_EVAL] compute=${COMPUTE_TAG}"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
