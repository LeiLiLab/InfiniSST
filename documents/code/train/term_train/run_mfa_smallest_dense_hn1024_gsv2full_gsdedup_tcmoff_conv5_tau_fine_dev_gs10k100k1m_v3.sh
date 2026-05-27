#!/bin/bash
#SBATCH --job-name=q3_gsdedup_tau_fine
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=260G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_gsdedup_tau_fine_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_gsdedup_tau_fine_%x.err

set -euo pipefail

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-512}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="tau_fine_gsdedup_conv5_dev_gs10k100k1m"
export VERSION="3var_gsv2full_gsfix_mfa_gsdedup_tcmoff_conv5_tau_fine_dev_gs10k100k1m_v3_${COMPUTE_TAG}_smallest_dense"
export WANDB_EXP_NAME="gsdedup_tcmoff_conv5_tau_fine_dev_gs10k100k1m_v3_${COMPUTE_TAG}"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_gsdedup_tcmoff_conv5_tau_fine_dev_gs10k100k1m_v3.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl"
export DEV_JSONL="/mnt/gemini/home/jiaxuanluo/noterm_dev/term_dev_with_gs_noterm_wiki_v3_balanced.jsonl"
export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv5_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt"

if [ ! -f "${RESUME}" ]; then
    echo "[FATAL] checkpoint not found: ${RESUME}" >&2
    exit 1
fi

export EVAL_ONLY=true
export TASK_TAG="eval"
export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=0
export NEG_BANK_SIZE=0
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_POS_THRESHOLD=0.85
export TCM_NEG_THRESHOLD=0.64
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
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false
export MASTER_PORT="${MASTER_PORT:-30346}"

export DATA_TAG="gsdedup_fine_dev_calib"
export EXPERIMENT_FAMILY="sst_ood_hardneg"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG} calibration:dev_only"
export BASELINE_RUN_IDS="7xu2b4so boykrco2 bdmc71za e9f4xnd1"
export SELECT_CLEAN_GPUS=true

export ACL_DEV_JSONL=""
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json"
export EVAL_GLOSSARY_SIZES="10000 100000 1000000"
export ACL_EVAL_WIKI_GLOSSARY=""
export ACL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/recall@10_gs1000000"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="0.65 0.66 0.67 0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90"
export TCM_SWEEP_FBETA="${TCM_SWEEP_FBETA:-3.0}"
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="Dev-only fine tau sweep for recall-retention calibration. Uses the P31 1M glossary with evaluation sizes 10k, 100k, and 1M. ACL is intentionally disabled."

echo "[GSDEDUP_TAU_FINE_DEV] ckpt=${RESUME}"
echo "[GSDEDUP_TAU_FINE_DEV] dev=${DEV_JSONL}"
echo "[GSDEDUP_TAU_FINE_DEV] acl=<disabled for dev-only calibration>"
echo "[GSDEDUP_TAU_FINE_DEV] dev_glossary=${EVAL_WIKI_GLOSSARY} sizes=${EVAL_GLOSSARY_SIZES}"
echo "[GSDEDUP_TAU_FINE_DEV] beta=${TCM_SWEEP_FBETA} taus=${TCM_SWEEP_THRESHOLDS}"
echo "[GSDEDUP_TAU_FINE_DEV] compute=${COMPUTE_TAG} batch=$((NUM_GPUS * PER_GPU_BATCH))"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
