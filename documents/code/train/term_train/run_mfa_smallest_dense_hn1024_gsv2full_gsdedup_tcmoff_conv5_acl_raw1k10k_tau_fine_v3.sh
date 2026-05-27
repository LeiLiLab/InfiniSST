#!/bin/bash
#SBATCH --job-name=q3_gsdedup_acl_fine
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_gsdedup_acl_fine_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_gsdedup_acl_fine_%x.err

set -euo pipefail

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-512}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="acl_tau_fine_gsdedup_conv5"
export VERSION="3var_gsv2full_gsfix_mfa_gsdedup_tcmoff_conv5_acl_tau_fine_v3_${COMPUTE_TAG}_smallest_dense"
export WANDB_EXP_NAME="gsdedup_tcmoff_conv5_acl_tau_fine_v3_${COMPUTE_TAG}"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_gsdedup_tcmoff_conv5_acl_tau_fine_v3.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl"
export DEV_JSONL=""
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
export MASTER_PORT="${MASTER_PORT:-30347}"

export DATA_TAG="acl_tau_fine"
export EXPERIMENT_FAMILY="sst_ood_hardneg"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG} readout:acl"
export BASELINE_RUN_IDS="7xu2b4so e9f4xnd1 bdmc71za"
export SELECT_CLEAN_GPUS=true

export ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
export EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"
export EVAL_GLOSSARY_SIZES="1000 10000"
export ACL_EVAL_WIKI_GLOSSARY=""
export ACL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_acl6060/topk10_filtered_recall@tau_0p74_gs10000"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="0.70 0.72 0.74 0.76 0.78 0.80 0.82 0.84 0.86"
export TCM_SWEEP_FBETA="${TCM_SWEEP_FBETA:-3.0}"
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="ACL-only tau readout after dev-only calibration. This run is for reporting candidate thresholds on ACL raw/gs1k/gs10k, not for choosing tau."

echo "[GSDEDUP_ACL_TAU_FINE] ckpt=${RESUME}"
echo "[GSDEDUP_ACL_TAU_FINE] dev=<disabled>"
echo "[GSDEDUP_ACL_TAU_FINE] acl=${ACL_DEV_JSONL}"
echo "[GSDEDUP_ACL_TAU_FINE] glossary=${EVAL_WIKI_GLOSSARY} sizes=${EVAL_GLOSSARY_SIZES}"
echo "[GSDEDUP_ACL_TAU_FINE] beta=${TCM_SWEEP_FBETA} taus=${TCM_SWEEP_THRESHOLDS}"
echo "[GSDEDUP_ACL_TAU_FINE] compute=${COMPUTE_TAG} batch=$((NUM_GPUS * PER_GPU_BATCH))"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
