#!/bin/bash
#SBATCH --job-name=q3_ep3_acl_ctx384
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ep3_acl_ctx384_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ep3_acl_ctx384_%x.err

set -euo pipefail

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-512}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="acl_ctx384_gsdedup_ep3best"
export VERSION="3var_gsv2full_gsfix_mfa_gsdedup_ep3best_acl_ctx384_${COMPUTE_TAG}_smallest_dense"
export WANDB_EXP_NAME="gsdedup_ep3best_acl_ctx384_${COMPUTE_TAG}"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_gsdedup_ep3best_acl_ctx3p84_eval_aries.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl"
export DEV_JSONL=""
export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep3_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt"

export EVAL_ONLY=true
export TASK_TAG="eval"
export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=0
export NEG_BANK_SIZE=0
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_POS_THRESHOLD=0.80
export TCM_NEG_THRESHOLD=0.60
export TCM_LOSS_FORM="hinge"
export TCM_REDUCTION="mean_viol"
export TCM_NEG_SCOPE="all"
export TCM_NEG_TOPK=0
export TCM_WARMUP_STEPS=0

export GRAD_CACHE_CHUNK_SIZE=256
export FIXED_AUDIO_SECONDS=3.84
export EVAL_FIXED_AUDIO_SECONDS=3.84
export EPOCHS=1
export SCHEDULER_EPOCHS=1
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false
export MASTER_PORT="${MASTER_PORT:-30384}"

export DATA_TAG="acl6060_ctx384_ep3best_eval"
export EXPERIMENT_FAMILY="sst_ood_hardneg"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG} readout:acl"
export BASELINE_RUN_IDS="ah9u1bao dxwrgbln 0k7s41qt bdmc71za"
export SELECT_CLEAN_GPUS=true

export ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_ctx3p84/acl6060_dev_dataset.jsonl"
export EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"
export EVAL_GLOSSARY_SIZES="10000"
export ACL_EVAL_WIKI_GLOSSARY=""
export ACL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_acl6060/recall@10_gs10000"
export BEST_METRIC_SECONDARY=""
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="0.85 0.80 0.75 0.70"
export TCM_SWEEP_FBETA="${TCM_SWEEP_FBETA:-3.0}"
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="One-shot ACL6060 3.84s readout for the ah9u1bao ep3 best checkpoint; compare to dxwrgbln inline ACL at matched steps."

for required_path in "${RESUME}" "${TRAIN_JSONL}" "${ACL_DEV_JSONL}" "${NOTES_FILE}"; do
    if [ ! -f "${required_path}" ]; then
        echo "[FATAL] required file missing: ${required_path}" >&2
        exit 1
    fi
done

echo "[EP3BEST_ACL_CTX384] ckpt=${RESUME}"
echo "[EP3BEST_ACL_CTX384] dev=<disabled>"
echo "[EP3BEST_ACL_CTX384] acl=${ACL_DEV_JSONL}"
echo "[EP3BEST_ACL_CTX384] glossary=${EVAL_WIKI_GLOSSARY} sizes=${EVAL_GLOSSARY_SIZES}"
echo "[EP3BEST_ACL_CTX384] fixed_audio_seconds=${FIXED_AUDIO_SECONDS} eval_fixed_audio_seconds=${EVAL_FIXED_AUDIO_SECONDS}"
echo "[EP3BEST_ACL_CTX384] beta=${TCM_SWEEP_FBETA} taus=${TCM_SWEEP_THRESHOLDS}"
echo "[EP3BEST_ACL_CTX384] compute=${COMPUTE_TAG} batch=$((NUM_GPUS * PER_GPU_BATCH))"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
