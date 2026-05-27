#!/bin/bash
#SBATCH --job-name=q3_gsdedup_tau_sweep
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_gsdedup_tau_sweep_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_gsdedup_tau_sweep_%x.err

set -euo pipefail

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-512}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"

export VARIANT_TAG="tau_sweep_gsdedup_conv5"
export VERSION="3var_gsv2full_gsfix_mfa_gsdedup_tcmoff_conv5_tau_sweep_v3_${COMPUTE_TAG}_smallest_dense"
export WANDB_EXP_NAME="gsdedup_tcmoff_conv5_tau_sweep_v3_${COMPUTE_TAG}"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_gsdedup_tcmoff_conv5_tau_sweep_acl_dev_v3.md"
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
export MASTER_PORT="${MASTER_PORT:-30343}"

export DATA_TAG="3variant_gsv2full_gsfix_mfa_gsdedup_tau_sweep_eval"
export EXPERIMENT_FAMILY="sst_ood_hardneg"
export EXTRA_WANDB_TAGS="variant:${VARIANT_TAG} compute:${COMPUTE_TAG} protocol:fixed-raw-denominator"
export BASELINE_RUN_IDS="7xu2b4so ig2mjmil ah9u1bao"
export SELECT_CLEAN_GPUS=true

export ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
export EVAL_METRIC_DENOMINATOR="fixed_raw"
export EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"
export EVAL_GLOSSARY_SIZES="1000 10000"
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/topk10_chunk_any_positive_filtered_recall@tau_0p75_gs10000"
export BEST_METRIC_SECONDARY="eval_acl6060/topk10_chunk_any_positive_filtered_recall@tau_0p75_gs10000"
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="0.65 0.66 0.67 0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90"
export TCM_SWEEP_FBETA="${TCM_SWEEP_FBETA:-3.0}"  # deprecated no-op; kept for launcher compatibility
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="Dense inference tau sweep for the gsdedup TCM-off conv5 best checkpoint. Metrics denominator is fixed to the strict raw/base glossary; retriever banks change with glossary size."

echo "[GSDEDUP_TAU_SWEEP] ckpt=${RESUME}"
echo "[GSDEDUP_TAU_SWEEP] dev=${DEV_JSONL}"
echo "[GSDEDUP_TAU_SWEEP] acl=${ACL_DEV_JSONL}"
echo "[GSDEDUP_TAU_SWEEP] eval_metric_denominator=${EVAL_METRIC_DENOMINATOR}"
echo "[GSDEDUP_TAU_SWEEP] glossary=${EVAL_WIKI_GLOSSARY} sizes=${EVAL_GLOSSARY_SIZES}"
echo "[GSDEDUP_TAU_SWEEP] beta=${TCM_SWEEP_FBETA} taus=${TCM_SWEEP_THRESHOLDS}"
echo "[GSDEDUP_TAU_SWEEP] compute=${COMPUTE_TAG} batch=$((NUM_GPUS * PER_GPU_BATCH))"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
