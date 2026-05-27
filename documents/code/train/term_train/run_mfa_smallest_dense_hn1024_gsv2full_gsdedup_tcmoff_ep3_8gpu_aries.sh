#!/bin/bash
#SBATCH --job-name=q3_hn1024_gsdedup_off3
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_gsdedup_off3_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_gsdedup_off3_%x.err

# Full GSV2 k=1024 TCM-off run with one random row per absolute GigaSpeech
# MFA event. Matches the ep3 baseline except for TRAIN_JSONL / tags / naming.

set -euo pipefail

export VARIANT_TAG="hn1024_gsv2full_gsdedup_tcmoff_ep3"
export VERSION="3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep3_bs12k_smallest_dense_normAGGR_8gpu_aries"
export WANDB_EXP_NAME="variantE_hn1024_gsv2full_gsdedup_tcmoff_ep3_8gpu_aries"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_gsdedup_tcmoff_ep3_aries.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl"

export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=1024
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_POS_THRESHOLD=0.80
export TCM_NEG_THRESHOLD=0.60
export TCM_WARMUP_STEPS=0

export NUM_GPUS=8
export PER_GPU_BATCH=1536
export GRAD_CACHE_CHUNK_SIZE=512
export EPOCHS=3
# Keep the LR schedule open for continuation, matching the ep3 baseline.
export SCHEDULER_EPOCHS=4
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export MASTER_PORT=29987
export DATA_TAG="3variant_gsv2full_gsfix_mfa_gsdedup"
export EXTRA_WANDB_TAGS="variant:hn1024_gsv2full_gsdedup_tcmoff_ep3 compute:aries-8gpu"
export BASELINE_RUN_IDS="ly6sc2mr fma3wmh2 5np0cxmq us4obwe3 058tdx9a"
export SELECT_CLEAN_GPUS=true

# Dev-only selection with untrained P31 distractors.
export ACL_DEV_JSONL=""
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
export EVAL_GLOSSARY_SIZES="10000"
export BEST_METRIC="eval_dev/recall@10_gs10000"
export BEST_METRIC_SECONDARY="eval_dev/topk10_filtered_recall@tau_0p80_gs10000"
export EVAL_STEPS_SAMPLE=80
export TCM_SWEEP_THRESHOLDS="0.85 0.80 0.75 0.70"

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
