#!/bin/bash
#SBATCH --job-name=q3_hn1024_smoke100k
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_smoke100k_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_smoke100k_%x.err

# Smoke test the two-tier eval path: one lightweight 10k eval plus one sparse
# 100k full eval immediately after resuming.

set -euo pipefail

export VARIANT_TAG="hn1024_gsv2full_r3_smoke100k_chunk"
export VERSION="3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_r3_smoke100k_chunk_bs12k_smallest_dense_normAGGR_8gpu_aries"
export WANDB_EXP_NAME="variantE_hn1024_gsv2full_tcmoff_resume3_full100k_chunked_smoke_8gpu_aries"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_tcmoff_resume3_conv_aries.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl"
export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep3_bs12k_smallest_dense_normAGGR_8gpu_aries_epoch_2.pt"

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
export EPOCHS=8
export SCHEDULER_EPOCHS=8
export RESET_SCHEDULER=true
# Resume checkpoint is global_step=1524; one update reaches the eval trigger.
export MAX_STEPS=1525
export MAX_TRAIN_SECONDS=0
export MASTER_PORT=29989
export DATA_TAG="3variant_gsv2full_gsfix_mfa"
export EXTRA_WANDB_TAGS="variant:hn1024_gsv2full_r3_smoke100k_chunk compute:aries-8gpu"
export BASELINE_RUN_IDS="k6e9askw ly6sc2mr fma3wmh2 x68wsne9"
export SELECT_CLEAN_GPUS=true

export ACL_DEV_JSONL=""
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
export EVAL_GLOSSARY_SIZES="10000"
export FULL_EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json"
export FULL_EVAL_GLOSSARY_SIZES="100000"
export FULL_EVAL_EVERY_N_EVALS=1
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="eval_dev/recall@10_gs10000"
export BEST_METRIC_SECONDARY="eval_dev_full/recall@10_gs100000"
export EVAL_STEPS_SAMPLE=25
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="0.85 0.80 0.75 0.70"
export RUN_VERDICT="SMOKE: verifies one lightweight 10k eval plus one sparse 100k full eval after resume."

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
