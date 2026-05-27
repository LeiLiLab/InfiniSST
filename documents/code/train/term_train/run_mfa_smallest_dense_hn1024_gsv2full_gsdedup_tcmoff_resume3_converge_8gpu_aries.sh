#!/bin/bash
#SBATCH --job-name=q3_hn1024_gsdedup_conv5
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_gsdedup_conv5_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_gsdedup_conv5_%x.err

# Resume the deduplicated full GSV2 k=1024 TCM-off run after the first
# continuation stopped too early at three stale evals. Hard-negative refreshes
# can temporarily depress recall, so allow five consecutive evals without a
# primary dev gs10000 refresh before stopping.

set -euo pipefail

export VARIANT_TAG="hn1024_gsv2full_gsdedup_tcmoff_conv5"
export VERSION="3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv5_bs12k_smallest_dense_normAGGR_8gpu_aries"
export WANDB_EXP_NAME="variantE_hn1024_gsv2full_gsdedup_tcmoff_conv5_8gpu_aries"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_gsdedup_tcmoff_conv5_aries.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl"
export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv_bs12k_smallest_dense_normAGGR_8gpu_aries.pt"

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
# The resumed checkpoint is the final save from job 45194 at step=1150. This
# runs through epoch=7 unless early-stop triggers first.
export EPOCHS=8
export SCHEDULER_EPOCHS=8
export RESET_SCHEDULER=false
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export MASTER_PORT=29989
export DATA_TAG="3variant_gsv2full_gsfix_mfa_gsdedup"
export EXTRA_WANDB_TAGS="variant:hn1024_gsv2full_gsdedup_tcmoff_conv5 compute:aries-8gpu"
export BASELINE_RUN_IDS="ig2mjmil ah9u1bao ly6sc2mr fma3wmh2 5np0cxmq us4obwe3 058tdx9a"
export SELECT_CLEAN_GPUS=true

# Dev selection keeps the untrained P31 distractor bank for BEST_METRIC.
# ACL6060 is evaluated in-process with its own raw 1k/10k glossary, so W&B
# shows cross-domain process metrics without changing the dev checkpoint target.
export ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
export EVAL_GLOSSARY_SIZES="10000"
export ACL_EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"
export ACL_EVAL_GLOSSARY_SIZES="1000 10000"
export BEST_METRIC="eval_dev/recall@10_gs10000"
export BEST_METRIC_SECONDARY="eval_dev/topk10_filtered_recall@tau_0p80_gs10000"
export EVAL_STEPS_SAMPLE=50
export EARLY_STOP_BEST_PATIENCE_EVALS=5
export EVAL_TOP100_SAMPLES=0
export TCM_SWEEP_THRESHOLDS="0.85 0.80 0.75 0.70"
export RUN_VERDICT="Continuation from job 45194 final checkpoint; stop after five additional consecutive dev gs10000 evals without a primary best refresh."

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
