#!/bin/bash
#SBATCH --job-name=q3_hn1024_ctx384_off6
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:8
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_ctx384_off6_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn1024_ctx384_off6_%x.err

# Full GSV2 k=1024 TCM-off run with GigaSpeech MFA-event dedup and 3.84s
# context chunks. GigaSpeech and wiki-synth rows are both recut to 3.84s
# windows by MFA before training.

set -euo pipefail

export VARIANT_TAG="hn1024_gsv2full_gsdedup_ctx384_tcmoff_ep6"
export VERSION="3var_gsv2full_gsfix_mfa_gsdedup_ctx384_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_bs12k_smallest_dense_normAGGR_8gpu_aries"
export WANDB_EXP_NAME="variantE_hn1024_gsv2full_gsdedup_ctx384_tcmoff_ep6_8gpu_aries"
export NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hn1024_gsv2full_gsdedup_ctx3p84_tcmoff_ep6_aries.md"
export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_gsctx3p84.jsonl"

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
# 3.84s doubles the encoder time axis; keep the effective batch fixed but use
# smaller GradCache chunks to reduce peak memory.
export GRAD_CACHE_CHUNK_SIZE=256
export FIXED_AUDIO_SECONDS=3.84
export EVAL_FIXED_AUDIO_SECONDS=3.84
export EPOCHS=6
export SCHEDULER_EPOCHS=6
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export MASTER_PORT=29989
export DATA_TAG="3variant_gsv2full_gsfix_mfa_gsdedup_ctx384"
export EXTRA_WANDB_TAGS="variant:hn1024_gsv2full_gsdedup_ctx384_tcmoff_ep6 compute:aries-8gpu"
export BASELINE_RUN_IDS="ah9u1bao 7xu2b4so ly6sc2mr fma3wmh2 5np0cxmq us4obwe3 058tdx9a"
export SELECT_CLEAN_GPUS=true

# 3.84s retriever eval chunks.
export DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_m4.jsonl"
export ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_ctx3p84/acl6060_dev_dataset.jsonl"
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json"
export EVAL_GLOSSARY_SIZES="10000"
export ACL_EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"
export ACL_EVAL_GLOSSARY_SIZES="10000"
export BEST_METRIC="eval_dev/recall@10_gs10000"
export BEST_METRIC_SECONDARY="eval_dev/topk10_filtered_recall@tau_0p80_gs10000"
export EVAL_STEPS_SAMPLE=80
export TCM_SWEEP_THRESHOLDS="0.85 0.80 0.75 0.70"

for required_path in "${TRAIN_JSONL}" "${DEV_JSONL}" "${ACL_DEV_JSONL}" "${NOTES_FILE}"; do
    if [ ! -f "${required_path}" ]; then
        echo "[ERROR] required file missing: ${required_path}" >&2
        if [ "${required_path}" = "${TRAIN_JSONL}" ]; then
            echo "[ERROR] build it first with: bash documents/code/data_pre/training_terms_for_retriever/run_expand_gsv2full_gsdedup_gsctx3p84.sh" >&2
        fi
        exit 2
    fi
done

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
