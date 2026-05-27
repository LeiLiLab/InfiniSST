#!/bin/bash
set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"

export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-aries}"
export RUN_STAMP="${RUN_STAMP:-ctx5p76_hn1024_latest_dev100k_fixedraw_sample100k_aries_gpu0_20260525T010522Z}"
export MODEL_TAG="hn1024_ctx5p76_latest_s1300"
export VARIANT_TAG="hn1024_ctx5p76_latest_s1300_dev100k_fixedraw"
export VERSION="3var_gsv2full_gsdedup_ctx5p76_hn1024_latest_s1300_dev100k_fixedraw_sample100k_${RUN_STAMP}"
export WANDB_EXP_NAME="variantE_hn1024_ctx5p76_latest_s1300_dev100k_fixedraw_sample100k_${RUN_STAMP}"
export DATA_TAG="ctx5p76_dev100k_fixedraw"
export EXPERIMENT_FAMILY="sst_ood_hardneg_eval_compare"
export TASK_TAG="eval"
export EXTRA_WANDB_TAGS="variant:hn1024_ctx5p76_latest_s1300_dev100k_fixedraw compute:aries-gpu0-shared source:jyb2u787 checkpoint:latest_s1300 protocol:dev-fixedraw readout:dev-only glossary:sample100k"
export BASELINE_RUN_IDS="jyb2u787 zseptpl0 lh1b88kw 1prwlh34"

export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0}"
export NUM_GPUS=1
export PER_GPU_BATCH=1
export BATCH_SIZE=1
export MASTER_PORT="${MASTER_PORT:-20048}"
export SELECT_CLEAN_GPUS=false
export WAIT_FOR_CLEAN_GPUS=false
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/tmp/jiaxuanluo_q3_ctx5p76_dev100k_eval_sample100k_${RUN_STAMP}}"

export WANDB_DIR="/mnt/gemini/data1/jiaxuanluo/wandb"
export WANDB_CACHE_DIR="/mnt/aries/data4/jiaxuanluo/cache/wandb"
export XDG_CACHE_HOME="/mnt/aries/data4/jiaxuanluo/cache"
export XDG_CONFIG_HOME="/mnt/aries/data4/jiaxuanluo/config"

export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_ctx5p76_bs8190_gc128_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_q3o_resume_s400_gpu012367_aries6_latest.pt"
export NOTES_FILE="${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260525__ctx5p76_hn1024_latest_dev100k_fixedraw_sample100k_eval.md"

export TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_ctx5p76.jsonl"
export DEV_JSONL="/mnt/gemini/home/jiaxuanluo/term_dev_dataset_ctx5p76_new_version.jsonl"
export EVAL_WIKI_GLOSSARY="/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json"
export EVAL_METRICS_GLOSSARY=""
export EVAL_GLOSSARY_SIZES="1000 10000 100000"
export EVAL_METRIC_DENOMINATOR="fixed_raw"
export BEST_METRIC="eval_dev/recall@10_gs100000"
export BEST_METRIC_SECONDARY=""

export FIXED_AUDIO_SECONDS=5.76
export EVAL_FIXED_AUDIO_SECONDS=5.76
export AUDIO_ENCODER_PRESET="qwen3-omni"
export AUDIO_ENCODER_TYPE="qwen3_omni"
export AUDIO_MODEL_ID="Atotti/Qwen3-Omni-AudioTransformer"
export AUDIO_FEATURE_EXTRACTOR_ID="openai/whisper-large-v3"
export AUDIO_INPUT_DTYPE="auto"
export AUDIO_HIDDEN_DIM=0
export TEXT_ENCODER_PRESET="bge-m3"
export TEXT_MODEL_ID="BAAI/bge-m3"
export TEXT_INPUT_PREFIX=""
export TEXT_POOLING="cls"
export TEXT_TARGET_MODULES="query key value dense"
export TEXT_LORA_RANK=128
export TEXT_LORA_ALPHA=256
export LORA_RANK=128
export LORA_ALPHA=256
export TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
export TARGET_DIM=1024
export POOLING_TYPE="transformer"
export USE_MAXSIM=true
export MFA_SUPERVISED=true
export MFA_WINDOW_SELECTION="smallest"
export MFA_POSITIVE_SCOPE="auto"
export MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"
export MAXSIM_STRIDE=2
export TERM_ID_NORMALIZE="aggressive"

export EVAL_ONLY=true
export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=0
export NEG_BANK_SIZE=0
export GLOSSARY_NEG_PATH=""
export GLOSSARY_NEG_REFRESH_STEPS=0
export NEG_BANK_REFRESH_STEPS=50
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
export TCM_SWEEP_THRESHOLDS="0.75"
export TCM_SWEEP_FBETA=3.0

export GRAD_CACHE_CHUNK_SIZE=128
export NUM_WORKERS=0
export EPOCHS=1
export SCHEDULER_EPOCHS=1
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESET_BEST_ON_RESUME=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false
export SAVE_STEPS=999999
export SAVE_LATEST_ON_EVAL=false
export KEEP_CHECKPOINTS=0
export EVAL_STEPS_SAMPLE=0
export EVAL_TOP100_SAMPLES=0
export EVAL_SAMPLE_LIMIT=0
export EVAL_SAMPLE_SEED=17
export EVAL_SCORE_DEVICE="cuda"
export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-128}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-512}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS=2

export ACL_DEV_JSONL=""
export TAGGED_ACL_DEV_JSONL=""
export MEDICINE_DEV_JSONL=""
export ACL_EVAL_WIKI_GLOSSARY=""
export ACL_EVAL_GLOSSARY_SIZES=""
export ACL_EVAL_METRICS_GLOSSARY=""
export TAGGED_ACL_EVAL_WIKI_GLOSSARY=""
export TAGGED_ACL_EVAL_GLOSSARY_SIZES=""
export TAGGED_ACL_EVAL_METRICS_GLOSSARY=""
export MEDICINE_EVAL_WIKI_GLOSSARY=""
export MEDICINE_EVAL_GLOSSARY_SIZES=""
export MEDICINE_EVAL_METRICS_GLOSSARY=""
export FULL_EVAL_WIKI_GLOSSARY=""
export FULL_EVAL_GLOSSARY_SIZES=""
export FULL_EVAL_EVERY_N_EVALS=0
export AUTO_FULL_EVAL_ON_BEST=false

export RUN_VERDICT="Eval-only fixed 5.76s HN1024 latest checkpoint on ctx5p76 dev, fixed raw denominator, retriever banks base/1k/10k/100k from the 100k glossary file. Dev only; ACL/tagged/medicine intentionally disabled."

for required_path in "${RESUME}" "${NOTES_FILE}" "${TRAIN_JSONL}" "${DEV_JSONL}" "${EVAL_WIKI_GLOSSARY}"; do
    if [ ! -f "${required_path}" ]; then
        echo "[FATAL] required input not found: ${required_path}" >&2
        exit 1
    fi
done

cd "${REPO_ROOT}"
source "${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
