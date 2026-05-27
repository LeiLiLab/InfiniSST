#!/bin/bash
set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_EVAL_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/launchers/2026/05/20260522__tau_delta_dev_acl_tagacl_medstrict_compare_aries1_eval.sh"

export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-aries}"
export SLURM_ALLOCATION_JOB_ID="${SLURM_ALLOCATION_JOB_ID:-45310}"
export RUN_STAMP="${RUN_STAMP:-hn512_latest_s640_devpr_fixedraw_100k_20260526T0052Z}"
export MODEL_TAG="${MODEL_TAG:-hn512_latest_s640}"
export VARIANT_TAG="${VARIANT_TAG:-hn512_latest_s640_devpr_fixedraw_100k}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_hn512_latest_s640_devpr_fixedraw_100k_${RUN_STAMP}}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn512_latest_s640_devpr_fixedraw_100k_${RUN_STAMP}}"
export DATA_TAG="${DATA_TAG:-vctx576_devpr_fixedraw_100k}"
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_hardneg_eval_compare}"
export TASK_TAG="${TASK_TAG:-eval}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn512_latest_s640_devpr_fixedraw_100k compute:aries45310-gpu2 source:bkcnqlg9 checkpoint:latest_s640 protocol:dev-fixedraw readout:dev-only glossary:dev100k}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-bkcnqlg9 8h9q0v4t 31xmxmdp ifs45d6j e8t8zdtd evcgcdlu}"

export RESUME="${RESUME:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu234567_taurus_latest.pt}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260526__hn512_latest_s640_devpr_fixedraw_100k_eval.md}"

export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-2}"
export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MASTER_PORT="${MASTER_PORT:-30452}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export WAIT_FOR_CLEAN_GPUS="${WAIT_FOR_CLEAN_GPUS:-true}"
export GPU_CLEAN_THRESHOLD_MIB="${GPU_CLEAN_THRESHOLD_MIB:-500}"
export GPU_WAIT_INTERVAL_SEC="${GPU_WAIT_INTERVAL_SEC:-30}"
export GPU_WAIT_TIMEOUT_SEC="${GPU_WAIT_TIMEOUT_SEC:-7200}"
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/tmp/hn512_eval_${USER}_${RUN_STAMP}}"

export WANDB_DIR="${WANDB_DIR:-/mnt/gemini/home/jiaxuanluo/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/aries/data4/jiaxuanluo/cache/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/aries/data4/jiaxuanluo/cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/mnt/aries/data4/jiaxuanluo/config}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${LOCAL_TMP_DIR}"

export TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl}"
export DEV_JSONL="${DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl}"
export EVAL_WIKI_GLOSSARY="${EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-10000 100000}"
export EVAL_METRIC_DENOMINATOR="${EVAL_METRIC_DENOMINATOR:-fixed_raw}"
export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs100000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-}"

export ACL_DEV_JSONL=""
export TAGGED_ACL_DEV_JSONL=""
export MEDICINE_DEV_JSONL=""
export ACL_EVAL_WIKI_GLOSSARY=""
export ACL_EVAL_GLOSSARY_SIZES=""
export TAGGED_ACL_EVAL_WIKI_GLOSSARY=""
export TAGGED_ACL_EVAL_GLOSSARY_SIZES=""
export MEDICINE_EVAL_WIKI_GLOSSARY=""
export MEDICINE_EVAL_GLOSSARY_SIZES=""

export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-64}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-2048}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.50 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.60 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90}"
export TCM_SWEEP_FBETA="${TCM_SWEEP_FBETA:-3.0}"
export RUN_VERDICT="${RUN_VERDICT:-Eval-only HN512 latest step640 checkpoint on dev fixed_raw; raw/gs10k/gs100k banks only for Figure 5 eligibility check. ACL/tagged/medicine are intentionally disabled.}"

for required_path in "${RESUME}" "${NOTES_FILE}" "${TRAIN_JSONL}" "${DEV_JSONL}" "${EVAL_WIKI_GLOSSARY}" "${BASE_EVAL_LAUNCHER}"; do
    if [ ! -f "${required_path}" ]; then
        echo "[FATAL] required input not found: ${required_path}" >&2
        exit 1
    fi
done

cd "${REPO_ROOT}"
exec bash "${BASE_EVAL_LAUNCHER}"
