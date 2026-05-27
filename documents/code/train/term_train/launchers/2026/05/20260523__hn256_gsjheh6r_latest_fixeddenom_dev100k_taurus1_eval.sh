#!/bin/bash
set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"

export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-taurus}"
export RUN_STAMP="${RUN_STAMP:-hn256_gsjheh6r_latest_fixeddenom_dev100k_taurus_gpu6_20260523T1622Z}"
export MODEL_TAG="hn256_gsjheh6r_latest_fixeddenom"
export VARIANT_TAG="hn256_gsjheh6r_latest_fixeddenom_dev100k_heldout"
export DATA_TAG="vctx576_hn256_latest_fixeddenom"
export EXPERIMENT_FAMILY="sst_ood_hardneg_eval_compare"
export EXTRA_WANDB_TAGS="variant:hn256_gsjheh6r_latest_fixeddenom compute:taurus-1gpu comparison:nohn_hn256_hn1024 calibration:dev_delta readout:dev100k_acl_tagged_medstrict protocol:fixed-raw-denominator hn:256 source:gsjheh6r"
export BASELINE_RUN_IDS="gsjheh6r lrdx14pm e981df6j lh1b88kw 40fgbr2y zji769ve ry8osg4u e8t8zdtd 31xmxmdp 9esujv2w ykwbip03"

export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-6}"
export NUM_GPUS=1
export PER_GPU_BATCH=1
export BATCH_SIZE=1
export MASTER_PORT="${MASTER_PORT:-30468}"
export SELECT_CLEAN_GPUS=false
export WAIT_FOR_CLEAN_GPUS=false
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/tmp/jiaxuanluo_q3_hn256_gsjheh6r_latest_eval_dev100k_${RUN_STAMP}}"

export WANDB_DIR="/mnt/gemini/data1/jiaxuanluo/wandb"
export WANDB_CACHE_DIR="/mnt/gemini/data1/jiaxuanluo/cache/wandb"
export XDG_CACHE_HOME="/mnt/gemini/data1/jiaxuanluo/cache"
export XDG_CONFIG_HOME="/mnt/gemini/data1/jiaxuanluo/config"

export RESUME="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012367_taurus_step1200_aclmetric_reset_latest.pt"
export NOTES_FILE="${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260523__hn256_gsjheh6r_latest_fixeddenom_dev100k_heldout_taurus1_eval.md"

export EVAL_GLOSSARY_SIZES="10000 100000"
export BEST_METRIC="eval_dev/recall@10_gs100000"
export BEST_METRIC_SECONDARY=""
export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-256}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-4096}"
export TCM_SWEEP_THRESHOLDS="0.50 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.60 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90"
export RUN_VERDICT="Eval-only HN256 latest fixed-denominator comparison. Dev tau selection uses raw/base plus gs10k and gs100k only; ACL6060, tagged ACL6060, and strict medicine are readout-only."

cd "${REPO_ROOT}"
exec bash documents/code/train/term_train/launchers/2026/05/20260522__tau_delta_dev_acl_tagacl_medstrict_compare_aries1_eval.sh
