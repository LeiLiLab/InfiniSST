#!/bin/bash
#SBATCH --job-name=q3_hn512_a6_resume
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=260G
#SBATCH --gres=gpu:6
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/home/jiaxuanluo/logs/%j_q3_hn512_a6_resume_%x.out
#SBATCH --error=/mnt/gemini/home/jiaxuanluo/logs/%j_q3_hn512_a6_resume_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh"

export PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin:${PATH}"

export RUN_STAMP="${RUN_STAMP:-hn512_gc128_resume640_aries6_allocsafe_20260526T0110Z}"
unset CUDA_DEVICE_LIST
unset GPU_LIST
export NUM_GPUS="${NUM_GPUS:-6}"
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-8190}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1365}"
export BATCH_SIZE="${BATCH_SIZE:-8190}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-128}"
export HARD_NEG_K="${HARD_NEG_K:-0}"
export HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-512}"

export RESUME="${RESUME:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu234567_taurus_latest.pt}"
export RESET_SCHEDULER="${RESET_SCHEDULER:-false}"
export RESET_BEST_ON_RESUME="${RESET_BEST_ON_RESUME:-false}"

export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export MASTER_PORT="${MASTER_PORT:-30456}"
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/tmp/hn512_a6_${USER}_${SLURM_JOB_ID:-local}_${RUN_STAMP}}"

export WANDB_DIR="${WANDB_DIR:-/mnt/gemini/home/jiaxuanluo/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/aries/data4/jiaxuanluo/cache/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/aries/data4/jiaxuanluo/cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/mnt/aries/data4/jiaxuanluo/config}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" /mnt/gemini/home/jiaxuanluo/logs

export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_STEPS="${MAX_STEPS:-0}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SAVE_LATEST_ON_EVAL="${SAVE_LATEST_ON_EVAL:-true}"
export EVAL_STEPS_SAMPLE="${EVAL_STEPS_SAMPLE:-100}"
export SAVE_LATEST_STEPS="${SAVE_LATEST_STEPS:-50}"
export SAVE_STEPS="${SAVE_STEPS:-999999}"
export KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"
export EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-0}"

export TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl}"
export DEV_JSONL="${DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl}"
export ACL_DEV_JSONL="${ACL_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl}"
export MEDICINE_DEV_JSONL="${MEDICINE_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl}"
export EVAL_WIKI_GLOSSARY="${EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-10000}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json}"
export ACL_EVAL_GLOSSARY_SIZES="${ACL_EVAL_GLOSSARY_SIZES:-10000}"
export MEDICINE_EVAL_WIKI_GLOSSARY="${MEDICINE_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json}"
export MEDICINE_EVAL_GLOSSARY_SIZES="${MEDICINE_EVAL_GLOSSARY_SIZES:-10000}"
export NLP_AI_CS_EVAL_GLOSSARY="${NLP_AI_CS_EVAL_GLOSSARY:-${REPO_ROOT}/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs_enriched.json}"
export TRAIN_EXCLUDE_TERM_GLOSSARIES="${TRAIN_EXCLUDE_TERM_GLOSSARIES:-${NLP_AI_CS_EVAL_GLOSSARY} ${MEDICINE_EVAL_WIKI_GLOSSARY}}"
export STRICT_TRAIN_EVAL_TERM_FILTER="${STRICT_TRAIN_EVAL_TERM_FILTER:-false}"

export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs10000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-eval_acl6060/recall@10}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS:-2}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.75}"

export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_hardneg}"
export TASK_TAG="${TASK_TAG:-train}"
export DATA_TAG="${DATA_TAG:-3variant_gsv2full_gsdedup_varctx576}"
export VARIANT_TAG="${VARIANT_TAG:-hn512_varctx576_v3_gc128_aries6_allocsafe}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_bs8190_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_aries6_allocsafe}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260526__varctx_lmlb_v3_hn512_gc128_resume640_aries6_allocsafe.md}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn512_gsv2full_gsdedup_varctx576_v3_bs8190_gc128_resume640_aries6_allocsafe_20260526T0110Z}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn512_varctx576_v3 compute:aries-6gpu ablation:hard_neg512 source:bkcnqlg9 resume_of:bkcnqlg9 gradcache:128 eval_steps:100 latest_steps:50 gpu:slurm6}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw 5fwrs7rh gasqw118 gsjheh6r yp0rmgrl bkcnqlg9 e981df6j 40fgbr2y bgz7akb6 ah9u1bao dxwrgbln}"
export RUN_VERDICT="${RUN_VERDICT:-HN512 allocation-compatible 6-GPU Aries resume from bkcnqlg9 step640 latest after step640 fixedraw Figure 5 check was not paper-worthy; preserve scheduler and best trackers.}"

cd "${REPO_ROOT}"
exec bash "${BASE_LAUNCHER}"
