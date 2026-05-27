#!/bin/bash
#SBATCH --job-name=thr_sweep_43827_best
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_thr_sweep_43827_best.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_thr_sweep_43827_best.err

# Threshold sweep on 43827 per-sample k=1024 best checkpoint (snapshot at
# step=1320, ACL6060/recall@10_gs10000=0.8775).  Same sweep grid as the
# pre-LR-shock baseline (variantE_best) so results are directly comparable:
#
# Baseline (variantE_hardneg_tcm, ACL gs10k=0.9085 at training-time):
#   output dir: /mnt/gemini/data2/jiaxuanluo/threshold_sweep/variantE_best_tau0p5_0p85
#
# This run (43827 per-sample k=1024 cold, ACL gs10k=0.8775 at training-time):
#   output dir: /mnt/gemini/data2/jiaxuanluo/threshold_sweep/43827_best_step1320_tau0p5_0p85
#
# Architecture is identical to variantE_hardneg_tcm (same launcher recipe
# except per-sample hard negatives replace pool mining); no architecture
# args change vs the existing variantE launcher.
#
# Submit: sbatch run_threshold_sweep_43827_best_taurus.sh

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Stable snapshot (MD5-verified copy of the live best_acl6060_gs10000.pt
# captured while 43827 was at step 1320 epoch 2).
MODEL_PATH="/mnt/gemini/home/jiaxuanluo/train_outputs/snapshots/43827_best_acl6060_gs10000_step1320_0p8775.pt"

DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"

OUTPUT_DIR="/mnt/gemini/data2/jiaxuanluo/threshold_sweep/43827_best_step1320_tau0p5_0p85"

# Same sweep grid as variantE baseline for apples-to-apples comparison.
ABS_TAU_VALUES="0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85"
GLOSSARY_SIZES="0 1000 10000"

# Retriever hyperparams (identical to variantE / 43827 training).
LORA_RANK=128
LORA_ALPHA=256
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
POOLING_TYPE="transformer"
TEXT_POOLING="cls"
MAXSIM_WINDOWS="6 10 16 24"
MAXSIM_STRIDE=2
SPARSE_WEIGHT="0.0"
TEMPERATURE="0.07"
TARGET_DIM=1024

SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/threshold_sweep_maxsim.py"
# ======Configuration=====

mkdir -p "${OUTPUT_DIR}"

echo "[EVAL] 43827 per-sample k=1024 best checkpoint threshold sweep"
echo "[EVAL] MODEL_PATH  = ${MODEL_PATH}"
echo "[EVAL] OUTPUT_DIR  = ${OUTPUT_DIR}"
echo "[EVAL] TAUS        = ${ABS_TAU_VALUES}"
echo "[EVAL] GLOSSARIES  = ${GLOSSARY_SIZES} (0=raw GT-only)"
echo "[EVAL] DEV_JSONL   = ${DEV_JSONL}"
echo "[EVAL] ACL_JSONL   = ${ACL_JSONL}"
echo "[EVAL] started at  $(date)"

python3 "${SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --dev_jsonl "${DEV_JSONL}" \
    --acl_jsonl "${ACL_JSONL}" \
    --wiki_glossary "${WIKI_GLOSSARY}" \
    --glossary_sizes ${GLOSSARY_SIZES} \
    --abs_tau_values ${ABS_TAU_VALUES} \
    --plot_histograms \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda:0" \
    --target_dim "${TARGET_DIM}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --pooling_type "${POOLING_TYPE}" \
    --temperature "${TEMPERATURE}" \
    --use_maxsim \
    --maxsim_windows "${MAXSIM_WINDOWS}" \
    --maxsim_stride "${MAXSIM_STRIDE}" \
    --text_lora_rank "${TEXT_LORA_RANK}" \
    --text_lora_alpha "${TEXT_LORA_ALPHA}" \
    --text_pooling "${TEXT_POOLING}" \
    --sparse_weight "${SPARSE_WEIGHT}"

echo "[EVAL] Done at $(date)"
