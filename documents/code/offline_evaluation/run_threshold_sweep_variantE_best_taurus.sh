#!/bin/bash
#SBATCH --job-name=thr_sweep_variantE_best
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_thr_sweep_variantE_best.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_thr_sweep_variantE_best.err

# Threshold sweep (absolute tau in [0.50, 0.85] step 0.05) for the variantE
# hardneg+TCM best checkpoint on DEV + ACL6060, across glossary sizes
# {raw(GT-only), 1k, 10k}.  Produces:
#   - per-(dataset, gs) TSV with P/R/F1/noise per tau
#   - per-(dataset, gs) score histograms (positive vs negative, top-10)
#   - per-(dataset, gs) P/R/F1 and noise curves vs tau
# for eyeballing before committing to a simuleval threshold.

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

# Pin to 1 GPU; SLURM --gres=gpu:1 already guarantees exclusivity.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- variantE best retriever checkpoint (pre-LR-shock, best ACL gs10k=0.9085). ---
MODEL_PATH="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_tcm_ep5_best_acl6060_gs10000.pt"

# --- Eval data. ---
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"

# --- Output: one dedicated dir per ckpt so sweeps are not mixed. ---
OUTPUT_DIR="/mnt/gemini/data2/jiaxuanluo/threshold_sweep/variantE_best_tau0p5_0p85"

# --- Sweep grid: 0.50 -> 0.85 step 0.05 (8 points).  gs 0 == raw (GT only). ---
ABS_TAU_VALUES="0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85"
GLOSSARY_SIZES="0 1000 10000"

# --- Retriever hyperparams (identical to variantE training). ---
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

echo "[EVAL] variantE best checkpoint threshold sweep"
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
