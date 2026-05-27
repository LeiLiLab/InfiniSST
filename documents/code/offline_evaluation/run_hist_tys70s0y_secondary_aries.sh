#!/bin/bash
#SBATCH --job-name=posneg_tys70s0y_sec
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_posneg_tys70s0y_sec.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_posneg_tys70s0y_sec.err

# Pos/Neg sim-distribution diagnostic for the _best_acl6060_gs10000.pt
# (best_secondary/step=520) snapshot of WandB run tys70s0y (A1 winner).
#
# Same settings as zv28ve3q but per-sample HN k=512 instead of pool HN k=64.
#
# tys70s0y = variantE smallest+dense MFA + per-sample HN k=512 + TCM(1.0, β=0.85, α=0.25).
# See documents/code/offline_evaluation/retriever_sim_dist_snapshot.md.

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

MODEL_PATH="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_ps_k512_tcm_ep3_cold_smallest_dense_normAGGR_6gpu_best_acl6060_gs10000.pt"

DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"

OUTPUT_DIR="/mnt/gemini/data2/jiaxuanluo/sim_dist_snapshot/tys70s0y_secondary"
GS_SIZE=10000
TOPK_NEG=50

# tys70s0y trained with alpha=0.25, beta=0.85 (same as zv28ve3q).
CURRENT_ALPHA=0.25
CURRENT_BETA=0.85

LORA_RANK=128
LORA_ALPHA=256
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
POOLING_TYPE="transformer"
TEXT_POOLING="cls"
MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"
MAXSIM_STRIDE=2
SPARSE_WEIGHT="0.0"
TEMPERATURE="0.07"
TARGET_DIM=1024

SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/retriever_posneg_dist.py"
# ======Configuration=====

mkdir -p "${OUTPUT_DIR}"

echo "[POSNEG] tys70s0y best_secondary pos/neg distribution"
echo "[POSNEG] MODEL_PATH     = ${MODEL_PATH}"
echo "[POSNEG] OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "[POSNEG] GS_SIZE        = ${GS_SIZE}"
echo "[POSNEG] TOPK_NEG       = ${TOPK_NEG}"
echo "[POSNEG] CURRENT_ALPHA  = ${CURRENT_ALPHA}  (training-config reference line)"
echo "[POSNEG] CURRENT_BETA   = ${CURRENT_BETA}"
echo "[POSNEG] started at     $(date)"

python3 "${SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --dev_jsonl "${DEV_JSONL}" \
    --acl_jsonl "${ACL_JSONL}" \
    --wiki_glossary "${WIKI_GLOSSARY}" \
    --gs_size "${GS_SIZE}" \
    --topk_neg "${TOPK_NEG}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda:0" \
    --current_alpha "${CURRENT_ALPHA}" \
    --current_beta "${CURRENT_BETA}" \
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

echo "[POSNEG] Done at $(date)"
