#!/bin/bash
#SBATCH --job-name=eval_retr_all
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_eval_retriever_all.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_eval_retriever_all.err

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0}}"
export PYTHONUNBUFFERED=1

SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/eval_prepare_retriever.py"
BUILD_SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/build_eval_glossaries.py"
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/tcr_fcr_eval"

DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20_final.jsonl"
ACL_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
ACL_GT_GLOSSARY="${OUTPUT_BASE}/acl_gt_terms_from_dev95.json"

GLOSSARY_SIZES=(100 1000 10000)
# ======Configuration=====

mkdir -p "${OUTPUT_BASE}"

echo "[EVAL] Starting retriever inference at $(date)"
echo "[EVAL] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "[EVAL] Rebuilding ACL eval glossaries from ACL dev GT terms..."
python3 "${BUILD_SCRIPT}"

# --- Dev dataset with 3 glossary sizes ---
for GS in "${GLOSSARY_SIZES[@]}"; do
    DEV_GLOSSARY="${OUTPUT_BASE}/dev_glossary_gs${GS}.json"
    DEV_OUTPUT="${OUTPUT_BASE}/dev_gs${GS}_retriever_results.jsonl"

    echo ""
    echo "======== Dev Retriever gs=${GS} ========"
    echo "  Glossary: ${DEV_GLOSSARY}"
    python3 "${SCRIPT}" \
        --dataset_type dev \
        --dataset_path "${DEV_JSONL}" \
        --glossary_json "${DEV_GLOSSARY}" \
        --output_path "${DEV_OUTPUT}" \
        --device "cuda:0"
    echo "[EVAL] Dev gs=${GS} retriever done."
done

# --- ACL dataset with 3 glossary sizes ---
for GS in "${GLOSSARY_SIZES[@]}"; do
    GLOSSARY="${OUTPUT_BASE}/acl_glossary_gs${GS}.json"
    OUTPUT="${OUTPUT_BASE}/acl_gs${GS}_retriever_results.jsonl"

    echo ""
    echo "======== ACL Retriever gs=${GS} ========"
    echo "  Glossary: ${GLOSSARY}"
    python3 "${SCRIPT}" \
        --dataset_type acl \
        --dataset_path "${ACL_JSONL}" \
        --glossary_json "${GLOSSARY}" \
        --output_path "${OUTPUT}" \
        --device "cuda:0" \
        --acl_gt_glossary "${ACL_GT_GLOSSARY}"

    echo "[EVAL] ACL gs=${GS} retriever done."
done

echo ""
echo "[EVAL] All retriever inference done at $(date)"
