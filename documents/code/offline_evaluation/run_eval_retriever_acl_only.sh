#!/bin/bash
#SBATCH --job-name=eval_retr_acl
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_eval_retriever_acl.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_eval_retriever_acl.err

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED=1

SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/eval_prepare_retriever.py"
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/tcr_fcr_eval"

ACL_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
ACL_GLOSSARY="${OUTPUT_BASE}/acl_expanded_combined_glossary.json"
ACL_GT_GLOSSARY="${OUTPUT_BASE}/acl_combined_glossary.json"
# ======Configuration=====

echo "[EVAL] Retriever inference for ACL dataset"
echo "  Glossary: ${ACL_GLOSSARY}"
echo "  GT glossary: ${ACL_GT_GLOSSARY}"

python3 "${SCRIPT}" \
    --dataset_type acl \
    --dataset_path "${ACL_JSONL}" \
    --glossary_json "${ACL_GLOSSARY}" \
    --output_path "${OUTPUT_BASE}/acl_retriever_results.jsonl" \
    --device "cuda:0" \
    --acl_gt_glossary "${ACL_GT_GLOSSARY}"

echo "[EVAL] ACL retriever inference done at $(date)"
