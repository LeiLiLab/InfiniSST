#!/bin/bash
#SBATCH --job-name=gen_termmap_varlen
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_gen_termmap_varlen.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_gen_termmap_varlen.err

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/hard_negative_jsonl_for_speech_llm/generate_termmap_maxsim.py"

CLEANED_JSONL="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_cleaned.jsonl"
GLOSSARY_JSON="/mnt/gemini/data1/jiaxuanluo/glossary_from_gt_cleaned.json"
MODEL_PATH="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"
OUTPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/train_cleaned_with_retriever_results_varlen.jsonl"

RETRIEVAL_DENSITY=10
MAX_CONVERSATIONS="${MAX_CONVERSATIONS:-0}"
# ======Configuration=====

mkdir -p "$(dirname "${OUTPUT_JSONL}")"
mkdir -p /mnt/gemini/data1/jiaxuanluo/logs

echo "[GEN] Starting generate_termmap_maxsim (variable-length)"
echo "[GEN] Cleaned JSONL: ${CLEANED_JSONL}"
echo "[GEN] Glossary: ${GLOSSARY_JSON}"
echo "[GEN] Model: ${MODEL_PATH}"
echo "[GEN] Output: ${OUTPUT_JSONL}"
echo "[GEN] Retrieval density: ${RETRIEVAL_DENSITY}"
echo "[GEN] Max conversations: ${MAX_CONVERSATIONS}"

python3 "${SCRIPT}" \
    --cleaned_jsonl "${CLEANED_JSONL}" \
    --glossary_json "${GLOSSARY_JSON}" \
    --model_path "${MODEL_PATH}" \
    --output_jsonl "${OUTPUT_JSONL}" \
    --device "cuda:0" \
    --retrieval_density "${RETRIEVAL_DENSITY}" \
    --max_conversations "${MAX_CONVERSATIONS}"

echo "[GEN] Done at $(date)"
