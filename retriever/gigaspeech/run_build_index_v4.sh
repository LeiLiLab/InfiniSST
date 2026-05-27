#!/bin/bash
#SBATCH --job-name=build_index_v4
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_build_index_v4.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_build_index_v4.err

set -euo pipefail

# 环境注入
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/mnt/taurus/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

echo "[INFO] Building New FAISS Index with Tuned Text Encoder (V4)..."

# Allow overriding via environment variables for language-specific pipelines.
# Example:
#   MODEL_PATH=... GLOSSARY_PATH=... OUTPUT_PATH=... TARGET_LANG_CODE=ja bash retriever/gigaspeech/run_build_index_v4.sh
MODEL_PATH="${MODEL_PATH:-/mnt/gemini/data2/jiaxuanluo/q3rag_unfrozen_lora-r32-tr16_bs4k_w1.0-0.0-ttm=query key value-temperature=0.03_sampled_best.pt}"
#MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/q3_rag_0.01_best_v1.pt"
#MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/q3rag_unfrozen_lora-r32-tr32_bs4k_w1.0-0.0_sampled_best_snapshot_v1.pt"
GLOSSARY_PATH="${GLOSSARY_PATH:-/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json}"
#OUTPUT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_index_v4_lora-r32-tr32.pkl"
#OUTPUT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_index_v4_lora-tr16.pkl"
#OUTPUT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_curated_index_v4.pkl"
#OUTPUT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_curated_index_v4_v1.pkl"
#OUTPUT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_index_v4_lora_from_paper-r32-tr16.pkl"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/gemini/data2/jiaxuanluo/q3rag_index_v4_extracted_glossary.pkl}"
TARGET_LANG_CODE="${TARGET_LANG_CODE:-zh}"
#GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
python /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/build_index_v4.py \
    --glossary_path "${GLOSSARY_PATH}" \
    --model_path "${MODEL_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --text_lora_r 16 \
    --device cuda:0 \
    --batch_size 1024 \
    --target_lang_code "${TARGET_LANG_CODE}"

echo "[INFO] Index building finished. Output saved to: ${OUTPUT_PATH}"
