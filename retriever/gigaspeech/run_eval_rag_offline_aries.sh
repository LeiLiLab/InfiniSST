#!/bin/bash
#SBATCH --job-name=eval_rag_qwen3
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_eval_rag_qwen3.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_eval_rag_qwen3.err

set -euo pipefail

# 环境注入
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/mnt/taurus/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

echo "[INFO] Running Offline RAG Evaluation on Single GPU..."

# 路径配置
#MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/88.38recall5_v1.0.pt"
MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/q3rag_lora-r32-all_bs8k_w1.0-1.0_sampled_best.pt"
INDEX_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_index.pkl"
GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060.json"
#GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/acl_terminology_glossary_lowercase.json"
#INDEX_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_curated_terms_index_bge_m3.pkl"

WAV_DIR="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/segmented_wavs/gold"
TXT_PATH="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt"

# RAG 配置
CHUNK_SIZE=1.92
HOP_SIZE=0.96
STRATEGY="max_pool" # "voting" 或 "max_pool"
TOP_K=5
VOTING_K=10
TPS=2.5

# 1. 直接运行评估脚本
# 单卡模式下，Python 脚本会自动打印最终结果表格
python /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/eval_rag_offline_qwen3_acl6060.py \
  --model_path "${MODEL_PATH}" \
  --index_path "${INDEX_PATH}" \
  --glossary_path "${GLOSSARY_PATH}" \
  --wav_dir "${WAV_DIR}" \
  --txt_path "${TXT_PATH}" \
  --rag_chunk_size "${CHUNK_SIZE}" \
  --rag_hop_size "${HOP_SIZE}" \
  --rag_strategy "${STRATEGY}" \
  --top_k "${TOP_K}" \
  --rag_voting_k "${VOTING_K}" \
  --terms_per_second "${TPS}" \
  --device cuda:0

echo "[INFO] Evaluation finished."
