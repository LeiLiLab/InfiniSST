#!/bin/bash
#SBATCH --job-name=eval_rag_sweep_v4
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --array=0-3
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_eval_rag_sweep_v4.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_eval_rag_sweep_v4.err

# 定义要扫描的阈值列表 (0.0 到 1.0, 步长 0.1)
THRESHOLDS=(0.1 0.2 0.3 0.4)
SCORE_THRESHOLD=${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}
set -euo pipefail

# 环境注入
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/mnt/taurus/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

echo "[INFO] Running Offline RAG Evaluation V4 (Tuned Text Encoder)..."

# 路径配置 - 使用新的最佳模型 (Unfrozen Text Encoder)
MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/q3rag_unfrozen_lora-r32-tr16_bs4k_w1.0-0.0_sampled_best_snapshot_v2.pt"

#INDEX_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_index_v4.pkl"
INDEX_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060_curated_index_v4.pkl"
#GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060.json"
#GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/acl_terminology_glossary_lowercase.json"

#MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/q3_rag_0.01_best_v1.pt"
GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
INDEX_PATH="/mnt/gemini/data2/jiaxuanluo/q3_rag_unfrozen_lora-r32-tr16_bs4k_w1.0-0.0_sampled_best_snapshot_v2_paper_extracted_glossary.pkl"


WAV_DIR="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/segmented_wavs/gold"
TXT_PATH="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt"

# RAG 配置
CHUNK_SIZE=1.92
HOP_SIZE=0.96
VLLM_INTERVAL=1.92
STRATEGY="max_pool" 
TOP_K=5
VOTING_K=20
VOTING_MIN_VOTES=2
MAX_SAMPLES=0 # 0 表示评估所有样本

# 运行评估脚本 V4
python /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/eval_rag_offline_qwen3_acl6060_v4.py \
  --model_path "${MODEL_PATH}" \
  --index_path "${INDEX_PATH}" \
  --glossary_path "${GLOSSARY_PATH}" \
  --wav_dir "${WAV_DIR}" \
  --txt_path "${TXT_PATH}" \
  --rag_chunk_size "${CHUNK_SIZE}" \
  --rag_hop_size "${HOP_SIZE}" \
  --vllm_interval "${VLLM_INTERVAL}" \
  --score_threshold "${SCORE_THRESHOLD}" \
  --rag_strategy "${STRATEGY}" \
  --top_k "${TOP_K}" \
  --rag_voting_k "${VOTING_K}" \
  --rag_voting_min_votes "${VOTING_MIN_VOTES}" \
  --max_samples "${MAX_SAMPLES}" \
  --rag_lora_r 32 \
  --rag_text_lora_r 16 \
  --device cuda:0

echo "[INFO] Evaluation finished. Threshold: ${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}"

