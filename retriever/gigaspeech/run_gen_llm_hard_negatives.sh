#!/bin/bash
#SBATCH --job-name=gen_llm_hn
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_gen_llm_hn.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_gen_llm_hn.err

set -euo pipefail

# ==================== 环境配置 ====================
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# vLLM 优化参数
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ==================== 路径与参数 ====================
TRAIN_JSONL="/mnt/data2/jiaxuanluo/local_train_dataset.jsonl"
OUTPUT_PATH="/mnt/gemini/data2/jiaxuanluo/llm_hard_negatives_v1.json"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

BATCH_SIZE=10          # 每个 Prompt 包含的 Term 数量
VLLM_BATCH_SIZE=4096   # 恢复到更稳健的 Batch Size
NUM_NEGATIVES=10       # 每个 Term 生成的负例数量
TP=8                   # 使用 8 张卡进行 Tensor Parallel

# ==================== 执行生成 ====================
echo "[INFO] Starting Multi-GPU Parallel Hard Negative Generation..."
echo "[INFO] Total Shards: 8"

for i in {0..7}; do
    echo "[INFO] Launching Shard $i on GPU $i..."
    CUDA_VISIBLE_DEVICES=$i python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/gen_llm_hard_negatives.py \
        --train_jsonl "${TRAIN_JSONL}" \
        --output_path "${OUTPUT_PATH}" \
        --model "${MODEL}" \
        --batch_size "${BATCH_SIZE}" \
        --vllm_batch_size "${VLLM_BATCH_SIZE}" \
        --num_negatives "${NUM_NEGATIVES}" \
        --tp 1 \
        --gpu_util 0.8 \
        --shard_id $i \
        --total_shards 8 &
done

# 等待所有后台任务完成
wait

echo "[INFO] All shards completed! Output saved as ${OUTPUT_PATH%.json}_shard_*.json"

echo "[INFO] Generation completed!"

