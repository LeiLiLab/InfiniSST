#!/bin/bash
#SBATCH --job-name=precompute_shard
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:6
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_precompute_shard.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_precompute_shard.err

set -euo pipefail

# 环境配置
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

INPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_dataset_final.jsonl"
OUTPUT_DIR="/mnt/gemini/data1/jiaxuanluo/precomputed_text_embs_v3"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/precompute_text_embeddings.py"
TOTAL_SHARDS=8

# 并行启动 8 个分片，每个分片锁定一张物理卡
for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
    echo "[INFO] Launching shard $i on GPU $i..."
    CUDA_VISIBLE_DEVICES=$i python "${SCRIPT_PATH}" \
        --input_jsonl "${INPUT_JSONL}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size 2048 \
        --shard_id $i \
        --total_shards ${TOTAL_SHARDS} &
    sleep 2
done

wait

echo "[INFO] All shards finished. Starting merge..."
# 合并模式不需要 GPU，直接 CPU 跑
python "${SCRIPT_PATH}" \
    --input_jsonl "${INPUT_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --total_shards ${TOTAL_SHARDS} \
    --merge

echo "[INFO] All done!"
