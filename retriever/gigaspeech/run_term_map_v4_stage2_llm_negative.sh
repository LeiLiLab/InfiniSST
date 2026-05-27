#!/bin/bash
#SBATCH --job-name=term_map_v4_stage2_llm_neg
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v4_stage2_llm_neg.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v4_stage2_llm_neg.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv

export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# vLLM/NCCL knobs
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- 配置区 ---
TOTAL_GPUS_LIMIT=${TOTAL_GPUS:-8} 
SKIP_GPUS="0,1,2,4,5,6,7"
# --------------

echo "[INFO] Individual GPU processing enabled (TP=1). Total limit: ${TOTAL_GPUS_LIMIT}, Skipping: ${SKIP_GPUS}"

AVAILABLE_GPUS=$(python3 - <<PY
import os
total_limit = int("$TOTAL_GPUS_LIMIT")
skip = [int(x) for x in "$SKIP_GPUS".split(",") if x.strip()]

usable = []
for i in range(total_limit):
    if i not in skip:
        usable.append(str(i))

print(" ".join(usable))
PY
)

if [ -z "${AVAILABLE_GPUS}" ]; then
    echo "[FATAL] No usable GPUs found."
    exit 1
fi

IFS=' ' read -r -a GPU_ARRAY <<< "${AVAILABLE_GPUS}"
TOTAL_SHARDS=${#GPU_ARRAY[@]}

echo "[INFO] Found ${TOTAL_SHARDS} usable GPUs: ${AVAILABLE_GPUS}"

# 修改输入文件为 v2 版本以获取最高术语覆盖率
INPUT_GT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl"
OUTPUT_BASE="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_freq_k20_final"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

# 1. 启动子任务 (每个 GPU 一个独立进程)
pids=()
for i in "${!GPU_ARRAY[@]}"; do
    CUR_GPU="${GPU_ARRAY[$i]}"
    echo "[INFO] Launching Shard $i / ${TOTAL_SHARDS} on physical GPU ${CUR_GPU}"
    
    CUDA_VISIBLE_DEVICES="${CUR_GPU}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_stage2_llm_negative.py \
      --input-gt-jsonl "${INPUT_GT}" \
      --output-base "${OUTPUT_BASE}" \
      --model "${MODEL}" \
      --gpu-id "$i" \
      --total-gpus "${TOTAL_SHARDS}" \
      --tensor-parallel-size 1 \
      --gpu-memory-util 0.90 \
      --batch-size 32 \
      --num-distractors 9 \
      --multiple-range 0 9 &
    pids+=($!)
done

echo "[INFO] All $TOTAL_SHARDS shards launched. Waiting..."
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[ERROR] Shard with PID $pid failed!"
    fi
done

# 2. 自动合并逻辑
FINAL_OUTPUT="${OUTPUT_BASE}.jsonl"
echo "[INFO] Merging shards into ${FINAL_OUTPUT}..."

# 清空或创建最终文件
: > "${FINAL_OUTPUT}"

for i in "${!GPU_ARRAY[@]}"; do
    SHARD_FILE="${OUTPUT_BASE}_gpu${i}.jsonl"
    if [ -f "${SHARD_FILE}" ]; then
        echo "[INFO] Appending shard $i..."
        cat "${SHARD_FILE}" >> "${FINAL_OUTPUT}"
        # 可选：合并后删除临时分片文件
        rm "${SHARD_FILE}"
    else
        echo "[WARN] Shard file ${SHARD_FILE} not found!"
    fi
done

echo "[INFO] All shards finished and merged."
