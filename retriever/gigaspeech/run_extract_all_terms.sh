#!/bin/bash
#SBATCH --job-name=extract_all_terms
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:6
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_extract_all_terms.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_extract_all_terms.err

set -euo pipefail

# 彻底放弃 source 脚本，改用手动环境变量注入
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="/mnt/gemini/data1/jiaxuanluo/xdg_cache"
mkdir -p "$XDG_CACHE_HOME"

# 验证 Python 位置
which python
python --version

export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/mnt/taurus/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# vLLM/NCCL knobs
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
export VLLM_CACHE="/mnt/gemini/data1/jiaxuanluo/vllm_cache"
mkdir -p "$HF_HOME" "$VLLM_CACHE"

# --- 配置区 ---
# 机器的总物理 GPU 数量
TOTAL_GPUS_LIMIT=6
# 需要跳过的物理 GPU ID，用逗号分隔
SKIP_GPUS="0,2" 
# --------------

echo "[INFO] Manual GPU selection enabled. Total limit: ${TOTAL_GPUS_LIMIT}, Skipping physical: ${SKIP_GPUS}"

# 使用 Python 脚本计算实际可用的物理 GPU ID 列表
AVAILABLE_GPUS=$(python3 - <<PY
import os
total_limit = int("$TOTAL_GPUS_LIMIT")
skip_str = "$SKIP_GPUS"
skip = [int(x) for x in skip_str.split(",") if x.strip()]

usable = []
for i in range(total_limit):
    if i not in skip:
        usable.append(str(i))

print(" ".join(usable))
PY
)

if [ -z "${AVAILABLE_GPUS}" ]; then
    echo "[FATAL] No usable GPUs found after skipping ${SKIP_GPUS}."
    exit 1
fi

IFS=' ' read -r -a GPU_ARRAY <<< "${AVAILABLE_GPUS}"
TOTAL_SHARDS=${#GPU_ARRAY[@]}

echo "[INFO] Found ${TOTAL_SHARDS} usable physical GPUs: ${AVAILABLE_GPUS}"
echo "[INFO] Launching parallel shards..."

INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
OUTPUT_DIR="/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks"
OUTPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_dataset.jsonl"
ALIGN_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

# 1. 启动分片任务 (TP=1)
for i in "${!GPU_ARRAY[@]}"; do
    CUR_PHYSICAL_GPU="${GPU_ARRAY[$i]}"
    echo "[INFO] Launching Shard $i / ${TOTAL_SHARDS} on physical GPU ${CUR_PHYSICAL_GPU}"
    
    # 强制覆盖 CUDA_VISIBLE_DEVICES 为指定的物理 ID
    CUDA_VISIBLE_DEVICES="${CUR_PHYSICAL_GPU}" python retriever/gigaspeech/extract_all_terms_from_tsv.py \
      --input-tsv "${INPUT_TSV}" \
      --output-dir "${OUTPUT_DIR}" \
      --output-jsonl "${OUTPUT_JSONL}" \
      --align-model "${ALIGN_MODEL}" \
      --gpu-memory-util 0.8 \
      --batch-size 32 \
      --tp-size 1 \
      --shard-id "$i" \
      --total-shards "${TOTAL_SHARDS}" &
done

echo "[INFO] All $TOTAL_SHARDS shards launched. Waiting..."
wait

# 2. 自动合并逻辑
FINAL_OUTPUT="${OUTPUT_JSONL}"
BASE_OUT_PATH="${OUTPUT_JSONL%.jsonl}"

if [ "${TOTAL_SHARDS}" -gt 1 ]; then
    echo "[INFO] Merging shards into ${FINAL_OUTPUT}..."
    # 清空或创建最终文件
    : > "${FINAL_OUTPUT}"

    for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
        SHARD_FILE="${BASE_OUT_PATH}_shard${i}.jsonl"
        if [ -f "${SHARD_FILE}" ]; then
            echo "[INFO] Appending shard $i..."
            cat "${SHARD_FILE}" >> "${FINAL_OUTPUT}"
            rm "${SHARD_FILE}"
        else
            echo "[WARN] Shard file ${SHARD_FILE} not found!"
        fi
    done
    echo "[INFO] All shards finished and merged into ${FINAL_OUTPUT}."
else
    # 只有一个分片的情况
    SHARD_FILE="${BASE_OUT_PATH}_shard0.jsonl"
    if [ -f "${SHARD_FILE}" ]; then
        echo "[INFO] Renaming single shard file to ${FINAL_OUTPUT}..."
        mv "${SHARD_FILE}" "${FINAL_OUTPUT}"
    else
        echo "[INFO] Single shard already wrote to ${FINAL_OUTPUT}."
    fi
fi
