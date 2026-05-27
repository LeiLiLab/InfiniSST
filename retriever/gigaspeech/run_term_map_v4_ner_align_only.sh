#!/bin/bash
#SBATCH --job-name=term_map_v4_ner_only
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v4_ner_only.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v4_ner_only.err

set -euo pipefail

# 彻底放弃 source 脚本，改用手动环境变量注入
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONHOME="$CONDA_PREFIX"
export XDG_CACHE_HOME="/mnt/gemini/data1/jiaxuanluo/xdg_cache"
mkdir -p "$XDG_CACHE_HOME"
# 验证 Python 位置
which python
python --version

export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# vLLM/NCCL knobs
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 重定向缓存目录到数据盘，防止 /home 空间不足
export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
export VLLM_CACHE="/mnt/gemini/data1/jiaxuanluo/vllm_cache"
mkdir -p "$HF_HOME" "$VLLM_CACHE"

# --- 配置区 ---
TOTAL_GPUS_LIMIT=${TOTAL_GPUS:-8} 
SKIP_GPUS="0,2"
# --------------

echo "[INFO] Manual GPU grouping enabled (Pure NER Mode). Total limit: ${TOTAL_GPUS_LIMIT}, Skipping physical: ${SKIP_GPUS}"

AVAILABLE_PAIRS=$(python3 - <<PY
import os
total_limit = int("$TOTAL_GPUS_LIMIT")
skip = [int(x) for x in "$SKIP_GPUS".split(",") if x.strip()]
usable = [str(i) for i in range(total_limit) if i not in skip]
pairs = [usable[i] + "," + usable[i+1] for i in range(0, len(usable) - 1, 2)]
print(" ".join(pairs))
PY
)

if [ -z "${AVAILABLE_PAIRS}" ]; then
    echo "[FATAL] No usable GPU pairs found."
    exit 1
fi

IFS=' ' read -r -a PAIR_ARRAY <<< "${AVAILABLE_PAIRS}"
TOTAL_SHARDS=${#PAIR_ARRAY[@]}

echo "[INFO] Found ${TOTAL_SHARDS} usable GPU pairs: ${AVAILABLE_PAIRS}"

INPUT_GT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
OUTPUT_GT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_only_aligned.jsonl"
ALIGN_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

# 1. 启动分片任务
for i in "${!PAIR_ARRAY[@]}"; do
    CUR_GPUS="${PAIR_ARRAY[$i]}"
    echo "[INFO] Launching Shard $i / ${TOTAL_SHARDS} on physical GPUs ${CUR_GPUS}"
    
    CUDA_VISIBLE_DEVICES="${CUR_GPUS}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_ner_align_only.py \
      --input-gt "${INPUT_GT}" \
      --input-tsv "${INPUT_TSV}" \
      --output-gt "${OUTPUT_GT}" \
      --align-model "${ALIGN_MODEL}" \
      --gpu-memory-util 0.90 \
      --batch-size 32 \
      --tensor-parallel-size 2 \
      --max-retries 1 \
      --gpu-id "$i" \
      --total-gpus "${TOTAL_SHARDS}" &
done

echo "[INFO] All $TOTAL_SHARDS shards launched. Waiting..."
wait

# 2. 自动合并逻辑
FINAL_OUTPUT="${OUTPUT_GT}"
echo "[INFO] Merging shards into ${FINAL_OUTPUT}..."

BASE_OUT_PATH="${OUTPUT_GT%.jsonl}"
: > "${FINAL_OUTPUT}"

for i in "${!PAIR_ARRAY[@]}"; do
    SHARD_FILE="${BASE_OUT_PATH}_gpu${i}.jsonl"
    if [ -f "${SHARD_FILE}" ]; then
        echo "[INFO] Appending shard $i..."
        cat "${SHARD_FILE}" >> "${FINAL_OUTPUT}"
        rm "${SHARD_FILE}"
    else
        echo "[WARN] Shard file ${SHARD_FILE} not found!"
    fi
done

echo "[INFO] All shards finished and merged into ${FINAL_OUTPUT}."

