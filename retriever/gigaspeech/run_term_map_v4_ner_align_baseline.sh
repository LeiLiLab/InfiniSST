#!/bin/bash
#SBATCH --job-name=term_map_v4_ner_baseline
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v4_ner_baseline.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_term_map_v4_ner_baseline.err

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

# 重定向缓存目录到数据盘，防止 /home 空间不足
export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
export VLLM_CACHE="/mnt/gemini/data1/jiaxuanluo/vllm_cache"
mkdir -p "$HF_HOME" "$VLLM_CACHE"

# --- 配置区 ---
TOTAL_GPUS_LIMIT=${TOTAL_GPUS:-2} 
SKIP_GPUS="2"
# 预算控制 (K)：每条 utterance 最多保留的术语数
MAX_TERMS_PER_UTTER=20
# 频率采样开关 (默认开启，若需关闭则在下方命令中加入 --no-freq-sampling)
ENABLE_FREQ_SAMPLING=false
# 备份采样率 (仅在关闭频率采样时生效)
SAMPLING_RATE=0.5
# --------------

# 构造文件名标识
if [ "$ENABLE_FREQ_SAMPLING" = true ]; then
    SAMPLING_STR="freq_k${MAX_TERMS_PER_UTTER}"
else
    RATE_STR=$(python3 -c "print(int(${SAMPLING_RATE} * 100))")
    SAMPLING_STR="${RATE_STR}percent"
fi

echo "[INFO] Manual GPU grouping enabled. Total limit: ${TOTAL_GPUS_LIMIT}, Skipping physical: ${SKIP_GPUS}"
echo "[INFO] Sampling Method: ${SAMPLING_STR}"

AVAILABLE_PAIRS=$(python3 - <<PY
import os
total_limit = int("$TOTAL_GPUS_LIMIT")
skip = [int(x) for x in "$SKIP_GPUS".split(",") if x.strip()]

usable = []
for i in range(total_limit):
    if i not in skip:
        usable.append(str(i))

# 两两配对 (TP=2)
pairs = []
for i in range(0, len(usable) - 1, 2):
    pairs.append(usable[i] + "," + usable[i+1])

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
# 自动在文件名中加入采样方式标识
OUTPUT_GT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_${SAMPLING_STR}.jsonl"
ALIGN_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

# 1. 启动分片任务
for i in "${!PAIR_ARRAY[@]}"; do
    CUR_GPUS="${PAIR_ARRAY[$i]}"
    echo "[INFO] Launching Shard $i / ${TOTAL_SHARDS} on physical GPUs ${CUR_GPUS}"
    
    # 根据配置决定是否加入 --no-freq-sampling
    FREQ_FLAG=""
    if [ "$ENABLE_FREQ_SAMPLING" = false ]; then
        FREQ_FLAG="--no-freq-sampling"
    fi

    CUDA_VISIBLE_DEVICES="${CUR_GPUS}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_ner_align_baseline.py \
      --input-gt "${INPUT_GT}" \
      --input-tsv "${INPUT_TSV}" \
      --output-gt "${OUTPUT_GT}" \
      --align-model "${ALIGN_MODEL}" \
      --gpu-memory-util 0.90 \
      --batch-size 32 \
      --tensor-parallel-size 2 \
      --max-retries 1 \
      --max-terms-per-utter "${MAX_TERMS_PER_UTTER}" \
      --sampling-rate "${SAMPLING_RATE}" \
      ${FREQ_FLAG} \
      --gpu-id "$i" \
      --total-gpus "${TOTAL_SHARDS}" &
done

echo "[INFO] All $TOTAL_SHARDS shards launched. Waiting..."
wait

# 2. 自动合并逻辑
FINAL_OUTPUT="${OUTPUT_GT}"
echo "[INFO] Merging shards into ${FINAL_OUTPUT}..."

BASE_OUT_PATH="${OUTPUT_GT%.jsonl}"

# 清空或创建最终文件
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
