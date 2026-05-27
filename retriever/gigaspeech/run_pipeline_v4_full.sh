#!/bin/bash
#SBATCH --job-name=v4_full_pipeline
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_v4_full_pipeline.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_v4_full_pipeline.err

set -euo pipefail

# --- 基础环境配置 ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
export VLLM_CACHE="/mnt/gemini/data1/jiaxuanluo/vllm_cache"
export XDG_CACHE_HOME="/mnt/gemini/data1/jiaxuanluo/xdg_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$HF_HOME" "$VLLM_CACHE" "$XDG_CACHE_HOME"

# --- 全局参数配置 ---
TOTAL_GPUS_LIMIT=${TOTAL_GPUS:-8} 
SKIP_GPUS="0,1,2,3"

# Stage 1 配置 (频率采样与预算)
MAX_TERMS_PER_UTTER=20
ENABLE_FREQ_SAMPLING=true
SAMPLING_RATE=0.5 # 仅在频率采样关闭时生效

# Stage 2 配置 (负采样)
NUM_DISTRACTORS=9
MULTIPLE_RANGE="0 9"

# 模型配置
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

# --- 自动计算标识 ---
if [ "$ENABLE_FREQ_SAMPLING" = true ]; then
    SAMPLING_STR="freq_k${MAX_TERMS_PER_UTTER}"
else
    RATE_STR=$(python3 -c "print(int(${SAMPLING_RATE} * 100))")
    SAMPLING_STR="${RATE_STR}percent"
fi

# 文件路径定义
INPUT_GT_STAGE1="/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
STAGE1_OUTPUT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_${SAMPLING_STR}.jsonl"
STAGE2_OUTPUT_BASE="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_${SAMPLING_STR}_final"

# --- GPU 分片逻辑 ---
AVAILABLE_PAIRS=$(python3 - <<PY
import os
total_limit = int("$TOTAL_GPUS_LIMIT")
skip = [int(x) for x in "$SKIP_GPUS".split(",") if x.strip()]
usable = [str(i) for i in range(total_limit) if i not in skip]
pairs = [usable[i] + "," + usable[i+1] for i in range(0, len(usable) - 1, 2)]
print(" ".join(pairs))
PY
)
IFS=' ' read -r -a PAIR_ARRAY <<< "${AVAILABLE_PAIRS}"
TOTAL_SHARDS=${#PAIR_ARRAY[@]}

echo "[INFO] Pipeline Started. Shards: ${TOTAL_SHARDS}, Method: ${SAMPLING_STR}"

# 跳转回项目根目录，确保相对路径 retriever/gigaspeech/... 正确
cd /home/jiaxuanluo/InfiniSST

# ==========================================
# STAGE 1: NER Alignment
# ==========================================
echo "[PIPELINE] Starting Stage 1: NER Alignment..."
FREQ_FLAG=""
if [ "$ENABLE_FREQ_SAMPLING" = false ]; then FREQ_FLAG="--no-freq-sampling"; fi

for i in "${!PAIR_ARRAY[@]}"; do
    CUDA_VISIBLE_DEVICES="${PAIR_ARRAY[$i]}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_ner_align_baseline.py \
      --input-gt "${INPUT_GT_STAGE1}" --input-tsv "${INPUT_TSV}" --output-gt "${STAGE1_OUTPUT}" \
      --align-model "${MODEL}" --max-terms-per-utter "${MAX_TERMS_PER_UTTER}" \
      --sampling-rate "${SAMPLING_RATE}" ${FREQ_FLAG} \
      --gpu-id "$i" --total-gpus "${TOTAL_SHARDS}" --tensor-parallel-size 2 --gpu-memory-util 0.90 &
done
wait

# 合并 Stage 1
echo "[PIPELINE] Merging Stage 1 shards..."
: > "${STAGE1_OUTPUT}"
for i in "${!PAIR_ARRAY[@]}"; do
    SHARD_FILE="${STAGE1_OUTPUT%.jsonl}_gpu${i}.jsonl"
    cat "${SHARD_FILE}" >> "${STAGE1_OUTPUT}"; rm "${SHARD_FILE}"
done

# ==========================================
# STAGE 2: LLM Negative Distractors
# ==========================================
echo "[PIPELINE] Starting Stage 2: Distractor Generation..."
for i in "${!PAIR_ARRAY[@]}"; do
    CUDA_VISIBLE_DEVICES="${PAIR_ARRAY[$i]}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_stage2_llm_negative.py \
      --input-gt-jsonl "${STAGE1_OUTPUT}" --output-base "${STAGE2_OUTPUT_BASE}" \
      --model "${MODEL}" --num-distractors "${NUM_DISTRACTORS}" --multiple-range ${MULTIPLE_RANGE} \
      --gpu-id "$i" --total-gpus "${TOTAL_SHARDS}" --tensor-parallel-size 2 --gpu-memory-util 0.90 &
done
wait

# 合并 Stage 2
echo "[PIPELINE] Merging Stage 2 shards..."
FINAL_JSONL="${STAGE2_OUTPUT_BASE}.jsonl"
: > "${FINAL_JSONL}"
for i in "${!PAIR_ARRAY[@]}"; do
    SHARD_FILE="${STAGE2_OUTPUT_BASE}_gpu${i}.jsonl"
    cat "${SHARD_FILE}" >> "${FINAL_JSONL}"; rm "${SHARD_FILE}"
done

echo "[PIPELINE] COMPLETE. Final result: ${FINAL_JSONL}"