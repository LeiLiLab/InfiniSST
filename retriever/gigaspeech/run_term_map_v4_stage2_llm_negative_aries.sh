#!/bin/bash
#SBATCH --job-name=term_map_v4_stage2_llm_neg
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_term_map_v4_stage2_llm_neg.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_term_map_v4_stage2_llm_neg.err

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
# 指定物理卡跳过列表（即使 Slurm 分配了这些卡，我们也主动不用它们）
SKIP_GPUS="0,2" 
# --------------

echo "[INFO] Multi-GPU parallel enabled (Slurm-based). Skipping physical GPUs: ${SKIP_GPUS}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES is not set. Are you running under Slurm with GPU gres?"
    exit 1
fi

# 获取 Slurm 分配的所有卡
IFS=',' read -r -a ALL_ALLOCATED <<< "${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Slurm allocated GPUs: ${CUDA_VISIBLE_DEVICES}"

# 过滤掉需要跳过的卡
GPU_LIST=()
for gpu in "${ALL_ALLOCATED[@]}"; do
    is_skip=false
    IFS=',' read -r -a SKIP_ARRAY <<< "${SKIP_GPUS}"
    for skip in "${SKIP_ARRAY[@]}"; do
        if [ "$gpu" == "$skip" ]; then
            is_skip=true
            break
        fi
    done
    if [ "$is_skip" = false ]; then
        GPU_LIST+=("$gpu")
    fi
done

TOTAL_SHARDS=${#GPU_LIST[@]}

if [ "${TOTAL_SHARDS}" -eq 0 ]; then
    echo "[FATAL] No usable GPUs left after skipping ${SKIP_GPUS}. Check your --gres and SKIP_GPUS settings."
    exit 1
fi

echo "[INFO] Final usable GPUs for this job: ${GPU_LIST[*]}"
echo "[INFO] Launching ${TOTAL_SHARDS} parallel shards..."

INPUT_GT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl"
OUTPUT_BASE="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_freq_k20_final"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

cd /home/jiaxuanluo/InfiniSST

# 1. 启动子任务 (每个 GPU 一个独立进程)
pids=()
for i in "${!GPU_LIST[@]}"; do
    CUR_PHYSICAL_GPU="${GPU_LIST[$i]}"
    echo "[INFO] Launching Shard $i / ${TOTAL_SHARDS} on physical GPU ${CUR_PHYSICAL_GPU}"
    
    CUDA_VISIBLE_DEVICES="${CUR_PHYSICAL_GPU}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_stage2_llm_negative.py \
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

for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
    SHARD_FILE="${OUTPUT_BASE}_gpu${i}.jsonl"
    if [ -f "${SHARD_FILE}" ]; then
        echo "[INFO] Appending shard $i..."
        cat "${SHARD_FILE}" >> "${FINAL_OUTPUT}"
        rm "${SHARD_FILE}"
    else
        echo "[WARN] Shard file ${SHARD_FILE} not found!"
    fi
done

echo "[INFO] All shards finished and merged into ${FINAL_OUTPUT}."
