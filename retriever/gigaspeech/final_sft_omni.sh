#!/bin/bash
#SBATCH --job-name=v4_dist_pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_v4_dist.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_v4_dist.err

set -euo pipefail

# ==========================================
# 1. 集群分布式配置 (由用户指定)
# ==========================================
TAURUS_PHYSICAL_GPUS=8
ARIES_PHYSICAL_GPUS=4
PROCESSES_PER_GPU=1  # 显存足够时可设为 2 提速

# 运行模式: all (默认), stage1, stage2, merge1, merge2
MODE=${1:-"all"}

# ==========================================
# 2. 自动检测环境与计算分片
# ==========================================
CLUSTER="taurus"
# 优先检查 hostname，备选检查 Slurm 队列名
if [[ $(hostname) == *aries* ]] || [[ "${SLURM_JOB_PARTITION:-}" == "aries" ]]; then
    CLUSTER="aries"
fi

# 全局总分片
GLOBAL_TOTAL_GPUS=$(( TAURUS_PHYSICAL_GPUS + ARIES_PHYSICAL_GPUS ))
GLOBAL_TOTAL_SHARDS=$(( GLOBAL_TOTAL_GPUS * PROCESSES_PER_GPU ))

# 根据集群决定 Offset 和 跳过列表
if [ "$CLUSTER" == "aries" ]; then
    GLOBAL_SHARD_OFFSET=$(( TAURUS_PHYSICAL_GPUS * PROCESSES_PER_GPU ))
    SKIP_GPUS=""
    # Aries 环境变量注入 (参考 run_extract_all_terms_aries.sh)
    export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
    export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
else
    GLOBAL_SHARD_OFFSET=0
    SKIP_GPUS=""
    # Taurus 环境变量配置
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate /mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv
    export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
fi

# 全局 vLLM/NCCL 优化 (确保所有集群都使用 spawn 方法以避免 CUDA 报错)
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
export VLLM_CACHE="/mnt/gemini/data1/jiaxuanluo/vllm_cache"
export XDG_CACHE_HOME="/mnt/gemini/data1/jiaxuanluo/xdg_cache"
mkdir -p "$HF_HOME" "$VLLM_CACHE" "$XDG_CACHE_HOME"

# ==========================================
# 3. 基础参数与路径
# ==========================================
BATCH_SIZE=4096      # 传入给 llm.generate 的 prompt 列表大小
SPACY_MODEL="en_core_web_trf" 
MAX_TERMS_PER_UTTER=20
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
SAMPLING_STR="freq_k${MAX_TERMS_PER_UTTER}"

# 路径定义
#INPUT_GT_STAGE1="/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline.jsonl"
INPUT_GT_STAGE1="/mnt/gemini/data1/jiaxuanluo/train_m_zh_baseline_simplified.jsonl"
INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
# 新增：NER 中间结果路径
NER_CANDIDATES_OUTPUT="/mnt/gemini/data1/jiaxuanluo/train_m_zh_ner_candidates_${SPACY_MODEL}.jsonl"

STAGE1_OUTPUT="/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_ner_baseline_aligned_${SAMPLING_STR}.jsonl"
STAGE2_OUTPUT_BASE="/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_ner_baseline_aligned_${SAMPLING_STR}_final"

# Conda 环境定义
VLLM_ENV="spaCyEnv"      # 这里的环境包含了 vLLM 0.13.0
SPACY_GPU_ENV="spacy_gpu_env" # 新环境，稍后执行创建脚本

# ==========================================
# 4. 获取本地可用 GPU 并构建逻辑分片
# ==========================================
# 获取 Slurm 分配的卡
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a ALL_ALLOCATED <<< "${CUDA_VISIBLE_DEVICES}"
else
    # 非 Slurm 环境下假设全部可用
    if [ "$CLUSTER" == "aries" ]; then
        MAX_LOCAL=$ARIES_PHYSICAL_GPUS
    else
        MAX_LOCAL=$TAURUS_PHYSICAL_GPUS
    fi
    ALL_ALLOCATED=($(seq 0 $((MAX_LOCAL - 1))))
fi

GPU_ARRAY=()
for gpu in "${ALL_ALLOCATED[@]}"; do
    is_skip=false
    IFS=',' read -r -a SKIP_ARRAY <<< "${SKIP_GPUS}"
    for skip in "${SKIP_ARRAY[@]}"; do
        if [ "$gpu" == "$skip" ]; then is_skip=true; break; fi
    done
    if [ "$is_skip" = false ]; then GPU_ARRAY+=("$gpu"); fi
done

LOGICAL_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
    for ((p=0; p<PROCESSES_PER_GPU; p++)); do LOGICAL_GPUS+=("${gpu}"); done
done
LOCAL_SHARDS=${#LOGICAL_GPUS[@]}
GPU_MEM_UTIL=$(python3 -c "print(0.90 / ${PROCESSES_PER_GPU})")

echo "[INFO] Cluster: ${CLUSTER}, Global Total Shards: ${GLOBAL_TOTAL_SHARDS}, Global Offset: ${GLOBAL_SHARD_OFFSET}"
echo "[INFO] Local Usable Shards: ${LOCAL_SHARDS}, Logical GPU Mapping: ${LOGICAL_GPUS[*]}"

cd /home/jiaxuanluo/InfiniSST

# 监控辅助函数
monitor_progress() {
    local phase=$1
    echo "[MONITOR] $phase: Monitoring $LOCAL_SHARDS local shards..."
    while true; do
        local running=$(jobs -r | wc -l)
        if [ "$running" -eq 0 ]; then break; fi
        echo -ne "[MONITOR] $phase: $running/$LOCAL_SHARDS local shards still running...\r"
        sleep 30
    done
    echo -e "\n[MONITOR] $phase: Local shards completed."
}

# 全局合并函数
merge_global_shards() {
    local base_out=$1
    echo "[PIPELINE] Global Merging for ${base_out}..."
    : > "${base_out}"
    # 按照全局分片顺序合并，确保结果顺序一致
    for i in $(seq 0 $((GLOBAL_TOTAL_SHARDS - 1))); do
        local shard="${base_out%.jsonl}_gpu${i}.jsonl"
        if [ -f "$shard" ]; then
            echo "[INFO] Appending shard $i..."
            cat "$shard" >> "${base_out}" && rm "$shard"
        else
            echo "[WARN] Shard file $shard not found, skipping."
        fi
    done
    echo "[PIPELINE] Merged into ${base_out}"
}

# ==========================================
# 5. 执行逻辑
# ==========================================

# --- STAGE 0: NER Extraction (spaCy GPU) ---
if [[ "$MODE" == "all" || "$MODE" == "stage0" ]]; then
    echo "[STAGE 0] Extracting NER candidates using spaCy GPU..."
    # 切换到 spaCy 专用环境
    conda activate ${SPACY_GPU_ENV} || { echo "Failed to activate ${SPACY_GPU_ENV}"; exit 1; }
    
    # 注入 LD_LIBRARY_PATH 以支持 pip 安装的 nvidia-* 运行库 (cupy 依赖)
    for d in $CONDA_PREFIX/lib/python*/site-packages/nvidia/*/lib; do
        if [ -d "$d" ]; then
            export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
        fi
    done
    
    for i in "${!LOGICAL_GPUS[@]}"; do
        GLOBAL_ID=$(( GLOBAL_SHARD_OFFSET + i ))
        CUR_GPU="${LOGICAL_GPUS[$i]}"
        CUDA_VISIBLE_DEVICES="${CUR_GPU}" python retriever/gigaspeech/extract_ner_candidates_v4.py \
          --input-gt "${INPUT_GT_STAGE1}" --input-tsv "${INPUT_TSV}" \
          --output-jsonl "${NER_CANDIDATES_OUTPUT}" \
          --spacy-model "${SPACY_MODEL}" \
          --gpu-id "$GLOBAL_ID" --total-gpus "${GLOBAL_TOTAL_SHARDS}" &
        sleep 1
    done
    monitor_progress "Stage 0"
fi

if [[ "$MODE" == "merge0" ]]; then
    merge_global_shards "${NER_CANDIDATES_OUTPUT}"
fi

# --- STAGE 1: NER Alignment (vLLM) ---
if [[ "$MODE" == "all" || "$MODE" == "stage1" ]]; then
    echo "[STAGE 1] Starting vLLM alignment..."
    # 切换回 vLLM 环境
    conda activate ${VLLM_ENV}
    
    for i in "${!LOGICAL_GPUS[@]}"; do
        GLOBAL_ID=$(( GLOBAL_SHARD_OFFSET + i ))
        CUR_GPU="${LOGICAL_GPUS[$i]}"
        CUDA_VISIBLE_DEVICES="${CUR_GPU}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_ner_align_baseline.py \
          --input-gt "${INPUT_GT_STAGE1}" --input-tsv "${INPUT_TSV}" --output-gt "${STAGE1_OUTPUT}" \
          --ner-candidates-path "${NER_CANDIDATES_OUTPUT}" \
          --align-model "${MODEL}" --max-terms-per-utter "${MAX_TERMS_PER_UTTER}" \
          --batch-size "${BATCH_SIZE}" \
          --gpu-id "$GLOBAL_ID" --total-gpus "${GLOBAL_TOTAL_SHARDS}" \
          --tensor-parallel-size 1 --gpu-memory-util "${GPU_MEM_UTIL}" &
        sleep 2
    done
    monitor_progress "Stage 1"
fi

if [[ "$MODE" == "merge1" ]]; then
    merge_global_shards "${STAGE1_OUTPUT}"
fi

# --- STAGE 2: LLM Negative Distractors ---
if [[ "$MODE" == "all" || "$MODE" == "stage2" ]]; then
    echo "[STAGE 2] Starting local shards..."
    for i in "${!LOGICAL_GPUS[@]}"; do
        GLOBAL_ID=$(( GLOBAL_SHARD_OFFSET + i ))
        CUR_GPU="${LOGICAL_GPUS[$i]}"
        CUDA_VISIBLE_DEVICES="${CUR_GPU}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_stage2_llm_negative.py \
          --input-gt-jsonl "${STAGE1_OUTPUT}" --output-base "${STAGE2_OUTPUT_BASE}" \
          --model "${MODEL}" --num-distractors 9 --multiple-range 0 9 \
          --batch-size "${BATCH_SIZE}" \
          --gpu-id "$GLOBAL_ID" --total-gpus "${GLOBAL_TOTAL_SHARDS}" \
          --tensor-parallel-size 1 --gpu-memory-util "${GPU_MEM_UTIL}" &
        sleep 2
    done
    monitor_progress "Stage 2"
fi

if [[ "$MODE" == "merge2" ]]; then
    merge_global_shards "${STAGE2_OUTPUT_BASE}.jsonl"
fi

if [[ "$MODE" == "all" ]]; then
    echo "[NOTICE] Mode 'all' finished local shards. Please run merge manually if running across multiple clusters."
fi

