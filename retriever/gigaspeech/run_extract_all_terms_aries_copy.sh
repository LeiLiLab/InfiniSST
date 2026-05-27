#!/bin/bash
#SBATCH --job-name=extract_all_terms
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
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
# 指定物理卡跳过列表
SKIP_GPUS="" 
# 是否启用断点续传 (true/false)
RESUME=true
# 是否强制重跑 (true/false)
FORCE=false

# 运行模式: all, stage0, stage1, merge0, merge
MODE=${1:-"all"}
# --------------

INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
OUTPUT_DIR="/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks"
OUTPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_dataset.jsonl"
NER_CANDIDATES_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_ner_candidates.jsonl"
ALIGN_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

VLLM_ENV="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
SPACY_GPU_ENV="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spacy_gpu_env"

EXTRA_ARGS=""
if [ "$RESUME" = true ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --resume"
fi
if [ "$FORCE" = true ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --force"
fi

echo "[INFO] Mode: ${MODE}, Resume: ${RESUME}, Force: ${FORCE}"

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
        if [ "$gpu" == "$skip" ]; then is_skip=true; break; fi
    done
    if [ "$is_skip" = false ]; then GPU_LIST+=("$gpu"); fi
done

TOTAL_SHARDS=${#GPU_LIST[@]}
if [ "${TOTAL_SHARDS}" -eq 0 ]; then
    echo "[FATAL] No usable GPUs left after skipping ${SKIP_GPUS}."
    exit 1
fi

echo "[INFO] Final usable GPUs: ${GPU_LIST[*]}"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

# --- STAGE 0: NER Extraction (spaCy GPU) ---
if [[ "$MODE" == "all" || "$MODE" == "stage0" ]]; then
    echo "[STAGE 0] Extracting NER candidates using spaCy GPU..."
    export CONDA_PREFIX="${SPACY_GPU_ENV}"
    export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    
    # 尝试注入 cupy 所需的库路径 (参考 run_pipeline_v4_full_distributed.sh)
    for d in $CONDA_PREFIX/lib/python*/site-packages/nvidia/*/lib; do
        if [ -d "$d" ]; then export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; fi
    done

    for i in "${!GPU_LIST[@]}"; do
        CUR_PHYSICAL_GPU="${GPU_LIST[$i]}"
        echo "[INFO] Launching Stage 0 Shard $i on GPU ${CUR_PHYSICAL_GPU}"
        CUDA_VISIBLE_DEVICES="${CUR_PHYSICAL_GPU}" python retriever/gigaspeech/extract_all_terms_from_tsv.py \
          --mode extract \
          --input-tsv "${INPUT_TSV}" \
          --output-dir "${OUTPUT_DIR}" \
          --output-jsonl "${OUTPUT_JSONL}" \
          --ner-candidates-path "${NER_CANDIDATES_JSONL}" \
          --batch-size 128 \
          --shard-id "$i" \
          --total-shards "${TOTAL_SHARDS}" \
          ${EXTRA_ARGS} &
        sleep 2
    done
    wait
    echo "[STAGE 0] Completed."
fi

if [[ "$MODE" == "all" || "$MODE" == "merge0" ]]; then
    echo "[INFO] Merging NER shards..."
    : > "${NER_CANDIDATES_JSONL}"
    BASE_NER_PATH="${NER_CANDIDATES_JSONL%.jsonl}"
    for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
        SHARD_FILE="${BASE_NER_PATH}_shard${i}.jsonl"
        if [ -f "${SHARD_FILE}" ]; then
            cat "${SHARD_FILE}" >> "${NER_CANDIDATES_JSONL}"
            rm "${SHARD_FILE}"
        fi
    done
fi

# --- STAGE 1: NER Alignment (vLLM) ---
if [[ "$MODE" == "all" || "$MODE" == "stage1" ]]; then
    echo "[STAGE 1] Starting vLLM alignment..."
    export CONDA_PREFIX="${VLLM_ENV}"
    export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

    # vLLM/NCCL knobs
    export VLLM_USE_V1=0
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1

    for i in "${!GPU_LIST[@]}"; do
        CUR_PHYSICAL_GPU="${GPU_LIST[$i]}"
        echo "[INFO] Launching Stage 1 Shard $i on GPU ${CUR_PHYSICAL_GPU}"
        CUDA_VISIBLE_DEVICES="${CUR_PHYSICAL_GPU}" python retriever/gigaspeech/extract_all_terms_from_tsv.py \
          --mode align \
          --input-tsv "${INPUT_TSV}" \
          --output-dir "${OUTPUT_DIR}" \
          --output-jsonl "${OUTPUT_JSONL}" \
          --ner-candidates-path "${NER_CANDIDATES_JSONL}" \
          --align-model "${ALIGN_MODEL}" \
          --gpu-memory-util 0.8 \
          --prompt-batch-size 4096 \
          --tp-size 1 \
          --shard-id "$i" \
          --total-shards "${TOTAL_SHARDS}" \
          ${EXTRA_ARGS} &
        sleep 2
    done
    wait
    echo "[STAGE 1] Completed."
fi

# --- Final Merge ---
if [[ "$MODE" == "all" || "$MODE" == "merge" ]]; then
    echo "[INFO] Merging final shards into ${OUTPUT_JSONL}..."
    FINAL_OUTPUT="${OUTPUT_JSONL}"
    BASE_OUT_PATH="${OUTPUT_JSONL%.jsonl}"
    # 这里需要临时文件合并，避免覆盖正在读取的文件（如果是在断点续传中）
    # 但由于已经 wait 了，可以直接合并
    
    # 找出所有 shard 并合并
    for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
        SHARD_FILE="${BASE_OUT_PATH}_shard${i}.jsonl"
        if [ -f "${SHARD_FILE}" ]; then
            cat "${SHARD_FILE}" >> "${FINAL_OUTPUT}"
            rm "${SHARD_FILE}"
        fi
    done
    echo "[INFO] All shards merged into ${FINAL_OUTPUT}."
fi
