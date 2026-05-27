#!/bin/bash
#SBATCH --job-name=v4_dev_pipeline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --partition=taurus
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_v4_dev.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_v4_dev.err

set -euo pipefail

# ==========================================
# 1. 基础配置
# ==========================================
TAURUS_PHYSICAL_GPUS=8
PROCESSES_PER_GPU=1 
SKIP_GPUS="0,1,2"    # 跳过的卡

# Force using physical GPU ids (ignore SLURM CUDA_VISIBLE_DEVICES).
FORCE_PHYSICAL_GPUS=${FORCE_PHYSICAL_GPUS:-1}

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 运行模式: all (默认), stage0, merge0, stage1, merge1, stage2, merge2
MODE=${1:-"all"}

# ==========================================
# 2. 获取实际可用 GPU 并构建逻辑分片
# ==========================================
if [[ "${FORCE_PHYSICAL_GPUS}" == "1" ]]; then
    unset CUDA_VISIBLE_DEVICES
    ALL_ALLOCATED=($(seq 0 $((TAURUS_PHYSICAL_GPUS - 1))))
else
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a ALL_ALLOCATED <<< "${CUDA_VISIBLE_DEVICES}"
    else
        ALL_ALLOCATED=($(seq 0 $((TAURUS_PHYSICAL_GPUS - 1))))
    fi
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

TOTAL_SHARDS=${#LOGICAL_GPUS[@]}

if [ "${TOTAL_SHARDS}" -eq 0 ]; then
    echo "[FATAL] No usable GPUs found after skipping ${SKIP_GPUS}."
    exit 1
fi

# ==========================================
# 3. 环境与路径
# ==========================================
source ~/miniconda3/etc/profile.d/conda.sh
VLLM_ENV_PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
SPACY_GPU_ENV_PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spacy_gpu_env"

conda activate "${VLLM_ENV_PATH}"
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# vLLM 优化
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
export VLLM_CACHE="/mnt/gemini/data1/jiaxuanluo/vllm_cache"
export XDG_CACHE_HOME="/mnt/gemini/data1/jiaxuanluo/xdg_cache"
mkdir -p "$HF_HOME" "$VLLM_CACHE" "$XDG_CACHE_HOME"

BATCH_SIZE=4096      
SPACY_MODEL="en_core_web_trf" 
MAX_TERMS_PER_UTTER=20
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
SAMPLING_STR="freq_k${MAX_TERMS_PER_UTTER}"

# DEV INPUTS
INPUT_GT_STAGE1="/mnt/gemini/data1/jiaxuanluo/train_s_zh_baseline_dev.jsonl"
INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"

# DEV OUTPUTS
NER_CANDIDATES_OUTPUT="/mnt/gemini/data1/jiaxuanluo/ner_candidates_dev_${SPACY_MODEL}.jsonl"
STAGE1_OUTPUT="/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_${SAMPLING_STR}.jsonl"
STAGE2_OUTPUT_BASE="/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_${SAMPLING_STR}_final"

VLLM_ENV="spaCyEnv"
SPACY_GPU_ENV="spacy_gpu_env"
GPU_MEM_UTIL=$(python3 -c "print(0.90 / ${PROCESSES_PER_GPU})")

echo "[INFO] Total Shards: ${TOTAL_SHARDS}, Using GPUs: ${GPU_ARRAY[*]}"

cd /home/jiaxuanluo/InfiniSST

# 监控与合并函数
monitor_progress() {
    local phase=$1
    echo "[MONITOR] $phase: Monitoring $TOTAL_SHARDS shards..."
    while true; do
        local running=$(jobs -r | wc -l)
        if [ "$running" -eq 0 ]; then break; fi
        echo -ne "[MONITOR] $phase: $running/$TOTAL_SHARDS shards still running...\r"
        sleep 30
    done
    echo -e "\n[MONITOR] $phase: Completed."
}

merge_shards() {
    local base_out=$1
    echo "[PIPELINE] Merging ${TOTAL_SHARDS} shards into ${base_out}..."
    : > "${base_out}"
    for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
        local shard="${base_out%.jsonl}_gpu${i}.jsonl"
        if [ -f "$shard" ]; then
            cat "$shard" >> "${base_out}" && rm "$shard"
        else
            echo "[WARN] Shard $shard missing!"
        fi
    done
}

preflight_spacy_gpu_env() {
    python - <<'PY'
import sys
missing = []
try:
    import cupy  # noqa: F401
except Exception as e:
    missing.append(f"cupy ({e})")
try:
    import regex  # noqa: F401
except Exception as e:
    missing.append(f"regex ({e})")

if missing:
    print("[FATAL] spacy_gpu_env dependency check failed: " + ", ".join(missing), file=sys.stderr)
    print("[HINT] Install in spacy_gpu_env:", file=sys.stderr)
    print("       pip install -U regex", file=sys.stderr)
    print("       pip install -U cupy-cuda12x  # or cupy-cuda11x depending on your CUDA", file=sys.stderr)
    sys.exit(2)
print("[INFO] spacy_gpu_env dependency check passed.")
PY
}

preflight_vllm_env() {
    python - <<'PY'
import sys
missing = []
try:
    import regex  # noqa: F401
except Exception as e:
    missing.append(f"regex ({e})")
if missing:
    print("[FATAL] spaCyEnv dependency check failed: " + ", ".join(missing), file=sys.stderr)
    print("[HINT] Install in spaCyEnv:", file=sys.stderr)
    print("       pip install -U regex", file=sys.stderr)
    sys.exit(2)
print("[INFO] spaCyEnv dependency check passed.")
PY
}

# ==========================================
# 5. 执行阶段
# ==========================================

# --- STAGE 0: NER ---
if [[ "$MODE" == "all" || "$MODE" == "stage0" ]]; then
    echo "[STAGE 0] Extracting NER candidates using spaCy GPU..."
    conda activate "${SPACY_GPU_ENV_PATH}" || { echo "Failed to activate ${SPACY_GPU_ENV_PATH}"; exit 1; }

    # Ensure pip-installed NVIDIA CUDA runtime libraries are discoverable.
    for d in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/lib; do
        if [ -d "$d" ]; then
            export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
        fi
    done

    preflight_spacy_gpu_env
    for i in "${!LOGICAL_GPUS[@]}"; do
        CUR_GPU="${LOGICAL_GPUS[$i]}"
        CUDA_VISIBLE_DEVICES="${CUR_GPU}" python retriever/gigaspeech/extract_ner_candidates_v4.py \
          --input-gt "${INPUT_GT_STAGE1}" --input-tsv "${INPUT_TSV}" \
          --output-jsonl "${NER_CANDIDATES_OUTPUT}" --spacy-model "${SPACY_MODEL}" \
          --gpu-id "$i" --total-gpus "${TOTAL_SHARDS}" &
        sleep 1
    done
    monitor_progress "Stage 0"
fi

if [[ "$MODE" == "all" || "$MODE" == "merge0" ]]; then merge_shards "${NER_CANDIDATES_OUTPUT}"; fi

# --- STAGE 1: Alignment ---
if [[ "$MODE" == "all" || "$MODE" == "stage1" ]]; then
    conda activate "${VLLM_ENV_PATH}"
    for d in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/lib; do
        if [ -d "$d" ]; then
            export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
        fi
    done
    preflight_vllm_env
    for i in "${!LOGICAL_GPUS[@]}"; do
        CUR_GPU="${LOGICAL_GPUS[$i]}"
        CUDA_VISIBLE_DEVICES="${CUR_GPU}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_ner_align_baseline.py \
          --input-gt "${INPUT_GT_STAGE1}" --input-tsv "${INPUT_TSV}" --output-gt "${STAGE1_OUTPUT}" \
          --ner-candidates-path "${NER_CANDIDATES_OUTPUT}" \
          --align-model "${MODEL}" --max-terms-per-utter "${MAX_TERMS_PER_UTTER}" \
          --batch-size "${BATCH_SIZE}" \
          --gpu-id "$i" --total-gpus "${TOTAL_SHARDS}" \
          --tensor-parallel-size 1 --gpu-memory-util "${GPU_MEM_UTIL}" &
        sleep 2
    done
    monitor_progress "Stage 1"
fi

if [[ "$MODE" == "all" || "$MODE" == "merge1" ]]; then merge_shards "${STAGE1_OUTPUT}"; fi

# --- STAGE 2: Negative ---
if [[ "$MODE" == "all" || "$MODE" == "stage2" ]]; then
    conda activate "${VLLM_ENV_PATH}"
    for d in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/lib; do
        if [ -d "$d" ]; then
            export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
        fi
    done
    preflight_vllm_env
    for i in "${!LOGICAL_GPUS[@]}"; do
        CUR_GPU="${LOGICAL_GPUS[$i]}"
        CUDA_VISIBLE_DEVICES="${CUR_GPU}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_stage2_llm_negative.py \
          --input-gt-jsonl "${STAGE1_OUTPUT}" --output-base "${STAGE2_OUTPUT_BASE}" \
          --model "${MODEL}" --num-distractors 9 --multiple-range 0 9 \
          --batch-size "${BATCH_SIZE}" \
          --gpu-id "$i" --total-gpus "${TOTAL_SHARDS}" \
          --tensor-parallel-size 1 --gpu-memory-util "${GPU_MEM_UTIL}" &
        sleep 2
    done
    monitor_progress "Stage 2"
fi

if [[ "$MODE" == "all" || "$MODE" == "merge2" ]]; then merge_shards "${STAGE2_OUTPUT_BASE}.jsonl"; fi

echo "[SUCCESS] Finished dev pipeline on Taurus."


















