#!/bin/bash
#SBATCH --job-name=v4_taurus_pipeline_zh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --partition=taurus
#SBATCH --time=48:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_v4_taurus_for_zh.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_v4_taurus_for_zh.err

set -euo pipefail

# ==========================================
# 1. 基础配置
# ==========================================
TAURUS_PHYSICAL_GPUS=8
PROCESSES_PER_GPU=1
SKIP_GPUS=""    # 跳过的卡

# Force using physical GPU ids (ignore SLURM CUDA_VISIBLE_DEVICES).
# - 1: always use 0..TAURUS_PHYSICAL_GPUS-1 (minus SKIP_GPUS)
# - 0: respect CUDA_VISIBLE_DEVICES provided by scheduler
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

# 【关键点】TOTAL_SHARDS 必须等于实际启动的进程总数，否则数据会漏掉
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

# --- 采样实验配置 ---
ENABLE_FREQ_SAMPLING=false
SAMPLING_RATE=1.0 # 仅在频率采样关闭时生效 (e.g. 0.3, 0.5, 1.0)
# --------------------

if [ "$ENABLE_FREQ_SAMPLING" = true ]; then
    SAMPLING_STR="freq_k${MAX_TERMS_PER_UTTER}"
else
    SAMPLING_STR="rate${SAMPLING_RATE}_k${MAX_TERMS_PER_UTTER}"
fi

# =======================
# ZH language configuration
# =======================
INPUT_GT_STAGE1="/mnt/gemini/data/jiaxuanluo/manifests_rag/train_s_zh_baseline.jsonl"
INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
NER_CANDIDATES_OUTPUT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_ner_candidates_${SPACY_MODEL}.jsonl"
STAGE1_OUTPUT="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_${SAMPLING_STR}.jsonl"
STAGE2_OUTPUT_BASE="/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_${SAMPLING_STR}_final"
TARGET_LANG_CODE="zh"

# Tuned model for Stage2 index + retriever (per paper_main_result)
RAG_MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"

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
    if [ -f "${NER_CANDIDATES_OUTPUT}" ] && [ -s "${NER_CANDIDATES_OUTPUT}" ]; then
        echo "[STAGE 0] NER candidates already exist: ${NER_CANDIDATES_OUTPUT} (skip)"
    else
        echo "[STAGE 0] Extracting NER candidates using spaCy GPU..."
        conda activate "${SPACY_GPU_ENV_PATH}" || { echo "Failed to activate ${SPACY_GPU_ENV_PATH}"; exit 1; }

        # Ensure pip-installed NVIDIA CUDA runtime libraries (e.g., libcudart.so.12) are discoverable.
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

    SAMPLING_ARGS=""
    if [ "$ENABLE_FREQ_SAMPLING" = false ]; then
        SAMPLING_ARGS="--no-freq-sampling --sampling-rate ${SAMPLING_RATE}"
    fi

    for i in "${!LOGICAL_GPUS[@]}"; do
        CUR_GPU="${LOGICAL_GPUS[$i]}"
        CUDA_VISIBLE_DEVICES="${CUR_GPU}" python retriever/gigaspeech/handle_train_dataset_for_term_map_v4_ner_align_baseline.py \
          --input-gt "${INPUT_GT_STAGE1}" --input-tsv "${INPUT_TSV}" --output-gt "${STAGE1_OUTPUT}" \
          --ner-candidates-path "${NER_CANDIDATES_OUTPUT}" \
          --align-model "${MODEL}" --max-terms-per-utter "${MAX_TERMS_PER_UTTER}" \
          --batch-size "${BATCH_SIZE}" \
          --gpu-id "$i" --total-gpus "${TOTAL_SHARDS}" \
          --tensor-parallel-size 1 --gpu-memory-util "${GPU_MEM_UTIL}" \
          --target-lang-code "${TARGET_LANG_CODE}" \
          ${SAMPLING_ARGS} &
        sleep 2
    done
    monitor_progress "Stage 1"
fi

if [[ "$MODE" == "all" || "$MODE" == "merge1" ]]; then merge_shards "${STAGE1_OUTPUT}"; fi

# --- STAGE 2: Negatives ---
if [[ "$MODE" == "all" || "$MODE" == "stage2" ]]; then
    conda activate "${VLLM_ENV_PATH}"
    for d in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/lib; do
        if [ -d "$d" ]; then
            export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
        fi
    done
    preflight_vllm_env

    # Build glossary + index once (ZH) from Stage1 aligned output
    ZH_GLOSSARY_JSON="/mnt/gemini/data1/jiaxuanluo/glossary_for_zh_${SAMPLING_STR}.json"
    ZH_INDEX_PKL="/mnt/gemini/data2/jiaxuanluo/index_cache_v4/glossary_for_zh_${SAMPLING_STR}.pkl"
    mkdir -p "$(dirname "${ZH_INDEX_PKL}")"

    echo "[STAGE 2] Extracting ZH glossary from ${STAGE1_OUTPUT} -> ${ZH_GLOSSARY_JSON}"
    python retriever/gigaspeech/extract_glossary_from_aligned_jsonl.py \
      --input-jsonl "${STAGE1_OUTPUT}" \
      --output-json "${ZH_GLOSSARY_JSON}" \
      --target-lang-code "${TARGET_LANG_CODE}"

    echo "[STAGE 2] Building FAISS index -> ${ZH_INDEX_PKL}"
    MODEL_PATH="${RAG_MODEL_PATH}" \
    GLOSSARY_PATH="${ZH_GLOSSARY_JSON}" \
    OUTPUT_PATH="${ZH_INDEX_PKL}" \
    TARGET_LANG_CODE="${TARGET_LANG_CODE}" \
    bash retriever/gigaspeech/run_build_index_v4.sh

    for i in "${!LOGICAL_GPUS[@]}"; do
        CUR_GPU="${LOGICAL_GPUS[$i]}"
        CUDA_VISIBLE_DEVICES="${CUR_GPU}" python /home/jiaxuanluo/InfiniSST/enrich_qwen3_rag_with_negatives_v2.py \
          --input-gt-jsonl "${STAGE1_OUTPUT}" --output-base "${STAGE2_OUTPUT_BASE}" \
          --index-path "${ZH_INDEX_PKL}" \
          --model-path "${RAG_MODEL_PATH}" \
          --target-lang-code "${TARGET_LANG_CODE}" \
          --gpu-id "$i" --total-gpus "${TOTAL_SHARDS}" \
          --window-batch-size 4096 \
          --top-k 20 \
          --score-threshold 0.0 \
          --max-neg-per-sec 9 &
        sleep 2
    done
    monitor_progress "Stage 2"
fi

if [[ "$MODE" == "all" || "$MODE" == "merge2" ]]; then merge_shards "${STAGE2_OUTPUT_BASE}.jsonl"; fi

echo "[SUCCESS] Finished on Taurus."


