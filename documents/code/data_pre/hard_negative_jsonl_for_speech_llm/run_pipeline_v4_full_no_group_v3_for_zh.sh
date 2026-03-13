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
SKIP_GPUS="0"    # 跳过的卡

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
MAX_NEG_PER_SEC=4.5
OLD_SYSTEM_PROMPT="You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text."
NEW_SYSTEM_PROMPT="You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text. Use the ‘term_map’ as a reference for terminology if provided."

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
STAGE2_OUTPUT_BASE="/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_${SAMPLING_STR}_final"
TARGET_LANG_CODE="zh"

# Tuned model for Stage2 index + retriever.
# Use the latest TTS+Text intersection-capable RAG checkpoint.
RAG_MODEL_PATH="/mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_v2_epoch_5.pt"
RAG_EVAL_MODE="intersection"
RAG_TTS_TERMS_NPY_PATH="/mnt/gemini/data/jiaxuanluo/tts_bank_from_term_train_v2/terms.npy"
RAG_TTS_WAV_DIR="/mnt/gemini/data/jiaxuanluo/tts_bank_from_term_train_v2/wav"
RAG_TTS_EMBEDDINGS_CACHE="/mnt/gemini/data/jiaxuanluo/tts_bank_from_term_train_v2/tts_embeddings_cache.npz"
RAG_TTS_EMBEDDING_BATCH_SIZE=512
RAG_TTS_MAX_PROTOTYPES_PER_TERM=8
RAG_TTS_SIMILARITY_TOP_K=20

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

normalize_system_prompt_in_jsonl() {
    local jsonl_path="$1"
    local old_prompt="$2"
    local new_prompt="$3"
    if [[ ! -f "${jsonl_path}" ]]; then
        echo "[WARN] Prompt normalization skipped: file not found -> ${jsonl_path}"
        return 0
    fi
    echo "[POST] Normalizing system prompt in ${jsonl_path}"
    python3 - "${jsonl_path}" "${old_prompt}" "${new_prompt}" <<'PY'
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
old_prompt = sys.argv[2]
new_prompt = sys.argv[3]
tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp_prompt")

changed = 0
total = 0

with jsonl_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        raw = line.rstrip("\n")
        if not raw:
            fout.write(line)
            continue

        start_idx = raw.find("{")
        if start_idx < 0:
            fout.write(line)
            continue

        prefix = raw[:start_idx]
        payload = raw[start_idx:]
        obj = json.loads(payload)
        total += 1

        local_changed = False
        for msg in obj.get("messages", []):
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content == old_prompt:
                msg["content"] = new_prompt
                local_changed = True

        if local_changed:
            changed += 1

        fout.write(prefix + json.dumps(obj, ensure_ascii=False) + "\n")

tmp_path.replace(jsonl_path)
print(f"[POST] Prompt normalization done: {changed}/{total} samples updated.")
PY
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
    RAG_MODEL_TAG="$(basename "${RAG_MODEL_PATH}" .pt)"
    ZH_INDEX_PKL="/mnt/gemini/data2/jiaxuanluo/index_cache_v4/${RAG_MODEL_TAG}__glossary_for_zh_${SAMPLING_STR}.pkl"
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

    # Pre-compute TTS embeddings once on GPU 0 (avoids 8x redundant encoding)
    if [ -n "${RAG_TTS_EMBEDDINGS_CACHE}" ] && [ ! -f "${RAG_TTS_EMBEDDINGS_CACHE}" ]; then
        echo "[STAGE 2] Pre-computing TTS embeddings -> ${RAG_TTS_EMBEDDINGS_CACHE}"
        CUDA_VISIBLE_DEVICES="${LOGICAL_GPUS[0]}" python \
          /home/jiaxuanluo/InfiniSST/documents/code/data_pre/data_convert/precompute_tts_embeddings.py \
          --terms-npy "${RAG_TTS_TERMS_NPY_PATH}" \
          --wav-dir "${RAG_TTS_WAV_DIR}" \
          --model-path "${RAG_MODEL_PATH}" \
          --glossary-json "${ZH_GLOSSARY_JSON}" \
          --output-npz "${RAG_TTS_EMBEDDINGS_CACHE}" \
          --target-lang-code "${TARGET_LANG_CODE}" \
          --batch-size "${RAG_TTS_EMBEDDING_BATCH_SIZE}" \
          --max-prototypes-per-term "${RAG_TTS_MAX_PROTOTYPES_PER_TERM}"
        echo "[STAGE 2] TTS embeddings cache ready."
    else
        echo "[STAGE 2] TTS embeddings cache already exists: ${RAG_TTS_EMBEDDINGS_CACHE}"
    fi

    # Pre-split the input file so each GPU reads only its own shard (true parallel I/O).
    STAGE2_SHARD_DIR="$(dirname "${STAGE2_OUTPUT_BASE}")/.stage2_shards_$$"
    mkdir -p "${STAGE2_SHARD_DIR}"
    echo "[STAGE 2] Splitting ${STAGE1_OUTPUT} into ${TOTAL_SHARDS} shards -> ${STAGE2_SHARD_DIR}/"
    python3 - "${STAGE1_OUTPUT}" "${STAGE2_SHARD_DIR}" "${TOTAL_SHARDS}" <<'SPLIT_PY'
import sys, os
input_path, shard_dir, n_shards = sys.argv[1], sys.argv[2], int(sys.argv[3])
handles = [open(os.path.join(shard_dir, f"shard_{i}.jsonl"), "w") for i in range(n_shards)]
with open(input_path) as f:
    for idx, line in enumerate(f):
        handles[idx % n_shards].write(line)
for h in handles:
    h.close()
for i in range(n_shards):
    path = os.path.join(shard_dir, f"shard_{i}.jsonl")
    n = sum(1 for _ in open(path))
    print(f"  shard_{i}.jsonl: {n} lines")
SPLIT_PY

    for i in "${!LOGICAL_GPUS[@]}"; do
        CUR_GPU="${LOGICAL_GPUS[$i]}"
        SHARD_INPUT="${STAGE2_SHARD_DIR}/shard_${i}.jsonl"
        CUDA_VISIBLE_DEVICES="${CUR_GPU}" python /home/jiaxuanluo/InfiniSST/documents/code/data_pre/hard_negative_jsonl_for_speech_llm/enrich_qwen3_rag_with_negatives_v2.py \
          --input-gt-jsonl "${SHARD_INPUT}" --output-base "${STAGE2_OUTPUT_BASE}" \
          --index-path "${ZH_INDEX_PKL}" \
          --model-path "${RAG_MODEL_PATH}" \
          --target-lang-code "${TARGET_LANG_CODE}" \
          --rag-eval-mode "${RAG_EVAL_MODE}" \
          --tts-terms-npy-path "${RAG_TTS_TERMS_NPY_PATH}" \
          --tts-wav-dir "${RAG_TTS_WAV_DIR}" \
          --tts-embeddings-cache "${RAG_TTS_EMBEDDINGS_CACHE}" \
          --tts-embedding-batch-size "${RAG_TTS_EMBEDDING_BATCH_SIZE}" \
          --tts-max-prototypes-per-term "${RAG_TTS_MAX_PROTOTYPES_PER_TERM}" \
          --tts-similarity-top-k "${RAG_TTS_SIMILARITY_TOP_K}" \
          --gpu-id "$i" --total-gpus 1 \
          --window-batch-size 4096 \
          --top-k 20 \
          --score-threshold 0.0 \
          --max-neg-per-sec "${MAX_NEG_PER_SEC}" &
        sleep 2
    done
    monitor_progress "Stage 2"
    rm -rf "${STAGE2_SHARD_DIR}"
    echo "[STAGE 2] Cleaned up shard dir: ${STAGE2_SHARD_DIR}"
fi

if [[ "$MODE" == "all" || "$MODE" == "merge2" ]]; then
    merge_shards "${STAGE2_OUTPUT_BASE}.jsonl"
    normalize_system_prompt_in_jsonl "${STAGE2_OUTPUT_BASE}.jsonl" "${OLD_SYSTEM_PROMPT}" "${NEW_SYSTEM_PROMPT}"
fi

echo "[SUCCESS] Finished on Taurus."


