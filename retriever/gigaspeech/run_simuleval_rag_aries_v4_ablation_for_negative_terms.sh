#!/bin/bash
#SBATCH --job-name=simuleval_rag_negterm_ablation
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
# 3 speech models * 2 glossaries = 6 tasks
#SBATCH --array=0-1
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_simuleval_rag_sweep.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_simuleval_rag_sweep.err

set -euo pipefail

# ==================== 环境配置 ====================
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# vLLM 配置
export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=0

# ==================== 扫参配置 ====================
# Indices must be built per (model checkpoint, glossary). If missing, build it first.
INDEX_CACHE_DIR="/mnt/gemini/data2/jiaxuanluo/index_cache_v4"
GLOSSARIES=(
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
)
# Fixed streaming window for this ablation (decoupled from speech chunking).
RAG_CHUNK_SIZE="1.92"
RAG_HOP_SIZE="0.48"

# Retriever checkpoints
RAG_MODEL_PATHS=(
  "/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"
)

# Speech models correspond to 3 negative-term selection strategies (Standard / None / All-Wrong).
# NOTE: The newest zh model expects the prompt to contain "term_map:NONE" when no terms are provided.
# Older models were trained without it; for them we pass --use-no-term-map-none to OMIT this marker.
SPEECH_MODELS=(
  "/mnt/gemini/data/jiaxuanluo/best_model_no_llm_zh_v2"
  # "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora-rank8/v0-20260103-221345-hf-v2"
  # "/mnt/gemini/data2/jiaxuanluo/models/owaski_gigaspeech_zh_k20_final_random"
  # "/mnt/gemini/data2/jiaxuanluo/models/owaski_gigaspeech_zh_k20_final_llm"
)

NUM_MODELS=${#SPEECH_MODELS[@]}
NUM_GLOSSARIES=${#GLOSSARIES[@]}

# Decode Task ID (NUM_MODELS * NUM_GLOSSARIES tasks)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
M_IDX=$((TASK_ID % NUM_MODELS))
G_IDX=$((TASK_ID / NUM_MODELS))

TOTAL_TASKS=$((NUM_MODELS * NUM_GLOSSARIES))
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "[ERROR] TASK_ID out of range: ${TASK_ID} (valid: 0..$((TOTAL_TASKS - 1)))"
  exit 2
fi

# This ablation uses a single retriever checkpoint for all chunk/hop settings.
RAG_MODEL_PATH="${RAG_MODEL_PATHS[0]}"
GLOSSARY_PATH="${GLOSSARIES[$G_IDX]}"
MODEL_NAME="${SPEECH_MODELS[$M_IDX]}"

# Fixed retrieval output size and voting size for this ablation.
FIXED_RECALL_K=2
FIXED_VOTING_K=1
FIXED_THRESHOLD=0.0

# Prompt compatibility toggle (see header comment).
USE_NO_TERM_MAP_NONE=0
if [[ "${MODEL_NAME}" == *"/owaski_gigaspeech_zh_k20_final_random"* ]] || [[ "${MODEL_NAME}" == *"/owaski_gigaspeech_zh_k20_final"* ]]; then
  USE_NO_TERM_MAP_NONE=1
fi

# Build / resolve index path for this (model, glossary)
MODEL_TAG="$(basename "${RAG_MODEL_PATH}" .pt)"
GLOSSARY_TAG="$(basename "${GLOSSARY_PATH}" .json)"
CUR_INDEX="${INDEX_CACHE_DIR}/${MODEL_TAG}__${GLOSSARY_TAG}__tr16.pkl"

# Index name tag for output folder
INDEX_NAME="curated"
[[ "${GLOSSARY_TAG}" == *"glossary_acl6060"* ]] && INDEX_NAME="raw"

echo "[INFO] Task ID: ${TASK_ID}"
echo "[INFO] GLOSSARY: ${GLOSSARY_PATH}"
echo "[INFO] INDEX: ${INDEX_NAME} (${CUR_INDEX})"
echo "[INFO] RAG_CHUNK_SIZE: ${RAG_CHUNK_SIZE}"
echo "[INFO] RAG_HOP_SIZE: ${RAG_HOP_SIZE}"
echo "[INFO] RAG_MODEL_PATH: ${RAG_MODEL_PATH}"
echo "[INFO] MODEL_NAME: ${MODEL_NAME}"
echo "[INFO] FIXED_RECALL_K: ${FIXED_RECALL_K}"
echo "[INFO] FIXED_VOTING_K: ${FIXED_VOTING_K}"
echo "[INFO] FIXED_THRESHOLD: ${FIXED_THRESHOLD}"
echo "[INFO] USE_NO_TERM_MAP_NONE: ${USE_NO_TERM_MAP_NONE}"

# ==================== 路径与参数 ====================
ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
LANG_CODE="zh"
SOURCE_LANG="English"
TARGET_LANG="Chinese"

# 输出目录 (包含超参信息)
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_sweep_v4_chunk_size_ablation_for_negative_terms"

# # 实验前清空之前的输出 (仅在 Task 0 执行一次)
# if [ "${TASK_ID}" -eq 0 ]; then
#   echo "[INFO] Cleaning old outputs in ${OUTPUT_BASE}..."
#   rm -rf "${OUTPUT_BASE}"/*
# fi

# Build index if missing (locked to avoid races)
if [ ! -f "${CUR_INDEX}" ]; then
  echo "[INFO] Index not found. Building: ${CUR_INDEX}"
  mkdir -p "${INDEX_CACHE_DIR}"
  LOCK_FILE="${CUR_INDEX}.lock"
  (
    exec 201>"${LOCK_FILE}"
    flock 201
    if [ ! -f "${CUR_INDEX}" ]; then
      MODEL_PATH="${RAG_MODEL_PATH}" \
      GLOSSARY_PATH="${GLOSSARY_PATH}" \
      OUTPUT_PATH="${CUR_INDEX}" \
      TARGET_LANG_CODE="${LANG_CODE}" \
      bash retriever/gigaspeech/run_build_index_v4.sh
    else
      echo "[INFO] Index already built by another process: ${CUR_INDEX}"
    fi
  )
fi

if [ ! -f "${CUR_INDEX}" ]; then
  echo "[ERROR] Index build failed or missing: ${CUR_INDEX}"
  exit 3
fi

CUR_CHUNK=1.92
# Cache chunk settings: divide by chunk_size, then floor.
MAX_CACHE_CHUNKS=$(python3 - <<PY
chunk=float("${CUR_CHUNK}")
print(int(80.0 / chunk))
PY
)
KEEP_CACHE_CHUNKS=$(python3 - <<PY
chunk=float("${CUR_CHUNK}")
print(int(60.0 / chunk))
PY
)

echo "[INFO] MAX_CACHE_CHUNKS: ${MAX_CACHE_CHUNKS}"
echo "[INFO] KEEP_CACHE_CHUNKS: ${KEEP_CACHE_CHUNKS}"

MODEL_TAG="$(basename "${MODEL_NAME}")"
OUTPUT_PATH="${OUTPUT_BASE}/${INDEX_NAME}_model${M_IDX}_${MODEL_TAG}_rk${FIXED_RECALL_K}_vk${FIXED_VOTING_K}_cs${RAG_CHUNK_SIZE}_hs${RAG_HOP_SIZE}"
# Same setting should overwrite previous runs
rm -rf "${OUTPUT_PATH}"
mkdir -p "${OUTPUT_PATH}"

# ==================== 准备临时数据 ====================
TMP_DATA_DIR="/tmp/${USER}/infinisst_eval_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
mkdir -p "${TMP_DATA_DIR}"
trap 'rm -rf "${TMP_DATA_DIR}"' EXIT

SOURCE_LIST="${TMP_DATA_DIR}/dev.source"
TARGET_LIST="${TMP_DATA_DIR}/dev.target.${LANG_CODE}"

cp "${ROOT}/dev.source" "${SOURCE_LIST}"
cp "${ROOT}/dev.target.zh" "${TARGET_LIST}"

# 修正音频路径
if grep -q "/mnt/data/siqiouyang" "${SOURCE_LIST}"; then
  sed -i 's|/mnt/data/siqiouyang|/mnt/taurus/data/siqiouyang|g' "${SOURCE_LIST}"
fi

# ==================== GPU 分配 ====================
# 使用 2 个 GPU：vLLM 使用 TP=2 占用 0,1；RAG 共享使用 cuda:1
GIDS=($(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '))
export CUDA_VISIBLE_DEVICES="${GIDS[0]},${GIDS[1]}"
RAG_GPU="cuda:1"

# ==================== 运行 SimulEval ====================
LATENCY_MULTIPLIER=1
SRC_SEGMENT_SIZE=$((LATENCY_MULTIPLIER * 480))
MAX_NEW_TOKENS=$((LATENCY_MULTIPLIER * 40))

python -u "$(which simuleval)" \
  --agent agents/infinisst_omni_vllm_rag_v4.py \
  --agent-class agents.infinisst_omni_vllm_rag_v4.InfiniSSTOmniVLLMRAGV4 \
  \
  --source "${SOURCE_LIST}" \
  --target "${TARGET_LIST}" \
  --output "${OUTPUT_PATH}" \
  \
  --source-segment-size "${SRC_SEGMENT_SIZE}" \
  --source-lang "${SOURCE_LANG}" \
  --target-lang "${TARGET_LANG}" \
  --min-start-sec 0 \
  \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --beam 1 \
  --no-repeat-ngram-lookback 100 \
  --no-repeat-ngram-size 5 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  \
  --use-vllm 1 \
  --gpu-memory-utilization 0.8 \
  --model-name "${MODEL_NAME}" \
  --max-cache-chunks "${MAX_CACHE_CHUNKS}" \
  --keep-cache-chunks "${KEEP_CACHE_CHUNKS}" \
  \
  --quality-metrics BLEU \
  --eval-latency-unit char \
  --sacrebleu-tokenizer zh \
  --rag-enabled \
  --rag-index-path "${CUR_INDEX}" \
  --rag-model-path "${RAG_MODEL_PATH}" \
  --rag-chunk-size "${RAG_CHUNK_SIZE}" \
  --rag-hop-size "${RAG_HOP_SIZE}" \
  --rag-device "${RAG_GPU}" \
  --rag-top-k "${FIXED_RECALL_K}" \
  --rag-voting-k "${FIXED_VOTING_K}" \
  --rag-confidence-threshold "${FIXED_THRESHOLD}" \
  --vllm-segment-sec "${RAG_CHUNK_SIZE}" \
  --rag-min-terms 0 \
  --log-sample 3 \
  $( [[ "${USE_NO_TERM_MAP_NONE}" -eq 1 ]] && echo "--use-no-term-map-none" )

echo "[INFO] SimulEval Task ${TASK_ID} DONE"
