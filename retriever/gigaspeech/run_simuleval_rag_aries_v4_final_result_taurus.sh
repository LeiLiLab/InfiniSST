#!/bin/bash
#SBATCH --job-name=simuleval_rag_sweep_taurus
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
# Default array covers ALL langs (3) * chunks (4) = 12 tasks.
# If you only want to run one language, set ONLY_LANG (zh/de/ja) when submitting:
#   ONLY_LANG=zh sbatch this_script.sh
# Tasks outside the computed TOTAL_TASKS will exit 0 (no error).
#SBATCH --array=0
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_simuleval_rag_sweep_taurus.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_simuleval_rag_sweep_taurus.err

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

# ==================== Main result config ====================
# Goal: H=1, 3 LLM checkpoints, 4 chunk sizes -> 12 runs.
# We report StreamLAAL / BLEU / TERM_ACC / RTF, using the curated glossary.

# Indices must be built per (retriever checkpoint, glossary). If missing, build it first.
INDEX_CACHE_DIR="/mnt/gemini/data2/jiaxuanluo/index_cache_v4"
GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
# Chunk / Hop sweep:
# - chunk_size: 0.96, 1.92, 2.88, 3.84
# - hop_size:   0.48, 0.96, 1.44, 1.92 (aligned)
CHUNK_SIZES=(3.84)

# Retriever checkpoint (fixed for main results)
RAG_MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"

# Fixed settings
FIXED_THRESHOLD=0.0
H=1

# 3 LLM checkpoints (zh/de/ja)
MODELS=(
  # "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora-rank8/v0-20260103-221345-hf-v2"
  "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora-v2/v2-20251211-045306-hf"
  "/mnt/gemini/data2/jiaxuanluo/models/owaski/owaski_de_final"
  "/mnt/gemini/data2/jiaxuanluo/models/owaski/owaski_ja_final"
)
LANG_CODES=("zh")
TARGET_LANGS=("Chinese")
ONLY_LANG=zh
# Optional: run only one language for quick iteration.
# Supported: zh / de / ja
ONLY_LANG="${ONLY_LANG:-}"
if [ -n "${ONLY_LANG}" ]; then
  if [ "${ONLY_LANG}" != "zh" ] && [ "${ONLY_LANG}" != "de" ] && [ "${ONLY_LANG}" != "ja" ]; then
    echo "[ERROR] ONLY_LANG must be one of {zh,de,ja}, got: ${ONLY_LANG}"
    exit 2
  fi
  if [ "${ONLY_LANG}" == "zh" ]; then
    MODELS=("/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora-v2/rate0.3_k20_final")
    LANG_CODES=("zh")
    TARGET_LANGS=("Chinese")
  elif [ "${ONLY_LANG}" == "de" ]; then
    MODELS=("/mnt/gemini/data2/jiaxuanluo/models/owaski/owaski_de_final")
    LANG_CODES=("de")
    TARGET_LANGS=("German")
  elif [ "${ONLY_LANG}" == "ja" ]; then
    MODELS=("/mnt/gemini/data2/jiaxuanluo/models/owaski/owaski_ja_final")
    LANG_CODES=("ja")
    TARGET_LANGS=("Japanese")
  fi
  echo "[INFO] ONLY_LANG enabled: ${ONLY_LANG}"
fi

NUM_CHUNKS=${#CHUNK_SIZES[@]}
NUM_MODELS=${#MODELS[@]}

# Decode Task ID (NUM_MODELS * NUM_CHUNKS tasks)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CHUNK_IDX=$((TASK_ID % NUM_CHUNKS))
MODEL_IDX=$((TASK_ID / NUM_CHUNKS))

TOTAL_TASKS=$((NUM_MODELS * NUM_CHUNKS))
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
  echo "[INFO] TASK_ID out of range (skip): ${TASK_ID} (valid: 0..$((TOTAL_TASKS - 1)))"
  exit 0
fi

CUR_CHUNK="${CHUNK_SIZES[$CHUNK_IDX]}"
CUR_HOP=0.48

MODEL_NAME="${MODELS[$MODEL_IDX]}"
LANG_CODE="${LANG_CODES[$MODEL_IDX]}"
TARGET_LANG="${TARGET_LANGS[$MODEL_IDX]}"

# Prompt compatibility:
# - zh model expects "term_map:NONE" when no terms are provided (default agent behavior).
# - non-zh models are trained without this marker; omit it via --use-no-term-map-none.
USE_NO_TERM_MAP_NONE=0
if [ "${LANG_CODE}" != "zh" ]; then
  USE_NO_TERM_MAP_NONE=1
fi

# Per-language tokenizer + latency unit (align with baseline script)
if [ "${LANG_CODE}" == "zh" ]; then
  CUR_TOKENIZER="zh"
  CUR_LATENCY_UNIT="char"
elif [ "${LANG_CODE}" == "ja" ]; then
  CUR_TOKENIZER="ja-mecab"
  CUR_LATENCY_UNIT="char"
elif [ "${LANG_CODE}" == "de" ]; then
  CUR_TOKENIZER="13a"
  CUR_LATENCY_UNIT="word"
else
  echo "[ERROR] Unsupported LANG_CODE: ${LANG_CODE}"
  exit 2
fi

# H=1 => dynamic top_k / voting_k based on chunk size
TOPK_AND_VK=$(python3 - <<PY
import math
chunk=float("${CUR_CHUNK}")
k=max(1, int(math.ceil(1.0 * chunk)))
vk=max(1, k//2)
print(f"{k} {vk}")
PY
)
# FIXED_RECALL_K="$(echo "${TOPK_AND_VK}" | awk '{print $1}')"
# FIXED_VOTING_K="$(echo "${TOPK_AND_VK}" | awk '{print $2}')"

FIXED_RECALL_K=10
FIXED_VOTING_K=5

# Build / resolve index path for this (model, glossary)
MODEL_TAG="$(basename "${RAG_MODEL_PATH}" .pt)"
GLOSSARY_TAG="$(basename "${GLOSSARY_PATH}" .json)"
CUR_INDEX="${INDEX_CACHE_DIR}/${MODEL_TAG}__${GLOSSARY_TAG}__tr16.pkl"

# Index name tag for output folder (main results use curated glossary)
INDEX_NAME="curated"

echo "[INFO] Task ID: ${TASK_ID}"
echo "[INFO] INDEX: ${INDEX_NAME} (${CUR_INDEX})"
echo "[INFO] CHUNK_SIZE: ${CUR_CHUNK}"
echo "[INFO] HOP_SIZE: ${CUR_HOP}"
echo "[INFO] H_MULT: ${H} (fixed)"
echo "[INFO] RAG_MODEL_PATH: ${RAG_MODEL_PATH}"
echo "[INFO] MODEL_NAME: ${MODEL_NAME}"
echo "[INFO] LANG_CODE: ${LANG_CODE}"
echo "[INFO] TOKENIZER: ${CUR_TOKENIZER}"
echo "[INFO] LATENCY_UNIT: ${CUR_LATENCY_UNIT}"
echo "[INFO] USE_NO_TERM_MAP_NONE: ${USE_NO_TERM_MAP_NONE}"
echo "[INFO] FIXED_RECALL_K: ${FIXED_RECALL_K}"
echo "[INFO] FIXED_VOTING_K: ${FIXED_VOTING_K}"
echo "[INFO] FIXED_THRESHOLD: ${FIXED_THRESHOLD}"

# ==================== 路径与参数 ====================
ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
SOURCE_LANG="English"
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_sweep_v4_main_result_final_taurus"

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

MODEL_SHORT="$(basename "${MODEL_NAME}")"
OUTPUT_PATH="${OUTPUT_BASE}/${LANG_CODE}/${MODEL_SHORT}_${INDEX_NAME}_cs${CUR_CHUNK}_hs${CUR_HOP}_H${H}_rk${FIXED_RECALL_K}_vk${FIXED_VOTING_K}"
# Same setting should overwrite previous runs (deterministic)
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

# Use language-specific target if present; otherwise fall back to full text reference (one sentence per line).
if [ -f "${ROOT}/dev.target.${LANG_CODE}" ]; then
  cp "${ROOT}/dev.target.${LANG_CODE}" "${TARGET_LIST}"
else
  REF_FALLBACK="${ROOT}/dev/text/txt/ACL.6060.dev.en-xx.${LANG_CODE}.txt"
  if [ ! -f "${REF_FALLBACK}" ]; then
    echo "[ERROR] Missing target file for LANG_CODE=${LANG_CODE}: ${ROOT}/dev.target.${LANG_CODE} and ${REF_FALLBACK} not found"
    exit 3
  fi
  cp "${REF_FALLBACK}" "${TARGET_LIST}"
fi

# 修正音频路径
if grep -q "/mnt/data/siqiouyang" "${SOURCE_LIST}"; then
  sed -i 's|/mnt/data/siqiouyang|/mnt/taurus/data/siqiouyang|g' "${SOURCE_LIST}"
fi

# ==================== GPU 分配 ====================
# Prefer 3 GPUs:
# - vLLM uses TP=2 on cuda:0,1
# - RAG runs on a separate gpu cuda:2
# Fallback to 2 GPUs if only 2 are allocated: RAG will share cuda:1.
# GIDS=($(echo "${CUDA_VISIBLE_DEVICES:-0,1,2}" | tr ',' ' '))
# if [ "${#GIDS[@]}" -ge 3 ]; then
#   export CUDA_VISIBLE_DEVICES="${GIDS[0]},${GIDS[1]},${GIDS[2]}"
#   RAG_GPU="cuda:2"
# else
#   export CUDA_VISIBLE_DEVICES="${GIDS[0]},${GIDS[1]}"
#   RAG_GPU="cuda:1"
# fi

RAG_GPU="cuda:1"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO] RAG_GPU: ${RAG_GPU}"

# ==================== 运行 SimulEval ====================
LATENCY_MULTIPLIER=1
SRC_SEGMENT_SIZE=$((LATENCY_MULTIPLIER * 480))
# Scale decoding budget with vLLM call interval to avoid truncation when chunk_size increases.
# Baseline: 40 tokens per 0.96s.
MAX_NEW_TOKENS=$(python3 - <<PY
import math
lat=float("${LATENCY_MULTIPLIER}")
chunk=float("${CUR_CHUNK}")
base_tokens=40.0
base_sec=0.96
print(int(max(1, math.ceil(lat * base_tokens * (chunk / base_sec)))))
PY
)
echo "[INFO] MAX_NEW_TOKENS (scaled): ${MAX_NEW_TOKENS}"

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
  --eval-latency-unit "${CUR_LATENCY_UNIT}" \
  --sacrebleu-tokenizer "${CUR_TOKENIZER}" \
  --rag-index-path "${CUR_INDEX}" \
  --rag-model-path "${RAG_MODEL_PATH}" \
  --rag-target-lang "${LANG_CODE}" \
  --rag-chunk-size 1.92 \
  --rag-hop-size 0.48 \
  --rag-device "${RAG_GPU}" \
  --rag-top-k "${FIXED_RECALL_K}" \
  --rag-voting-k "${FIXED_VOTING_K}" \
  --rag-confidence-threshold 0.0 \
  --vllm-segment-sec "${CUR_CHUNK}" \
  --rag-min-terms 0 \
  --log-sample 3 \
  $( [[ "${USE_NO_TERM_MAP_NONE}" -eq 1 ]] && echo "--use-no-term-map-none" ) \
  2>&1 | tee "${OUTPUT_PATH}/simuleval.log"

echo "[INFO] SimulEval Task ${TASK_ID} DONE"

# ==================== Post-eval + summary (StreamLAAL / BLEU / TERM_ACC / RTF) ====================
REF_FILE="${ROOT}/dev/text/txt/ACL.6060.dev.en-xx.${LANG_CODE}.txt"
AUDIO_YAML="${ROOT}/dev.yaml"

EVAL_OUT="$(
python /home/jiaxuanluo/FBK-fairseq/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py \
  --simuleval-instances "${OUTPUT_PATH}/instances.log" \
  --reference "${REF_FILE}" \
  --audio-yaml "${AUDIO_YAML}" \
  --sacrebleu-tokenizer "${CUR_TOKENIZER}" \
  --latency-unit "${CUR_LATENCY_UNIT}" \
  --glossary "${GLOSSARY_PATH}" \
  --term-lang "${LANG_CODE}" \
  --term-mismatch-examples 0 2>&1
)"

METRIC_LINE="$(echo "${EVAL_OUT}" | awk 'match($0,/^[0-9]+\\.[0-9]+[[:space:]]+[0-9]+\\.[0-9]+[[:space:]]+[0-9]+\\.[0-9]+[[:space:]]*$/){print; exit}')"
BLEU="$(echo "${METRIC_LINE}" | awk '{print $1}')"
STREAM_LAAL="$(echo "${METRIC_LINE}" | awk '{print $2}')"
STREAM_LAAL_CA="$(echo "${METRIC_LINE}" | awk '{print $3}')"

TERM_LINE="$(echo "${EVAL_OUT}" | awk '/^TERM_ACC[[:space:]]/{print; exit}')"
TERM_ACC="$(echo "${TERM_LINE}" | awk '{print $2}')"
TERM_CORRECT="$(echo "${TERM_LINE}" | awk '{print $4}')"
TERM_TOTAL="$(echo "${TERM_LINE}" | awk '{print $6}')"

RTF_TOTAL="$(grep -oP 'rtf_total=\\K[0-9.]+' \"${OUTPUT_PATH}/simuleval.log\" | tail -n 1 || true)"

SUMMARY_TSV="${OUTPUT_BASE}/main_result_h1_summary.tsv"
mkdir -p "$(dirname "${SUMMARY_TSV}")"

# Append one line per run (locked)
{
  flock 200
  if [ ! -f "${SUMMARY_TSV}" ]; then
    echo -e "timestamp\\tlang\\tmodel\\tchunk_size\\thop_size\\ttop_k\\tvoting_k\\tBLEU\\tStreamLAAL\\tStreamLAAL_CA\\tTERM_ACC\\tTERM_CORRECT\\tTERM_TOTAL\\tRTF\\toutput_path" > "${SUMMARY_TSV}"
  fi
  echo -e "$(date +'%Y-%m-%d %H:%M:%S')\\t${LANG_CODE}\\t${MODEL_SHORT}\\t${CUR_CHUNK}\\t${CUR_HOP}\\t${FIXED_RECALL_K}\\t${FIXED_VOTING_K}\\t${BLEU}\\t${STREAM_LAAL}\\t${STREAM_LAAL_CA}\\t${TERM_ACC}\\t${TERM_CORRECT}\\t${TERM_TOTAL}\\t${RTF_TOTAL}\\t${OUTPUT_PATH}" >> "${SUMMARY_TSV}"
} 200>"${SUMMARY_TSV}.lock"

echo "[INFO] Summary appended: ${SUMMARY_TSV}"
