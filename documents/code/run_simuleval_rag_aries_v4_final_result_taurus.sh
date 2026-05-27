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
# Note: You can also override the array range at submit time, e.g.:
#   sbatch --array=0-3 this_script.sh
# Tasks outside the computed TOTAL_TASKS will exit 0 (no error).
#SBATCH --array=0-3
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_simuleval_rag_sweep_taurus.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%A_%a_simuleval_rag_sweep_taurus.err

set -euo pipefail

# ==================== 环境配置 ====================
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# Post-eval dependency (StreamLAAL/TERM_ACC script uses mwerSegmenter)
# If not set, stream_laal_term.py will assert and the script will exit before writing summary.
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
export MWERSEGMENTER_ROOT
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

# Post-eval script root (fallback to /home if /mnt/taurus/home is not available on the node)
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
if [ ! -f "${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py" ]; then
  FBK_FAIRSEQ_ROOT="/home/jiaxuanluo/FBK-fairseq"
fi

# vLLM 配置
export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=0
# If set to 1, disable vLLM torch.compile/cudagraph paths (reduces startup latency; helps avoid rare init hangs).
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"

# Optional: blacklist physical GPU ids (as shown in CUDA_VISIBLE_DEVICES set by Slurm).
# Example: DISABLE_GPU_IDS="1,2,3"
DISABLE_GPU_IDS="${DISABLE_GPU_IDS:-}"

# ==================== Main result config ====================
# Goal: H=1, 3 LLM checkpoints, 4 chunk sizes -> 12 runs.
# We report StreamLAAL / BLEU / TERM_ACC / RTF, using the curated glossary.

# Indices must be built per (retriever checkpoint, glossary). If missing, build it first.
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/index_cache_v4}"
GLOSSARY_PATH="${GLOSSARY_PATH_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json}"
# Chunk / Hop sweep:
# - chunk_size: 0.96, 1.92, 2.88, 3.84
# - hop_size:   0.48 (fixed)
if [ -n "${CHUNK_SIZES_OVERRIDE:-}" ]; then
  read -r -a CHUNK_SIZES <<< "${CHUNK_SIZES_OVERRIDE}"
else
  CHUNK_SIZES=(0.96 1.92 2.88 3.84)
fi

# Retriever checkpoint (fixed for main results)
RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt}"

# Fixed settings
FIXED_THRESHOLD=0.0
H=1

# 3 LLM checkpoints (zh/de/ja)
MODELS=(
  # "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora-rank8/v0-20260103-221345-hf-v2"
  "/mnt/gemini/data2/jiaxuanluo/models/owaski/gigaspeech-zh-m_v4_final_merged_shuffled_sample0.5_NONE"
  "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora-hope-final/v3-20260102-235532-hf"
  "/mnt/gemini/data2/jiaxuanluo/models/owaski/gigaspeech-zh"
  "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora-v2/v2-20251211-045306-hf"

  #"/mnt/gemini/data2/jiaxuanluo/models/owaski/owaski_de_final"
  #"/mnt/gemini/data2/jiaxuanluo/models/owaski/owaski_ja_final"
)
# Optional: run only one language for quick iteration.
# Supported: zh / de / ja
ONLY_LANG="${ONLY_LANG:-}"
if [ -z "${ONLY_LANG}" ]; then
  # Default: run all languages (must align with MODELS)
  LANG_CODES=("zh" "de" "ja")
  TARGET_LANGS=("Chinese" "German" "Japanese")
else
  if [ "${ONLY_LANG}" != "zh" ] && [ "${ONLY_LANG}" != "de" ] && [ "${ONLY_LANG}" != "ja" ]; then
    echo "[ERROR] ONLY_LANG must be one of {zh,de,ja}, got: ${ONLY_LANG}"
    exit 2
  fi
  if [ "${ONLY_LANG}" == "zh" ]; then
    # Allow overriding model path for quick experiments (e.g., sampling sweeps).
    # Usage:
    #   ONLY_LANG=zh MODEL_NAME_OVERRIDE=/path/to/model-hf sbatch this_script.sh
    if [ -n "${MODEL_NAME_OVERRIDE:-}" ]; then
      MODELS=("${MODEL_NAME_OVERRIDE}")
    else
      MODELS=(
          "/mnt/gemini/data2/jiaxuanluo/models/owaski/gigaspeech-zh-m_v4_final_merged_shuffled_sample0.5_NONE"
  "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora-hope-final/v3-20260102-235532-hf"
  "/mnt/gemini/data2/jiaxuanluo/models/owaski/gigaspeech-zh"
  "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora-v2/v2-20251211-045306-hf"
        "/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct-lora-v2/rate0.3_k20_final")
    fi
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
# TODO test this
USE_NO_TERM_MAP_NONE=1
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

FIXED_RECALL_K="${FIXED_RECALL_K:-5}"
FIXED_VOTING_K="${FIXED_VOTING_K:-10}"

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
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_sweep_v4_main_result_final_taurus}"

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
# NOTE:
# - Slurm sets CUDA_VISIBLE_DEVICES to allocated *physical* GPU ids (e.g., "1,2,3").
# - Inside the process, torch/vLLM sees them as cuda:0, cuda:1, ... (remapped indices).
#
# This section optionally filters out blacklisted physical GPU ids, then chooses:
# - vLLM uses TP=VLLM_TP_SIZE on cuda:0..cuda:(TP-1)
# - RAG uses cuda:TP if available, otherwise shares cuda:1

if [ -n "${DISABLE_GPU_IDS}" ]; then
  OLD_CVD="${CUDA_VISIBLE_DEVICES:-}"
  NEW_CVD="$(DISABLE_GPU_IDS="${DISABLE_GPU_IDS}" CUDA_VISIBLE_DEVICES="${OLD_CVD}" python3 - <<'PY'
import os
cvd=(os.environ.get("CUDA_VISIBLE_DEVICES","") or "").strip()
ban=(os.environ.get("DISABLE_GPU_IDS","") or "").strip()
ban_set=set(x.strip() for x in ban.split(",") if x.strip()!="")
kept=[x.strip() for x in cvd.split(",") if x.strip()!="" and x.strip() not in ban_set]
print(",".join(kept))
PY
)"
  if [ -z "${NEW_CVD}" ]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES filtered to empty. old='${OLD_CVD}' ban='${DISABLE_GPU_IDS}'" >&2
    echo "[ERROR] Hint: resubmit to get different GPUs/nodes, or request more GPUs." >&2
    exit 3
  fi
  export CUDA_VISIBLE_DEVICES="${NEW_CVD}"
  echo "[INFO] CUDA_VISIBLE_DEVICES filtered: old='${OLD_CVD}' ban='${DISABLE_GPU_IDS}' new='${CUDA_VISIBLE_DEVICES}'"
fi

GIDS=($(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' ' '))
if [ "${#GIDS[@]}" -lt "${VLLM_TP_SIZE}" ]; then
  echo "[ERROR] Not enough GPUs after filtering. Need VLLM_TP_SIZE=${VLLM_TP_SIZE}, got CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'"
  echo "[ERROR] Hint: resubmit to get different GPUs/nodes, or request more GPUs."
  exit 3
fi

# Choose RAG device
if [ "${#GIDS[@]}" -ge "$((VLLM_TP_SIZE + 1))" ]; then
  RAG_GPU="cuda:${VLLM_TP_SIZE}"
else
  # Share with the second vLLM GPU (cuda:1) when only TP GPUs are available.
  RAG_GPU="cuda:1"
fi

echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO] VLLM_TP_SIZE: ${VLLM_TP_SIZE}"
echo "[INFO] RAG_GPU: ${RAG_GPU}"

# ==================== 运行 SimulEval ====================
LATENCY_MULTIPLIER=1
SRC_SEGMENT_SIZE=$((LATENCY_MULTIPLIER * 480))
# Scale decoding budget with vLLM call interval to avoid truncation when chunk_size increases.
# Baseline: 40 tokens per 0.96s.
MAX_NEW_TOKENS=$((LATENCY_MULTIPLIER * 40))
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
  --vllm-enforce-eager "${VLLM_ENFORCE_EAGER}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --model-name "${MODEL_NAME}" \
  --max-cache-chunks "${MAX_CACHE_CHUNKS}" \
  --keep-cache-chunks "${KEEP_CACHE_CHUNKS}" \
  \
  --quality-metrics BLEU \
  --eval-latency-unit "${CUR_LATENCY_UNIT}" \
  --sacrebleu-tokenizer "${CUR_TOKENIZER}" \
  --rag-enabled \
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
  --debug-llm-io \
  --debug-llm-io-file "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/logs/vllm_logs/vllm_debug.jsonl" \
  --rag-min-terms 0 \
  --log-sample 3 \
  2>&1 | tee "${OUTPUT_PATH}/simuleval.log"

#$( [[ "${USE_NO_TERM_MAP_NONE}" -eq 1 ]] && echo "--use-no-term-map-none" ) \

echo "[INFO] SimulEval Task ${TASK_ID} DONE"

# ==================== Post-eval + summary (StreamLAAL / BLEU / TERM_ACC / RTF) ====================
REF_FILE="${ROOT}/dev/text/txt/ACL.6060.dev.en-xx.${LANG_CODE}.txt"
SOURCE_TEXT_FILE="${ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt"
AUDIO_YAML="${ROOT}/dev.yaml"

EVAL_OUT="$(
python "${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py" \
  --simuleval-instances "${OUTPUT_PATH}/instances.log" \
  --reference "${REF_FILE}" \
  --audio-yaml "${AUDIO_YAML}" \
  --sacrebleu-tokenizer "${CUR_TOKENIZER}" \
  --latency-unit "${CUR_LATENCY_UNIT}" \
  --glossary "${GLOSSARY_PATH}" \
  --term-lang "${LANG_CODE}" \
  --term-mismatch-examples 0 2>&1
)"

# Save post-eval output for debugging / reproducibility.
echo "${EVAL_OUT}" > "${OUTPUT_PATH}/post_eval.log"

TERM_ADOPTION_OUT="$(
python "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_sst_eval/compute_sentence_term_adoption.py" \
  --instances-log "${OUTPUT_PATH}/instances.log" \
  --source-file "${SOURCE_TEXT_FILE}" \
  --reference-file "${REF_FILE}" \
  --glossary-path "${GLOSSARY_PATH}" \
  --target-lang "${LANG_CODE}" \
  --output-json "${OUTPUT_PATH}/term_adoption.json" 2>&1 || true
)"
echo "${TERM_ADOPTION_OUT}" > "${OUTPUT_PATH}/term_adoption.log"

# Append raw post-eval output into TSV safely (single TSV field):
# encode as JSON string so tabs/newlines are escaped.
EVAL_OUT_JSON="$(
  printf '%s' "${EVAL_OUT}" | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read(), ensure_ascii=False))'
)"

# Robust metric parsing:
# Prefer the first line whose first 3 fields are numeric (int/float), then take them as:
#   BLEU StreamLAAL StreamLAAL_CA
METRIC_LINE="$(
echo "${EVAL_OUT}" | awk '
function isnum(x){ return (x ~ /^[0-9]+(\.[0-9]+)?$/) }
NF>=3 && isnum($1) && isnum($2) && isnum($3) { print $1"\t"$2"\t"$3; exit }
'
)"
BLEU="$(echo "${METRIC_LINE}" | awk -F'\t' '{print $1}')"
STREAM_LAAL="$(echo "${METRIC_LINE}" | awk -F'\t' '{print $2}')"
STREAM_LAAL_CA="$(echo "${METRIC_LINE}" | awk -F'\t' '{print $3}')"

# Fallback: parse from labeled lines if the numeric triple-line format is absent.
if [ -z "${BLEU}" ]; then
  BLEU="$(echo "${EVAL_OUT}" | grep -oP '(?i)\bBLEU\b[[:space:]]*[:=][[:space:]]*\K[0-9]+(\.[0-9]+)?' | head -n 1 || true)"
fi
if [ -z "${STREAM_LAAL}" ]; then
  STREAM_LAAL="$(echo "${EVAL_OUT}" | grep -oP '(?i)\bStreamLAAL\b[[:space:]]*[:=][[:space:]]*\K[0-9]+(\.[0-9]+)?' | head -n 1 || true)"
fi
if [ -z "${STREAM_LAAL_CA}" ]; then
  STREAM_LAAL_CA="$(echo "${EVAL_OUT}" | grep -oP '(?i)\bStreamLAAL_CA\b[[:space:]]*[:=][[:space:]]*\K[0-9]+(\.[0-9]+)?' | head -n 1 || true)"
fi

if [ -z "${BLEU}" ] || [ -z "${STREAM_LAAL}" ] || [ -z "${STREAM_LAAL_CA}" ]; then
  echo "[WARN] Failed to parse BLEU/StreamLAAL metrics. See: ${OUTPUT_PATH}/post_eval.log"
fi

TERM_LINE="$(echo "${EVAL_OUT}" | awk '/^TERM_ACC[[:space:]]/{print; exit}')"
TERM_ACC="$(echo "${TERM_LINE}" | awk '{print $2}')"
TERM_CORRECT="$(echo "${TERM_LINE}" | awk '{print $4}')"
TERM_TOTAL="$(echo "${TERM_LINE}" | awk '{print $6}')"

TERM_ADOPTION_LINE="$(echo "${TERM_ADOPTION_OUT}" | awk '/^TERM_ADOPTION[[:space:]]/{print; exit}')"
TERM_ADOPTION="$(echo "${TERM_ADOPTION_LINE}" | awk '{print $2}')"
TERM_ADOPTED="$(echo "${TERM_ADOPTION_LINE}" | awk '{print $4}')"
TERM_ADOPTION_TOTAL="$(echo "${TERM_ADOPTION_LINE}" | awk '{print $6}')"
TERM_ADOPTION_SENTENCES="$(echo "${TERM_ADOPTION_LINE}" | awk '{print $8}')"
TERM_ADOPTION_MICRO="$(echo "${TERM_ADOPTION_LINE}" | awk '{print $10}')"

RTF_TOTAL="$(grep -oP 'rtf_total=\\K[0-9.]+' \"${OUTPUT_PATH}/simuleval.log\" | tail -n 1 || true)"

SUMMARY_TSV="${OUTPUT_BASE}/main_result_h1_summary.tsv"
mkdir -p "$(dirname "${SUMMARY_TSV}")"

# Append one line per run (locked)
{
  flock 200
  if [ ! -f "${SUMMARY_TSV}" ]; then
    echo -e "timestamp\\tlang\\tmodel\\tchunk_size\\thop_size\\ttop_k\\tvoting_k\\tBLEU\\tStreamLAAL\\tStreamLAAL_CA\\tTERM_ACC\\tTERM_CORRECT\\tTERM_TOTAL\\tTERM_ADOPTION\\tTERM_ADOPTED\\tTERM_ADOPTION_TOTAL\\tTERM_ADOPTION_SENTENCES\\tTERM_ADOPTION_MICRO\\tRTF\\toutput_path\\tpost_eval_raw_json" > "${SUMMARY_TSV}"
  fi
  echo -e "$(date +'%Y-%m-%d %H:%M:%S')\\t${LANG_CODE}\\t${MODEL_SHORT}\\t${CUR_CHUNK}\\t${CUR_HOP}\\t${FIXED_RECALL_K}\\t${FIXED_VOTING_K}\\t${BLEU}\\t${STREAM_LAAL}\\t${STREAM_LAAL_CA}\\t${TERM_ACC}\\t${TERM_CORRECT}\\t${TERM_TOTAL}\\t${TERM_ADOPTION}\\t${TERM_ADOPTED}\\t${TERM_ADOPTION_TOTAL}\\t${TERM_ADOPTION_SENTENCES}\\t${TERM_ADOPTION_MICRO}\\t${RTF_TOTAL}\\t${OUTPUT_PATH}\\t${EVAL_OUT_JSON}" >> "${SUMMARY_TSV}"
} 200>"${SUMMARY_TSV}.lock"

echo "[INFO] Summary appended: ${SUMMARY_TSV}"
