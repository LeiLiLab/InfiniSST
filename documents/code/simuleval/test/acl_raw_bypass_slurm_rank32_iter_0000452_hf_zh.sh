#!/usr/bin/env bash
set -euo pipefail

# Run SimulEval locally (no Slurm) for a fixed HF checkpoint (zh only).
# This script selects physical GPUs via CUDA_VISIBLE_DEVICES.

# ======Configuration=====
# Exit codes
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

# Repo
ROOT_DIR="/home/jiaxuanluo/InfiniSST"

# Optional: activate a conda env that has simuleval installed (recommended).
# Set CONDA_ENV_NAME empty to disable auto-activation.
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"
CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"

# Fixed model (rank32, iter_0000452 exported to HF)
MODEL_NAME="/mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107/keep1.0_r32/v3-20260108-045345/iter_0000452-hf"
# Dataset
DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
SOURCE_LANG="English"
TARGET_LANG="Chinese"
LANG_CODE="zh"

# Output
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_acl6060_glossary_acl_raw_test"
CLEAN_OUTPUT_DIR="1"

# GPU selection (physical GPU ids on this machine, e.g. "0,1" or "2,3")
CUDA_VISIBLE_DEVICES_PHYSICAL="6,7"

# vLLM call cadence
VLLM_SEGMENT_SEC="1.92"
# If set to 1, disable vLLM torch.compile/cudagraph paths (reduces startup latency; helps avoid rare init hangs).
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

# RAG
RAG_ENABLED="1"
RAG_MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"
#GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
INDEX_CACHE_DIR="/mnt/gemini/data2/jiaxuanluo/index_cache_v4"
RAG_CHUNK_SIZE="1.92"
RAG_HOP_SIZE="0.48"
RAG_TOP_K="10"
RAG_VOTING_K="5"
RAG_CONFIDENCE_THRESHOLD="0.0"
RAG_MIN_TERMS="0"

# Runtime toggles
USE_VLLM="1"

# Tokenizer/latency unit (zh)
SACREBLEU_TOKENIZER="zh"
LATENCY_UNIT="char"

# Decode
BEAM="1"
TEMPERATURE="0.6"
TOP_P="0.95"
TOP_K="20"
NO_REPEAT_NGRAM_LOOKBACK="100"
NO_REPEAT_NGRAM_SIZE="5"
MIN_START_SEC="0"

# Logging
DEBUG_LLM_IO="1"
DEBUG_LLM_IO_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/logs/vllm_logs/vllm_debug_rank32_iter_0000452.jsonl"

# vLLM memory
GPU_MEMORY_UTILIZATION="0.8"

# Cache window in agent (seconds -> chunks, derived in script)
MAX_CACHE_SECONDS="80.0"
KEEP_CACHE_SECONDS="60.0"
MIN_CACHE_CHUNKS="1"
# ======Configuration=====

echo "[INFO] CUDA_VISIBLE_DEVICES_PHYSICAL=${CUDA_VISIBLE_DEVICES_PHYSICAL}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] VLLM_SEGMENT_SEC=${VLLM_SEGMENT_SEC}"

cd "${ROOT_DIR}"
echo "[INFO] CWD=$(pwd)"

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    echo "[INFO] Activated conda env: ${CONDA_ENV_NAME}"
  else
    # Fallback: prepend env bin to PATH (may be enough if packages are installed in that env).
    if [[ -d "${CONDA_PREFIX}/bin" ]]; then
      export PATH="${CONDA_PREFIX}/bin:${PATH}"
      echo "[WARN] conda.sh not found; prepended PATH with: ${CONDA_PREFIX}/bin"
    else
      echo "[WARN] CONDA env bin not found: ${CONDA_PREFIX}/bin"
    fi
  fi
fi

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "[ERROR] ROOT_DIR not found: ${ROOT_DIR}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "[ERROR] HF model dir not found: ${MODEL_NAME}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

if [[ "${RAG_ENABLED}" == "1" ]] && [[ ! -f "${GLOSSARY_PATH}" ]]; then
  echo "[ERROR] GLOSSARY_PATH not found: ${GLOSSARY_PATH}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

if [[ "${RAG_ENABLED}" == "1" ]] && [[ ! -f "${RAG_MODEL_PATH}" ]]; then
  echo "[ERROR] RAG_MODEL_PATH not found: ${RAG_MODEL_PATH}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

SIMULEVAL_BIN="$(command -v simuleval || true)"
# Prefer python -m simuleval if available, otherwise fall back to simuleval binary.
SIMULEVAL_MODE="python_module"
if python -m simuleval --help >/dev/null 2>&1; then
  SIMULEVAL_MODE="python_module"
elif [[ -n "${SIMULEVAL_BIN}" ]]; then
  SIMULEVAL_MODE="binary"
else
  echo "[ERROR] simuleval not found. Install it in your current python env (or set CONDA_PREFIX to an env that has it)." >&2
  echo "[ERROR] Quick check: python -c 'import simuleval' should work." >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export VLLM_USE_V1="0"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="0"

# Use physical GPU ids directly.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_PHYSICAL}"

MODEL_TAG="$(basename "${RAG_MODEL_PATH}" .pt)"
GLOSSARY_TAG="$(basename "${GLOSSARY_PATH}" .json)"
CUR_INDEX="${INDEX_CACHE_DIR}/${MODEL_TAG}__${GLOSSARY_TAG}__tr16.pkl"

if [[ "${RAG_ENABLED}" == "1" ]] && [[ ! -f "${CUR_INDEX}" ]]; then
  echo "[ERROR] RAG index not found: ${CUR_INDEX}" >&2
  echo "[ERROR] Build it first (example):" >&2
  echo "[ERROR]   MODEL_PATH='${RAG_MODEL_PATH}' GLOSSARY_PATH='${GLOSSARY_PATH}' OUTPUT_PATH='${CUR_INDEX}' TARGET_LANG_CODE='${LANG_CODE}' bash ${ROOT_DIR}/retriever/gigaspeech/run_build_index_v4.sh" >&2
  exit "${EXIT_DATA_ERROR}"
fi

SRC_LIST="${DATA_ROOT}/dev.source"
TGT_LIST="${DATA_ROOT}/dev.target.${LANG_CODE}"
if [[ ! -f "${TGT_LIST}" ]]; then
  TGT_LIST="${DATA_ROOT}/dev.target.zh"
fi
if [[ ! -f "${SRC_LIST}" ]] || [[ ! -f "${TGT_LIST}" ]]; then
  echo "[ERROR] Missing source/target list: ${SRC_LIST} or ${TGT_LIST}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

# Derive cache chunk sizes from seconds and VLLM_SEGMENT_SEC.
MAX_CACHE_CHUNKS="$(
python3 - <<PY
chunk=float("${VLLM_SEGMENT_SEC}")
print(max(int("${MIN_CACHE_CHUNKS}"), int(float("${MAX_CACHE_SECONDS}") / chunk)))
PY
)"
KEEP_CACHE_CHUNKS="$(
python3 - <<PY
chunk=float("${VLLM_SEGMENT_SEC}")
print(max(int("${MIN_CACHE_CHUNKS}"), int(float("${KEEP_CACHE_SECONDS}") / chunk)))
PY
)"

MODEL_SHORT="$(basename "${MODEL_NAME}")"
OUTPUT_DIR="${OUTPUT_BASE}/${LANG_CODE}/${MODEL_SHORT}_cs${VLLM_SEGMENT_SEC}_hs${RAG_HOP_SIZE}_rk${RAG_TOP_K}_vk${RAG_VOTING_K}"
mkdir -p "${OUTPUT_DIR}"
if [[ "${CLEAN_OUTPUT_DIR}" == "1" ]]; then
  rm -rf "${OUTPUT_DIR:?}/"*
fi

echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] MAX_CACHE_CHUNKS=${MAX_CACHE_CHUNKS} KEEP_CACHE_CHUNKS=${KEEP_CACHE_CHUNKS}"
echo "[INFO] CUR_INDEX=${CUR_INDEX}"

# RAG GPU selection within process (after CUDA_VISIBLE_DEVICES remap).
# vLLM uses TP=2 => cuda:0,cuda:1. If a third GPU is visible, use it for RAG; else share cuda:1.
VISIBLE_GPU_COUNT="$(
python3 - <<PY
cvd="${CUDA_VISIBLE_DEVICES_PHYSICAL}".strip()
print(len([x for x in cvd.split(",") if x.strip() != ""]))
PY
)"
VLLM_TP_SIZE="2"
if [[ "${VISIBLE_GPU_COUNT}" -ge "$((VLLM_TP_SIZE + 1))" ]]; then
  RAG_GPU="cuda:${VLLM_TP_SIZE}"
else
  RAG_GPU="cuda:1"
fi
echo "[INFO] RAG_GPU=${RAG_GPU}"

# SimulEval segmenting: 480 samples per read => 0.03s at 16k, matches previous setup.
LATENCY_MULTIPLIER="1"
SRC_SEGMENT_SIZE="$((LATENCY_MULTIPLIER * 480))"
MAX_NEW_TOKENS="$((LATENCY_MULTIPLIER * 40))"
echo "[INFO] SRC_SEGMENT_SIZE=${SRC_SEGMENT_SIZE} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"

if [[ "${SIMULEVAL_MODE}" == "python_module" ]]; then
  SIM_CMD=(python -u -m simuleval)
else
  SIM_CMD=(python -u "${SIMULEVAL_BIN}")
fi

# Use absolute agent path to avoid dependence on caller CWD.
AGENT_FILE="${ROOT_DIR}/agents/infinisst_omni_vllm_rag_v4.py"
AGENT_CLASS="agents.infinisst_omni_vllm_rag_v4.InfiniSSTOmniVLLMRAGV4"
if [[ ! -f "${AGENT_FILE}" ]]; then
  echo "[ERROR] Agent file not found: ${AGENT_FILE}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

"${SIM_CMD[@]}" \
  --agent "${AGENT_FILE}" \
  --agent-class "${AGENT_CLASS}" \
  --source "${SRC_LIST}" \
  --target "${TGT_LIST}" \
  --output "${OUTPUT_DIR}" \
  --source-segment-size "${SRC_SEGMENT_SIZE}" \
  --source-lang "${SOURCE_LANG}" \
  --target-lang "${TARGET_LANG}" \
  --min-start-sec "${MIN_START_SEC}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --beam "${BEAM}" \
  --no-repeat-ngram-lookback "${NO_REPEAT_NGRAM_LOOKBACK}" \
  --no-repeat-ngram-size "${NO_REPEAT_NGRAM_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --use-vllm "${USE_VLLM}" \
  --vllm-enforce-eager "${VLLM_ENFORCE_EAGER}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --model-name "${MODEL_NAME}" \
  --max-cache-chunks "${MAX_CACHE_CHUNKS}" \
  --keep-cache-chunks "${KEEP_CACHE_CHUNKS}" \
  --quality-metrics BLEU \
  --eval-latency-unit "${LATENCY_UNIT}" \
  --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
  $( [[ "${RAG_ENABLED}" == "1" ]] && echo "--rag-enabled" ) \
  $( [[ "${RAG_ENABLED}" == "1" ]] && echo "--rag-index-path" "${CUR_INDEX}" ) \
  $( [[ "${RAG_ENABLED}" == "1" ]] && echo "--rag-model-path" "${RAG_MODEL_PATH}" ) \
  $( [[ "${RAG_ENABLED}" == "1" ]] && echo "--rag-target-lang" "${LANG_CODE}" ) \
  $( [[ "${RAG_ENABLED}" == "1" ]] && echo "--rag-chunk-size" "${RAG_CHUNK_SIZE}" ) \
  $( [[ "${RAG_ENABLED}" == "1" ]] && echo "--rag-hop-size" "${RAG_HOP_SIZE}" ) \
  $( [[ "${RAG_ENABLED}" == "1" ]] && echo "--rag-device" "${RAG_GPU}" ) \
  $( [[ "${RAG_ENABLED}" == "1" ]] && echo "--rag-top-k" "${RAG_TOP_K}" ) \
  $( [[ "${RAG_ENABLED}" == "1" ]] && echo "--rag-voting-k" "${RAG_VOTING_K}" ) \
  $( [[ "${RAG_ENABLED}" == "1" ]] && echo "--rag-confidence-threshold" "${RAG_CONFIDENCE_THRESHOLD}" ) \
  --vllm-segment-sec "${VLLM_SEGMENT_SEC}" \
  --rag-min-terms "${RAG_MIN_TERMS}" \
  $( [[ "${DEBUG_LLM_IO}" == "1" ]] && echo "--debug-llm-io" ) \
  $( [[ "${DEBUG_LLM_IO}" == "1" ]] && echo "--debug-llm-io-file" "${DEBUG_LLM_IO_FILE}" ) \
  2>&1 | tee "${OUTPUT_DIR}/simuleval.log"

echo "[INFO] Done. Output: ${OUTPUT_DIR}"


