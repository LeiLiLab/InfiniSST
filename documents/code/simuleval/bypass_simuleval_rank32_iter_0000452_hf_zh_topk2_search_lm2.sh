#!/usr/bin/env bash
set -euo pipefail

# Run SimulEval locally (no Slurm) for Top-K2 (recall_k) search around 10 at fixed latency=2x.
# Fixed: vllm_segment_sec=1.92 (2 * 0.96), hop_size=0.48, voting_k=5.
# All user-facing strings are in English.

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
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_topk2_search_lm2"
CLEAN_OUTPUT_DIR="1"

# GPU selection (physical GPU ids on this machine, e.g. "0,1" or "2,3")
CUDA_VISIBLE_DEVICES_PHYSICAL="6,7"

# vLLM call cadence (fixed for latency=2x)
VLLM_SEGMENT_SEC="1.92"
# If set to 1, disable vLLM torch.compile/cudagraph paths (reduces startup latency; helps avoid rare init hangs).
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

# Runtime toggles
USE_VLLM="1"
VLLM_USE_V1="0"
NCCL_P2P_DISABLE="1"
NCCL_IB_DISABLE="1"
VLLM_ALLOW_RUNTIME_LORA_UPDATING="0"

# RAG
RAG_ENABLED="1"
RAG_MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"
GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
INDEX_CACHE_DIR="/mnt/gemini/data2/jiaxuanluo/index_cache_v4"
RAG_CHUNK_SIZE="1.92"
RAG_HOP_SIZE="0.48"
RAG_K2_VALUES=("6" "7" "8" "9" "10" "11" "12" "13" "14")
RAG_K1_FIXED="5"
RAG_CONFIDENCE_THRESHOLD="0.0"
RAG_MIN_TERMS="0"

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
DEBUG_LLM_IO_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/logs/vllm_logs/vllm_debug_rank32_iter_0000452_topk2_search_lm2.jsonl"

# vLLM memory
GPU_MEMORY_UTILIZATION="0.8"

# Cache window in agent (seconds -> chunks, derived in script)
MAX_CACHE_SECONDS="80.0"
KEEP_CACHE_SECONDS="60.0"
MIN_CACHE_CHUNKS="1"

# SimulEval audio segment size is fixed (480 samples per read => 0.03s at 16k).
SRC_SEGMENT_SIZE_SAMPLES="480"
MAX_NEW_TOKENS="40"

# vLLM tensor parallel (for selecting RAG GPU)
VLLM_TP_SIZE="2"
RAG_EXTRA_GPU_COUNT="1"
RAG_FALLBACK_GPU_INDEX="1"
# ======Configuration=====

echo "[INFO] CUDA_VISIBLE_DEVICES_PHYSICAL=${CUDA_VISIBLE_DEVICES_PHYSICAL}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] VLLM_SEGMENT_SEC=${VLLM_SEGMENT_SEC}"
echo "[INFO] RAG_K2_VALUES=${RAG_K2_VALUES[*]} RAG_K1_FIXED=${RAG_K1_FIXED}"

cd "${ROOT_DIR}"
echo "[INFO] CWD=$(pwd)"

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    echo "[INFO] Activated conda env: ${CONDA_ENV_NAME}"
  else
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

SIMULEVAL_BIN="$(command -v simuleval || true)"
SIMULEVAL_MODE="python_module"
if python -m simuleval --help >/dev/null 2>&1; then
  SIMULEVAL_MODE="python_module"
elif [[ -n "${SIMULEVAL_BIN}" ]]; then
  SIMULEVAL_MODE="binary"
else
  echo "[ERROR] simuleval not found. Install it in your current python env (or set CONDA_PREFIX to an env that has it)." >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export VLLM_USE_V1="${VLLM_USE_V1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_PHYSICAL}"

MODEL_TAG="$(basename "${RAG_MODEL_PATH}" .pt)"
GLOSSARY_TAG="$(basename "${GLOSSARY_PATH}" .json)"
CUR_INDEX="${INDEX_CACHE_DIR}/${MODEL_TAG}__${GLOSSARY_TAG}__tr16.pkl"

if [[ "${RAG_ENABLED}" == "1" ]] && [[ ! -f "${CUR_INDEX}" ]]; then
  echo "[ERROR] RAG index not found: ${CUR_INDEX}" >&2
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

VISIBLE_GPU_COUNT="$(
python3 - <<PY
cvd="${CUDA_VISIBLE_DEVICES_PHYSICAL}".strip()
print(len([x for x in cvd.split(",") if x.strip() != ""]))
PY
)"
if [[ "${VISIBLE_GPU_COUNT}" -ge "$((VLLM_TP_SIZE + RAG_EXTRA_GPU_COUNT))" ]]; then
  RAG_GPU="cuda:${VLLM_TP_SIZE}"
else
  RAG_GPU="cuda:${RAG_FALLBACK_GPU_INDEX}"
fi
echo "[INFO] RAG_GPU=${RAG_GPU}"

if [[ "${SIMULEVAL_MODE}" == "python_module" ]]; then
  SIM_CMD=(python -u -m simuleval)
else
  SIM_CMD=(python -u "${SIMULEVAL_BIN}")
fi

AGENT_FILE="${ROOT_DIR}/agents/infinisst_omni_vllm_rag_v4.py"
AGENT_CLASS="agents.infinisst_omni_vllm_rag_v4.InfiniSSTOmniVLLMRAGV4"
if [[ ! -f "${AGENT_FILE}" ]]; then
  echo "[ERROR] Agent file not found: ${AGENT_FILE}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

MODEL_SHORT="$(basename "${MODEL_NAME}")"
mkdir -p "${OUTPUT_BASE}/${LANG_CODE}"

for RAG_K2 in "${RAG_K2_VALUES[@]}"; do
  OUTPUT_DIR="${OUTPUT_BASE}/${LANG_CODE}/${MODEL_SHORT}_cs${VLLM_SEGMENT_SEC}_hs${RAG_HOP_SIZE}_lm2_rk${RAG_K2}_vk${RAG_K1_FIXED}"
  mkdir -p "${OUTPUT_DIR}"
  if [[ "${CLEAN_OUTPUT_DIR}" == "1" ]]; then
    rm -rf "${OUTPUT_DIR:?}/"*
  fi

  echo "[INFO] ============================================================"
  echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
  echo "[INFO] RAG_K2=${RAG_K2} RAG_K1_FIXED=${RAG_K1_FIXED}"
  echo "[INFO] SRC_SEGMENT_SIZE_SAMPLES=${SRC_SEGMENT_SIZE_SAMPLES} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
  echo "[INFO] MAX_CACHE_CHUNKS=${MAX_CACHE_CHUNKS} KEEP_CACHE_CHUNKS=${KEEP_CACHE_CHUNKS}"

  SIM_ARGS=(
    --agent "${AGENT_FILE}"
    --agent-class "${AGENT_CLASS}"
    --source "${SRC_LIST}"
    --target "${TGT_LIST}"
    --output "${OUTPUT_DIR}"
    --source-segment-size "${SRC_SEGMENT_SIZE_SAMPLES}"
    --source-lang "${SOURCE_LANG}"
    --target-lang "${TARGET_LANG}"
    --min-start-sec "${MIN_START_SEC}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --beam "${BEAM}"
    --no-repeat-ngram-lookback "${NO_REPEAT_NGRAM_LOOKBACK}"
    --no-repeat-ngram-size "${NO_REPEAT_NGRAM_SIZE}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --top-k "${TOP_K}"
    --use-vllm "${USE_VLLM}"
    --vllm-enforce-eager "${VLLM_ENFORCE_EAGER}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --model-name "${MODEL_NAME}"
    --max-cache-chunks "${MAX_CACHE_CHUNKS}"
    --keep-cache-chunks "${KEEP_CACHE_CHUNKS}"
    --quality-metrics BLEU
    --eval-latency-unit "${LATENCY_UNIT}"
    --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}"
    --vllm-segment-sec "${VLLM_SEGMENT_SEC}"
  )

  if [[ "${RAG_ENABLED}" == "1" ]]; then
    SIM_ARGS+=(
      --rag-enabled
      --rag-index-path "${CUR_INDEX}"
      --rag-model-path "${RAG_MODEL_PATH}"
      --rag-target-lang "${LANG_CODE}"
      --rag-chunk-size "${RAG_CHUNK_SIZE}"
      --rag-hop-size "${RAG_HOP_SIZE}"
      --rag-device "${RAG_GPU}"
      --rag-top-k "${RAG_K2}"
      --rag-voting-k "${RAG_K1_FIXED}"
      --rag-confidence-threshold "${RAG_CONFIDENCE_THRESHOLD}"
      --rag-min-terms "${RAG_MIN_TERMS}"
    )
  fi

  if [[ "${DEBUG_LLM_IO}" == "1" ]]; then
    SIM_ARGS+=(
      --debug-llm-io
      --debug-llm-io-file "${DEBUG_LLM_IO_FILE}"
    )
  fi

  "${SIM_CMD[@]}" "${SIM_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/simuleval.log"
done

echo "[INFO] Done. Output base: ${OUTPUT_BASE}"


