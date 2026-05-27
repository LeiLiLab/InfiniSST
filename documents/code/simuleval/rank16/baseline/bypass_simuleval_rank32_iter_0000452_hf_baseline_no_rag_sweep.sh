#!/usr/bin/env bash
set -euo pipefail

# Baseline (NO RAG) SimulEval local sweep (rank16 HF model).
#
# This script is intentionally compatible with the per-paper wrapper that passes:
# - GLOSSARY_PATHS_OVERRIDE (used ONLY for output directory naming / bookkeeping)
# - SRC_LIST_OVERRIDE / TGT_LIST_OVERRIDE
# - MODEL_NAME_OVERRIDE / OUTPUT_BASE_OVERRIDE
#
# IMPORTANT:
# - RAG is disabled: we do NOT pass --rag-enabled nor any --rag-* args to SimulEval.
# - All user-facing strings are in English.

# ======Configuration=====
# Exit codes
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

# Resume behavior
RESUME_MODE="${RESUME_MODE:-0}"
BACKUP_PARTIAL_RUNS="${BACKUP_PARTIAL_RUNS:-1}"
BACKUP_DIR_SUFFIX_PREFIX="partial_backup"
BACKUP_TIMESTAMP_FORMAT="%Y%m%d_%H%M%S"
INSTANCES_FILE_NAME="instances.log"

# Repo
ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"

# Optional: activate a conda env that has simuleval installed.
CONDA_BASE="${CONDA_BASE_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME_OVERRIDE:-spaCyEnv}"
DEFAULT_CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
if [[ -n "${CONDA_PREFIX_OVERRIDE:-}" ]]; then
  CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE}"
elif [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV_NAME}" ]]; then
  CONDA_PREFIX="${CONDA_PREFIX}"
else
  CONDA_PREFIX="${DEFAULT_CONDA_PREFIX}"
fi

# Default model (rank16 exported to HF)
MODEL_NAME_DEFAULT="/mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107/keep1.0_r16/v3-20260121-021342-hf"
MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE:-}"

# Dataset
DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/taurus/data/siqiouyang/datasets/acl6060}"
SOURCE_LANG="English"
TARGET_LANG="Chinese"
LANG_CODE="zh"
LANG_CODE_OVERRIDE="${LANG_CODE_OVERRIDE:-}"
TARGET_LANG_OVERRIDE="${TARGET_LANG_OVERRIDE:-}"

# Optional dataset list overrides (for per-talk / per-paper runs)
SRC_LIST_OVERRIDE="${SRC_LIST_OVERRIDE:-}"
TGT_LIST_OVERRIDE="${TGT_LIST_OVERRIDE:-}"

# Output
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rank16_baseline_no_rag"
CLEAN_OUTPUT_DIR="1"

# Optional overrides (space-separated lists).
LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIERS_OVERRIDE:-}"
RAG_K2_VALUES_OVERRIDE="${RAG_K2_VALUES_OVERRIDE:-}"      # accepted for API compatibility; used for output naming only
GLOSSARY_PATHS_OVERRIDE="${GLOSSARY_PATHS_OVERRIDE:-}"    # accepted for API compatibility; used for output naming only
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE:-}"
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-}"

# GPU selection (physical GPU ids on this machine, e.g. "0,1" or "2,3")
CUDA_VISIBLE_DEVICES_PHYSICAL="6,7"
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-}"

# vLLM segment sec schedule (latency multiplier * base)
BASE_VLLM_SEGMENT_SEC="0.96"
LATENCY_MULTIPLIERS=("1" "2" "3" "4")
VLLM_SEGMENT_SEC_FORMAT_DECIMALS="2"

# If set to 1, disable vLLM torch.compile/cudagraph paths.
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

# Glossary list (optional, for naming / bookkeeping only)
GLOSSARY_PATHS=(
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
)

# These are kept to preserve output directory naming compatibility.
RAG_CHUNK_SIZE="1.92"
RAG_HOP_SIZE="0.48"
RAG_K2_VALUES=("10")
RAG_K1_FIXED="10"
RAG_CONFIDENCE_THRESHOLD="0.0"

# Tokenizer/latency unit (will be set based on LANG_CODE)
SACREBLEU_TOKENIZER=""
LATENCY_UNIT=""

# Decode
BEAM="1"
TEMPERATURE="0.6"
TOP_P="0.95"
TOP_K="20"
NO_REPEAT_NGRAM_LOOKBACK="100"
NO_REPEAT_NGRAM_SIZE="5"
MIN_START_SEC="0"

# Determinism / overrides
DEFAULT_GEN_SEED="998244353"
SEED_OVERRIDE="${SEED_OVERRIDE:-}"
TEMPERATURE_OVERRIDE="${TEMPERATURE_OVERRIDE:-}"

# Logging
DEBUG_LLM_IO="0"
DEBUG_LLM_IO_FILE_NAME="vllm_debug.jsonl"
DEBUG_LLM_IO_OVERRIDE="${DEBUG_LLM_IO_OVERRIDE:-}"

# vLLM memory
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.8}"

# Cache window in agent (seconds -> chunks, derived in script)
MAX_CACHE_SECONDS="${MAX_CACHE_SECONDS_OVERRIDE:-80.0}"
KEEP_CACHE_SECONDS="${KEEP_CACHE_SECONDS_OVERRIDE:-60.0}"
MIN_CACHE_CHUNKS="1"

# SimulEval audio segment size is fixed (480 samples per read => 0.03s at 16k).
SRC_SEGMENT_SIZE_SAMPLES="480"
MAX_NEW_TOKENS="40"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-}"

# vLLM tensor parallel (affects vLLM init only)
VLLM_TP_SIZE="${VLLM_TP_SIZE_OVERRIDE:-2}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN_OVERRIDE:-32768}"
VLLM_LIMIT_AUDIO="${VLLM_LIMIT_AUDIO_OVERRIDE:-}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0}"

# Runtime toggles
USE_VLLM="1"
VLLM_USE_V1="0"
NCCL_P2P_DISABLE="1"
NCCL_IB_DISABLE="1"
VLLM_ALLOW_RUNTIME_LORA_UPDATING="0"
# ======Configuration=====

if [[ -n "${OUTPUT_BASE_OVERRIDE}" ]]; then
  OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE}"
fi

MODEL_NAME="${MODEL_NAME_DEFAULT}"
if [[ -n "${MODEL_NAME_OVERRIDE}" ]]; then
  MODEL_NAME="${MODEL_NAME_OVERRIDE}"
fi

if [[ -n "${CLEAN_OUTPUT_DIR_OVERRIDE}" ]]; then
  CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR_OVERRIDE}"
fi

if [[ -n "${LATENCY_MULTIPLIERS_OVERRIDE}" ]]; then
  # shellcheck disable=SC2206
  LATENCY_MULTIPLIERS=(${LATENCY_MULTIPLIERS_OVERRIDE})
fi

# We keep K2 / glossary overrides ONLY for naming compatibility.
if [[ -n "${RAG_K2_VALUES_OVERRIDE}" ]]; then
  # shellcheck disable=SC2206
  RAG_K2_VALUES=(${RAG_K2_VALUES_OVERRIDE})
fi
if [[ -n "${GLOSSARY_PATHS_OVERRIDE}" ]]; then
  # shellcheck disable=SC2206
  GLOSSARY_PATHS=(${GLOSSARY_PATHS_OVERRIDE})
fi

if [[ -n "${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}" ]]; then
  CUDA_VISIBLE_DEVICES_PHYSICAL="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}"
fi

if [[ -n "${LANG_CODE_OVERRIDE}" ]]; then
  LANG_CODE="${LANG_CODE_OVERRIDE}"
fi

if [[ -n "${MAX_NEW_TOKENS_OVERRIDE}" ]]; then
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS_OVERRIDE}"
fi

if [[ -n "${DEBUG_LLM_IO_OVERRIDE}" ]]; then
  DEBUG_LLM_IO="${DEBUG_LLM_IO_OVERRIDE}"
fi

SEED="${DEFAULT_GEN_SEED}"
if [[ -n "${SEED_OVERRIDE}" ]]; then
  SEED="${SEED_OVERRIDE}"
fi
if [[ -n "${TEMPERATURE_OVERRIDE}" ]]; then
  TEMPERATURE="${TEMPERATURE_OVERRIDE}"
fi

# Decide TARGET_LANG from override or LANG_CODE.
if [[ -n "${TARGET_LANG_OVERRIDE}" ]]; then
  TARGET_LANG="${TARGET_LANG_OVERRIDE}"
else
  if [[ "${LANG_CODE}" == "zh" ]]; then
    TARGET_LANG="Chinese"
  elif [[ "${LANG_CODE}" == "ja" ]]; then
    TARGET_LANG="Japanese"
  elif [[ "${LANG_CODE}" == "de" ]]; then
    TARGET_LANG="German"
  else
    echo "[ERROR] Unsupported LANG_CODE: ${LANG_CODE} (set TARGET_LANG_OVERRIDE if needed)" >&2
    exit "${EXIT_CONFIG_ERROR}"
  fi
fi

# Tokenizer/latency unit (baseline-aligned)
if [[ "${LANG_CODE}" == "zh" ]]; then
  SACREBLEU_TOKENIZER="zh"
  LATENCY_UNIT="char"
elif [[ "${LANG_CODE}" == "ja" ]]; then
  SACREBLEU_TOKENIZER="ja-mecab"
  LATENCY_UNIT="char"
elif [[ "${LANG_CODE}" == "de" ]]; then
  SACREBLEU_TOKENIZER="13a"
  LATENCY_UNIT="word"
else
  echo "[ERROR] Unsupported LANG_CODE for tokenizer/latency unit: ${LANG_CODE}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

echo "[INFO] CUDA_VISIBLE_DEVICES_PHYSICAL=${CUDA_VISIBLE_DEVICES_PHYSICAL}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] LANG_CODE=${LANG_CODE} TARGET_LANG=${TARGET_LANG} TOKENIZER=${SACREBLEU_TOKENIZER} LATENCY_UNIT=${LATENCY_UNIT}"
echo "[INFO] BASE_VLLM_SEGMENT_SEC=${BASE_VLLM_SEGMENT_SEC}"
echo "[INFO] LATENCY_MULTIPLIERS=${LATENCY_MULTIPLIERS[*]}"
echo "[INFO] RESUME_MODE=${RESUME_MODE} CLEAN_OUTPUT_DIR=${CLEAN_OUTPUT_DIR} BACKUP_PARTIAL_RUNS=${BACKUP_PARTIAL_RUNS}"
echo "[INFO] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} MAX_CACHE_SECONDS=${MAX_CACHE_SECONDS} KEEP_CACHE_SECONDS=${KEEP_CACHE_SECONDS}"
echo "[INFO] VLLM_TP_SIZE=${VLLM_TP_SIZE} VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN} VLLM_LIMIT_AUDIO=${VLLM_LIMIT_AUDIO:-auto} VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE}"
echo "[INFO] Baseline mode: RAG is disabled."

cd "${ROOT_DIR}"
echo "[INFO] CWD=$(pwd)"

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  # On aries, conda.sh can exist but contain node-local hardcoded /home paths.
  # Prefer direct PATH setup when the target env is available.
  if [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
    echo "[INFO] Prepended conda env directly: ${CONDA_PREFIX}"
  elif [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    echo "[INFO] Activated conda env: ${CONDA_ENV_NAME}"
  else
    echo "[WARN] CONDA env bin not found: ${CONDA_PREFIX}/bin"
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
export VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE}"
export VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE}"
if [[ -n "${VLLM_LIMIT_AUDIO}" ]]; then
  export VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_PHYSICAL}"

SRC_LIST="${DATA_ROOT}/dev.source"
TGT_LIST="${DATA_ROOT}/dev.target.${LANG_CODE}"
if [[ ! -f "${TGT_LIST}" ]]; then
  TGT_LIST="${DATA_ROOT}/dev.target.zh"
fi
if [[ -n "${SRC_LIST_OVERRIDE}" ]]; then
  SRC_LIST="${SRC_LIST_OVERRIDE}"
fi
if [[ -n "${TGT_LIST_OVERRIDE}" ]]; then
  TGT_LIST="${TGT_LIST_OVERRIDE}"
fi

if [[ ! -f "${SRC_LIST}" ]] || [[ ! -f "${TGT_LIST}" ]]; then
  echo "[ERROR] Missing source/target list: ${SRC_LIST} or ${TGT_LIST}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

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

for GLOSSARY_PATH in "${GLOSSARY_PATHS[@]}"; do
  # In baseline mode, glossary path is only used for output naming (paper bookkeeping).
  GLOSSARY_TAG="$(basename "${GLOSSARY_PATH}" .json)"

  for LATENCY_MULTIPLIER in "${LATENCY_MULTIPLIERS[@]}"; do
    VLLM_SEGMENT_SEC="$(
    python3 - <<PY
base=float("${BASE_VLLM_SEGMENT_SEC}")
mult=float("${LATENCY_MULTIPLIER}")
decimals=int("${VLLM_SEGMENT_SEC_FORMAT_DECIMALS}")
print(f"{base*mult:.{decimals}f}")
PY
    )"

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

    for RAG_K2 in "${RAG_K2_VALUES[@]}"; do
      THRESHOLD_TAG="${RAG_CONFIDENCE_THRESHOLD//./p}"
      OUTPUT_DIR="${OUTPUT_BASE}/${LANG_CODE}/${MODEL_SHORT}_g${GLOSSARY_TAG}_cs${VLLM_SEGMENT_SEC}_hs${RAG_HOP_SIZE}_lm${LATENCY_MULTIPLIER}_k2${RAG_K2}_k1${RAG_K1_FIXED}_th${THRESHOLD_TAG}"
      INSTANCES_PATH="${OUTPUT_DIR}/${INSTANCES_FILE_NAME}"

      if [[ "${RESUME_MODE}" == "1" ]]; then
        if [[ -f "${INSTANCES_PATH}" ]] && [[ -s "${INSTANCES_PATH}" ]]; then
          echo "[INFO] Skip (already has non-empty ${INSTANCES_FILE_NAME}): ${OUTPUT_DIR}"
          continue
        fi

        if [[ -d "${OUTPUT_DIR}" ]] && [[ "${BACKUP_PARTIAL_RUNS}" == "1" ]]; then
          TS="$(date +${BACKUP_TIMESTAMP_FORMAT})"
          BACKUP_DIR="${OUTPUT_DIR}_${BACKUP_DIR_SUFFIX_PREFIX}_${TS}"
          echo "[WARN] Incomplete run detected (missing/empty ${INSTANCES_FILE_NAME}). Backing up to: ${BACKUP_DIR}"
          mv "${OUTPUT_DIR}" "${BACKUP_DIR}"
        fi

        mkdir -p "${OUTPUT_DIR}"
      else
        mkdir -p "${OUTPUT_DIR}"
        if [[ "${CLEAN_OUTPUT_DIR}" == "1" ]]; then
          rm -rf "${OUTPUT_DIR:?}/"*
        fi
      fi

      DEBUG_LLM_IO_FILE="${OUTPUT_DIR}/${DEBUG_LLM_IO_FILE_NAME}"

      echo "[INFO] ============================================================"
      echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
      echo "[INFO] Baseline mode: RAG disabled (no term map / no retrieval)."
      echo "[INFO] LATENCY_MULTIPLIER=${LATENCY_MULTIPLIER} VLLM_SEGMENT_SEC=${VLLM_SEGMENT_SEC}"
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
        --seed "${SEED}"
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
        --runtime-log-dir "${OUTPUT_DIR}"
      )

      if [[ "${DEBUG_LLM_IO}" == "1" ]]; then
        SIM_ARGS+=(
          --debug-llm-io
          --debug-llm-io-file "${DEBUG_LLM_IO_FILE}"
        )
      fi

      "${SIM_CMD[@]}" "${SIM_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/simuleval.log"
    done
  done
done

echo "[INFO] Done. Output base: ${OUTPUT_BASE}"
