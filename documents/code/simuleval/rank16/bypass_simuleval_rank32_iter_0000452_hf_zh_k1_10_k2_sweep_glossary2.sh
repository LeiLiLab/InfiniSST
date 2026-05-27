#!/usr/bin/env bash
set -euo pipefail

# Run SimulEval locally (no Slurm) for:
# - latency multipliers: 1,2,3,4 (implemented via VLLM_SEGMENT_SEC = 0.96 * multiplier)
# - fixed K1 (voting_k) = 10
# - sweep K2 (recall_k) in {5,10,15,20}
# - two glossaries: ACL6060 raw glossary + paper extracted glossary
#
# All user-facing strings are in English.

# ======Configuration=====
# Exit codes
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

# Resume behavior
# - RESUME_MODE=1: skip runs that already have a non-empty instances.log; re-run only missing/empty ones.
# - CLEAN_OUTPUT_DIR=1 will delete existing outputs. For resume, set CLEAN_OUTPUT_DIR_OVERRIDE=0 (recommended).
RESUME_MODE="${RESUME_MODE:-0}"
BACKUP_PARTIAL_RUNS="${BACKUP_PARTIAL_RUNS:-1}"
BACKUP_DIR_SUFFIX_PREFIX="partial_backup"
BACKUP_TIMESTAMP_FORMAT="%Y%m%d_%H%M%S"
INSTANCES_FILE_NAME="instances.log"

# Repo
ROOT_DIR="/home/jiaxuanluo/InfiniSST"

# Optional: activate a conda env that has simuleval installed (recommended).
# Set CONDA_ENV_NAME empty to disable auto-activation.
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"
CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"

# Default model (rank16 exported to HF)
MODEL_NAME_DEFAULT="/mnt/gemini/data/jiaxuanluo/Omni-30B-sampling-0107/keep1.0_r16/v3-20260121-021342-hf"
MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE:-}"

# Dataset
DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
SOURCE_LANG="English"
TARGET_LANG="Chinese"
LANG_CODE="zh"
LANG_CODE_OVERRIDE="${LANG_CODE_OVERRIDE:-}"
TARGET_LANG_OVERRIDE="${TARGET_LANG_OVERRIDE:-}"

# Optional dataset list overrides (for per-talk / per-paper runs)
SRC_LIST_OVERRIDE="${SRC_LIST_OVERRIDE:-}"
TGT_LIST_OVERRIDE="${TGT_LIST_OVERRIDE:-}"

# Output
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank16_v3-20260121-021342-hf_zh_k1_10_k2_sweep_glossary2"
CLEAN_OUTPUT_DIR="1"

# Optional overrides (space-separated lists). Useful for splitting the sweep across machines.
# Examples:
#   LATENCY_MULTIPLIERS_OVERRIDE="1"                  (run only latency multiplier 1)
#   RAG_K2_VALUES_OVERRIDE="5 10"                    (run only K2 in {5,10})
#   GLOSSARY_PATHS_OVERRIDE="/path/a.json /path/b.json" (run only selected glossaries)
#   OUTPUT_BASE_OVERRIDE="/mnt/.../my_custom_output_base"
#
# Model override:
#   MODEL_NAME_OVERRIDE="/path/to/my_hf_model_dir"
LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIERS_OVERRIDE:-}"
RAG_K2_VALUES_OVERRIDE="${RAG_K2_VALUES_OVERRIDE:-}"
GLOSSARY_PATHS_OVERRIDE="${GLOSSARY_PATHS_OVERRIDE:-}"
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE:-}"
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-}"
RAG_CONFIDENCE_THRESHOLD_OVERRIDE="${RAG_CONFIDENCE_THRESHOLD_OVERRIDE:-}"
RAG_CONFIDENCE_THRESHOLD_MODE_OVERRIDE="${RAG_CONFIDENCE_THRESHOLD_MODE_OVERRIDE:-}"
RAG_ENABLED_OVERRIDE="${RAG_ENABLED_OVERRIDE:-}"

# GPU selection (physical GPU ids on this machine, e.g. "0,1" or "2,3")
CUDA_VISIBLE_DEVICES_PHYSICAL="6,7"
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-}"

# vLLM segment sec schedule (latency multiplier * base)
BASE_VLLM_SEGMENT_SEC="0.96"
LATENCY_MULTIPLIERS=("1" "2" "3" "4")
VLLM_SEGMENT_SEC_FORMAT_DECIMALS="2"

# If set to 1, disable vLLM torch.compile/cudagraph paths (reduces startup latency; helps avoid rare init hangs).
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

# RAG
DEFAULT_RAG_ENABLED="1"
RAG_ENABLED="${DEFAULT_RAG_ENABLED}"
RAG_MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/final_main_result_model_v1.pt"
INDEX_CACHE_DIR="/mnt/gemini/data2/jiaxuanluo/index_cache_v4"
# RAG index naming:
# - "lang": include LANG_CODE in index filename (recommended to avoid cross-lang reuse)
# - "legacy": use old naming (no lang tag)
RAG_INDEX_NAMING_MODE="lang"
# If 1, build RAG index automatically when missing.
AUTO_BUILD_RAG_INDEX="1"
RAG_BUILD_INDEX_SCRIPT="${ROOT_DIR}/retriever/gigaspeech/run_build_index_v4.sh"

# Glossaries (two settings)
GLOSSARY_PATHS=(
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
)

RAG_CHUNK_SIZE="1.92"
RAG_HOP_SIZE="0.48"

# K naming:
# - K2: recall top-k (simuleval arg: --rag-top-k)
# - K1: voting-k (simuleval arg: --rag-voting-k)
RAG_K2_VALUES=("5" "10" "15" "20")
RAG_K1_FIXED="10"

RAG_CONFIDENCE_THRESHOLD="0.0"
RAG_CONFIDENCE_THRESHOLD_MODE="absolute"
RAG_MIN_TERMS="0"

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

# Logging
DEBUG_LLM_IO="0"
DEBUG_LLM_IO_FILE_NAME="vllm_debug.jsonl"
DEBUG_LLM_IO_OVERRIDE="${DEBUG_LLM_IO_OVERRIDE:-}"

# vLLM memory
GPU_MEMORY_UTILIZATION="0.8"

# Cache window in agent (seconds -> chunks, derived in script)
MAX_CACHE_SECONDS="80.0"
KEEP_CACHE_SECONDS="60.0"
MIN_CACHE_CHUNKS="1"

# SimulEval audio segment size is fixed (480 samples per read => 0.03s at 16k).
SRC_SEGMENT_SIZE_SAMPLES="480"
MAX_NEW_TOKENS="40"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-}"

# vLLM tensor parallel (for selecting RAG GPU)
VLLM_TP_SIZE="2"
RAG_EXTRA_GPU_COUNT="1"
RAG_FALLBACK_GPU_INDEX="1"

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

if [[ -n "${RAG_K2_VALUES_OVERRIDE}" ]]; then
  # shellcheck disable=SC2206
  RAG_K2_VALUES=(${RAG_K2_VALUES_OVERRIDE})
fi

if [[ -n "${GLOSSARY_PATHS_OVERRIDE}" ]]; then
  # shellcheck disable=SC2206
  GLOSSARY_PATHS=(${GLOSSARY_PATHS_OVERRIDE})
fi
if [[ -n "${RAG_CONFIDENCE_THRESHOLD_OVERRIDE}" ]]; then
  RAG_CONFIDENCE_THRESHOLD="${RAG_CONFIDENCE_THRESHOLD_OVERRIDE}"
fi
if [[ -n "${RAG_CONFIDENCE_THRESHOLD_MODE_OVERRIDE}" ]]; then
  RAG_CONFIDENCE_THRESHOLD_MODE="${RAG_CONFIDENCE_THRESHOLD_MODE_OVERRIDE}"
fi
if [[ -n "${RAG_ENABLED_OVERRIDE}" ]]; then
  RAG_ENABLED="${RAG_ENABLED_OVERRIDE}"
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
echo "[INFO] RAG_ENABLED=${RAG_ENABLED}"
echo "[INFO] RAG_K2_VALUES=${RAG_K2_VALUES[*]} RAG_K1_FIXED=${RAG_K1_FIXED}"
echo "[INFO] RAG_CONFIDENCE_THRESHOLD=${RAG_CONFIDENCE_THRESHOLD} MODE=${RAG_CONFIDENCE_THRESHOLD_MODE}"
echo "[INFO] GLOSSARY_PATHS=${GLOSSARY_PATHS[*]}"
echo "[INFO] RESUME_MODE=${RESUME_MODE} CLEAN_OUTPUT_DIR=${CLEAN_OUTPUT_DIR} BACKUP_PARTIAL_RUNS=${BACKUP_PARTIAL_RUNS}"

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
if [[ "${RAG_ENABLED}" == "1" ]] && [[ ! -f "${RAG_MODEL_PATH}" ]]; then
  echo "[ERROR] RAG_MODEL_PATH not found: ${RAG_MODEL_PATH}" >&2
  exit "${EXIT_DATA_ERROR}"
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

for GLOSSARY_PATH in "${GLOSSARY_PATHS[@]}"; do
  if [[ "${RAG_ENABLED}" == "1" ]] && [[ ! -f "${GLOSSARY_PATH}" ]]; then
    echo "[ERROR] GLOSSARY_PATH not found: ${GLOSSARY_PATH}" >&2
    exit "${EXIT_DATA_ERROR}"
  fi

  MODEL_TAG="$(basename "${RAG_MODEL_PATH}" .pt)"
  GLOSSARY_TAG="$(basename "${GLOSSARY_PATH}" .json)"
INDEX_LANG_TAG=""
if [[ "${RAG_INDEX_NAMING_MODE}" == "lang" ]]; then
  INDEX_LANG_TAG="__${LANG_CODE}"
elif [[ "${RAG_INDEX_NAMING_MODE}" == "legacy" ]]; then
  INDEX_LANG_TAG=""
else
  echo "[ERROR] Unsupported RAG_INDEX_NAMING_MODE: ${RAG_INDEX_NAMING_MODE}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
CUR_INDEX="${INDEX_CACHE_DIR}/${MODEL_TAG}__${GLOSSARY_TAG}${INDEX_LANG_TAG}__tr16.pkl"

  if [[ "${RAG_ENABLED}" == "1" ]] && [[ ! -f "${CUR_INDEX}" ]]; then
    if [[ "${AUTO_BUILD_RAG_INDEX}" == "1" ]]; then
      if [[ ! -f "${RAG_BUILD_INDEX_SCRIPT}" ]]; then
        echo "[ERROR] RAG build script not found: ${RAG_BUILD_INDEX_SCRIPT}" >&2
        exit "${EXIT_CONFIG_ERROR}"
      fi

      echo "[WARN] RAG index not found. Building now: ${CUR_INDEX}" >&2
      MODEL_PATH="${RAG_MODEL_PATH}" \
      GLOSSARY_PATH="${GLOSSARY_PATH}" \
      OUTPUT_PATH="${CUR_INDEX}" \
      TARGET_LANG_CODE="${LANG_CODE}" \
      bash "${RAG_BUILD_INDEX_SCRIPT}"

      if [[ ! -f "${CUR_INDEX}" ]]; then
        echo "[ERROR] RAG index build finished but output not found: ${CUR_INDEX}" >&2
        exit "${EXIT_DATA_ERROR}"
      fi
    else
      echo "[ERROR] RAG index not found: ${CUR_INDEX}" >&2
      echo "[ERROR] Build it first (example):" >&2
      echo "[ERROR]   MODEL_PATH='${RAG_MODEL_PATH}' GLOSSARY_PATH='${GLOSSARY_PATH}' OUTPUT_PATH='${CUR_INDEX}' TARGET_LANG_CODE='${LANG_CODE}' bash ${RAG_BUILD_INDEX_SCRIPT}" >&2
      exit "${EXIT_DATA_ERROR}"
    fi
  fi

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
      echo "[INFO] GLOSSARY_TAG=${GLOSSARY_TAG}"
      echo "[INFO] LATENCY_MULTIPLIER=${LATENCY_MULTIPLIER} VLLM_SEGMENT_SEC=${VLLM_SEGMENT_SEC}"
      echo "[INFO] K2=${RAG_K2} K1=${RAG_K1_FIXED}"
      echo "[INFO] MAX_CACHE_CHUNKS=${MAX_CACHE_CHUNKS} KEEP_CACHE_CHUNKS=${KEEP_CACHE_CHUNKS}"
      echo "[INFO] CUR_INDEX=${CUR_INDEX}"

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
          --rag-confidence-threshold-mode "${RAG_CONFIDENCE_THRESHOLD_MODE}"
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
  done
done

echo "[INFO] Done. Output base: ${OUTPUT_BASE}"



