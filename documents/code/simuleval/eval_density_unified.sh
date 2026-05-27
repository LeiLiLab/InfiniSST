#!/usr/bin/env bash
set -euo pipefail

# Unified evaluation pipeline for one Speech LLM model + one glossary + one latency_multiplier.
#
# Supports both tagged (acl6060, 5 talks) and per-paper (1 talk) modes.
#
# Flow:
#   1. Determine glossary path (default or override)
#   2. Build MaxSim text embeddings index (.pt) if not cached
#   3. Run SimulEval with MaxSim RAG agent
#   4. Run offline_streamlaal_eval.py on instances.log
#
# All user-facing strings are in English.

# ======Configuration=====
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"

CONDA_BASE="${CONDA_BASE:-/mnt/taurus/home/jiaxuanluo/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-spaCyEnv}"

# Dataset
DATA_ROOT="${DATA_ROOT:-/mnt/taurus/data/siqiouyang/datasets/acl6060}"
SOURCE_LANG="English"
TARGET_LANG="Chinese"
LANG_CODE="zh"

# MaxSim retriever checkpoint
RAG_MODEL_PATH="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"
RAG_LORA_R="128"
RAG_TEXT_LORA_R="128"
RAG_TEXT_LORA_ALPHA="256"

# Default glossary
GLOSSARY_ACL6060="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060.json"

# Index cache
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-${INDEX_CACHE_DIR:-/mnt/gemini/data2/jiaxuanluo/maxsim_index_cache}}"

# Output
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim"

# GPU selection
CUDA_VISIBLE_DEVICES_PHYSICAL="2,3,4"

# vLLM
BASE_VLLM_SEGMENT_SEC="0.96"
LATENCY_MULTIPLIER="1"
VLLM_ENFORCE_EAGER="1"
GPU_MEMORY_UTILIZATION="0.8"
USE_VLLM="1"

# Tokenizer/latency
SACREBLEU_TOKENIZER="zh"
LATENCY_UNIT="char"

# Decode
BEAM="1"
TEMPERATURE="0.6"
TOP_P="0.95"
TOP_K_DECODE="20"
MIN_START_SEC="0"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS_OVERRIDE:-40}"

# Cache
MAX_CACHE_SECONDS="80.0"
KEEP_CACHE_SECONDS="60.0"
MIN_CACHE_CHUNKS="1"

# Eval mode for offline_streamlaal_eval.py
EVAL_MODE="acl6060"

# Offline eval
OFFLINE_EVAL_SCRIPT="${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT_OVERRIDE:-${FBK_FAIRSEQ_ROOT:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}}"
STREAM_LAAL_TOOL_REL="${STREAM_LAAL_TOOL_REL_OVERRIDE:-${STREAM_LAAL_TOOL_REL:-examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py}}"

# Override env vars (set by caller)
MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE:-}"
RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH_OVERRIDE:-}"
RAG_TOP_K_OVERRIDE="${RAG_TOP_K_OVERRIDE:-10}"
EVAL_MODE_OVERRIDE="${EVAL_MODE_OVERRIDE:-}"
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE:-}"
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-}"
LATENCY_MULTIPLIER_OVERRIDE="${LATENCY_MULTIPLIER_OVERRIDE:-}"
RAG_LORA_R_OVERRIDE="${RAG_LORA_R_OVERRIDE:-}"
RAG_TEXT_LORA_R_OVERRIDE="${RAG_TEXT_LORA_R_OVERRIDE:-}"
RAG_TEXT_LORA_ALPHA_OVERRIDE="${RAG_TEXT_LORA_ALPHA_OVERRIDE:-}"
RAG_MAXSIM_WINDOWS_OVERRIDE="${RAG_MAXSIM_WINDOWS_OVERRIDE:-2 3 4 5 6 7 8 10 12 16 20 24}"
RAG_MAXSIM_STRIDE_OVERRIDE="${RAG_MAXSIM_STRIDE_OVERRIDE:-2}"
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-0}"
RAG_RETRIEVE_STRIDE_SEC_OVERRIDE="${RAG_RETRIEVE_STRIDE_SEC_OVERRIDE:-}"
RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE:-1.92}"
RAG_STREAMING_MODE_OVERRIDE="${RAG_STREAMING_MODE_OVERRIDE:-timeline}"
RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD_OVERRIDE:-0.73}"
TERM_MAP_FORMAT_OVERRIDE="${TERM_MAP_FORMAT_OVERRIDE:-plain}"
ORACLE_TERM_MAP_PATH_OVERRIDE="${ORACLE_TERM_MAP_PATH_OVERRIDE:-}"
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-}"
USE_VLLM_OVERRIDE="${USE_VLLM_OVERRIDE:-}"
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_OVERRIDE:-${TMPDIR:-/mnt/gemini/data1/jiaxuanluo/tmp}}"
DENSITY_TAG="${DENSITY_TAG:-default}"
SOURCE_LANG_OVERRIDE="${SOURCE_LANG_OVERRIDE:-}"
TARGET_LANG_OVERRIDE="${TARGET_LANG_OVERRIDE:-}"
LANG_CODE_OVERRIDE="${LANG_CODE_OVERRIDE:-}"
SACREBLEU_TOKENIZER_OVERRIDE="${SACREBLEU_TOKENIZER_OVERRIDE:-}"
LATENCY_UNIT_OVERRIDE="${LATENCY_UNIT_OVERRIDE:-}"

# Per-paper overrides
GLOSSARY_PATH_OVERRIDE="${GLOSSARY_PATH_OVERRIDE:-}"
EVAL_GLOSSARY_PATH_OVERRIDE="${EVAL_GLOSSARY_PATH_OVERRIDE:-}"
SRC_LIST_OVERRIDE="${SRC_LIST_OVERRIDE:-}"
TGT_LIST_OVERRIDE="${TGT_LIST_OVERRIDE:-}"
PAPER_ID_TAG="${PAPER_ID_TAG:-}"
REF_FILE_OVERRIDE="${REF_FILE_OVERRIDE:-}"
SOURCE_TEXT_FILE_OVERRIDE="${SOURCE_TEXT_FILE_OVERRIDE:-}"
AUDIO_YAML_OVERRIDE="${AUDIO_YAML_OVERRIDE:-}"

# Skip offline eval (orchestrator handles it)
SKIP_OFFLINE_EVAL="${SKIP_OFFLINE_EVAL:-0}"
TERM_FCR_POLICY="${TERM_FCR_POLICY:-term_map_if_available}"
STRIP_OUTPUT_TAGS="${STRIP_OUTPUT_TAGS_OVERRIDE:-none}"
EMPTY_TERM_MAP_POLICY="${EMPTY_TERM_MAP_POLICY_OVERRIDE:-none_block}"
SYSTEM_PROMPT_STYLE="${SYSTEM_PROMPT_STYLE_OVERRIDE:-translate_task}"

# Pre-built index override (skip auto-build)
INDEX_PATH_OVERRIDE="${INDEX_PATH_OVERRIDE:-}"
# ======Configuration=====

# Apply overrides
if [[ -n "${LANG_CODE_OVERRIDE}" ]]; then
  LANG_CODE="${LANG_CODE_OVERRIDE}"
  case "${LANG_CODE}" in
    zh)
      TARGET_LANG="Chinese"
      SACREBLEU_TOKENIZER="zh"
      LATENCY_UNIT="char"
      ;;
    ja)
      TARGET_LANG="Japanese"
      SACREBLEU_TOKENIZER="ja-mecab"
      LATENCY_UNIT="char"
      ;;
    de)
      TARGET_LANG="German"
      SACREBLEU_TOKENIZER="13a"
      LATENCY_UNIT="word"
      ;;
    *)
      echo "[ERROR] Unsupported LANG_CODE_OVERRIDE=${LANG_CODE}" >&2
      exit "${EXIT_CONFIG_ERROR}"
      ;;
  esac
fi
if [[ -n "${SOURCE_LANG_OVERRIDE}" ]]; then
  SOURCE_LANG="${SOURCE_LANG_OVERRIDE}"
fi
if [[ -n "${TARGET_LANG_OVERRIDE}" ]]; then
  TARGET_LANG="${TARGET_LANG_OVERRIDE}"
fi
if [[ -n "${SACREBLEU_TOKENIZER_OVERRIDE}" ]]; then
  SACREBLEU_TOKENIZER="${SACREBLEU_TOKENIZER_OVERRIDE}"
fi
if [[ -n "${LATENCY_UNIT_OVERRIDE}" ]]; then
  LATENCY_UNIT="${LATENCY_UNIT_OVERRIDE}"
fi

MODEL_NAME="${MODEL_NAME_OVERRIDE}"
if [[ -z "${MODEL_NAME}" ]]; then
  echo "[ERROR] MODEL_NAME_OVERRIDE is required." >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

if [[ -n "${RAG_MODEL_PATH_OVERRIDE}" ]]; then
  RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE}"
fi
RAG_TOP_K="${RAG_TOP_K_OVERRIDE}"
if [[ -n "${EVAL_MODE_OVERRIDE}" ]]; then
  EVAL_MODE="${EVAL_MODE_OVERRIDE}"
fi
if [[ -n "${OUTPUT_BASE_OVERRIDE}" ]]; then
  OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}" ]]; then
  CUDA_VISIBLE_DEVICES_PHYSICAL="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}"
fi
if [[ -n "${LATENCY_MULTIPLIER_OVERRIDE}" ]]; then
  LATENCY_MULTIPLIER="${LATENCY_MULTIPLIER_OVERRIDE}"
fi
if [[ -n "${RAG_LORA_R_OVERRIDE}" ]]; then
  RAG_LORA_R="${RAG_LORA_R_OVERRIDE}"
fi
if [[ -n "${RAG_TEXT_LORA_R_OVERRIDE}" ]]; then
  RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE}"
fi
if [[ -n "${RAG_TEXT_LORA_ALPHA_OVERRIDE}" ]]; then
  RAG_TEXT_LORA_ALPHA="${RAG_TEXT_LORA_ALPHA_OVERRIDE}"
fi
if [[ -n "${GPU_MEMORY_UTILIZATION_OVERRIDE}" ]]; then
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_OVERRIDE}"
fi
if [[ -n "${USE_VLLM_OVERRIDE}" ]]; then
  USE_VLLM="${USE_VLLM_OVERRIDE}"
fi
if [[ -n "${MAX_CACHE_SECONDS_OVERRIDE:-}" ]]; then
  MAX_CACHE_SECONDS="${MAX_CACHE_SECONDS_OVERRIDE}"
fi
if [[ -n "${KEEP_CACHE_SECONDS_OVERRIDE:-}" ]]; then
  KEEP_CACHE_SECONDS="${KEEP_CACHE_SECONDS_OVERRIDE}"
fi

# Determine glossary path
if [[ -n "${GLOSSARY_PATH_OVERRIDE}" ]]; then
  GLOSSARY_PATH="${GLOSSARY_PATH_OVERRIDE}"
else
  GLOSSARY_PATH="${GLOSSARY_ACL6060}"
fi
if [[ -n "${EVAL_GLOSSARY_PATH_OVERRIDE}" ]]; then
  EVAL_GLOSSARY_PATH="${EVAL_GLOSSARY_PATH_OVERRIDE}"
else
  EVAL_GLOSSARY_PATH="${GLOSSARY_PATH}"
fi

# Determine src/tgt lists
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

VLLM_SEGMENT_SEC="$(python3 -c "print(f'{float(\"${BASE_VLLM_SEGMENT_SEC}\") * float(\"${LATENCY_MULTIPLIER}\"):.2f}')")"

# Timeline mode retrieves once per vLLM generation step.  The stride variable is
# only used by legacy/debug modes.
if [[ -n "${RAG_RETRIEVE_STRIDE_SEC_OVERRIDE}" ]]; then
  RAG_RETRIEVE_STRIDE_SEC="${RAG_RETRIEVE_STRIDE_SEC_OVERRIDE}"
else
  RAG_RETRIEVE_STRIDE_SEC="${VLLM_SEGMENT_SEC}"
fi

GLOSSARY_TAG="$(basename "${GLOSSARY_PATH}" .json)"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE}"
if [[ -n "${ORACLE_TERM_MAP_PATH_OVERRIDE}" ]]; then
  OUTPUT_DIR_SUFFIX="d${DENSITY_TAG}_oraclegt_lm${LATENCY_MULTIPLIER}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${GLOSSARY_TAG}"
else
  OUTPUT_DIR_SUFFIX="d${DENSITY_TAG}_lm${LATENCY_MULTIPLIER}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${GLOSSARY_TAG}"
fi
if [[ -n "${PAPER_ID_TAG}" ]]; then
  OUTPUT_DIR_SUFFIX="${OUTPUT_DIR_SUFFIX}_pp${PAPER_ID_TAG}"
fi
OUTPUT_DIR="${OUTPUT_BASE}/${LANG_CODE}/${OUTPUT_DIR_SUFFIX}"

echo "[INFO] ============================================================"
echo "[INFO] Unified MaxSim Evaluation Pipeline"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] GLOSSARY_PATH=${GLOSSARY_PATH}"
echo "[INFO] EVAL_GLOSSARY_PATH=${EVAL_GLOSSARY_PATH}"
echo "[INFO] RAG_TOP_K=${RAG_TOP_K} RAG_LORA_R=${RAG_LORA_R} RAG_TEXT_LORA_R=${RAG_TEXT_LORA_R} RAG_TEXT_LORA_ALPHA=${RAG_TEXT_LORA_ALPHA} RAG_SCORE_THRESHOLD=${RAG_SCORE_THRESHOLD}"
echo "[INFO] RAG_STREAMING_MODE=${RAG_STREAMING_MODE_OVERRIDE}"
echo "[INFO] RAG_TIMELINE_LOOKBACK_SEC=${RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE}"
echo "[INFO] ORACLE_TERM_MAP_PATH=${ORACLE_TERM_MAP_PATH_OVERRIDE:-<none>}"
echo "[INFO] RAG_MAXSIM_WINDOWS=${RAG_MAXSIM_WINDOWS_OVERRIDE} RAG_MAXSIM_STRIDE=${RAG_MAXSIM_STRIDE_OVERRIDE}"
echo "[INFO] DENSITY_TAG=${DENSITY_TAG} PAPER_ID_TAG=${PAPER_ID_TAG:-<none>}"
echo "[INFO] VLLM_SEGMENT_SEC=${VLLM_SEGMENT_SEC} (lm=${LATENCY_MULTIPLIER}) RAG_STRIDE=${RAG_RETRIEVE_STRIDE_SEC}s"
echo "[INFO] SRC_LIST=${SRC_LIST}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES_PHYSICAL=${CUDA_VISIBLE_DEVICES_PHYSICAL}"
echo "[INFO] ============================================================"

# Conda activation
# The cluster-wide miniconda has hardcoded shebangs pointing at /home/... which
# is node-local and broken on aries (see user rule on non-portable paths).
# We therefore skip `conda activate` when the caller already prepared a working
# env via PATH/LD_LIBRARY_PATH, and only fall back to `conda activate` when no
# env is on PATH.
cd "${ROOT_DIR}"
TARGET_ENV_DIR="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
if [[ -n "${CONDA_ENV_NAME}" ]] && [[ -x "${TARGET_ENV_DIR}/bin/python" ]] \
   && [[ ":${PATH}:" == *":${TARGET_ENV_DIR}/bin:"* ]]; then
  echo "[INFO] Conda env already on PATH, skipping activate: ${TARGET_ENV_DIR}"
elif [[ -n "${CONDA_ENV_NAME}" ]] && [[ -x "${TARGET_ENV_DIR}/bin/python" ]]; then
  echo "[INFO] Prepending env to PATH directly (no conda activate): ${TARGET_ENV_DIR}"
  export PATH="${TARGET_ENV_DIR}/bin:${PATH}"
  export LD_LIBRARY_PATH="${TARGET_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
  export CONDA_PREFIX="${TARGET_ENV_DIR}"
  export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
elif [[ -n "${CONDA_ENV_NAME}" ]] && [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
  echo "[INFO] Activated conda env via conda.sh: ${CONDA_ENV_NAME}"
fi

# Validation
if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "[ERROR] HF model dir not found: ${MODEL_NAME}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
if [[ -n "${ORACLE_TERM_MAP_PATH_OVERRIDE}" ]]; then
  if [[ ! -f "${ORACLE_TERM_MAP_PATH_OVERRIDE}" ]]; then
    echo "[ERROR] Oracle term_map not found: ${ORACLE_TERM_MAP_PATH_OVERRIDE}" >&2
    exit "${EXIT_DATA_ERROR}"
  fi
elif [[ ! -f "${RAG_MODEL_PATH}" ]]; then
  echo "[ERROR] RAG model not found: ${RAG_MODEL_PATH}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${GLOSSARY_PATH}" ]]; then
  echo "[ERROR] Glossary not found: ${GLOSSARY_PATH}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${EVAL_GLOSSARY_PATH}" ]]; then
  echo "[ERROR] Eval glossary not found: ${EVAL_GLOSSARY_PATH}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${SRC_LIST}" ]] || [[ ! -f "${TGT_LIST}" ]]; then
  echo "[ERROR] Missing source/target list: ${SRC_LIST} or ${TGT_LIST}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

# ---- Step 1: Build MaxSim index (.pt) ----
mkdir -p "${INDEX_CACHE_DIR}"
INDEX_PATH=""
INDEX_MANIFEST_PATH=""
if [[ -n "${ORACLE_TERM_MAP_PATH_OVERRIDE}" ]]; then
  echo "[INFO] Oracle term_map mode: skipping MaxSim index build."
else
  BUILD_INDEX_SCRIPT="${ROOT_DIR}/retriever/gigaspeech/build_maxsim_index.py"
  INDEX_KEY_TOOL="${ROOT_DIR}/documents/code/simuleval/tools/maxsim_index_cache_key.py"
  if [[ ! -f "${BUILD_INDEX_SCRIPT}" ]]; then
    echo "[ERROR] Index build script not found: ${BUILD_INDEX_SCRIPT}" >&2
    exit "${EXIT_CONFIG_ERROR}"
  fi

  if [[ -n "${INDEX_PATH_OVERRIDE}" ]]; then
    INDEX_PATH="${INDEX_PATH_OVERRIDE}"
    echo "[INFO] Using explicit override index: ${INDEX_PATH}"
  else
    if [[ ! -x "${INDEX_KEY_TOOL}" ]]; then
      echo "[ERROR] MaxSim index key tool not found/executable: ${INDEX_KEY_TOOL}" >&2
      exit "${EXIT_CONFIG_ERROR}"
    fi
    INDEX_RESOLVE_ENV="$(
      python3 "${INDEX_KEY_TOOL}" resolve \
        --cache-dir "${INDEX_CACHE_DIR}" \
        --model-path "${RAG_MODEL_PATH}" \
        --glossary-path "${GLOSSARY_PATH}" \
        --builder-script "${BUILD_INDEX_SCRIPT}" \
        --glossary-tag "${GLOSSARY_TAG}" \
        --text-lora-rank "${RAG_TEXT_LORA_R}" \
        --text-lora-alpha "${RAG_TEXT_LORA_ALPHA}" \
        --output-format shell
    )"
    eval "${INDEX_RESOLVE_ENV}"
    echo "[INFO] Resolved hashed MaxSim index: ${INDEX_PATH}"
    echo "[INFO] MaxSim index manifest: ${INDEX_MANIFEST_PATH}"
  fi

  if [[ ! -f "${INDEX_PATH}" ]]; then
    echo "[INFO] Building MaxSim index: ${INDEX_PATH}"

    INDEX_BUILD_GPU="$(python3 -c "print('${CUDA_VISIBLE_DEVICES_PHYSICAL}'.split(',')[-1])")"

    CUDA_VISIBLE_DEVICES="${INDEX_BUILD_GPU}" python3 "${BUILD_INDEX_SCRIPT}" \
      --model-path "${RAG_MODEL_PATH}" \
      --glossary-path "${GLOSSARY_PATH}" \
      --output-path "${INDEX_PATH}" \
      --device "cuda:0" \
      --text-lora-rank "${RAG_TEXT_LORA_R}" \
      --text-lora-alpha "${RAG_TEXT_LORA_ALPHA}"
    echo "[INFO] Index built: ${INDEX_PATH}"
    if [[ -n "${INDEX_MANIFEST_PATH}" ]]; then
      python3 "${INDEX_KEY_TOOL}" finalize \
        --model-path "${RAG_MODEL_PATH}" \
        --glossary-path "${GLOSSARY_PATH}" \
        --builder-script "${BUILD_INDEX_SCRIPT}" \
        --glossary-tag "${GLOSSARY_TAG}" \
        --text-lora-rank "${RAG_TEXT_LORA_R}" \
        --text-lora-alpha "${RAG_TEXT_LORA_ALPHA}" \
        --index-path "${INDEX_PATH}" \
        --manifest-path "${INDEX_MANIFEST_PATH}"
    fi
  else
    echo "[INFO] Using cached MaxSim index: ${INDEX_PATH}"
  fi
fi

# ---- Step 2: Run SimulEval with MaxSim agent ----
echo "[INFO] Runtime EVAL_TMPDIR_OVERRIDE=${EVAL_TMPDIR_OVERRIDE}"
mkdir -p "${EVAL_TMPDIR_OVERRIDE}" "${EVAL_TMPDIR_OVERRIDE}/torchinductor" "${EVAL_TMPDIR_OVERRIDE}/triton"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TMPDIR="${EVAL_TMPDIR_OVERRIDE}"
export TMP="${EVAL_TMPDIR_OVERRIDE}"
export TEMP="${EVAL_TMPDIR_OVERRIDE}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${EVAL_TMPDIR_OVERRIDE}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${EVAL_TMPDIR_OVERRIDE}/triton}"
export VLLM_USE_V1="0"
export VLLM_NO_USAGE_STATS="1"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME="${VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME:-VLLM_OBJECT_STORAGE_SHM_BUFFER_${SLURM_JOB_ID:-$$}}"
export MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_PHYSICAL}"
echo "[INFO] Runtime TMPDIR=${TMPDIR}"
echo "[INFO] TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"

DEFAULT_MAX_CACHE_CHUNKS="$(python3 -c "print(max(${MIN_CACHE_CHUNKS}, int(80.0 / float('${VLLM_SEGMENT_SEC}'))))")"
DEFAULT_KEEP_CACHE_CHUNKS="$(python3 -c "print(max(${MIN_CACHE_CHUNKS}, int(60.0 / float('${VLLM_SEGMENT_SEC}'))))")"
CACHE_POLICY_NOTE="default_seconds_80_60"
if [[ -n "${MAX_CACHE_CHUNKS_OVERRIDE:-}" || -n "${KEEP_CACHE_CHUNKS_OVERRIDE:-}" ]]; then
  MAX_CACHE_CHUNKS="${DEFAULT_MAX_CACHE_CHUNKS}"
  KEEP_CACHE_CHUNKS="${DEFAULT_KEEP_CACHE_CHUNKS}"
  if [[ -n "${MAX_CACHE_CHUNKS_OVERRIDE:-}" ]]; then
    MAX_CACHE_CHUNKS="${MAX_CACHE_CHUNKS_OVERRIDE}"
  fi
  if [[ -n "${KEEP_CACHE_CHUNKS_OVERRIDE:-}" ]]; then
    KEEP_CACHE_CHUNKS="${KEEP_CACHE_CHUNKS_OVERRIDE}"
  fi
  CACHE_POLICY_NOTE="chunk_override"
elif [[ -n "${MAX_CACHE_SECONDS_OVERRIDE:-}" || -n "${KEEP_CACHE_SECONDS_OVERRIDE:-}" ]]; then
  MAX_CACHE_CHUNKS="$(python3 -c "print(max(${MIN_CACHE_CHUNKS}, int(float('${MAX_CACHE_SECONDS}') / float('${VLLM_SEGMENT_SEC}'))))")"
  KEEP_CACHE_CHUNKS="$(python3 -c "print(max(${MIN_CACHE_CHUNKS}, int(float('${KEEP_CACHE_SECONDS}') / float('${VLLM_SEGMENT_SEC}'))))")"
  CACHE_POLICY_NOTE="seconds_override"
else
  MAX_CACHE_CHUNKS="${DEFAULT_MAX_CACHE_CHUNKS}"
  KEEP_CACHE_CHUNKS="${DEFAULT_KEEP_CACHE_CHUNKS}"
fi
echo "[INFO] CACHE_CHUNKS=${MAX_CACHE_CHUNKS}/${KEEP_CACHE_CHUNKS} policy=${CACHE_POLICY_NOTE} default_seconds_80_60=${DEFAULT_MAX_CACHE_CHUNKS}/${DEFAULT_KEEP_CACHE_CHUNKS}"

mkdir -p "${OUTPUT_DIR}"
if [[ "${CLEAN_OUTPUT_DIR_OVERRIDE}" == "1" ]]; then
  rm -rf "${OUTPUT_DIR:?}/"*
fi

# RAG GPU: vLLM uses cuda:0..TP-1; put MaxSim on cuda:TP when available.
VISIBLE_GPU_COUNT="$(python3 -c "print(len('${CUDA_VISIBLE_DEVICES_PHYSICAL}'.split(',')))")"
VLLM_TP_SIZE="${VLLM_TP_SIZE_OVERRIDE:-2}"
export VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE}"
if [[ -n "${ORACLE_TERM_MAP_PATH_OVERRIDE}" ]]; then
  RAG_GPU="cuda:0"
elif [[ -n "${RAG_GPU_OVERRIDE:-}" ]]; then
  RAG_GPU="${RAG_GPU_OVERRIDE}"
  echo "[WARN] RAG_GPU_OVERRIDE=${RAG_GPU}; MaxSim may share a GPU with vLLM/Transformers."
elif [[ "${VISIBLE_GPU_COUNT}" -ge "$((VLLM_TP_SIZE + 1))" ]]; then
  RAG_GPU="cuda:${VLLM_TP_SIZE}"
else
  echo "[ERROR] MaxSim eval needs at least VLLM_TP_SIZE+1 visible GPUs; got ${VISIBLE_GPU_COUNT} for TP=${VLLM_TP_SIZE}." >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

SRC_SEGMENT_SIZE="$((LATENCY_MULTIPLIER * 480))"
AGENT_FILE="${ROOT_DIR}/agents/infinisst_omni_vllm_maxsim_rag.py"
AGENT_CLASS="agents.infinisst_omni_vllm_maxsim_rag.InfiniSSTOmniVLLMMaxSimRAG"

if [[ ! -f "${AGENT_FILE}" ]]; then
  echo "[ERROR] Agent file not found: ${AGENT_FILE}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

read -r -a RAG_MAXSIM_WINDOWS_ARGS <<< "${RAG_MAXSIM_WINDOWS_OVERRIDE}"
AGENT_TERM_ARGS=()
if [[ -n "${ORACLE_TERM_MAP_PATH_OVERRIDE}" ]]; then
  AGENT_TERM_ARGS=(
    --oracle-term-map-path "${ORACLE_TERM_MAP_PATH_OVERRIDE}"
    --rag-top-k "${RAG_TOP_K}"
    --rag-target-lang "${LANG_CODE}"
    --term-map-format "${TERM_MAP_FORMAT_OVERRIDE}"
    --system-prompt-style "${SYSTEM_PROMPT_STYLE}"
  )
else
  AGENT_TERM_ARGS=(
    --rag-enabled
    --rag-index-path "${INDEX_PATH}"
    --rag-model-path "${RAG_MODEL_PATH}"
    --rag-device "${RAG_GPU}"
    --rag-top-k "${RAG_TOP_K}"
    --rag-score-threshold "${RAG_SCORE_THRESHOLD}"
    --rag-target-lang "${LANG_CODE}"
    --rag-lora-r "${RAG_LORA_R}"
    --rag-text-lora-r "${RAG_TEXT_LORA_R}"
    --rag-maxsim-windows "${RAG_MAXSIM_WINDOWS_ARGS[@]}"
    --rag-maxsim-stride "${RAG_MAXSIM_STRIDE_OVERRIDE}"
    --rag-retrieve-stride-sec "${RAG_RETRIEVE_STRIDE_SEC}"
    --rag-timeline-lookback-sec "${RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE}"
    --rag-streaming-mode "${RAG_STREAMING_MODE_OVERRIDE}"
    --term-map-format "${TERM_MAP_FORMAT_OVERRIDE}"
    --empty-term-map-policy "${EMPTY_TERM_MAP_POLICY}"
    --system-prompt-style "${SYSTEM_PROMPT_STYLE}"
  )
fi

echo "[INFO] RAG_GPU=${RAG_GPU} SRC_SEGMENT_SIZE=${SRC_SEGMENT_SIZE}"
echo "[INFO] VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME=${VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME}"

INSTANCES_LOG="${OUTPUT_DIR}/instances.log"

if [[ -f "${INSTANCES_LOG}" ]] && [[ -s "${INSTANCES_LOG}" ]]; then
  echo "[INFO] instances.log already exists, skipping SimulEval: ${INSTANCES_LOG}"
else
  simuleval \
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
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K_DECODE}" \
    --use-vllm "${USE_VLLM}" \
    --vllm-enforce-eager "${VLLM_ENFORCE_EAGER}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --model-name "${MODEL_NAME}" \
    --max-cache-chunks "${MAX_CACHE_CHUNKS}" \
    --keep-cache-chunks "${KEEP_CACHE_CHUNKS}" \
    --quality-metrics BLEU \
    --eval-latency-unit "${LATENCY_UNIT}" \
    --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
    --vllm-segment-sec "${VLLM_SEGMENT_SEC}" \
    "${AGENT_TERM_ARGS[@]}" \
    --runtime-log-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/simuleval.log"
fi

# ---- Step 3: Run offline StreamLAAL eval ----
if [[ "${SKIP_OFFLINE_EVAL}" == "1" ]]; then
  echo "[INFO] Skipping offline eval (SKIP_OFFLINE_EVAL=1)."
  echo "[INFO] Done. Output: ${OUTPUT_DIR}"
  exit 0
fi

if [[ ! -f "${INSTANCES_LOG}" ]] || [[ ! -s "${INSTANCES_LOG}" ]]; then
  echo "[ERROR] instances.log not found or empty: ${INSTANCES_LOG}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

EVAL_TSV="${OUTPUT_DIR}/eval_results.tsv"
EVAL_LOG="${OUTPUT_DIR}/eval_results.log"

echo "[INFO] Running offline StreamLAAL evaluation (mode=${EVAL_MODE})..."
export INFINISST_ROOT="${ROOT_DIR}"
python3 "${OFFLINE_EVAL_SCRIPT}" \
  --mode "${EVAL_MODE}" \
  --instances-log "${INSTANCES_LOG}" \
  --lang-code "${LANG_CODE}" \
  --ref-file "${REF_FILE_OVERRIDE}" \
  --source-file "${SOURCE_TEXT_FILE_OVERRIDE}" \
  --audio-yaml "${AUDIO_YAML_OVERRIDE:-${DATA_ROOT}/dev.yaml}" \
  --sentence-term-map "${ORACLE_TERM_MAP_PATH_OVERRIDE}" \
  --glossary-acl6060 "${EVAL_GLOSSARY_PATH}" \
  --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
  --stream-laal-tool-rel "${STREAM_LAAL_TOOL_REL}" \
  --strip-output-tags "${STRIP_OUTPUT_TAGS}" \
  --term-fcr-policy "${TERM_FCR_POLICY}" \
  --output-tsv "${EVAL_TSV}" \
  --output-log "${EVAL_LOG}"

echo "[INFO] Eval results:"
if [[ -f "${EVAL_TSV}" ]]; then
  cat "${EVAL_TSV}"
fi

echo "[INFO] Done. Output: ${OUTPUT_DIR}"
