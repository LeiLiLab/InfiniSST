#!/usr/bin/env bash
set -euo pipefail

# Standalone batched-vLLM RAG evaluator.
# This does not modify or source the existing serial SimulEval launchers.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

CONDA_BASE="${CONDA_BASE:-/mnt/taurus/home/jiaxuanluo/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-spaCyEnv}"
TARGET_ENV_DIR="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-${TARGET_ENV_DIR}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Python env not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export LD_LIBRARY_PATH="${TARGET_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${TARGET_ENV_DIR}"
export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

PY_SCRIPT="${ROOT_DIR}/documents/code/simuleval/src/batched_vllm_rag_eval.py"
OFFLINE_EVAL_SCRIPT="${OFFLINE_EVAL_SCRIPT_OVERRIDE:-documents/code/offline_sst_eval/offline_streamlaal_eval.py}"
WANDB_LOGGER="${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py"

LANG_CODE="${LANG_CODE_OVERRIDE:-de}"
LMS_OVERRIDE="${LMS_OVERRIDE:-1 2 3 4}"
RUN_TAG="${RUN_TAG_OVERRIDE:-hn1024_tau078_raw_batchvllm_$(date -u +%Y%m%dT%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/batched_vllm_rag_eval_${RUN_TAG}}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-batchvllm_hn1024_tau078}"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-}"
SOURCE_LIST="${SRC_LIST_OVERRIDE:-}"
TARGET_LIST="${TGT_LIST_OVERRIDE:-}"
SOURCE_TEXT_FILE="${SOURCE_TEXT_FILE_OVERRIDE:-}"
REF_FILE="${REF_FILE_OVERRIDE:-}"
AUDIO_YAML="${AUDIO_YAML_OVERRIDE:-}"
GLOSSARY_PATH="${GLOSSARY_PATH_OVERRIDE:-}"
EVAL_GLOSSARY_PATH="${EVAL_GLOSSARY_PATH_OVERRIDE:-${GLOSSARY_PATH}}"

HN1024_CKPT="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt"
RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-${HN1024_CKPT}}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:-0.78}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE:-1.92}"
RAG_LORA_R="${RAG_LORA_R_OVERRIDE:-128}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE:-128}"
RAG_TEXT_LORA_ALPHA="${RAG_TEXT_LORA_ALPHA_OVERRIDE:-256}"
RAG_DEVICE="${RAG_DEVICE_OVERRIDE:-cuda:1}"
RAG_BATCH_RETRIEVAL="${RAG_BATCH_RETRIEVAL_OVERRIDE:-1}"
DISABLE_RAG="${DISABLE_RAG_OVERRIDE:-0}"

GPU_PAIR="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-}"
VLLM_TP_SIZE="${VLLM_TP_SIZE_OVERRIDE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}"
MAX_NUM_SEQS="${MAX_NUM_SEQS_OVERRIDE:-8}"
SCHEDULER_BATCH_SIZE="${SCHEDULER_BATCH_SIZE_OVERRIDE:-8}"
SCHEDULE_MODE="${SCHEDULE_MODE_OVERRIDE:-round_robin}"
MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-${VLLM_MAX_MODEL_LEN_OVERRIDE:-32768}}"
VLLM_LIMIT_AUDIO="${VLLM_LIMIT_AUDIO_OVERRIDE:-128}"
DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER_OVERRIDE:-0}"
VLLM_ENABLE_PREFIX_CACHING_VALUE="${VLLM_ENABLE_PREFIX_CACHING:-1}"
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY_OVERRIDE:-lazy}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS_OVERRIDE:-40}"
MAX_NEW_TOKENS_POLICY="${MAX_NEW_TOKENS_POLICY_OVERRIDE:-lm_scaled}"
TEMPERATURE="${TEMPERATURE_OVERRIDE:-0.6}"
TOP_P="${TOP_P_OVERRIDE:-0.95}"
TOP_K_DECODE="${TOP_K_DECODE_OVERRIDE:-20}"
MAX_CACHE_CHUNKS="${MAX_CACHE_CHUNKS_OVERRIDE:-16}"
KEEP_CACHE_CHUNKS="${KEEP_CACHE_CHUNKS_OVERRIDE:-8}"
MAX_CACHE_SECONDS="${MAX_CACHE_SECONDS_OVERRIDE:-0}"
KEEP_CACHE_SECONDS="${KEEP_CACHE_SECONDS_OVERRIDE:-0}"
MIN_CACHE_CHUNKS="${MIN_CACHE_CHUNKS_OVERRIDE:-1}"
TERM_MAP_FORMAT="${TERM_MAP_FORMAT_OVERRIDE:-plain}"
EMPTY_TERM_MAP_POLICY="${EMPTY_TERM_MAP_POLICY_OVERRIDE:-none_block}"
RAG_PROMPT_POLICY="${RAG_PROMPT_POLICY_OVERRIDE:-translate_task}"
NORAG_PROMPT_POLICY="${NORAG_PROMPT_POLICY_OVERRIDE:-term_map_if_available}"

EVAL_MODE="${EVAL_MODE_OVERRIDE:-acl6060}"
STRIP_OUTPUT_TAGS="${STRIP_OUTPUT_TAGS_OVERRIDE:-term}"
TERM_FCR_POLICY="${TERM_FCR_POLICY_OVERRIDE:-term_map_source_ref_negative_sentence}"
SKIP_OFFLINE_EVAL="${SKIP_OFFLINE_EVAL_OVERRIDE:-0}"
DRY_RUN="${DRY_RUN_OVERRIDE:-0}"
WANDB_LOG="${WANDB_LOG_OVERRIDE:-0}"
WANDB_PYTHON="${WANDB_PYTHON:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
WANDB_HOME="${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX_OVERRIDE:-batchvllm_hn1024_tau078}"
WANDB_EXPERIMENT_FAMILY="${WANDB_EXPERIMENT_FAMILY_OVERRIDE:-tagged_acl_batchvllm_hn1024_tau078}"
WANDB_VARIANT_PREFIX="${WANDB_VARIANT_PREFIX_OVERRIDE:-batchvllm_hn1024_tau078}"
WANDB_COMPUTE_TAG="${WANDB_COMPUTE_TAG_OVERRIDE:-compute:taurus8_batchvllm}"
WANDB_RUNTIME_GLOSSARY_LABEL="${WANDB_RUNTIME_GLOSSARY_LABEL_OVERRIDE:-raw}"
WANDB_DATA_TAG="${WANDB_DATA_TAG_OVERRIDE:-tagged_acl_strict_raw_${LANG_CODE}}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__batched_vllm_rag_eval.md}"

INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/batched_vllm}"
INDEX_BUILDER="${ROOT_DIR}/retriever/gigaspeech/build_maxsim_index.py"
INDEX_CACHE_TOOL="${ROOT_DIR}/documents/code/simuleval/tools/maxsim_index_cache_key.py"
INDEX_BUILD_DEVICE="${INDEX_BUILD_DEVICE_OVERRIDE:-${RAG_DEVICE}}"
GLOSSARY_TAG="${GLOSSARY_TAG_OVERRIDE:-$(basename "${GLOSSARY_PATH:-glossary}" .json)}"

LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/batched_vllm_rag_eval}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_BASE}"

# Keep vLLM IPC socket paths short.
EVAL_TMPDIR="${EVAL_TMPDIR_OVERRIDE:-/tmp/jx_bvllm_${USER:-u}_$(date -u +%H%M%S)}"
mkdir -p "${EVAL_TMPDIR}"
export TMPDIR="${EVAL_TMPDIR}"
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"

if [[ -n "${GPU_PAIR}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_PAIR}"
fi

for name in MODEL_NAME SOURCE_LIST TARGET_LIST SOURCE_TEXT_FILE REF_FILE AUDIO_YAML GLOSSARY_PATH; do
  value="${!name}"
  if [[ -z "${value}" ]]; then
    echo "[ERROR] ${name} is required; set ${name}_OVERRIDE where applicable." >&2
    exit 2
  fi
done
if [[ "${DISABLE_RAG}" != "1" && -z "${RAG_MODEL_PATH}" ]]; then
  echo "[ERROR] RAG_MODEL_PATH is required unless DISABLE_RAG_OVERRIDE=1." >&2
  exit 2
fi

required_paths=("${PY_SCRIPT}" "${SOURCE_LIST}" "${TARGET_LIST}" "${SOURCE_TEXT_FILE}" "${REF_FILE}" "${AUDIO_YAML}" "${GLOSSARY_PATH}" "${EVAL_GLOSSARY_PATH}")
if [[ "${DISABLE_RAG}" != "1" ]]; then
  required_paths+=("${RAG_MODEL_PATH}")
fi
for p in "${required_paths[@]}"; do
  if [[ ! -s "${p}" ]]; then
    echo "[ERROR] Missing or empty required file: ${p}" >&2
    exit 3
  fi
done

df -h /mnt/gemini/data1 || true

if [[ "${DISABLE_RAG}" == "1" ]]; then
  INDEX_PATH=""
  echo "[INDEX] skipped because DISABLE_RAG_OVERRIDE=1"
else
  resolve_out="$(
    "${PYTHON_BIN}" "${INDEX_CACHE_TOOL}" resolve \
      --model-path "${RAG_MODEL_PATH}" \
      --glossary-path "${GLOSSARY_PATH}" \
      --builder-script "${INDEX_BUILDER}" \
      --cache-dir "${INDEX_CACHE_DIR}" \
      --glossary-tag "${GLOSSARY_TAG}" \
      --text-lora-rank "${RAG_TEXT_LORA_R}" \
      --text-lora-alpha "${RAG_TEXT_LORA_ALPHA}"
  )"
  eval "${resolve_out}"

  if [[ ! -s "${INDEX_PATH}" ]]; then
    echo "[INDEX] building ${INDEX_PATH}"
    mkdir -p "$(dirname "${INDEX_PATH}")"
    "${PYTHON_BIN}" "${INDEX_BUILDER}" \
      --model-path "${RAG_MODEL_PATH}" \
      --glossary-path "${GLOSSARY_PATH}" \
      --output-path "${INDEX_PATH}" \
      --device "${INDEX_BUILD_DEVICE}" \
      --text-lora-rank "${RAG_TEXT_LORA_R}" \
      --text-lora-alpha "${RAG_TEXT_LORA_ALPHA}"
    "${PYTHON_BIN}" "${INDEX_CACHE_TOOL}" finalize \
      --model-path "${RAG_MODEL_PATH}" \
      --glossary-path "${GLOSSARY_PATH}" \
      --builder-script "${INDEX_BUILDER}" \
      --index-path "${INDEX_PATH}" \
      --manifest-path "${INDEX_MANIFEST_PATH}" \
      --glossary-tag "${GLOSSARY_TAG}" \
      --text-lora-rank "${RAG_TEXT_LORA_R}" \
      --text-lora-alpha "${RAG_TEXT_LORA_ALPHA}"
  else
    echo "[INDEX] reusing ${INDEX_PATH}"
  fi
fi

args=(
  --source-list "${SOURCE_LIST}"
  --target-list "${TARGET_LIST}"
  --source-text-file "${SOURCE_TEXT_FILE}"
  --ref-file "${REF_FILE}"
  --audio-yaml "${AUDIO_YAML}"
  --glossary "${GLOSSARY_PATH}"
  --eval-glossary "${EVAL_GLOSSARY_PATH}"
  --output-base "${OUTPUT_BASE}"
  --run-tag "${RUN_TAG}"
  --density-tag "${DENSITY_TAG}"
  --glossary-tag "${GLOSSARY_TAG}"
  --lang-code "${LANG_CODE}"
  --lms ${LMS_OVERRIDE}
  --model-name "${MODEL_NAME}"
  --vllm-tp-size "${VLLM_TP_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --scheduler-batch-size "${SCHEDULER_BATCH_SIZE}"
  --schedule-mode "${SCHEDULE_MODE}"
  --vllm-limit-audio "${VLLM_LIMIT_AUDIO}"
  --vllm-enforce-eager "${VLLM_ENFORCE_EAGER}"
  --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}"
  --enable-prefix-caching "${VLLM_ENABLE_PREFIX_CACHING_VALUE}"
  --disable-custom-all-reduce "${DISABLE_CUSTOM_ALL_REDUCE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-new-tokens-policy "${MAX_NEW_TOKENS_POLICY}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K_DECODE}"
  --max-cache-chunks "${MAX_CACHE_CHUNKS}"
  --keep-cache-chunks "${KEEP_CACHE_CHUNKS}"
  --max-cache-seconds "${MAX_CACHE_SECONDS}"
  --keep-cache-seconds "${KEEP_CACHE_SECONDS}"
  --min-cache-chunks "${MIN_CACHE_CHUNKS}"
  --rag-model-path "${RAG_MODEL_PATH}"
  --rag-index-path "${INDEX_PATH}"
  --rag-device "${RAG_DEVICE}"
  --rag-top-k "${RAG_TOP_K}"
  --rag-score-threshold "${RAG_SCORE_THRESHOLD}"
  --rag-timeline-lookback-sec "${RAG_TIMELINE_LOOKBACK_SEC}"
  --rag-lora-r "${RAG_LORA_R}"
  --rag-text-lora-r "${RAG_TEXT_LORA_R}"
  --rag-batch-retrieval "${RAG_BATCH_RETRIEVAL}"
  --term-map-format "${TERM_MAP_FORMAT}"
  --empty-term-map-policy "${EMPTY_TERM_MAP_POLICY}"
  --rag-prompt-policy "${RAG_PROMPT_POLICY}"
  --norag-prompt-policy "${NORAG_PROMPT_POLICY}"
  --offline-eval-script "${OFFLINE_EVAL_SCRIPT}"
  --eval-mode "${EVAL_MODE}"
  --strip-output-tags "${STRIP_OUTPUT_TAGS}"
  --term-fcr-policy "${TERM_FCR_POLICY}"
)

if [[ "${DISABLE_RAG}" == "1" ]]; then
  args+=(--disable-rag)
fi
if [[ "${SKIP_OFFLINE_EVAL}" == "1" ]]; then
  args+=(--skip-offline-eval)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

echo "[RUN] output_base=${OUTPUT_BASE}"
echo "[RUN] lang=${LANG_CODE} lms=${LMS_OVERRIDE} model=${MODEL_NAME}"
echo "[RUN] term_map_format=${TERM_MAP_FORMAT} empty_term_map_policy=${EMPTY_TERM_MAP_POLICY}"
echo "[RUN] disable_rag=${DISABLE_RAG} rag_prompt_policy=${RAG_PROMPT_POLICY} norag_prompt_policy=${NORAG_PROMPT_POLICY}"
"${PYTHON_BIN}" "${PY_SCRIPT}" "${args[@]}"

if [[ "${WANDB_LOG}" == "1" && "${DRY_RUN}" != "1" && "${SKIP_OFFLINE_EVAL}" != "1" ]]; then
  tau_tag="tau${RAG_SCORE_THRESHOLD/./}"
  for lm in ${LMS_OVERRIDE}; do
    HOME="${WANDB_HOME}" \
    WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME}/.config/wandb}" \
    "${WANDB_PYTHON}" "${WANDB_LOGGER}" \
      --project simuleval_eval \
      --run-name "${WANDB_RUN_PREFIX}__tagged_acl__${LANG_CODE}__lm${lm}__${tau_tag}__${WANDB_RUNTIME_GLOSSARY_LABEL}__batchvllm" \
      --experiment-family "${WANDB_EXPERIMENT_FAMILY}" \
      --data-tag "${WANDB_DATA_TAG}" \
      --task-tag eval \
      --notes-file "${NOTES_FILE}" \
      --extra-tags "variant:${WANDB_VARIANT_PREFIX}_${LANG_CODE}_${WANDB_RUNTIME_GLOSSARY_LABEL}_lm${lm}" "${WANDB_COMPUTE_TAG}" "tau:${tau_tag}" "glossary:${WANDB_RUNTIME_GLOSSARY_LABEL}" "lang:${LANG_CODE}" \
      --density "${DENSITY_TAG}" \
      --rag-top-k "${RAG_TOP_K}" \
      --rag-score-threshold "${RAG_SCORE_THRESHOLD}" \
      --output-base "${OUTPUT_BASE}" \
      --lang-code "${LANG_CODE}" \
      --latency-multipliers "${lm}" \
      --glossary-tag "${GLOSSARY_TAG}" \
      --model-name "${MODEL_NAME}" \
      --rag-model-path "${RAG_MODEL_PATH}" \
      --verdict "Batched vLLM prototype: ${LANG_CODE}, lm=${lm}, ${WANDB_RUNTIME_GLOSSARY_LABEL} tagged ACL; HN1024 tau=${RAG_SCORE_THRESHOLD}, lookback=${RAG_TIMELINE_LOOKBACK_SEC}s. Compare against serial SimulEval before using as truth."
  done
fi
