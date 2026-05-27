#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BATCH_LAUNCHER="${BATCH_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
RUN_TAG="${RUN_TAG_OVERRIDE:-20260524T175015_medicine_norag_de_lm4_batch_max80_aries67}"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4}"
HARD_GLOSSARY="${HARD_GLOSSARY_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_hard_manual_glossary_streamlaal_20260524.json}"
INPUT_SOURCE_DIR="${INPUT_SOURCE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm4_aries67/de/__medicine_inputs__/combined}"

OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_de_lm4_batch_max80_aries67_${RUN_TAG}}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-${OUT_ROOT}/batch_eval}"
INPUT_DIR="${INPUT_DIR_OVERRIDE:-${OUT_ROOT}/de/__medicine_inputs__/combined}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_norag_baseline_de_lm4_batch_max80_aries67_${RUN_TAG}}"
CACHE_BASE="${CACHE_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/cache/medicine_norag_de_lm4_batch_max80_aries67_${RUN_TAG}}"
EVAL_TMPDIR="${EVAL_TMPDIR_OVERRIDE:-/dev/shm/jxdbm80}"

GPU_PAIR="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-6,7}"
VLLM_TP_SIZE="${VLLM_TP_SIZE_OVERRIDE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN_OVERRIDE:-8192}"

SOURCE_LIST="${INPUT_DIR}/medicine5.source.txt"
TARGET_LIST="${INPUT_DIR}/medicine5.target.de.txt"
SOURCE_TEXT_FILE="${INPUT_DIR}/medicine5.source_text.en.sentences.txt"
REF_FILE="${INPUT_DIR}/medicine5.ref.de.sentences.txt"
AUDIO_YAML="${INPUT_DIR}/medicine5.audio.yaml"
DENSITY_TAG="medicine_norag_baseline_batch_max80"
GLOSSARY_TAG="hard_llm_manual_check"
BATCH_DIR="${OUTPUT_BASE}/de/d${DENSITY_TAG}_lm4_k0_th0.0_g${GLOSSARY_TAG}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

gpu_pair_is_free() {
  local gpu_comma="${GPU_PAIR//:/,}"
  local gpu
  local used
  IFS=',' read -r -a gpu_ids <<< "${gpu_comma}"
  for gpu in "${gpu_ids[@]}"; do
    used="$(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, -v target="${gpu}" '$1 ~ "^[[:space:]]*" target "[[:space:]]*$" {gsub(/ /, "", $2); print $2}'
    )"
    [[ -n "${used}" ]] || fail "Could not read memory for GPU ${gpu}"
    [[ "${used}" -le "${GPU_BUSY_THRESHOLD_MIB_OVERRIDE:-1000}" ]] || fail "GPU ${gpu} is busy (${used} MiB)"
  done
}

copy_inputs() {
  mkdir -p "${INPUT_DIR}"
  for name in \
    medicine5.source.txt \
    medicine5.target.de.txt \
    medicine5.source_text.en.sentences.txt \
    medicine5.ref.de.sentences.txt \
    medicine5.audio.yaml; do
    [[ -s "${INPUT_SOURCE_DIR}/${name}" ]] || fail "missing prepared input: ${INPUT_SOURCE_DIR}/${name}"
    cp "${INPUT_SOURCE_DIR}/${name}" "${INPUT_DIR}/${name}"
  done
}

validate_inputs() {
  for p in "${MODEL_NAME}/config.json" "${HARD_GLOSSARY}" "${BATCH_LAUNCHER}" \
    "${SOURCE_LIST}" "${TARGET_LIST}" "${SOURCE_TEXT_FILE}" "${REF_FILE}" "${AUDIO_YAML}"; do
    [[ -s "${p}" ]] || fail "missing/empty required path: ${p}"
  done

  local source_count
  local target_count
  source_count="$(wc -l < "${SOURCE_LIST}")"
  target_count="$(wc -l < "${TARGET_LIST}")"
  [[ "${source_count}" == "5" ]] || fail "expected 5 source rows, got ${source_count}"
  [[ "${target_count}" == "5" ]] || fail "expected 5 target rows, got ${target_count}"
  while IFS= read -r wav_path; do
    [[ -s "${wav_path}" ]] || fail "missing wav from source list: ${wav_path}"
  done < "${SOURCE_LIST}"
}

write_meta() {
  mkdir -p "${OUT_ROOT}" "${OUTPUT_BASE}" "${LOG_ROOT}" "${CACHE_BASE}" "${EVAL_TMPDIR}"
  {
    echo "run_tag=${RUN_TAG}"
    echo "host=$(hostname -s)"
    echo "launcher=${BATCH_LAUNCHER}"
    echo "model=${MODEL_NAME}"
    echo "input_source_dir=${INPUT_SOURCE_DIR}"
    echo "input_dir=${INPUT_DIR}"
    echo "output_base=${OUTPUT_BASE}"
    echo "batch_dir=${BATCH_DIR}"
    echo "hard_glossary=${HARD_GLOSSARY}"
    echo "lang=de"
    echo "lm=4"
    echo "sample_count=5"
    echo "gpu_pair=${GPU_PAIR}"
    echo "vllm_tp_size=${VLLM_TP_SIZE}"
    echo "max_num_seqs=5"
    echo "scheduler_batch_size=5"
    echo "schedule_mode=round_robin"
    echo "disable_rag=1"
    echo "max_new_tokens=80"
    echo "max_new_tokens_policy=fixed"
    echo "max_model_len=${MAX_MODEL_LEN}"
  } | tee "${OUT_ROOT}/run_meta.txt"
}

main() {
  if [[ -e "${OUT_ROOT}/.started" && "${FORCE_RERUN_OVERRIDE:-0}" != "1" ]]; then
    fail "output root already started: ${OUT_ROOT}"
  fi
  gpu_pair_is_free
  copy_inputs
  validate_inputs
  write_meta
  date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.started"

  export XDG_CACHE_HOME="${CACHE_BASE}/xdg"
  export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/torchinductor"
  export CUDA_CACHE_PATH="${CACHE_BASE}/cuda"
  export HF_HOME="${CACHE_BASE}/hf"
  export HF_HUB_CACHE="${CACHE_BASE}/hf/hub"
  export TRANSFORMERS_CACHE="${CACHE_BASE}/hf/transformers"
  export VLLM_CACHE_ROOT="${CACHE_BASE}/vllm"
  export NUMBA_CACHE_DIR="${CACHE_BASE}/numba"
  mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" \
    "${NUMBA_CACHE_DIR}"

  cd "${ROOT_DIR}"
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  RUN_TAG_OVERRIDE="${RUN_TAG}" \
  LANG_CODE_OVERRIDE="de" \
  LMS_OVERRIDE="4" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE}" \
  MAX_NUM_SEQS_OVERRIDE="5" \
  SCHEDULER_BATCH_SIZE_OVERRIDE="5" \
  SCHEDULE_MODE_OVERRIDE="round_robin" \
  DISABLE_RAG_OVERRIDE="1" \
  RAG_TOP_K_OVERRIDE="0" \
  RAG_SCORE_THRESHOLD_OVERRIDE="0" \
  RAG_BATCH_RETRIEVAL_OVERRIDE="0" \
  MAX_NEW_TOKENS_OVERRIDE="80" \
  MAX_NEW_TOKENS_POLICY_OVERRIDE="fixed" \
  MAX_CACHE_SECONDS_OVERRIDE="80" \
  KEEP_CACHE_SECONDS_OVERRIDE="60" \
  MIN_CACHE_CHUNKS_OVERRIDE="1" \
  VLLM_LIMIT_AUDIO_OVERRIDE="64" \
  VLLM_ENFORCE_EAGER_OVERRIDE="1" \
  VLLM_ENABLE_PREFIX_CACHING="1" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${MAX_MODEL_LEN}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE="1" \
  VLLM_MOE_USE_DEEP_GEMM="0" \
  VLLM_USE_FUSED_MOE_GROUPED_TOPK="0" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION}" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  SRC_LIST_OVERRIDE="${SOURCE_LIST}" \
  TGT_LIST_OVERRIDE="${TARGET_LIST}" \
  SOURCE_TEXT_FILE_OVERRIDE="${SOURCE_TEXT_FILE}" \
  REF_FILE_OVERRIDE="${REF_FILE}" \
  AUDIO_YAML_OVERRIDE="${AUDIO_YAML}" \
  GLOSSARY_PATH_OVERRIDE="${HARD_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${HARD_GLOSSARY}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
  GLOSSARY_TAG_OVERRIDE="${GLOSSARY_TAG}" \
  TERM_FCR_POLICY_OVERRIDE="source_ref_negative_sentence" \
  STRIP_OUTPUT_TAGS_OVERRIDE="term" \
  DRY_RUN_OVERRIDE="${DRY_RUN_OVERRIDE:-0}" \
  WANDB_LOG_OVERRIDE="${WANDB_LOG_OVERRIDE:-0}" \
  LOG_ROOT_OVERRIDE="${LOG_ROOT}" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
  bash "${BATCH_LAUNCHER}" \
    > "${LOG_ROOT}/launcher.out" \
    2> "${LOG_ROOT}/launcher.err"

  if [[ "${DRY_RUN_OVERRIDE:-0}" == "1" ]]; then
    echo "[DRY RUN] batch launcher validation complete"
    exit 0
  fi

  for p in "${BATCH_DIR}/eval_results.tsv" "${BATCH_DIR}/instances.log" \
    "${BATCH_DIR}/runtime_omni_vllm_maxsim_rag_batched_lm4.jsonl"; do
    [[ -s "${p}" ]] || fail "missing/empty batch output: ${p}"
  done
  cp -f "${BATCH_DIR}/eval_results.tsv" \
    "${BATCH_DIR}/eval_results_streamlaal_term.hard_llm_manual_check.tsv"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.success"
  echo "[ALL DONE] de lm4 no-RAG batch max80: ${BATCH_DIR}"
}

main "$@"
