#!/usr/bin/env bash
# Run a 3-model x 3-setting tagged ACL probe for V3 speech-LLM SFT variants.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260521__tagged_acl_v3_speech_llm_3x3.md}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"
MODEL_ROOT="${MODEL_ROOT_OVERRIDE:-/mnt/aries/data7/jiaxuanluo/slm/v3_speech_llm}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_v3_speech_llm_3x3_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/aries/data7/jiaxuanluo/slm/maxsim_index_cache}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_v3_speech_llm_3x3_${RUN_STAMP}}"
MAX_PARALLEL="${MAX_PARALLEL_SETTINGS_OVERRIDE:-4}"
GPU_PAIRS_CSV="${GPU_PAIRS_CSV_OVERRIDE:-0,1;2,3;4,5;6,7}"
RUN_TAGS="${RUN_TAGS_OVERRIDE:-}"
SKIP_COMPLETED="${SKIP_COMPLETED_OVERRIDE:-1}"
RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
RAG_TEXT_LORA_R="${RAG_TEXT_LORA_R_OVERRIDE:-128}"
RAW_GLOSSARY="${RAW_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"

declare -a VARIANTS=("real" "tagged" "adv")
declare -a MODEL_DIRS=(
  "${MODEL_ROOT}/real_r8a32_iter793-hf"
  "${MODEL_ROOT}/tagged_r8a32_iter923-hf"
  "${MODEL_ROOT}/adv_r8a32_iter803-hf"
)
declare -a TERM_FORMATS=("plain" "tagged" "plain")

declare -a SETTINGS=(
  "zh 2 raw"
  "de 3 raw"
  "ja 1 gs10k"
)

if [[ ! -f "${BASE_LAUNCHER}" ]]; then
  echo "[ERROR] Missing base launcher: ${BASE_LAUNCHER}" >&2
  exit 3
fi
if [[ ! -f "${NOTES_FILE}" ]]; then
  echo "[ERROR] Missing notes file: ${NOTES_FILE}" >&2
  exit 3
fi
if (( MAX_PARALLEL < 1 )); then
  echo "[ERROR] MAX_PARALLEL_SETTINGS_OVERRIDE must be >= 1" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${INDEX_CACHE_DIR}" "${LOG_ROOT}"

check_hf_dir() {
  local model_dir="$1"
  if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "[ERROR] HF export missing config.json: ${model_dir}" >&2
    return 3
  fi
  local shard_count
  shard_count="$(find "${model_dir}" -maxdepth 1 -name '*.safetensors' | wc -l)"
  if (( shard_count < 10 )); then
    echo "[ERROR] HF export looks incomplete: ${model_dir} shard_count=${shard_count}" >&2
    return 3
  fi
}

index_path_for_glossary() {
  local glossary_path="$1"
  local model_hash glossary_hash glossary_tag
  model_hash="$(printf '%s' "${RAG_MODEL_PATH}" | sha1sum | awk '{print substr($1,1,10)}')"
  glossary_hash="$(printf '%s' "${glossary_path}" | sha1sum | awk '{print substr($1,1,10)}')"
  glossary_tag="$(basename "${glossary_path}" .json)"
  printf '%s/lh1b88kw_%s__%s__%s__maxsim.pt\n' \
    "${INDEX_CACHE_DIR}" "${model_hash}" "${glossary_tag}" "${glossary_hash}"
}

prebuild_index() {
  local kind="$1" glossary_path="$2" gpu_pair="$3"
  local index_path log_prefix
  index_path="$(index_path_for_glossary "${glossary_path}")"
  if [[ -f "${index_path}" ]]; then
    echo "[INDEX] exists kind=${kind}: ${index_path}"
    return 0
  fi
  log_prefix="${LOG_ROOT}/prebuild_index_${kind}_${RUN_STAMP}"
  echo "[INDEX] build kind=${kind}: ${index_path}"
  CUDA_VISIBLE_DEVICES="${gpu_pair}" CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${gpu_pair}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/retriever/gigaspeech/build_maxsim_index.py" \
      --model-path "${RAG_MODEL_PATH}" \
      --glossary-path "${glossary_path}" \
      --output-path "${index_path}" \
      --device cuda:0 \
      --text-lora-rank "${RAG_TEXT_LORA_R}" \
      > "${log_prefix}.out" 2> "${log_prefix}.err"
}

glossary_tag_for_kind() {
  local kind="$1"
  case "${kind}" in
    raw) basename "${RAW_GLOSSARY}" .json ;;
    gs10k) basename "${GS10K_GLOSSARY}" .json ;;
    *)
      echo "[ERROR] unsupported glossary kind for result path: ${kind}" >&2
      return 2
      ;;
  esac
}

result_tsv_for() {
  local variant_idx="$1" setting_idx="$2"
  local variant="${VARIANTS[$variant_idx]}"
  local lang lm kind glossary_tag
  read -r lang lm kind <<< "${SETTINGS[$setting_idx]}"
  glossary_tag="$(glossary_tag_for_kind "${kind}")"
  printf '%s/%s/%s/dtagacl_v3_%s_%s_lm%s_%s_tau073_lm%s_k10_th0.73_g%s/eval_results.tsv\n' \
    "${OUT_ROOT}" "${variant}" "${lang}" "${variant}" "${lang}" "${lm}" "${kind}" "${lm}" "${glossary_tag}"
}

should_run_tag() {
  local run_tag="$1"
  [[ -z "${RUN_TAGS}" ]] && return 0
  local requested
  for requested in ${RUN_TAGS}; do
    [[ "${requested}" == "${run_tag}" ]] && return 0
  done
  return 1
}

wait_for_slot() {
  while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
    local pid
    pid="$(jobs -pr | head -n 1)"
    wait "${pid}"
  done
}

run_one() {
  local variant_idx="$1"
  local setting_idx="$2"
  local gpu_pair="$3"

  local variant="${VARIANTS[$variant_idx]}"
  local model_dir="${MODEL_DIRS[$variant_idx]}"
  local term_format="${TERM_FORMATS[$variant_idx]}"
  local lang lm kind
  read -r lang lm kind <<< "${SETTINGS[$setting_idx]}"

  check_hf_dir "${model_dir}"

  local run_tag="v3_${variant}_${lang}_lm${lm}_${kind}"
  local out_base="${OUT_ROOT}/${variant}"
  local log_dir="${LOG_ROOT}/${run_tag}"

  echo "[RUN] variant=${variant} lang=${lang} lm=${lm} glossary=${kind} gpu=${gpu_pair}"
  echo "[RUN] model=${model_dir}"
  echo "[RUN] term_format=${term_format}"

  env \
    MODE=full \
    RUN_GRANULARITY=full_corpus \
    HOLD_JOB_ID=0 \
    INSIDE_HOLD_STEP=1 \
    MAX_PARALLEL_OVERRIDE=1 \
    RUN_STAMP="${RUN_STAMP}_${run_tag}" \
    LANGS_OVERRIDE="${lang}" \
    LMS_OVERRIDE="${lm}" \
    GLOSSARY_KINDS_OVERRIDE="${kind}" \
    GPU_PAIRS_CSV_OVERRIDE="${gpu_pair}" \
    MODEL_NAME_OVERRIDE="${model_dir}" \
    TERM_MAP_FORMAT_OVERRIDE="${term_format}" \
    OUTPUT_BASE_OVERRIDE="${out_base}" \
    INPUT_ROOT_OVERRIDE="${OUT_ROOT}/__inputs__" \
    LOG_DIR_OVERRIDE="${log_dir}" \
    SUMMARY_DIR_OVERRIDE="${out_base}/__summary__" \
    INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
    DENSITY_TAG_OVERRIDE="tagacl_${run_tag}_tau073" \
    NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
    GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
    bash "${BASE_LAUNCHER}"

  echo "[DONE] ${run_tag}"
}

IFS=';' read -r -a GPU_PAIRS <<< "${GPU_PAIRS_CSV}"
if (( ${#GPU_PAIRS[@]} < 1 )); then
  echo "[ERROR] Empty GPU_PAIRS_CSV_OVERRIDE" >&2
  exit 2
fi

# Build shared retriever text indices once before parallel eval workers.
# Otherwise multiple workers can race on the same index path.
prebuild_index "raw" "${RAW_GLOSSARY}" "${GPU_PAIRS[0]}"
prebuild_index "gs10k" "${GS10K_GLOSSARY}" "${GPU_PAIRS[0]}"

setting_counter=0
for vi in "${!VARIANTS[@]}"; do
  for si in "${!SETTINGS[@]}"; do
    variant="${VARIANTS[$vi]}"
    read -r lang lm kind <<< "${SETTINGS[$si]}"
    run_tag="v3_${variant}_${lang}_lm${lm}_${kind}"
    if ! should_run_tag "${run_tag}"; then
      echo "[SKIP] ${run_tag}: not requested by RUN_TAGS_OVERRIDE"
      continue
    fi
    result_tsv="$(result_tsv_for "${vi}" "${si}")"
    if [[ "${SKIP_COMPLETED}" == "1" && -s "${result_tsv}" ]]; then
      echo "[SKIP] ${run_tag}: completed result exists at ${result_tsv}"
      continue
    fi
    wait_for_slot
    gpu_pair="${GPU_PAIRS[$((setting_counter % ${#GPU_PAIRS[@]}))]}"
    run_one "${vi}" "${si}" "${gpu_pair}" &
    setting_counter=$((setting_counter + 1))
  done
done

failed=0
for pid in $(jobs -pr); do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if (( failed )); then
  echo "[ERROR] one or more tagged ACL V3 settings failed" >&2
  exit 1
fi

echo "[ALL DONE] output_root=${OUT_ROOT}"
