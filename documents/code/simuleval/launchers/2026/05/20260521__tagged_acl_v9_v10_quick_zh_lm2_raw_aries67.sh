#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"

V9_ROOT="${V9_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_v9_dense_top10_notau_backfill_zh_lh1b88kw_srcmatch100k_r8a32_aries2/keep1.0_r8}"
V10_ROOT="${V10_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_v10_marker_dense_top10_notau_backfill_zh_lh1b88kw_srcmatch100k_r8a32_aries2/keep1.0_r8}"
OUT_ROOT="${OUT_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_v9_v10_quick_zh_lm2_raw_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_v9_v10_quick_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/aries/data7/jiaxuanluo/slm/maxsim_index_cache}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260521__tagged_acl_v9_v10_quick_zh_lm2_raw.md}"
GPU_PAIR="${GPU_PAIR:-6,7}"
EVAL_VARIANTS="${EVAL_VARIANTS:-v9_dense v10_marker}"

latest_hf_dir() {
  local root="$1"
  local found
  found="$(find "${root}" -maxdepth 1 -type d -name '*-hf' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${found}" ]]; then
    echo "[ERROR] No exported HF checkpoint found under ${root}" >&2
    return 2
  fi
  if [[ ! -f "${found}/config.json" ]]; then
    echo "[ERROR] HF checkpoint is missing config.json: ${found}" >&2
    return 2
  fi
  printf '%s\n' "${found}"
}

run_one() {
  local variant="$1"
  local model_dir="$2"
  local output_base="${OUT_ROOT}/${variant}"
  local input_root="${OUT_ROOT}/__inputs__/${variant}"
  local summary_dir="${output_base}/__summary__"
  local log_dir="${LOG_ROOT}/${variant}"

  mkdir -p "${output_base}" "${input_root}" "${summary_dir}" "${log_dir}"
  echo "[INFO] Launching ${variant}: model=${model_dir} gpu_pair=${GPU_PAIR}"

  env \
    ROOT_DIR="${ROOT_DIR}" \
    RUN_STAMP="${RUN_STAMP}_${variant}" \
    MODE=full \
    RUN_GRANULARITY=full_corpus \
    HOLD_JOB_ID=0 \
    INSIDE_HOLD_STEP=1 \
    MAX_PARALLEL_OVERRIDE=1 \
    LANGS_OVERRIDE="zh" \
    LMS_OVERRIDE="2" \
    GLOSSARY_KINDS_OVERRIDE="raw" \
    GPU_PAIRS_CSV_OVERRIDE="${GPU_PAIR}" \
    MODEL_NAME_OVERRIDE="${model_dir}" \
    TERM_MAP_FORMAT_OVERRIDE="plain" \
    OUTPUT_BASE_OVERRIDE="${output_base}" \
    INPUT_ROOT_OVERRIDE="${input_root}" \
    LOG_DIR_OVERRIDE="${log_dir}" \
    SUMMARY_DIR_OVERRIDE="${summary_dir}" \
    INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
    NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
    DENSITY_TAG_OVERRIDE="tagacl_${variant}_tau073" \
    WANDB_RUN_PREFIX_OVERRIDE="${variant}" \
    WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_v9_v10_quick" \
    WANDB_VARIANT_PREFIX_OVERRIDE="${variant}" \
    WANDB_COMPUTE_TAG_OVERRIDE="compute:aries_direct" \
    TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
    bash "${BASE_LAUNCHER}" \
      > "${log_dir}/launcher.out" \
      2> "${log_dir}/launcher.err"
}

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "out_root=${OUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
  echo "gpu_pair=${GPU_PAIR}"
  echo "eval_variants=${EVAL_VARIANTS}"
} | tee "${OUT_ROOT}/run_meta.txt"

for variant in ${EVAL_VARIANTS}; do
  case "${variant}" in
    v9_dense)
      V9_MODEL="$(latest_hf_dir "${V9_ROOT}")"
      echo "v9_model=${V9_MODEL}" | tee -a "${OUT_ROOT}/run_meta.txt"
      run_one "v9_dense" "${V9_MODEL}"
      ;;
    v10_marker)
      V10_MODEL="$(latest_hf_dir "${V10_ROOT}")"
      echo "v10_model=${V10_MODEL}" | tee -a "${OUT_ROOT}/run_meta.txt"
      run_one "v10_marker" "${V10_MODEL}"
      ;;
    *)
      echo "[ERROR] Unknown EVAL_VARIANTS entry: ${variant}" >&2
      exit 2
      ;;
  esac
done

echo "[INFO] Done. Output root: ${OUT_ROOT}"
