#!/usr/bin/env bash
set -euo pipefail

# Quick tagged-ACL eval for the two refmatch Speech LLM variants:
#   V7: plain term_map
#   V8: XML-style <term>source => target</term> term_map
#
# Run after both training launchers have exported their HF checkpoints.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"

V7_ROOT="${V7_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_v7_refmatch_r95_zh_lh1b88kw_tau073_srcmatch100k_r8a32_aries2/keep1.0_r8}"
V8_ROOT="${V8_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_v8_refmatch_r95_xmlterm_zh_lh1b88kw_tau073_srcmatch100k_r8a32_aries2/keep1.0_r8}"
OUT_ROOT="${OUT_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_v7_v8_refmatch_quick_zh_lm2_raw_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_v7_v8_refmatch_quick_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/aries/data7/jiaxuanluo/slm/maxsim_index_cache}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260521__tagged_acl_v7_v8_refmatch_quick_zh_lm2_raw.md}"
V7_GPU_PAIR="${V7_GPU_PAIR:-4,5}"
V8_GPU_PAIR="${V8_GPU_PAIR:-2,3}"
RUN_SERIAL="${RUN_SERIAL:-0}"

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
  local term_format="$3"
  local gpu_pair="$4"
  local output_base="${OUT_ROOT}/${variant}"
  local input_root="${OUT_ROOT}/__inputs__/${variant}"
  local summary_dir="${output_base}/__summary__"
  local log_dir="${LOG_ROOT}/${variant}"

  mkdir -p "${output_base}" "${input_root}" "${summary_dir}" "${log_dir}"
  echo "[INFO] Launching ${variant}: model=${model_dir} term_format=${term_format} gpu_pair=${gpu_pair}"

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
    GPU_PAIRS_CSV_OVERRIDE="${gpu_pair}" \
    MODEL_NAME_OVERRIDE="${model_dir}" \
    TERM_MAP_FORMAT_OVERRIDE="${term_format}" \
    OUTPUT_BASE_OVERRIDE="${output_base}" \
    INPUT_ROOT_OVERRIDE="${input_root}" \
    LOG_DIR_OVERRIDE="${log_dir}" \
    SUMMARY_DIR_OVERRIDE="${summary_dir}" \
    INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
    NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
    DENSITY_TAG_OVERRIDE="tagacl_${variant}_tau073" \
    WANDB_RUN_PREFIX_OVERRIDE="${variant}" \
    WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_v7_v8_refmatch_quick" \
    WANDB_VARIANT_PREFIX_OVERRIDE="${variant}" \
    WANDB_COMPUTE_TAG_OVERRIDE="compute:aries_direct" \
    TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
    bash "${BASE_LAUNCHER}" \
      > "${log_dir}/launcher.out" \
      2> "${log_dir}/launcher.err"
}

V7_MODEL="$(latest_hf_dir "${V7_ROOT}")"
V8_MODEL="$(latest_hf_dir "${V8_ROOT}")"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "v7_model=${V7_MODEL}"
  echo "v8_model=${V8_MODEL}"
  echo "out_root=${OUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
} | tee "${OUT_ROOT}/run_meta.txt"

if [[ "${RUN_SERIAL}" == "1" ]]; then
  echo "[INFO] RUN_SERIAL=1"
  run_one "v7_plain" "${V7_MODEL}" "plain" "${V7_GPU_PAIR}"
  run_one "v8_xml" "${V8_MODEL}" "xml_tagged" "${V8_GPU_PAIR}"
else
  run_one "v7_plain" "${V7_MODEL}" "plain" "${V7_GPU_PAIR}" &
  pid_v7=$!
  run_one "v8_xml" "${V8_MODEL}" "xml_tagged" "${V8_GPU_PAIR}" &
  pid_v8=$!

  echo "${pid_v7}" > "${OUT_ROOT}/v7_plain.pid"
  echo "${pid_v8}" > "${OUT_ROOT}/v8_xml.pid"
  echo "[INFO] PIDs: v7=${pid_v7} v8=${pid_v8}"

  wait "${pid_v7}"
  wait "${pid_v8}"
fi

echo "[INFO] Done. Output root: ${OUT_ROOT}"
