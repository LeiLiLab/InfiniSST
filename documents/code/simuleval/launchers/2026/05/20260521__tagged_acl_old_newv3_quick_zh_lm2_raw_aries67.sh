#!/usr/bin/env bash
set -euo pipefail

# Quick tagged-ACL readout for older Speech LLM new_v3 term-map SFT checkpoints.
# Runs serially on one 2-GPU pair, defaulting to Aries GPUs 6,7.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"

NEWV3_MODEL="${NEWV3_MODEL:-/mnt/gemini/data2/jiaxuanluo/speech_llm_tcmw100kgt_tau075_new_v3_r32a64_taurus8/keep1.0_r32/v0-20260508-122348-hf}"
NEWV3_RANDOM_MODEL="${NEWV3_RANDOM_MODEL:-/mnt/gemini/data2/jiaxuanluo/speech_llm_tcmw100kgt_tau075_new_v3_random_r32a64_aries8/keep1.0_r32/v1-20260508-123645-hf}"

OUT_ROOT="${OUT_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_old_newv3_quick_zh_lm2_raw_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_old_newv3_quick_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/aries/data7/jiaxuanluo/slm/maxsim_index_cache}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260521__tagged_acl_old_newv3_quick_zh_lm2_raw.md}"
GPU_PAIR="${GPU_PAIR:-6,7}"

check_hf_dir() {
  local model_dir="$1"
  if [[ ! -d "${model_dir}" ]]; then
    echo "[ERROR] Missing HF checkpoint dir: ${model_dir}" >&2
    return 2
  fi
  if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "[ERROR] HF checkpoint missing config.json: ${model_dir}" >&2
    return 2
  fi
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
    WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_old_newv3_quick" \
    WANDB_VARIANT_PREFIX_OVERRIDE="${variant}" \
    WANDB_COMPUTE_TAG_OVERRIDE="compute:aries_direct" \
    TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
    bash "${BASE_LAUNCHER}" \
      > "${log_dir}/launcher.out" \
      2> "${log_dir}/launcher.err"
}

check_hf_dir "${NEWV3_MODEL}"
check_hf_dir "${NEWV3_RANDOM_MODEL}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "newv3_model=${NEWV3_MODEL}"
  echo "newv3_random_model=${NEWV3_RANDOM_MODEL}"
  echo "out_root=${OUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
  echo "gpu_pair=${GPU_PAIR}"
} | tee "${OUT_ROOT}/run_meta.txt"

run_one "old_newv3_r32a64" "${NEWV3_MODEL}"
run_one "old_newv3_random_r32a64" "${NEWV3_RANDOM_MODEL}"

echo "[INFO] Done. Output root: ${OUT_ROOT}"
