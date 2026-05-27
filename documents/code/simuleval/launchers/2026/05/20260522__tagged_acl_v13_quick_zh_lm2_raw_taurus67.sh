#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"

V13_ROOT="${V13_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_v13_lm1to6_retriever_timeline_zh_lh1b88kw_tau073_r8a32_taurus2/keep1.0_r8}"
OUT_ROOT="${OUT_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_v13_quick_zh_lm2_raw_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_v13_quick_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/aries/data7/jiaxuanluo/slm/maxsim_index_cache}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260522__tagged_acl_v13_quick_zh_lm2_raw.md}"
GPU_PAIR="${GPU_PAIR:-6,7}"
WAIT_FOR_HF_SECS="${WAIT_FOR_HF_SECS:-0}"

latest_hf_dir() {
  local root="$1"
  local found
  local waited=0
  while true; do
    found="$(find "${root}" -maxdepth 1 -type d -name '*-hf' 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${found}" && -f "${found}/config.json" ]]; then
      local shard_count
      shard_count="$(find "${found}" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
      if [[ "${shard_count}" == "15" ]]; then
        printf '%s\n' "${found}"
        return 0
      fi
      echo "[WAIT] HF checkpoint incomplete: ${found} has ${shard_count}/15 safetensor shards" >&2
    else
      echo "[WAIT] No complete HF checkpoint found under ${root}" >&2
    fi
    if (( WAIT_FOR_HF_SECS <= 0 || waited >= WAIT_FOR_HF_SECS )); then
      break
    fi
    sleep 60
    waited=$((waited + 60))
  done

  if [[ -z "${found}" ]]; then
    echo "[ERROR] No exported HF checkpoint found under ${root}" >&2
    return 2
  fi
  if [[ ! -f "${found}/config.json" ]]; then
    echo "[ERROR] HF checkpoint is missing config.json: ${found}" >&2
    return 2
  fi
  shard_count="$(find "${found}" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${shard_count}" != "15" ]]; then
    echo "[ERROR] HF checkpoint is incomplete: ${found} has ${shard_count}/15 safetensor shards" >&2
    return 2
  fi
  printf '%s\n' "${found}"
}

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
V13_MODEL="$(latest_hf_dir "${V13_ROOT}")"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "model=${V13_MODEL}"
  echo "out_root=${OUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
  echo "gpu_pair=${GPU_PAIR}"
  echo "wait_for_hf_secs=${WAIT_FOR_HF_SECS}"
} | tee "${OUT_ROOT}/run_meta.txt"

env \
  ROOT_DIR="${ROOT_DIR}" \
  RUN_STAMP="${RUN_STAMP}_v13_lm1to6" \
  MODE=full \
  RUN_GRANULARITY=full_corpus \
  HOLD_JOB_ID=0 \
  INSIDE_HOLD_STEP=1 \
  MAX_PARALLEL_OVERRIDE=1 \
  LANGS_OVERRIDE="zh" \
  LMS_OVERRIDE="2" \
  GLOSSARY_KINDS_OVERRIDE="raw" \
  GPU_PAIRS_CSV_OVERRIDE="${GPU_PAIR}" \
  MODEL_NAME_OVERRIDE="${V13_MODEL}" \
  TERM_MAP_FORMAT_OVERRIDE="plain" \
  OUTPUT_BASE_OVERRIDE="${OUT_ROOT}/v13_lm1to6" \
  INPUT_ROOT_OVERRIDE="${OUT_ROOT}/__inputs__/v13_lm1to6" \
  LOG_DIR_OVERRIDE="${LOG_ROOT}/v13_lm1to6" \
  SUMMARY_DIR_OVERRIDE="${OUT_ROOT}/v13_lm1to6/__summary__" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  DENSITY_TAG_OVERRIDE="tagacl_v13_lm1to6_tau073" \
  WANDB_RUN_PREFIX_OVERRIDE="v13_lm1to6" \
  WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_v13_quick" \
  WANDB_VARIANT_PREFIX_OVERRIDE="v13_lm1to6" \
  WANDB_COMPUTE_TAG_OVERRIDE="compute:taurus_direct" \
  TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
  bash "${BASE_LAUNCHER}" \
    > "${LOG_ROOT}/v13_lm1to6_launcher.out" \
    2> "${LOG_ROOT}/v13_lm1to6_launcher.err"

echo "[INFO] Done. Output root: ${OUT_ROOT}"
