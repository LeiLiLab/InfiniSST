#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"

MODEL_ROOT="${MODEL_ROOT:?MODEL_ROOT is required}"
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
GPU_PAIR="${GPU_PAIR:-0,1}"
MODE="${MODE:-full}"
WAIT_FOR_HF_SECS="${WAIT_FOR_HF_SECS:-28800}"
OUT_ROOT="${OUT_ROOT:-/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_${MODEL_LABEL}_quick_zh_lm2_raw_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_${MODEL_LABEL}_quick_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/aries/data7/jiaxuanluo/slm/maxsim_index_cache}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260522__tagged_acl_quick_v15_v16_newv4_zh_lm2_raw.md}"
LANGS="${LANGS:-zh}"
LMS="${LMS:-2}"
GLOSSARY_KINDS="${GLOSSARY_KINDS:-raw}"
RUN_GRANULARITY="${RUN_GRANULARITY:-full_corpus}"
PAPERS="${PAPERS:-2022.acl-long.268 2022.acl-long.367 2022.acl-long.590 2022.acl-long.110 2022.acl-long.117}"
EVAL_GLOSSARY_PATH_GLOBAL="${EVAL_GLOSSARY_PATH_GLOBAL:-}"
EVAL_GLOSSARY_FOLLOWS_KIND="${EVAL_GLOSSARY_FOLLOWS_KIND:-0}"
EXTRACTED_GS10K_GLOSSARY="${EXTRACTED_GS10K_GLOSSARY:-}"

latest_hf_dir() {
  local root="$1"
  local found=""
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

  echo "[ERROR] No complete HF checkpoint found under ${root} after ${waited}s" >&2
  return 2
}

for p in "${BASE_LAUNCHER}" "${NOTES_FILE}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing required file: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
MODEL_DIR="$(latest_hf_dir "${MODEL_ROOT}")"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "model_label=${MODEL_LABEL}"
  echo "model=${MODEL_DIR}"
  echo "model_root=${MODEL_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
  echo "gpu_pair=${GPU_PAIR}"
  echo "mode=${MODE}"
  echo "wait_for_hf_secs=${WAIT_FOR_HF_SECS}"
  echo "langs=${LANGS}"
  echo "lms=${LMS}"
  echo "glossary_kinds=${GLOSSARY_KINDS}"
  echo "run_granularity=${RUN_GRANULARITY}"
  echo "papers=${PAPERS}"
  echo "eval_glossary_path_global=${EVAL_GLOSSARY_PATH_GLOBAL}"
  echo "eval_glossary_follows_kind=${EVAL_GLOSSARY_FOLLOWS_KIND}"
  echo "extracted_gs10k_glossary=${EXTRACTED_GS10K_GLOSSARY}"
} | tee "${OUT_ROOT}/run_meta.txt"

env \
  ROOT_DIR="${ROOT_DIR}" \
  RUN_STAMP="${RUN_STAMP}_${MODEL_LABEL}" \
  MODE="${MODE}" \
  RUN_GRANULARITY=full_corpus \
  HOLD_JOB_ID=0 \
  INSIDE_HOLD_STEP=1 \
  MAX_PARALLEL_OVERRIDE=1 \
  LANGS_OVERRIDE="${LANGS}" \
  LMS_OVERRIDE="${LMS}" \
  GLOSSARY_KINDS_OVERRIDE="${GLOSSARY_KINDS}" \
  RUN_GRANULARITY="${RUN_GRANULARITY}" \
  PAPERS_OVERRIDE="${PAPERS}" \
  GPU_PAIRS_CSV_OVERRIDE="${GPU_PAIR}" \
  MODEL_NAME_OVERRIDE="${MODEL_DIR}" \
  TERM_MAP_FORMAT_OVERRIDE="plain" \
  OUTPUT_BASE_OVERRIDE="${OUT_ROOT}/${MODEL_LABEL}" \
  INPUT_ROOT_OVERRIDE="${OUT_ROOT}/__inputs__/${MODEL_LABEL}" \
  LOG_DIR_OVERRIDE="${LOG_ROOT}/${MODEL_LABEL}" \
  SUMMARY_DIR_OVERRIDE="${OUT_ROOT}/${MODEL_LABEL}/__summary__" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  EVAL_GLOSSARY_FOLLOWS_KIND_OVERRIDE="${EVAL_GLOSSARY_FOLLOWS_KIND}" \
  ${EVAL_GLOSSARY_PATH_GLOBAL:+EVAL_GLOSSARY_PATH_GLOBAL_OVERRIDE="${EVAL_GLOSSARY_PATH_GLOBAL}"} \
  ${EXTRACTED_GS10K_GLOSSARY:+EXTRACTED_GS10K_GLOSSARY_OVERRIDE="${EXTRACTED_GS10K_GLOSSARY}"} \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_OVERRIDE:-${TMPDIR:-}}" \
  DENSITY_TAG_OVERRIDE="${DENSITY_TAG_OVERRIDE:-tagacl_${MODEL_LABEL}_quick_tau073}" \
  WANDB_RUN_PREFIX_OVERRIDE="${WANDB_RUN_PREFIX_OVERRIDE:-${MODEL_LABEL}}" \
  WANDB_EXPERIMENT_FAMILY_OVERRIDE="${WANDB_EXPERIMENT_FAMILY_OVERRIDE:-tagged_acl_speech_llm_quick}" \
  WANDB_VARIANT_PREFIX_OVERRIDE="${WANDB_VARIANT_PREFIX_OVERRIDE:-${MODEL_LABEL}}" \
  WANDB_COMPUTE_TAG_OVERRIDE="${WANDB_COMPUTE_TAG_OVERRIDE:-compute:aries_direct}" \
  TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
  bash "${BASE_LAUNCHER}" \
    > "${LOG_ROOT}/${MODEL_LABEL}_launcher.out" \
    2> "${LOG_ROOT}/${MODEL_LABEL}_launcher.err"

echo "[INFO] Done. Output root: ${OUT_ROOT}"
