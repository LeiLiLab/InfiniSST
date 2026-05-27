#!/usr/bin/env bash
# Quick tagged ACL readouts for V16-family SLMs with fixed metric denominators.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
QUICK="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__tagged_acl_speech_llm_quick_zh_lm2_raw_wait_hf.sh"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"
GPU_PAIR="${GPU_PAIR:-6,7}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260522__tagged_acl_v16_fixeddenom_raw10k_extracted10k.md}"

RAW_DENOM="${RAW_DENOM_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
EXTRACTED_DENOM="${EXTRACTED_DENOM_OVERRIDE:-${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper/extracted_glossary__2022.acl-long.110.json}"
EXTRACTED_GS10K="${EXTRACTED_GS10K_OVERRIDE:-${ROOT_DIR}/documents/data/data_pre/expanded_glossaries_by_paper/expanded_glossary__2022.acl-long.110_gs10000.json}"

V16_ROOT="${V16_ROOT_OVERRIDE:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_v16_llm_variant_aug_retriever_timeline_zh_lh1b88kw_tau073_r8a32_aries2/keep1.0_r8}"
V16_NO_GT_ZERO_ROOT="${V16_NO_GT_ZERO_ROOT_OVERRIDE:-/mnt/aries/data7/jiaxuanluo/slm/speech_llm_v16_no_gt_zero_llm_variant_aug_retriever_timeline_zh_lh1b88kw_tau073_r8a32_aries2/keep1.0_r8}"

for p in "${QUICK}" "${NOTES_FILE}" "${RAW_DENOM}" "${EXTRACTED_DENOM}" "${EXTRACTED_GS10K}" "${V16_ROOT}" "${V16_NO_GT_ZERO_ROOT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

run_panel() {
  local label="$1"
  local model_root="$2"
  local panel="$3"
  local glossary_kinds="$4"
  local run_granularity="$5"
  local papers="$6"
  local eval_denom="$7"
  local out_root="/mnt/aries/data7/jiaxuanluo/slm/tagged_acl_${label}_${panel}_${RUN_STAMP}"
  local log_root="/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_${label}_${panel}_${RUN_STAMP}"

  echo "[PANEL] label=${label} panel=${panel} glossaries=${glossary_kinds} denom=${eval_denom}"
  ROOT_DIR="${ROOT_DIR}" \
  MODEL_ROOT="${model_root}" \
  MODEL_LABEL="${label}_${panel}" \
  GPU_PAIR="${GPU_PAIR}" \
  RUN_STAMP="${RUN_STAMP}_${label}_${panel}" \
  OUT_ROOT="${out_root}" \
  LOG_ROOT="${log_root}" \
  NOTES_FILE="${NOTES_FILE}" \
  GLOSSARY_KINDS="${glossary_kinds}" \
  RUN_GRANULARITY="${run_granularity}" \
  PAPERS="${papers}" \
  EVAL_GLOSSARY_PATH_GLOBAL="${eval_denom}" \
  EVAL_GLOSSARY_FOLLOWS_KIND=0 \
  EXTRACTED_GS10K_GLOSSARY="${EXTRACTED_GS10K}" \
  WAIT_FOR_HF_SECS=0 \
  bash "${QUICK}"
}

for item in \
  "v16_llmvariant ${V16_ROOT}" \
  "v16_no_gt_zero ${V16_NO_GT_ZERO_ROOT}"
do
  label="${item%% *}"
  root="${item#* }"
  run_panel "${label}" "${root}" "rawfixed_raw_gs10k" "raw gs10k" "full_corpus" "2022.acl-long.268 2022.acl-long.367 2022.acl-long.590 2022.acl-long.110 2022.acl-long.117" "${RAW_DENOM}"
  run_panel "${label}" "${root}" "extracted110_fixed_extracted_gs10k" "extracted extracted_gs10k" "per_paper" "2022.acl-long.110" "${EXTRACTED_DENOM}"
done

echo "[DONE] V16 fixed-denominator evals finished. run_stamp=${RUN_STAMP}"
