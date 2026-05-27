#!/usr/bin/env bash
set -euo pipefail

# Batch wrapper for the remaining ESO medicine oracle-GT readout.  It reuses
# the single-sample SFT launcher so each sample gets its own output directory
# and W&B run.

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
SINGLE_LAUNCHER="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260519__medicine_onetalk_oracle_gt_sft_oraclegt_r32a64_taurus_hold.sh"
NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260519__medicine_remaining_oracle_gt_sft_oraclegt_r32a64.md"

TARGET_SAMPLES="${TARGET_SAMPLES_OVERRIDE:-596001 606 545006}"
TARGET_LM="${TARGET_LM_OVERRIDE:-2}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/medicine_remaining_oracle_gt_sft_oraclegt_r32a64_20260519}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-medicine_remaining_oraclegt_r32a64}"
GPU_CSV="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-6:7}"
COMPUTE_TAG="${COMPUTE_TAG_OVERRIDE:-aries_direct_gpu6_7}"
DATA_TAG="${DATA_TAG_OVERRIDE:-medicine_remaining_zh}"
GLOSSARY_TAG_PATTERN="${GLOSSARY_TAG_PATTERN_OVERRIDE-}"
if [[ -z "${GLOSSARY_TAG_PATTERN}" ]]; then
    GLOSSARY_TAG_PATTERN="medicine_gt_strict_translated__medicine_{sample}"
fi

for p in "${SINGLE_LAUNCHER}" "${NOTES_FILE}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

render_sample_pattern() {
    local pattern="$1"
    local sample="$2"
    printf '%s' "${pattern}" | sed "s/{sample}/${sample}/g"
}

echo "[INFO] TARGET_SAMPLES=${TARGET_SAMPLES}"
echo "[INFO] TARGET_LM=${TARGET_LM}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] GPU_CSV=${GPU_CSV}"

for sample in ${TARGET_SAMPLES}; do
    echo "[BATCH] Running medicine sample=${sample}"
    glossary_tag="$(render_sample_pattern "${GLOSSARY_TAG_PATTERN}" "${sample}")"
    oracle_term_map_tag="${ORACLE_TERM_MAP_TAG_PATTERN_OVERRIDE-}"
    if [[ -z "${oracle_term_map_tag}" ]]; then
        oracle_term_map_tag="medicine.oracle_term_map__medicine_{sample}"
    fi
    oracle_term_map_tag="$(render_sample_pattern "${oracle_term_map_tag}" "${sample}")"
    TARGET_SAMPLE_OVERRIDE="${sample}" \
  TARGET_LM_OVERRIDE="${TARGET_LM}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
    DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
    TERM_SOURCE_OVERRIDE="${TERM_SOURCE_OVERRIDE:-sentence_terms}" \
    ORACLE_GLOSSARY_OVERRIDE="${ORACLE_GLOSSARY_OVERRIDE:-}" \
    EVAL_GLOSSARY_OVERRIDE="${EVAL_GLOSSARY_OVERRIDE:-}" \
    GLOSSARY_SOURCE_FILTER_OVERRIDE="${GLOSSARY_SOURCE_FILTER_OVERRIDE:-}" \
    GLOSSARY_TAG_OVERRIDE="${glossary_tag}" \
    ORACLE_TERM_MAP_TAG_OVERRIDE="${oracle_term_map_tag}" \
    TERM_FCR_POLICY_OVERRIDE="${TERM_FCR_POLICY_OVERRIDE:-term_map_if_available}" \
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${GPU_CSV}" \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  COMPUTE_TAG_OVERRIDE="${COMPUTE_TAG}" \
  DATA_TAG_OVERRIDE="${DATA_TAG}" \
  bash "${SINGLE_LAUNCHER}"
done

echo "[ALL DONE] Remaining medicine oracle-GT SFT readout complete."
