#!/usr/bin/env bash
set -euo pipefail

# Panel A medicine oracle-GT readout for all-GT SFT.  This keeps the metric
# denominator fixed to translated medicine_gt entries and builds oracle term maps
# from source/reference glossary matching instead of ESO sentence.terms.

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_SWEEP="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260519__medicine_lm1to4_oracle_gt_sft_oraclegt_r32a64_aries2.sh"
NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260519__medicine_panelA_fixedgt_oraclegt_r32a64.md"

MEDICINE_TRANSLATED_GLOSSARY="${MEDICINE_TRANSLATED_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000_translated.json}"

for p in "${BASE_SWEEP}" "${NOTES_FILE}" "${MEDICINE_TRANSLATED_GLOSSARY}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

TERM_SOURCE_OVERRIDE="glossary_match" \
ORACLE_GLOSSARY_OVERRIDE="${MEDICINE_TRANSLATED_GLOSSARY}" \
EVAL_GLOSSARY_OVERRIDE="${MEDICINE_TRANSLATED_GLOSSARY}" \
GLOSSARY_SOURCE_FILTER_OVERRIDE="medicine_gt" \
TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
GLOSSARY_TAG_PATTERN_OVERRIDE="medicine_panelA_fixed_gt__medicine_{sample}" \
ORACLE_TERM_MAP_TAG_PATTERN_OVERRIDE="medicine.oracle_term_map__panelA_fixed_gt__medicine_{sample}" \
ONETALK_OUTPUT_BASE_OVERRIDE="/mnt/gemini/data2/jiaxuanluo/medicine_onetalk_panelA_fixedgt_oraclegt_sft_r32a64_20260519" \
REMAINING_OUTPUT_BASE_OVERRIDE="/mnt/gemini/data2/jiaxuanluo/medicine_remaining_panelA_fixedgt_oraclegt_sft_r32a64_20260519" \
AGG_OUTPUT_BASE_OVERRIDE="/mnt/gemini/data2/jiaxuanluo/medicine4_panelA_fixedgt_oraclegt_sft_r32a64_lm_sweep_20260519" \
ONETALK_DENSITY_OVERRIDE="medicine1_panelA_fixedgt_r32a64" \
REMAINING_DENSITY_OVERRIDE="medicine_remaining_panelA_fixedgt_r32a64" \
AGG_DENSITY_OVERRIDE="medicine4_panelA_fixedgt_r32a64" \
COMBINED_GLOSSARY_TAG_OVERRIDE="medicine_panelA_fixed_gt_four_samples" \
TARGET_SAMPLES_OVERRIDE="${TARGET_SAMPLES_OVERRIDE:-404 596001 606 545006}" \
TARGET_LMS_OVERRIDE="${TARGET_LMS_OVERRIDE:-1 2 3 4}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-6:7}" \
COMPUTE_TAG_OVERRIDE="${COMPUTE_TAG_OVERRIDE:-aries_direct_gpu6_7}" \
DATA_TAG_OVERRIDE="medicine4_panelA_fixedgt_zh" \
NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
FORCE_RERUN_OVERRIDE="${FORCE_RERUN_OVERRIDE:-0}" \
bash "${BASE_SWEEP}"
