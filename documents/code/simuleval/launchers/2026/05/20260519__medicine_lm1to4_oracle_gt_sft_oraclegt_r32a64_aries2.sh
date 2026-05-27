#!/usr/bin/env bash
set -euo pipefail

# Reusable ESO medicine oracle-GT lm sweep for the all-GT SFT checkpoint.
#
# This launcher runs missing per-talk SimulEval jobs and then writes a combined
# four-talk summary per LM.  It keeps lm2 results if already complete, so reruns
# do not duplicate old per-talk W&B runs.

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
SINGLE_LAUNCHER="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260519__medicine_onetalk_oracle_gt_sft_oraclegt_r32a64_taurus_hold.sh"
AGG_SCRIPT="${ROOT_DIR}/documents/code/simuleval/aggregate_medicine_oracle_lm_sweep.py"
WANDB_LOGGER="${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py"
NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260519__medicine_lm1to4_oracle_gt_sft_oraclegt_r32a64.md"
WANDB_PYTHON="${WANDB_PYTHON:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
WANDB_HOME="${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_r32a64_taurus4/keep1.0_r32/v1-20260519-105111-hf}"
TARGET_SAMPLES="${TARGET_SAMPLES_OVERRIDE:-404 596001 606 545006}"
TARGET_LMS="${TARGET_LMS_OVERRIDE:-1 2 3 4}"
LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:-1.0}"
GPU_CSV="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-6:7}"
COMPUTE_TAG="${COMPUTE_TAG_OVERRIDE:-aries_direct_gpu6_7}"
DATA_TAG="${DATA_TAG_OVERRIDE:-medicine4_zh}"
TERM_FCR_POLICY="${TERM_FCR_POLICY_OVERRIDE:-term_map_if_available}"

ONETALK_OUTPUT_BASE="${ONETALK_OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/medicine_onetalk_oracle_gt_sft_oraclegt_r32a64_20260519}"
REMAINING_OUTPUT_BASE="${REMAINING_OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/medicine_remaining_oracle_gt_sft_oraclegt_r32a64_20260519}"
AGG_OUTPUT_BASE="${AGG_OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/medicine4_oracle_gt_sft_oraclegt_r32a64_lm_sweep_20260519}"
ONETALK_DENSITY="${ONETALK_DENSITY_OVERRIDE:-medicine1_oraclegt_r32a64}"
REMAINING_DENSITY="${REMAINING_DENSITY_OVERRIDE:-medicine_remaining_oraclegt_r32a64}"
AGG_DENSITY="${AGG_DENSITY_OVERRIDE:-medicine4_oraclegt_r32a64}"
COMBINED_GLOSSARY_TAG="${COMBINED_GLOSSARY_TAG_OVERRIDE:-medicine_gt_strict_translated_four_samples}"
GLOSSARY_TAG_PATTERN="${GLOSSARY_TAG_PATTERN_OVERRIDE-}"
if [[ -z "${GLOSSARY_TAG_PATTERN}" ]]; then
  GLOSSARY_TAG_PATTERN="medicine_gt_strict_translated__medicine_{sample}"
fi
FORCE_RERUN="${FORCE_RERUN_OVERRIDE:-0}"

for p in "${SINGLE_LAUNCHER}" "${AGG_SCRIPT}" "${WANDB_LOGGER}" "${NOTES_FILE}" "${MODEL_NAME}"; do
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

sample_output_dir() {
  local sample="$1"
  local lm="$2"
  local base density
  if [[ "${sample}" == "404" ]]; then
    base="${ONETALK_OUTPUT_BASE}"
    density="${ONETALK_DENSITY}"
  else
    base="${REMAINING_OUTPUT_BASE}"
    density="${REMAINING_DENSITY}"
  fi
  local glossary_tag
  glossary_tag="$(render_sample_pattern "${GLOSSARY_TAG_PATTERN}" "${sample}")"
  echo "${base}/${LANG_CODE}/d${density}_oraclegt_lm${lm}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${glossary_tag}_ppmedicine_${sample}"
}

run_one_if_needed() {
  local sample="$1"
  local lm="$2"
  local output_base density data_tag output_dir eval_tsv glossary_tag oracle_term_map_tag
  if [[ "${sample}" == "404" ]]; then
    output_base="${ONETALK_OUTPUT_BASE}"
    density="${ONETALK_DENSITY}"
    data_tag="medicine_onetalk_zh"
  else
    output_base="${REMAINING_OUTPUT_BASE}"
    density="${REMAINING_DENSITY}"
    data_tag="medicine_remaining_zh"
  fi

  output_dir="$(sample_output_dir "${sample}" "${lm}")"
  eval_tsv="${output_dir}/eval_results.tsv"
  if [[ "${FORCE_RERUN}" != "1" && -s "${eval_tsv}" ]]; then
    echo "[SKIP] sample=${sample} lm=${lm}: ${eval_tsv}"
    return 0
  fi

  echo "[RUN] sample=${sample} lm=${lm}"
  glossary_tag="$(render_sample_pattern "${GLOSSARY_TAG_PATTERN}" "${sample}")"
  oracle_term_map_tag="${ORACLE_TERM_MAP_TAG_PATTERN_OVERRIDE-}"
  if [[ -z "${oracle_term_map_tag}" ]]; then
    oracle_term_map_tag="medicine.oracle_term_map__medicine_{sample}"
  fi
  oracle_term_map_tag="$(render_sample_pattern "${oracle_term_map_tag}" "${sample}")"
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  TARGET_SAMPLE_OVERRIDE="${sample}" \
  TARGET_LM_OVERRIDE="${lm}" \
  OUTPUT_BASE_OVERRIDE="${output_base}" \
  DENSITY_TAG_OVERRIDE="${density}" \
  LANG_CODE_OVERRIDE="${LANG_CODE}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  TERM_SOURCE_OVERRIDE="${TERM_SOURCE_OVERRIDE:-sentence_terms}" \
  ORACLE_GLOSSARY_OVERRIDE="${ORACLE_GLOSSARY_OVERRIDE:-}" \
  EVAL_GLOSSARY_OVERRIDE="${EVAL_GLOSSARY_OVERRIDE:-}" \
  GLOSSARY_SOURCE_FILTER_OVERRIDE="${GLOSSARY_SOURCE_FILTER_OVERRIDE:-}" \
  GLOSSARY_TAG_OVERRIDE="${glossary_tag}" \
  ORACLE_TERM_MAP_TAG_OVERRIDE="${oracle_term_map_tag}" \
  TERM_FCR_POLICY_OVERRIDE="${TERM_FCR_POLICY}" \
  FORCE_RERUN_OVERRIDE="${FORCE_RERUN}" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${GPU_CSV}" \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  COMPUTE_TAG_OVERRIDE="${COMPUTE_TAG}" \
  DATA_TAG_OVERRIDE="${data_tag}" \
  bash "${SINGLE_LAUNCHER}"
}

echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] TARGET_SAMPLES=${TARGET_SAMPLES}"
echo "[INFO] TARGET_LMS=${TARGET_LMS}"
echo "[INFO] GPU_CSV=${GPU_CSV}"
echo "[INFO] ONETALK_OUTPUT_BASE=${ONETALK_OUTPUT_BASE}"
echo "[INFO] REMAINING_OUTPUT_BASE=${REMAINING_OUTPUT_BASE}"
echo "[INFO] AGG_OUTPUT_BASE=${AGG_OUTPUT_BASE}"

for lm in ${TARGET_LMS}; do
  for sample in ${TARGET_SAMPLES}; do
    run_one_if_needed "${sample}" "${lm}"
  done
done

"${WANDB_PYTHON}" "${AGG_SCRIPT}" \
  --lms ${TARGET_LMS} \
  --samples ${TARGET_SAMPLES} \
  --lang-code "${LANG_CODE}" \
  --rag-top-k "${RAG_TOP_K}" \
  --rag-score-threshold "${RAG_SCORE_THRESHOLD}" \
  --onetalk-output-base "${ONETALK_OUTPUT_BASE}" \
  --remaining-output-base "${REMAINING_OUTPUT_BASE}" \
  --onetalk-density "${ONETALK_DENSITY}" \
  --remaining-density "${REMAINING_DENSITY}" \
  --aggregate-output-base "${AGG_OUTPUT_BASE}" \
  --aggregate-density "${AGG_DENSITY}" \
  --combined-glossary-tag "${COMBINED_GLOSSARY_TAG}" \
  --glossary-tag-pattern "${GLOSSARY_TAG_PATTERN}" \
  --oracle-term-map-tag-pattern "${ORACLE_TERM_MAP_TAG_PATTERN_OVERRIDE:-medicine.oracle_term_map__medicine_{sample}}" \
  --term-fcr-policy "${TERM_FCR_POLICY}"

HOME="${WANDB_HOME}" \
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME}/.config/wandb}" \
"${WANDB_PYTHON}" "${WANDB_LOGGER}" \
  --project simuleval_eval \
  --run-name "oraclegt_r32a64__medicine4__lm1to4__oracle_gt" \
  --experiment-family speech_llm_oracle_gt_sft_readout \
  --data-tag "${DATA_TAG}" \
  --task-tag eval \
  --notes-file "${NOTES_FILE}" \
  --extra-tags "variant:oraclegt_r32a64" "compute:${COMPUTE_TAG}" "oracle:gt" "sample:medicine4" \
  --density "${AGG_DENSITY}" \
  --rag-top-k "${RAG_TOP_K}" \
  --output-base "${AGG_OUTPUT_BASE}" \
  --lang-code "${LANG_CODE}" \
  --latency-multipliers ${TARGET_LMS} \
  --glossary-tag "${COMBINED_GLOSSARY_TAG}" \
  --model-name "${MODEL_NAME}" \
  --trained-from-run 3h4wm92o \
  --oracle-term-map \
  --verdict "Logged four-talk medicine oracle-GT lm2-4 readout for all-GT SFT oraclegt_r32a64."

HOME="${WANDB_HOME}" \
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME}/.config/wandb}" \
"${WANDB_PYTHON}" "${ROOT_DIR}/documents/code/general/wandb_tool.py" --project simuleval_eval db-sync \
  --family speech_llm_oracle_gt_sft_readout --best-bundles --limit 30 || true

echo "[ALL DONE] medicine oracle-GT lm sweep complete: ${AGG_OUTPUT_BASE}/${LANG_CODE}/summary_lm_sweep.tsv"
