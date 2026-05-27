#!/usr/bin/env bash
set -euo pipefail

# Direct launcher for Taurus hold allocation 45269. It evaluates the all-GT
# term_map SFT HF export on the same one-talk medicine oracle readout used for
# the origin_bsz4 zero-shot baseline.

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_LAUNCHER="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260518__medicine_onetalk_oracle_gt_aries.sh"
WANDB_LOGGER="${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260519__medicine_onetalk_oracle_gt_sft_oraclegt_r32a64.md}"
WANDB_PYTHON="${WANDB_PYTHON:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
WANDB_HOME="${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}"
PER_SAMPLE_DB_SYNC="${PER_SAMPLE_DB_SYNC_OVERRIDE:-0}"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/speech_llm_oracle_gt_termmap_zh_r32a64_taurus4/keep1.0_r32/v1-20260519-105111-hf}"
TARGET_SAMPLE="${TARGET_SAMPLE_OVERRIDE:-404}"
TARGET_LM="${TARGET_LM_OVERRIDE:-2}"
LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:-1.0}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-medicine1_oraclegt_r32a64}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/medicine_onetalk_oracle_gt_sft_oraclegt_r32a64_20260519}"
GLOSSARY_TAG="${GLOSSARY_TAG_OVERRIDE:-medicine_gt_strict_translated__medicine_${TARGET_SAMPLE}}"
GPU_CSV="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-4:5}"
COMPUTE_TAG="${COMPUTE_TAG_OVERRIDE:-taurus_hold45269}"
DATA_TAG="${DATA_TAG_OVERRIDE:-medicine_onetalk_zh}"

for p in "${BASE_LAUNCHER}" "${WANDB_LOGGER}" "${NOTES_FILE}" "${MODEL_NAME}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] TARGET_SAMPLE=${TARGET_SAMPLE} TARGET_LM=${TARGET_LM}"
echo "[INFO] GPU_CSV=${GPU_CSV}"

MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
TARGET_SAMPLE_OVERRIDE="${TARGET_SAMPLE}" \
TARGET_LM="${TARGET_LM}" \
LANG_CODE_OVERRIDE="${LANG_CODE}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${GPU_CSV}" \
RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
TERM_SOURCE_OVERRIDE="${TERM_SOURCE_OVERRIDE:-sentence_terms}" \
ORACLE_GLOSSARY_OVERRIDE="${ORACLE_GLOSSARY_OVERRIDE:-}" \
EVAL_GLOSSARY_OVERRIDE="${EVAL_GLOSSARY_OVERRIDE:-}" \
GLOSSARY_SOURCE_FILTER_OVERRIDE="${GLOSSARY_SOURCE_FILTER_OVERRIDE:-}" \
GLOSSARY_TAG_OVERRIDE="${GLOSSARY_TAG}" \
ORACLE_TERM_MAP_TAG_OVERRIDE="${ORACLE_TERM_MAP_TAG_OVERRIDE:-medicine.oracle_term_map__medicine_${TARGET_SAMPLE}}" \
TERM_FCR_POLICY_OVERRIDE="${TERM_FCR_POLICY_OVERRIDE:-term_map_if_available}" \
FORCE_RERUN_OVERRIDE="${FORCE_RERUN_OVERRIDE:-0}" \
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
MAX_SENTENCES_OVERRIDE="${MAX_SENTENCES_OVERRIDE:-0}" \
bash "${BASE_LAUNCHER}"

HOME="${WANDB_HOME}" \
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME}/.config/wandb}" \
"${WANDB_PYTHON}" "${WANDB_LOGGER}" \
  --project simuleval_eval \
  --run-name "oraclegt_r32a64__medicine_${TARGET_SAMPLE}__lm${TARGET_LM}__oracle_gt" \
  --experiment-family speech_llm_oracle_gt_sft_readout \
  --data-tag "${DATA_TAG}" \
  --task-tag eval \
  --notes-file "${NOTES_FILE}" \
  --extra-tags "variant:oraclegt_r32a64" "compute:${COMPUTE_TAG}" "oracle:gt" "sample:medicine_${TARGET_SAMPLE}" \
  --density "${DENSITY_TAG}" \
  --rag-top-k "${RAG_TOP_K}" \
  --rag-score-threshold "${RAG_SCORE_THRESHOLD}" \
  --paper-id "medicine_${TARGET_SAMPLE}" \
  --output-base "${OUTPUT_BASE}" \
  --lang-code "${LANG_CODE}" \
  --latency-multipliers "${TARGET_LM}" \
  --glossary-tag "${GLOSSARY_TAG}" \
  --model-name "${MODEL_NAME}" \
  --trained-from-run 3h4wm92o \
  --oracle-term-map \
  --verdict "Logged oracle term_map medicine readout for all-GT SFT oraclegt_r32a64."

if [[ "${PER_SAMPLE_DB_SYNC}" == "1" ]]; then
  HOME="${WANDB_HOME}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME}/.config/wandb}" \
  "${WANDB_PYTHON}" "${ROOT_DIR}/documents/code/general/wandb_tool.py" --project simuleval_eval db-sync \
    --family speech_llm_oracle_gt_sft_readout --best-bundles --limit 20 || true
else
  echo "[WARN] Skipping per-sample family db-sync (PER_SAMPLE_DB_SYNC_OVERRIDE=${PER_SAMPLE_DB_SYNC}); final aggregate sync or a later run-level sync should refresh SQLite."
fi

echo "[ALL DONE] oraclegt_r32a64 oracle medicine evaluation logged."
