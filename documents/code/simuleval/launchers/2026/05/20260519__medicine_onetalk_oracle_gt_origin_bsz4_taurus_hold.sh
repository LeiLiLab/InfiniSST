#!/usr/bin/env bash
set -euo pipefail

# Direct launcher for Taurus hold allocation 45269.  It evaluates the existing
# pure-streaming HF baseline in oracle term_map mode, without additional SFT.

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_LAUNCHER="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260518__medicine_onetalk_oracle_gt_aries.sh"
WANDB_LOGGER="${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py"
NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260519__medicine_onetalk_oracle_gt_origin_bsz4_zeroshot.md"
WANDB_PYTHON="${WANDB_PYTHON:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
WANDB_HOME="${WANDB_HOME:-/mnt/taurus/home/jiaxuanluo}"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4}"
TARGET_SAMPLE="${TARGET_SAMPLE_OVERRIDE:-404}"
TARGET_LM="${TARGET_LM_OVERRIDE:-2}"
LANG_CODE="${LANG_CODE_OVERRIDE:-zh}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD_OVERRIDE:-1.0}"
DENSITY_TAG="${DENSITY_TAG_OVERRIDE:-medicine1_origin_bsz4}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data2/jiaxuanluo/medicine_onetalk_oracle_gt_origin_bsz4_20260519}"
GPU_CSV="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-4:5}"

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
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
MAX_SENTENCES_OVERRIDE="${MAX_SENTENCES_OVERRIDE:-0}" \
bash "${BASE_LAUNCHER}"

glossary_tag="medicine_gt_strict_translated__medicine_${TARGET_SAMPLE}"
HOME="${WANDB_HOME}" \
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME}/.config/wandb}" \
"${WANDB_PYTHON}" "${WANDB_LOGGER}" \
  --project simuleval_eval \
  --run-name "origin_bsz4__medicine_${TARGET_SAMPLE}__lm${TARGET_LM}__oracle_gt" \
  --experiment-family speech_llm_oracle_gt_zeroshot \
  --data-tag "medicine_onetalk_zh" \
  --task-tag eval \
  --notes-file "${NOTES_FILE}" \
  --extra-tags "variant:origin_bsz4" "compute:taurus_hold45269" "oracle:gt" "sample:medicine_${TARGET_SAMPLE}" \
  --density "${DENSITY_TAG}" \
  --rag-top-k "${RAG_TOP_K}" \
  --rag-score-threshold "${RAG_SCORE_THRESHOLD}" \
  --paper-id "medicine_${TARGET_SAMPLE}" \
  --output-base "${OUTPUT_BASE}" \
  --lang-code "${LANG_CODE}" \
  --latency-multipliers "${TARGET_LM}" \
  --glossary-tag "${glossary_tag}" \
  --model-name "${MODEL_NAME}" \
  --oracle-term-map \
  --verdict "Logged zero-shot oracle term_map medicine readout for origin_bsz4 baseline."

HOME="${WANDB_HOME}" \
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_HOME}/.config/wandb}" \
"${WANDB_PYTHON}" "${ROOT_DIR}/documents/code/general/wandb_tool.py" --project simuleval_eval db-sync \
  --family speech_llm_oracle_gt_zeroshot --best-bundles --limit 20 || true

echo "[ALL DONE] origin_bsz4 oracle medicine zero-shot evaluation logged."
