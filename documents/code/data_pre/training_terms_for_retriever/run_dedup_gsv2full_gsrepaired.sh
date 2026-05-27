#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
SCRIPT_PATH="${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/dedup_gigaspeech_mfa_events.py"

INPUT_JSONL="${INPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl}"
STATS_JSON="${STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_stats.json}"
SEED="${SEED:-20260509}"

echo "[GSDEDUP] input=${INPUT_JSONL}"
echo "[GSDEDUP] output=${OUTPUT_JSONL}"
echo "[GSDEDUP] stats=${STATS_JSON}"
echo "[GSDEDUP] seed=${SEED}"

python "${SCRIPT_PATH}" \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_JSONL}" \
  --stats-json "${STATS_JSON}" \
  --seed "${SEED}" \
  --chunk-stride-sec 0.96 \
  --round-digits 4
