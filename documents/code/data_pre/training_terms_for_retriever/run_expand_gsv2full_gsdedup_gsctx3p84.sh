#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
SCRIPT_PATH="${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/expand_gigaspeech_context_3p84.py"

INPUT_JSONL="${INPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_gsctx3p84.jsonl}"
AUDIO_OUTPUT_DIR="${AUDIO_OUTPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_gsctx3p84}"
WIKI_AUDIO_OUTPUT_DIR="${WIKI_AUDIO_OUTPUT_DIR:-${AUDIO_OUTPUT_DIR}/wiki_synth}"
STATS_JSON="${STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_gsctx3p84_stats.json}"

MAX_LINES="${MAX_LINES:-0}"
MAX_GS_GROUPS="${MAX_GS_GROUPS:-0}"
OVERWRITE_AUDIO="${OVERWRITE_AUDIO:-false}"
DRY_RUN="${DRY_RUN:-false}"

echo "[GSCTX3P84] input=${INPUT_JSONL}"
echo "[GSCTX3P84] output=${OUTPUT_JSONL}"
echo "[GSCTX3P84] audio_output_dir=${AUDIO_OUTPUT_DIR}"
echo "[GSCTX3P84] wiki_audio_output_dir=${WIKI_AUDIO_OUTPUT_DIR}"
echo "[GSCTX3P84] stats=${STATS_JSON}"
echo "[GSCTX3P84] max_lines=${MAX_LINES} max_gs_groups=${MAX_GS_GROUPS}"
echo "[GSCTX3P84] overwrite_audio=${OVERWRITE_AUDIO} dry_run=${DRY_RUN}"

extra_args=()
if [ "${MAX_LINES}" != "0" ]; then
  extra_args+=(--max-lines "${MAX_LINES}")
fi
if [ "${MAX_GS_GROUPS}" != "0" ]; then
  extra_args+=(--max-gs-groups "${MAX_GS_GROUPS}")
fi
if [ "${OVERWRITE_AUDIO}" = "true" ]; then
  extra_args+=(--overwrite-audio)
fi
if [ "${DRY_RUN}" = "true" ]; then
  extra_args+=(--dry-run)
fi

python "${SCRIPT_PATH}" \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_JSONL}" \
  --audio-output-dir "${AUDIO_OUTPUT_DIR}" \
  --wiki-audio-output-dir "${WIKI_AUDIO_OUTPUT_DIR}" \
  --stats-json "${STATS_JSON}" \
  --old-chunk-sec 1.92 \
  --new-chunk-sec 3.84 \
  --stride-sec 0.96 \
  --include-mode overlap \
  "${extra_args[@]}"
