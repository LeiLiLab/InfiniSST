#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
SCRIPT_PATH="${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/build_variable_gigaspeech_context.py"

INPUT_JSONL="${INPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx0p96_1p92_2p88_3p84.jsonl}"
AUDIO_OUTPUT_DIR="${AUDIO_OUTPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_varctx0p96_1p92_2p88_3p84}"
WIKI_AUDIO_OUTPUT_DIR="${WIKI_AUDIO_OUTPUT_DIR:-${AUDIO_OUTPUT_DIR}/wiki_synth}"
STATS_JSON="${STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx0p96_1p92_2p88_3p84_stats.json}"

DURATION_SECS="${DURATION_SECS:-0.96 1.92 2.88 3.84}"
DURATION_ASSIGNMENT="${DURATION_ASSIGNMENT:-balance_rows}"
MAX_LINES="${MAX_LINES:-0}"
MAX_GS_GROUPS="${MAX_GS_GROUPS:-0}"
OVERWRITE_AUDIO="${OVERWRITE_AUDIO:-false}"
DRY_RUN="${DRY_RUN:-false}"
REUSE_OLD_AUDIO_FOR_1P92="${REUSE_OLD_AUDIO_FOR_1P92:-true}"

echo "[VARCTX] input=${INPUT_JSONL}"
echo "[VARCTX] output=${OUTPUT_JSONL}"
echo "[VARCTX] audio_output_dir=${AUDIO_OUTPUT_DIR}"
echo "[VARCTX] wiki_audio_output_dir=${WIKI_AUDIO_OUTPUT_DIR}"
echo "[VARCTX] stats=${STATS_JSON}"
echo "[VARCTX] duration_secs=${DURATION_SECS} assignment=${DURATION_ASSIGNMENT}"
echo "[VARCTX] max_lines=${MAX_LINES} max_gs_groups=${MAX_GS_GROUPS}"
echo "[VARCTX] overwrite_audio=${OVERWRITE_AUDIO} dry_run=${DRY_RUN} reuse_old_audio_for_1p92=${REUSE_OLD_AUDIO_FOR_1P92}"

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
if [ "${REUSE_OLD_AUDIO_FOR_1P92}" = "true" ]; then
  extra_args+=(--reuse-old-audio-for-1p92)
else
  extra_args+=(--no-reuse-old-audio-for-1p92)
fi

python "${SCRIPT_PATH}" \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_JSONL}" \
  --audio-output-dir "${AUDIO_OUTPUT_DIR}" \
  --wiki-audio-output-dir "${WIKI_AUDIO_OUTPUT_DIR}" \
  --stats-json "${STATS_JSON}" \
  --old-chunk-sec 1.92 \
  --stride-sec 0.96 \
  --duration-secs "${DURATION_SECS}" \
  --duration-assignment "${DURATION_ASSIGNMENT}" \
  --include-mode overlap \
  "${extra_args[@]}"
