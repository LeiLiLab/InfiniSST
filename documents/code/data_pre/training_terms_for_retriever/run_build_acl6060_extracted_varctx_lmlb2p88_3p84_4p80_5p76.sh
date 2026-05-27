#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76}"
CHUNK_AUDIO_DIR="${CHUNK_AUDIO_DIR:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_varctx2p88_3p84_4p80_5p76/audio_chunks}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/acl6060_dev_dataset.jsonl}"
STATS_JSON="${STATS_JSON:-${OUTPUT_DIR}/acl6060_dev_dataset_stats.json}"
DIAG_JSON="${DIAG_JSON:-${OUTPUT_DIR}/acl6060_dev_dataset_diag.json}"
DURATION_SECS="${DURATION_SECS:-2.88 3.84 4.80 5.76}"
OVERWRITE_AUDIO="${OVERWRITE_AUDIO:-false}"

extra_args=()
if [ "${OVERWRITE_AUDIO}" = "true" ]; then
  extra_args+=(--overwrite-audio)
fi

echo "[ACL-VARCTX] output_dir=${OUTPUT_DIR}"
echo "[ACL-VARCTX] chunk_audio_dir=${CHUNK_AUDIO_DIR}"
echo "[ACL-VARCTX] durations=${DURATION_SECS}"
echo "[ACL-VARCTX] overwrite_audio=${OVERWRITE_AUDIO}"

python "${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/prepare_acl6060_extracted_variable_context.py" \
  --output-dir "${OUTPUT_DIR}" \
  --chunk-audio-dir "${CHUNK_AUDIO_DIR}" \
  --duration-secs "${DURATION_SECS}" \
  --stats-json "${STATS_JSON}" \
  "${extra_args[@]}"

python "${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/diagnose_variable_context_jsonl.py" \
  --input "${OUTPUT_JSONL}" \
  --stats-json "${STATS_JSON}" \
  --expected-duration-secs "${DURATION_SECS}" \
  --report-json "${DIAG_JSON}"

echo "[ACL-VARCTX] DONE diag=${DIAG_JSON}"
