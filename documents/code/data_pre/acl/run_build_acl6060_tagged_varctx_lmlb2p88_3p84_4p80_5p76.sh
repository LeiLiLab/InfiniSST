#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_tagged_glossary_varctx2p88_3p84_4p80_5p76}"
CHUNK_AUDIO_DIR="${CHUNK_AUDIO_DIR:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_tagged_varctx2p88_3p84_4p80_5p76/audio_chunks}"
OUTPUT_JSONL_NAME="${OUTPUT_JSONL_NAME:-acl6060_tagged_dev_dataset.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/${OUTPUT_JSONL_NAME}}"
STATS_JSON="${STATS_JSON:-${OUTPUT_DIR}/acl6060_tagged_dev_dataset_stats.json}"
DIAG_JSON="${DIAG_JSON:-${OUTPUT_DIR}/acl6060_tagged_dev_dataset_diag.json}"
EVAL_GLOSSARY_OUTPUT="${EVAL_GLOSSARY_OUTPUT:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
TAGGED_GLOSSARY_JSON="${TAGGED_GLOSSARY_JSON:-${REPO_ROOT}/documents/data/data_pre/glossary_acl6060.json}"
TAGGED_TEXT="${TAGGED_TEXT:-/mnt/data/siqiouyang/datasets/acl6060/dev/text/tagged_terminology/ACL.6060.dev.tagged.en-xx.en.txt}"
WIKI_GLOSSARY_JSON="${WIKI_GLOSSARY_JSON:-${REPO_ROOT}/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs_enriched.json}"
DURATION_SECS="${DURATION_SECS:-2.88 3.84 4.80 5.76}"
DIAG_TARGET_FRAC="${DIAG_TARGET_FRAC:-}"
MIN_NORM_CHARS="${MIN_NORM_CHARS:-2}"
OVERWRITE_AUDIO="${OVERWRITE_AUDIO:-false}"

extra_args=()
if [ "${OVERWRITE_AUDIO}" = "true" ]; then
  extra_args+=(--overwrite-audio)
fi

echo "[ACL-TAGGED-VARCTX] output_dir=${OUTPUT_DIR}"
echo "[ACL-TAGGED-VARCTX] output_jsonl=${OUTPUT_JSONL}"
echo "[ACL-TAGGED-VARCTX] chunk_audio_dir=${CHUNK_AUDIO_DIR}"
echo "[ACL-TAGGED-VARCTX] eval_glossary=${EVAL_GLOSSARY_OUTPUT}"
echo "[ACL-TAGGED-VARCTX] durations=${DURATION_SECS}"
echo "[ACL-TAGGED-VARCTX] diag_target_frac=${DIAG_TARGET_FRAC:-default}"
echo "[ACL-TAGGED-VARCTX] min_norm_chars=${MIN_NORM_CHARS}"
echo "[ACL-TAGGED-VARCTX] overwrite_audio=${OVERWRITE_AUDIO}"

python "${REPO_ROOT}/documents/code/data_pre/acl/prepare_acl6060_tagged_variable_context.py" \
  --output-dir "${OUTPUT_DIR}" \
  --output-jsonl-name "${OUTPUT_JSONL_NAME}" \
  --chunk-audio-dir "${CHUNK_AUDIO_DIR}" \
  --tagged-text "${TAGGED_TEXT}" \
  --tagged-glossary-json "${TAGGED_GLOSSARY_JSON}" \
  --wiki-glossary-json "${WIKI_GLOSSARY_JSON}" \
  --eval-glossary-output "${EVAL_GLOSSARY_OUTPUT}" \
  --duration-secs "${DURATION_SECS}" \
  --min-norm-chars "${MIN_NORM_CHARS}" \
  --stats-json "${STATS_JSON}" \
  "${extra_args[@]}"

diag_args=()
if [ -n "${DIAG_TARGET_FRAC}" ]; then
  diag_args+=(--target-frac "${DIAG_TARGET_FRAC}")
fi

python "${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/diagnose_variable_context_jsonl.py" \
  --input "${OUTPUT_JSONL}" \
  --stats-json "${STATS_JSON}" \
  --expected-duration-secs "${DURATION_SECS}" \
  --report-json "${DIAG_JSON}" \
  "${diag_args[@]}"

echo "[ACL-TAGGED-VARCTX] DONE diag=${DIAG_JSON}"
