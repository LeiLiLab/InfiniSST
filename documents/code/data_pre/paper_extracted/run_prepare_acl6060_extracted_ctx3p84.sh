#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
SCRIPT_PATH="${REPO_ROOT}/documents/code/data_pre/paper_extracted/prepare_acl6060_extracted_paper_glossary.py"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_ctx3p84}"
CHUNK_AUDIO_DIR="${CHUNK_AUDIO_DIR:-/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_ctx3p84/audio_chunks}"
CHUNK_SEC="${CHUNK_SEC:-3.84}"
STRIDE_SEC="${STRIDE_SEC:-1.92}"
OVERWRITE_AUDIO="${OVERWRITE_AUDIO:-false}"

echo "[ACL_CTX3P84] output_dir=${OUTPUT_DIR}"
echo "[ACL_CTX3P84] chunk_audio_dir=${CHUNK_AUDIO_DIR}"
echo "[ACL_CTX3P84] chunk_sec=${CHUNK_SEC} stride_sec=${STRIDE_SEC}"
echo "[ACL_CTX3P84] overwrite_audio=${OVERWRITE_AUDIO}"

extra_args=()
if [ "${OVERWRITE_AUDIO}" = "true" ]; then
  extra_args+=(--overwrite-audio)
fi

python "${SCRIPT_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --chunk-audio-dir "${CHUNK_AUDIO_DIR}" \
  --chunk-sec "${CHUNK_SEC}" \
  --stride-sec "${STRIDE_SEC}" \
  "${extra_args[@]}"
