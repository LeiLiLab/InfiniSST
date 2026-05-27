#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="${INPUT_DIR:-/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test}"
MFA_TEXTGRID_DIR="${MFA_TEXTGRID_DIR:-/home/jiaxingxu/rag-sst/eso-dataset/mfa_v1/textgrids}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76}"
CHUNK_AUDIO_DIR="${CHUNK_AUDIO_DIR:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76/audio_chunks}"
FILLER_GLOSSARY="${FILLER_GLOSSARY:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_medicine_enriched.json}"
FILLER_SOURCE="${FILLER_SOURCE:-medicine_wiki_filler}"
DURATION_SECS="${DURATION_SECS:-2.88 3.84 4.80 5.76}"
MAX_BASE_CHUNKS_PER_SAMPLE="${MAX_BASE_CHUNKS_PER_SAMPLE:-0}"
OVERWRITE_AUDIO="${OVERWRITE_AUDIO:-false}"
UNMATCHED_TERM_POLICY="${UNMATCHED_TERM_POLICY:-drop}"
DROPPED_TERMS_JSON="${DROPPED_TERMS_JSON:-}"
ALLOWED_LOCATE_METHODS="${ALLOWED_LOCATE_METHODS:-mfa_exact char_proportional}"

OPTS=()
if [ "${MAX_BASE_CHUNKS_PER_SAMPLE}" -gt 0 ]; then
  OPTS+=(--max-base-chunks-per-sample "${MAX_BASE_CHUNKS_PER_SAMPLE}")
fi
if [ "${OVERWRITE_AUDIO}" = "true" ]; then
  OPTS+=(--overwrite-audio)
fi
if [ -n "${DROPPED_TERMS_JSON}" ]; then
  OPTS+=(--dropped-terms-json "${DROPPED_TERMS_JSON}")
fi

echo "[MED-VARCTX] input=${INPUT_DIR}"
echo "[MED-VARCTX] mfa=${MFA_TEXTGRID_DIR}"
echo "[MED-VARCTX] output=${OUTPUT_DIR}/medicine_dev_dataset.jsonl"
echo "[MED-VARCTX] filler_glossary=${FILLER_GLOSSARY}"
echo "[MED-VARCTX] glossary=${OUTPUT_DIR}/medicine_glossary_gt_plus_medicine_wiki_gs10000.json"
echo "[MED-VARCTX] audio=${CHUNK_AUDIO_DIR}"
echo "[MED-VARCTX] duration_secs=${DURATION_SECS}"
echo "[MED-VARCTX] max_base_chunks_per_sample=${MAX_BASE_CHUNKS_PER_SAMPLE}"
echo "[MED-VARCTX] unmatched_term_policy=${UNMATCHED_TERM_POLICY}"
echo "[MED-VARCTX] allowed_locate_methods=${ALLOWED_LOCATE_METHODS}"
if [ -n "${DROPPED_TERMS_JSON}" ]; then
  echo "[MED-VARCTX] dropped_terms_json=${DROPPED_TERMS_JSON}"
fi

python3 "${SCRIPT_DIR}/prepare_medicine_variable_context.py" \
  --input-dir "${INPUT_DIR}" \
  --mfa-textgrid-dir "${MFA_TEXTGRID_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --chunk-audio-dir "${CHUNK_AUDIO_DIR}" \
  --duration-secs "${DURATION_SECS}" \
  --filler-glossary "${FILLER_GLOSSARY}" \
  --filler-source "${FILLER_SOURCE}" \
  --unmatched-term-policy "${UNMATCHED_TERM_POLICY}" \
  --allowed-locate-methods "${ALLOWED_LOCATE_METHODS}" \
  "${OPTS[@]}"
