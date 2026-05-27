#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)"
PREP_DIR="${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever"

export INPUT_DIR="${INPUT_DIR:-/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2/test}"
export MFA_TEXTGRID_DIR="${MFA_TEXTGRID_DIR:-/home/jiaxingxu/rag-sst/eso-dataset/mfa_v1/textgrids}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only}"
export CHUNK_AUDIO_DIR="${CHUNK_AUDIO_DIR:-${OUTPUT_DIR}/audio_chunks}"
export FILLER_GLOSSARY="${FILLER_GLOSSARY:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_medicine_enriched.json}"
export FILLER_SOURCE="${FILLER_SOURCE:-medicine_wiki_filler}"
export DURATION_SECS="${DURATION_SECS:-2.88 3.84 4.80 5.76}"
export UNMATCHED_TERM_POLICY="${UNMATCHED_TERM_POLICY:-drop}"
export ALLOWED_LOCATE_METHODS="${ALLOWED_LOCATE_METHODS:-mfa_exact}"
export DROPPED_TERMS_JSON="${DROPPED_TERMS_JSON:-${OUTPUT_DIR}/medicine_dev_dataset_dropped_terms.json}"
export MAX_BASE_CHUNKS_PER_SAMPLE="${MAX_BASE_CHUNKS_PER_SAMPLE:-0}"
export OVERWRITE_AUDIO="${OVERWRITE_AUDIO:-false}"

exec bash "${PREP_DIR}/run_prepare_medicine_varctx_lmlb2p88_3p84_4p80_5p76.sh"
