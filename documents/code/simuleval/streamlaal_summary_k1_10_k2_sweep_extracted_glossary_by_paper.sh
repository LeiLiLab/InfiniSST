#!/usr/bin/env bash
set -euo pipefail

# Summarize StreamLAAL + TERM metrics for the "paper extracted glossary" runs,
# but compute TERM metrics using a per-paper glossary chosen by each instance's talk id.
#
# This script calls:
#   streamlaal_summary_k1_10_k2_sweep_extracted_glossary_by_paper.py
#
# All user-facing strings are in English.

# ======Configuration=====
# Exit codes
EXIT_CONFIG_ERROR="2"

# Conda
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"

# SimulEval outputs
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2"
LANG_CODE="zh"

# Dataset / references
DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
REF_FILE="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.zh.txt"
AUDIO_YAML="${DATA_ROOT}/dev.yaml"

# Extracted glossary (contains source_paper)
EXTRACTED_GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"

# FBK fairseq tool
FBK_FAIRSEQ_ROOT="/mnt/taurus/home/jiaxuanluo/FBK-fairseq"
STREAM_LAAL_TOOL_REL="examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"

# MWER segmenter (required by stream_laal_term.py)
MWERSEGMENTER_ROOT="/mnt/taurus/home/jiaxuanluo/mwerSegmenter"

# Metrics options
SACREBLEU_TOKENIZER="zh"
LATENCY_UNIT="char"
TERM_LANG="zh"
TERM_MISMATCH_EXAMPLES="0"

# Limit work (0 means no limit). Useful for quick preview.
MAX_OUTPUT_DIRS="${MAX_OUTPUT_DIRS:-0}"

# Output naming (TSV will be written under ${OUTPUT_BASE}/${LANG_CODE}/)
SUMMARY_TSV_NAME="k1_10_k2_sweep_extracted_glossary_by_paper_streamlaal_summary.tsv"

# Script path
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
PY_SCRIPT="${ROOT_DIR}/documents/code/simuleval/streamlaal_summary_k1_10_k2_sweep_extracted_glossary_by_paper.py"
# ======Configuration=====

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Python script not found: ${PY_SCRIPT}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
echo "[INFO] Activated conda env: ${CONDA_ENV_NAME}"

export MWERSEGMENTER_ROOT
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

python3 "${PY_SCRIPT}" \
  --output-base "${OUTPUT_BASE}" \
  --lang-code "${LANG_CODE}" \
  --ref-file "${REF_FILE}" \
  --audio-yaml "${AUDIO_YAML}" \
  --extracted-glossary "${EXTRACTED_GLOSSARY_PATH}" \
  --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
  --stream-laal-tool-rel "${STREAM_LAAL_TOOL_REL}" \
  --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
  --latency-unit "${LATENCY_UNIT}" \
  --term-lang "${TERM_LANG}" \
  --term-mismatch-examples "${TERM_MISMATCH_EXAMPLES}" \
  --max-output-dirs "${MAX_OUTPUT_DIRS}" \
  --summary-tsv-name "${SUMMARY_TSV_NAME}"

SUMMARY_TSV="${OUTPUT_BASE}/${LANG_CODE}/${SUMMARY_TSV_NAME}"
echo "[INFO] Summary TSV: ${SUMMARY_TSV}"
if [[ -f "${SUMMARY_TSV}" ]]; then
  column -t -s $'\t' "${SUMMARY_TSV}" | head -n 30
fi


