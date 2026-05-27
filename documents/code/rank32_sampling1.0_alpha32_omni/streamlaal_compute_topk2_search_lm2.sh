#!/usr/bin/env bash
set -euo pipefail

# Compute StreamLAAL/TERM metrics for the "topk2 search (lm2)" runs and summarize into TSV.
# All user-facing strings are in English.

# ======Configuration=====
# Exit codes
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

# Conda
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"

# SimulEval outputs (topk2 search lm2)
OUTPUT_ZH_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_topk2_search_lm2/zh"
OUTPUT_DIR_GLOB="iter_0000452-hf_cs1.92_hs0.48_lm2_rk*_vk5"

# Dataset / references
DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
REF_FILE="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.zh.txt"
AUDIO_YAML="${DATA_ROOT}/dev.yaml"

# Term evaluation
GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
TERM_LANG="zh"

# FBK fairseq tool
FBK_FAIRSEQ_ROOT="/mnt/taurus/home/jiaxuanluo/FBK-fairseq"
STREAM_LAAL_TOOL_REL="examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"

# MWER segmenter
MWERSEGMENTER_ROOT="/mnt/taurus/home/jiaxuanluo/mwerSegmenter"

# SimulEval log file naming
SIMULEVAL_INSTANCES_FILE="instances.log"
SIMULEVAL_STDOUT_LOG="simuleval.log"
POST_EVAL_LOG="post_eval.log"

# Output
SUMMARY_TSV="${OUTPUT_ZH_BASE}/topk2_search_lm2_streamlaal_summary.tsv"

# Limit work (0 means no limit). Useful for quick preview.
MAX_OUTPUT_DIRS="${MAX_OUTPUT_DIRS:-0}"

# Parsing / display
SACREBLEU_TOKENIZER="zh"
LATENCY_UNIT="char"
TERM_MISMATCH_EXAMPLES="0"

# sed extraction patterns for fields from dirname
SED_EXTRACT_CS='s/.*_cs\([0-9.]*\)_.*/\1/p'
SED_EXTRACT_HS='s/.*_hs\([0-9.]*\)_.*/\1/p'
SED_EXTRACT_LM='s/.*_lm\([0-9]*\)_.*/\1/p'
SED_EXTRACT_RK='s/.*_rk\([0-9]*\)_.*/\1/p'
SED_EXTRACT_VK='s/.*_vk\([0-9]*\)$/\1/p'

# Grep regex (PCRE) for rtf_total
RTF_TOTAL_REGEX='rtf_total=\\K[0-9.]+'
# ======Configuration=====

# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export MWERSEGMENTER_ROOT
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

STREAM_LAAL_TOOL="${FBK_FAIRSEQ_ROOT}/${STREAM_LAAL_TOOL_REL}"
if [[ ! -f "${STREAM_LAAL_TOOL}" ]]; then
  echo "[ERROR] stream_laal_term.py not found: ${STREAM_LAAL_TOOL}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
if [[ ! -f "${REF_FILE}" ]]; then
  echo "[ERROR] REF_FILE missing: ${REF_FILE}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${AUDIO_YAML}" ]]; then
  echo "[ERROR] AUDIO_YAML missing: ${AUDIO_YAML}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${GLOSSARY_PATH}" ]]; then
  echo "[ERROR] GLOSSARY_PATH missing: ${GLOSSARY_PATH}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

{
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "vllm_segment_sec" \
    "chunk_hop_sec" \
    "latency_multiplier" \
    "recall_k" \
    "voting_k" \
    "eval_rc" \
    "BLEU" \
    "StreamLAAL" \
    "StreamLAAL_CA" \
    "TERM_ACC" \
    "TERM_CORRECT" \
    "TERM_TOTAL" \
    "RTF" \
    "output_path"
} > "${SUMMARY_TSV}"

shopt -s nullglob
PROCESSED_DIRS="0"
for out in "${OUTPUT_ZH_BASE}"/${OUTPUT_DIR_GLOB}; do
  [[ -d "${out}" ]] || continue
  if [[ "${MAX_OUTPUT_DIRS}" != "0" ]] && [[ "${PROCESSED_DIRS}" -ge "${MAX_OUTPUT_DIRS}" ]]; then
    echo "[INFO] Reached MAX_OUTPUT_DIRS=${MAX_OUTPUT_DIRS}, stop."
    break
  fi

  INSTANCES_PATH="${out}/${SIMULEVAL_INSTANCES_FILE}"
  if [[ ! -f "${INSTANCES_PATH}" ]] || [[ ! -s "${INSTANCES_PATH}" ]]; then
    echo "[WARN] Missing/empty ${SIMULEVAL_INSTANCES_FILE} (skip): ${out}" >&2
    continue
  fi

  echo "[INFO] Post-eval: ${out}"

  set +e
  EVAL_OUT="$(
    python "${STREAM_LAAL_TOOL}" \
      --simuleval-instances "${INSTANCES_PATH}" \
      --reference "${REF_FILE}" \
      --audio-yaml "${AUDIO_YAML}" \
      --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
      --latency-unit "${LATENCY_UNIT}" \
      --glossary "${GLOSSARY_PATH}" \
      --term-lang "${TERM_LANG}" \
      --term-mismatch-examples "${TERM_MISMATCH_EXAMPLES}" 2>&1
  )"
  EVAL_RC="$?"
  set -e

  echo "${EVAL_OUT}" > "${out}/${POST_EVAL_LOG}"

  METRIC_LINE="$(
    echo "${EVAL_OUT}" | awk '
    function isnum(x){ return (x ~ /^[0-9]+(\.[0-9]+)?$/) }
    NF>=3 && isnum($1) && isnum($2) && isnum($3) { print $1"\t"$2"\t"$3; exit }
    '
  )"
  BLEU="$(echo "${METRIC_LINE}" | awk -F'\t' '{print $1}')"
  STREAM_LAAL="$(echo "${METRIC_LINE}" | awk -F'\t' '{print $2}')"
  STREAM_LAAL_CA="$(echo "${METRIC_LINE}" | awk -F'\t' '{print $3}')"

  TERM_LINE="$(echo "${EVAL_OUT}" | awk '/^TERM_ACC[[:space:]]/{print; exit}')"
  TERM_ACC="$(echo "${TERM_LINE}" | awk '{print $2}')"
  TERM_CORRECT="$(echo "${TERM_LINE}" | awk '{print $4}')"
  TERM_TOTAL="$(echo "${TERM_LINE}" | awk '{print $6}')"

  RTF_TOTAL=""
  if [[ -f "${out}/${SIMULEVAL_STDOUT_LOG}" ]]; then
    RTF_TOTAL="$(grep -oP "${RTF_TOTAL_REGEX}" "${out}/${SIMULEVAL_STDOUT_LOG}" 2>/dev/null | tail -n 1 || true)"
  fi

  BASE="$(basename "${out}")"
  VLLM_SEGMENT_SEC="$(echo "${BASE}" | sed -n "${SED_EXTRACT_CS}")"
  HOP_SEC="$(echo "${BASE}" | sed -n "${SED_EXTRACT_HS}")"
  LATENCY_MULTIPLIER="$(echo "${BASE}" | sed -n "${SED_EXTRACT_LM}")"
  RECALL_K="$(echo "${BASE}" | sed -n "${SED_EXTRACT_RK}")"
  VOTING_K="$(echo "${BASE}" | sed -n "${SED_EXTRACT_VK}")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${VLLM_SEGMENT_SEC}" \
    "${HOP_SEC}" \
    "${LATENCY_MULTIPLIER}" \
    "${RECALL_K}" \
    "${VOTING_K}" \
    "${EVAL_RC}" \
    "${BLEU}" \
    "${STREAM_LAAL}" \
    "${STREAM_LAAL_CA}" \
    "${TERM_ACC}" \
    "${TERM_CORRECT}" \
    "${TERM_TOTAL}" \
    "${RTF_TOTAL}" \
    "${out}" >> "${SUMMARY_TSV}"

  PROCESSED_DIRS="$((PROCESSED_DIRS + 1))"
done

echo "[INFO] Summary written: ${SUMMARY_TSV}"
column -t -s $'\t' "${SUMMARY_TSV}" | head -n 30


