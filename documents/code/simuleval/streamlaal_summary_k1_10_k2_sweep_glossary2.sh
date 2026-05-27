#!/usr/bin/env bash
set -euo pipefail

# Post-eval SimulEval outputs with stream_laal_term.py and summarize TERM_ACC into a TSV.
# This script matches the output naming from:
#   bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh
#
# All user-facing strings are in English.

# ======Configuration=====
# Exit codes
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

# Conda
CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="spaCyEnv"

# SimulEval outputs
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2"
LANG_CODE="zh"
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE:-}"
LANG_CODE_OVERRIDE="${LANG_CODE_OVERRIDE:-}"

# Dataset / references
DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
REF_FILE="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.${LANG_CODE}.txt"
AUDIO_YAML="${DATA_ROOT}/dev.yaml"

# Glossaries used during simuleval (must match the run script)
GLOSSARY_PATH_ACL6060="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
GLOSSARY_PATH_EXTRACTED="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"

# Optional overrides to support custom glossaries:
# - If GLOSSARY_PATH_OVERRIDE is set, use it for ALL outputs regardless of glossary_tag.
# - Otherwise, we try to resolve glossary_tag by searching for "${glossary_tag}.json" in GLOSSARY_SEARCH_DIRS.
#   You can override the search dirs via GLOSSARY_SEARCH_DIRS_OVERRIDE (space-separated).
GLOSSARY_PATH_OVERRIDE="${GLOSSARY_PATH_OVERRIDE:-}"
GLOSSARY_SEARCH_DIRS=(
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre"
  "/home/jiaxuanluo/InfiniSST/documents/data/data_pre"
)
GLOSSARY_SEARCH_DIRS_OVERRIDE="${GLOSSARY_SEARCH_DIRS_OVERRIDE:-}"

# FBK fairseq tool
FBK_FAIRSEQ_ROOT="/mnt/taurus/home/jiaxuanluo/FBK-fairseq"
STREAM_LAAL_TOOL_REL="examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"

# MWER segmenter
MWERSEGMENTER_ROOT="/mnt/taurus/home/jiaxuanluo/mwerSegmenter"

# Input/Output filenames under each run directory
INSTANCES_FILE="instances.log"
SIMULEVAL_STDOUT_LOG="simuleval.log"
POST_EVAL_LOG="post_eval.log"

# Glossary normalization:
# stream_laal_term.py expects a flat JSON dict: { "<key>": {"term":..., "target_translations": {...}}, ... }
# Some project glossaries are wrapped as: { "meta": {...}, "terms": { ... } }
GLOSSARY_TERMS_FIELD_NAME="terms"
GLOSSARY_META_FIELD_NAME="meta"
NORMALIZED_GLOSSARY_FILE_NAME="glossary_for_streamlaal_eval.json"

# Summary output (derived again after applying overrides)
SUMMARY_TSV="${OUTPUT_BASE}/${LANG_CODE}/k1_10_k2_sweep_glossary2_streamlaal_summary.tsv"

# Skip directories (avoid scanning backup dirs created by resume mode)
SKIP_DIR_GLOB="*_partial_backup_*"

# Parsing / display
SACREBLEU_TOKENIZER="zh"
LATENCY_UNIT="char"
TERM_LANG="zh"
TERM_MISMATCH_EXAMPLES="0"

# Limit work (0 means no limit). Useful for quick preview.
MAX_OUTPUT_DIRS="${MAX_OUTPUT_DIRS:-0}"

# Grep regex (PCRE) for rtf_total (optional)
RTF_TOTAL_REGEX='rtf_total=\\K[0-9.]+'
# ======Configuration=====

if [[ -n "${OUTPUT_BASE_OVERRIDE}" ]]; then
  OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE}"
fi
if [[ -n "${LANG_CODE_OVERRIDE}" ]]; then
  LANG_CODE="${LANG_CODE_OVERRIDE}"
fi

OUTPUT_LANG_BASE="${OUTPUT_BASE}/${LANG_CODE}"
SUMMARY_TSV="${OUTPUT_BASE}/${LANG_CODE}/k1_10_k2_sweep_glossary2_streamlaal_summary.tsv"

# Reference + tokenizer + term language should follow LANG_CODE.
REF_FILE="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.${LANG_CODE}.txt"
case "${LANG_CODE}" in
  zh)
    SACREBLEU_TOKENIZER="zh"
    LATENCY_UNIT="char"
    TERM_LANG="zh"
    ;;
  de)
    # stream_laal_term.py does not accept "de" as a sacrebleu tokenizer.
    # Use a supported tokenizer for German.
    SACREBLEU_TOKENIZER="13a"
    LATENCY_UNIT="word"
    TERM_LANG="de"
    ;;
  ja)
    # sacrebleu Japanese tokenization
    SACREBLEU_TOKENIZER="ja-mecab"
    LATENCY_UNIT="char"
    TERM_LANG="ja"
    ;;
  *)
    # best-effort: many sacrebleu tokenizers use language code
    SACREBLEU_TOKENIZER="${LANG_CODE}"
    # stream_laal_term.py supports latency units {word,char}. Default to word.
    LATENCY_UNIT="word"
    TERM_LANG="${LANG_CODE}"
    ;;
esac

# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
echo "[INFO] Activated conda env: ${CONDA_ENV_NAME}"

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
if [[ ! -f "${GLOSSARY_PATH_ACL6060}" ]]; then
  echo "[ERROR] GLOSSARY_PATH_ACL6060 missing: ${GLOSSARY_PATH_ACL6060}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${GLOSSARY_PATH_EXTRACTED}" ]]; then
  echo "[ERROR] GLOSSARY_PATH_EXTRACTED missing: ${GLOSSARY_PATH_EXTRACTED}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ -n "${GLOSSARY_SEARCH_DIRS_OVERRIDE}" ]]; then
  # shellcheck disable=SC2206
  GLOSSARY_SEARCH_DIRS=(${GLOSSARY_SEARCH_DIRS_OVERRIDE})
fi
if [[ -n "${GLOSSARY_PATH_OVERRIDE}" ]] && [[ ! -f "${GLOSSARY_PATH_OVERRIDE}" ]]; then
  echo "[ERROR] GLOSSARY_PATH_OVERRIDE not found: ${GLOSSARY_PATH_OVERRIDE}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

mkdir -p "${OUTPUT_LANG_BASE}"

{
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "timestamp" \
    "glossary_tag" \
    "vllm_segment_sec" \
    "latency_multiplier" \
    "K2" \
    "K1" \
    "BLEU" \
    "StreamLAAL" \
    "StreamLAAL_CA" \
    "TERM_ACC" \
    "TERM_CORRECT" \
    "TERM_TOTAL" \
    "output_path"
} > "${SUMMARY_TSV}"

shopt -s nullglob
PROCESSED_DIRS="0"
for out in "${OUTPUT_LANG_BASE}"/*_g*_cs*_hs*_lm*_k2*_k1*_th*; do
  [[ -d "${out}" ]] || continue
  BASE="$(basename "${out}")"
  if [[ "${BASE}" == ${SKIP_DIR_GLOB} ]]; then
    echo "[INFO] Skip backup dir: ${out}"
    continue
  fi
  if [[ "${MAX_OUTPUT_DIRS}" != "0" ]] && [[ "${PROCESSED_DIRS}" -ge "${MAX_OUTPUT_DIRS}" ]]; then
    echo "[INFO] Reached MAX_OUTPUT_DIRS=${MAX_OUTPUT_DIRS}, stop."
    break
  fi

  INSTANCES_PATH="${out}/${INSTANCES_FILE}"
  if [[ ! -f "${INSTANCES_PATH}" ]] || [[ ! -s "${INSTANCES_PATH}" ]]; then
    echo "[WARN] Missing/empty ${INSTANCES_FILE} (skip): ${out}" >&2
    continue
  fi

  # Extract meta from dirname (avoid truncation when tags contain underscores)
  # Example dirname:
  #   iter_0000452-hf_gglossary_acl6060_cs1.92_hs0.48_lm2_k210_k110
  GLOSSARY_TAG="${BASE#*_g}"
  GLOSSARY_TAG="${GLOSSARY_TAG%%_cs*}"

  VLLM_SEGMENT_SEC="$(echo "${BASE}" | sed -n 's/.*_cs\([0-9.]*\)_.*/\1/p')"
  LATENCY_MULTIPLIER="$(echo "${BASE}" | sed -n 's/.*_lm\([0-9]*\)_.*/\1/p')"
  K2="$(echo "${BASE}" | sed -n 's/.*_k2\([0-9]*\)_.*/\1/p')"
  # dirname now ends with "_k1<k1>_th<th>", so parse k1 from that pattern.
  K1="$(echo "${BASE}" | sed -n 's/.*_k1\([0-9]*\)_th.*/\1/p')"

  # Choose glossary path
  GLOSSARY_PATH=""
  if [[ -n "${GLOSSARY_PATH_OVERRIDE}" ]]; then
    GLOSSARY_PATH="${GLOSSARY_PATH_OVERRIDE}"
  else
    # Prefer tag-matched file under search dirs.
    for d in "${GLOSSARY_SEARCH_DIRS[@]}"; do
      cand="${d}/${GLOSSARY_TAG}.json"
      if [[ -f "${cand}" ]]; then
        GLOSSARY_PATH="${cand}"
        break
      fi
    done
    # Fallback to known defaults if we still cannot resolve.
    if [[ -z "${GLOSSARY_PATH}" ]]; then
      if [[ "${GLOSSARY_TAG}" == "$(basename "${GLOSSARY_PATH_ACL6060}" .json)" ]]; then
        GLOSSARY_PATH="${GLOSSARY_PATH_ACL6060}"
      elif [[ "${GLOSSARY_TAG}" == "$(basename "${GLOSSARY_PATH_EXTRACTED}" .json)" ]]; then
        GLOSSARY_PATH="${GLOSSARY_PATH_EXTRACTED}"
      else
        echo "[WARN] Unknown glossary_tag='${GLOSSARY_TAG}'. Using extracted glossary by default: ${GLOSSARY_PATH_EXTRACTED}" >&2
        GLOSSARY_PATH="${GLOSSARY_PATH_EXTRACTED}"
      fi
    fi
  fi

  if [[ ! -f "${GLOSSARY_PATH}" ]]; then
    echo "[ERROR] Resolved glossary path not found: ${GLOSSARY_PATH} (tag='${GLOSSARY_TAG}')" >&2
    exit "${EXIT_DATA_ERROR}"
  fi

  # Normalize glossary if it uses a wrapped format with {"meta":..., "terms":{...}}.
  # Write the normalized flat dict next to the run outputs for transparency/debugging.
  NORMALIZED_GLOSSARY_PATH="${out}/${NORMALIZED_GLOSSARY_FILE_NAME}"
  python3 - <<PY
import json
from pathlib import Path

src = Path("${GLOSSARY_PATH}")
dst = Path("${NORMALIZED_GLOSSARY_PATH}")
terms_field = "${GLOSSARY_TERMS_FIELD_NAME}"
meta_field = "${GLOSSARY_META_FIELD_NAME}"

obj = json.loads(src.read_text(encoding="utf-8"))
normalized = obj

if isinstance(obj, dict) and terms_field in obj and isinstance(obj.get(terms_field), dict):
    # Wrapped format -> extract terms dict
    normalized = obj[terms_field]
    if not normalized:
        raise SystemExit(f"[ERROR] Glossary has '{terms_field}' field but it's empty: {src}")

# Validate normalized structure is a dict
if not isinstance(normalized, dict):
    raise SystemExit(f"[ERROR] Unsupported glossary JSON format (expected dict): {src}")

dst.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  GLOSSARY_PATH="${NORMALIZED_GLOSSARY_PATH}"

  echo "[INFO] Post-eval: ${out} glossary_tag=${GLOSSARY_TAG}"

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
  if [[ "${EVAL_RC}" != "0" ]]; then
    echo "[WARN] Post-eval returned non-zero (rc=${EVAL_RC}): ${out}" >&2
  fi

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

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +'%Y-%m-%d %H:%M:%S')" \
    "${GLOSSARY_TAG}" \
    "${VLLM_SEGMENT_SEC}" \
    "${LATENCY_MULTIPLIER}" \
    "${K2}" \
    "${K1}" \
    "${BLEU}" \
    "${STREAM_LAAL}" \
    "${STREAM_LAAL_CA}" \
    "${TERM_ACC}" \
    "${TERM_CORRECT}" \
    "${TERM_TOTAL}" \
    "${out}" >> "${SUMMARY_TSV}"

  PROCESSED_DIRS="$((PROCESSED_DIRS + 1))"
done

echo "[INFO] Summary written: ${SUMMARY_TSV}"
column -t -s $'\t' "${SUMMARY_TSV}" | head -n 30



