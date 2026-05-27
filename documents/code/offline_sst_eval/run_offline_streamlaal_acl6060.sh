#!/usr/bin/env bash
set -euo pipefail

# Compute StreamLAAL + BLEU + TERM metrics for offline instances.log using ACL6060 glossary.
#
# This script writes per-language TSV/log files into:
#   <offline_sst_eval>/<pair>/offline/
#
# All user-facing strings are in English.

# ======Configuration=====
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

ROOT_DIR="/home/jiaxuanluo/InfiniSST"
OFFLINE_ROOT="${ROOT_DIR}/documents/code/offline_sst_eval"

DATA_ROOT="/mnt/taurus/data/siqiouyang/datasets/acl6060"
AUDIO_YAML="${DATA_ROOT}/dev.yaml"

GLOSSARY_ACL6060="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060.json"

FBK_FAIRSEQ_ROOT="/mnt/taurus/home/jiaxuanluo/FBK-fairseq"
STREAM_LAAL_TOOL_REL="examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"

MWERSEGMENTER_ROOT="/mnt/taurus/home/jiaxuanluo/mwerSegmenter"

# Optional conda activation (recommended)
CONDA_BASE="/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="infinisst"

PY_SCRIPT="${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py"
# ======Configuration=====

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Python script not found: ${PY_SCRIPT}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
if [[ ! -f "${AUDIO_YAML}" ]]; then
  echo "[ERROR] AUDIO_YAML not found: ${AUDIO_YAML}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${GLOSSARY_ACL6060}" ]]; then
  echo "[ERROR] GLOSSARY_ACL6060 not found: ${GLOSSARY_ACL6060}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

export MWERSEGMENTER_ROOT
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

PYTHON_BIN="$(command -v python3 || true)"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if conda activate "${CONDA_ENV_NAME}" >/dev/null 2>&1; then
    echo "[INFO] Activated conda env: ${CONDA_ENV_NAME}"
    PYTHON_BIN="$(python -c 'import sys; print(sys.executable)')"
  else
    echo "[WARN] Failed to activate conda env: ${CONDA_ENV_NAME}. Continue with system python."
  fi
else
  echo "[WARN] CONDA_BASE not found: ${CONDA_BASE}. Continue with system python."
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[ERROR] python3 not found in PATH." >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

run_one() {
  local pair="$1"
  local lang_code="$2"

  local offline_dir="${OFFLINE_ROOT}/${pair}/offline"
  local instances_log="${offline_dir}/instances.log"
  local ref_file="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.${lang_code}.txt"

  if [[ ! -f "${instances_log}" ]] || [[ ! -s "${instances_log}" ]]; then
    echo "[WARN] Missing/empty instances.log (skip): ${instances_log}" >&2
    return 0
  fi
  if [[ ! -f "${ref_file}" ]]; then
    echo "[ERROR] REF file not found: ${ref_file}" >&2
    return "${EXIT_DATA_ERROR}"
  fi

  local out_tsv="${offline_dir}/streamlaal_acl6060.tsv"
  local out_log="${offline_dir}/streamlaal_acl6060.log"

  echo "[INFO] Running acl6060 eval: pair=${pair} lang=${lang_code}"
  "${PYTHON_BIN}" "${PY_SCRIPT}" \
    --mode acl6060 \
    --instances-log "${instances_log}" \
    --lang-code "${lang_code}" \
    --ref-file "${ref_file}" \
    --audio-yaml "${AUDIO_YAML}" \
    --glossary-acl6060 "${GLOSSARY_ACL6060}" \
    --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
    --stream-laal-tool-rel "${STREAM_LAAL_TOOL_REL}" \
    --python-bin "${PYTHON_BIN}" \
    --output-tsv "${out_tsv}" \
    --output-log "${out_log}"

  echo "[INFO] Done: ${out_tsv}"
}

run_one "en-zh" "zh"
run_one "en-ja" "ja"
run_one "en-de" "de"

echo "[INFO] All done."


