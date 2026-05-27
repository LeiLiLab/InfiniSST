#!/usr/bin/env bash
set -euo pipefail

# Run SimulEval for RealSI en->zh using ONE merged glossary, then post-evaluate with StreamLAAL + term accuracy.
#
# All user-facing strings are in English.

# ======Configuration=====
# Exit codes
EXIT_CONFIG_ERROR="2"
EXIT_DATA_ERROR="3"

# Repo
ROOT_DIR="/home/jiaxuanluo/InfiniSST"

# Base SimulEval sweep script (rank16 v3 HF model)
BASE_SIMULEVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/rank16/bypass_simuleval_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2.sh"

# RealSI dataset (en->zh)
REALSI_WAV_DIR="/mnt/gemini/data/jiaxuanluo/RealSI/data/en2zh/wav"
REALSI_JSON_DIR="/mnt/gemini/data/jiaxuanluo/RealSI/data/en2zh/json"
REALSI_JSON_GLOB="en2zh-*.json"

# Merged glossary path
MERGED_GLOSSARY_PATH="${ROOT_DIR}/documents/data/data_pre/RealSI/extracted_glossary.json"

# Output base (override via OUTPUT_BASE_OVERRIDE)
OUTPUT_BASE_DEFAULT="/mnt/gemini/data2/jiaxuanluo/realsi_rank16_v3_k1_10_k2_sweep_merged_glossary"
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE:-}"
LANG_CODE_DEFAULT="zh"
LANG_CODE_OVERRIDE="${LANG_CODE_OVERRIDE:-}"

# Where to put generated input artifacts (lists + streamlaal files + mapping)
INPUTS_DIRNAME="__realsi_inputs__"

# Pass-through overrides for the base sweep script
LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIERS_OVERRIDE:-}"
RAG_K2_VALUES_OVERRIDE="${RAG_K2_VALUES_OVERRIDE:-}"
RAG_CONFIDENCE_THRESHOLD_OVERRIDE="${RAG_CONFIDENCE_THRESHOLD_OVERRIDE:-}"
RAG_CONFIDENCE_THRESHOLD_MODE_OVERRIDE="${RAG_CONFIDENCE_THRESHOLD_MODE_OVERRIDE:-}"
MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE:-}"
RESUME_MODE="${RESUME_MODE:-0}"
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-}"
BACKUP_PARTIAL_RUNS="${BACKUP_PARTIAL_RUNS:-}"
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-4,5}"

# StreamLAAL tool
FBK_FAIRSEQ_ROOT="/mnt/taurus/home/jiaxuanluo/FBK-fairseq"
STREAM_LAAL_TOOL_REL="examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
TERM_MISMATCH_EXAMPLES="${TERM_MISMATCH_EXAMPLES:-0}"

# MWER segmenter (required by stream_laal_term.py)
MWERSEGMENTER_ROOT="/mnt/taurus/home/jiaxuanluo/mwerSegmenter"

# Fixed settings (must match the base SimulEval script)
RAG_HOP_SIZE="0.48"
RAG_K1_FIXED="10"
RAG_CONFIDENCE_THRESHOLD_DEFAULT="0.0"

# Conda env (must have dependencies used by stream_laal_term.py)
CONDA_BASE="/home/jiaxuanluo/miniconda3"
CONDA_ENV_NAME="infinisst"
# ======Configuration=====

OUTPUT_BASE="${OUTPUT_BASE_DEFAULT}"
if [[ -n "${OUTPUT_BASE_OVERRIDE}" ]]; then
  OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE}"
fi

LANG_CODE="${LANG_CODE_DEFAULT}"
if [[ -n "${LANG_CODE_OVERRIDE}" ]]; then
  LANG_CODE="${LANG_CODE_OVERRIDE}"
fi

if [[ ! -f "${BASE_SIMULEVAL_SCRIPT}" ]]; then
  echo "[ERROR] Base SimulEval script not found: ${BASE_SIMULEVAL_SCRIPT}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
if [[ ! -d "${REALSI_WAV_DIR}" ]]; then
  echo "[ERROR] REALSI_WAV_DIR not found: ${REALSI_WAV_DIR}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -d "${REALSI_JSON_DIR}" ]]; then
  echo "[ERROR] REALSI_JSON_DIR not found: ${REALSI_JSON_DIR}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${MERGED_GLOSSARY_PATH}" ]]; then
  echo "[ERROR] MERGED_GLOSSARY_PATH not found: ${MERGED_GLOSSARY_PATH}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

STREAM_LAAL_TOOL="${FBK_FAIRSEQ_ROOT}/${STREAM_LAAL_TOOL_REL}"
if [[ ! -f "${STREAM_LAAL_TOOL}" ]]; then
  echo "[ERROR] stream_laal_term.py not found: ${STREAM_LAAL_TOOL}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

# Tokenizer/latency unit by language
SACREBLEU_TOKENIZER=""
LATENCY_UNIT=""
if [[ "${LANG_CODE}" == "zh" ]]; then
  SACREBLEU_TOKENIZER="zh"
  LATENCY_UNIT="char"
else
  echo "[ERROR] Unsupported LANG_CODE for RealSI: ${LANG_CODE} (expected zh)" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

INPUTS_ROOT="${OUTPUT_BASE}/${LANG_CODE}/${INPUTS_DIRNAME}"
mkdir -p "${INPUTS_ROOT}"

echo "[INFO] Preparing RealSI inputs under: ${INPUTS_ROOT}"
python3 "${ROOT_DIR}/documents/data/data_pre/RealSI/prepare_realsi_simuleval_inputs.py" \
  --wav-dir "${REALSI_WAV_DIR}" \
  --json-dir "${REALSI_JSON_DIR}" \
  --json-glob "${REALSI_JSON_GLOB}" \
  --output-dir "${INPUTS_ROOT}" \
  --lang-code "${LANG_CODE}" \
  --merged-glossary-path "${MERGED_GLOSSARY_PATH}"

MAP_JSON="${INPUTS_ROOT}/talk_inputs_map.json"
if [[ ! -f "${MAP_JSON}" ]]; then
  echo "[ERROR] Mapping JSON not found: ${MAP_JSON}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

SRC_LIST_OVERRIDE_LOCAL="$(
python3 - <<PY
import json
mp="${MAP_JSON}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("merged",{}).get("src_list",""))
PY
)"
TGT_LIST_OVERRIDE_LOCAL="$(
python3 - <<PY
import json
mp="${MAP_JSON}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("merged",{}).get("tgt_list",""))
PY
)"
MERGED_AUDIO_YAML="$(
python3 - <<PY
import json
mp="${MAP_JSON}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("merged",{}).get("streamlaal_audio_yaml",""))
PY
)"
MERGED_REF_TGT="$(
python3 - <<PY
import json
mp="${MAP_JSON}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("merged",{}).get("streamlaal_ref",""))
PY
)"
MERGED_REF_SRC="$(
python3 - <<PY
import json
mp="${MAP_JSON}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("merged",{}).get("streamlaal_source_ref",""))
PY
)"

if [[ -z "${SRC_LIST_OVERRIDE_LOCAL}" ]] || [[ ! -f "${SRC_LIST_OVERRIDE_LOCAL}" ]]; then
  echo "[ERROR] merged src_list missing: ${SRC_LIST_OVERRIDE_LOCAL}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ -z "${TGT_LIST_OVERRIDE_LOCAL}" ]] || [[ ! -f "${TGT_LIST_OVERRIDE_LOCAL}" ]]; then
  echo "[ERROR] merged tgt_list missing: ${TGT_LIST_OVERRIDE_LOCAL}" >&2
  exit "${EXIT_DATA_ERROR}"
fi
if [[ ! -f "${MERGED_AUDIO_YAML}" ]] || [[ ! -f "${MERGED_REF_TGT}" ]]; then
  echo "[ERROR] Missing merged streamlaal inputs: ${MERGED_AUDIO_YAML} or ${MERGED_REF_TGT}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] LANG_CODE=${LANG_CODE}"
echo "[INFO] Merged glossary: ${MERGED_GLOSSARY_PATH}"
echo "[INFO] SRC_LIST_OVERRIDE=${SRC_LIST_OVERRIDE_LOCAL}"
echo "[INFO] TGT_LIST_OVERRIDE=${TGT_LIST_OVERRIDE_LOCAL}"

# Run SimulEval once with the merged glossary
GLOSSARY_PATHS_OVERRIDE="${MERGED_GLOSSARY_PATH}" \
SRC_LIST_OVERRIDE="${SRC_LIST_OVERRIDE_LOCAL}" \
TGT_LIST_OVERRIDE="${TGT_LIST_OVERRIDE_LOCAL}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
MODEL_NAME_OVERRIDE="${MODEL_NAME_OVERRIDE}" \
LANG_CODE_OVERRIDE="${LANG_CODE}" \
LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIERS_OVERRIDE}" \
RAG_K2_VALUES_OVERRIDE="${RAG_K2_VALUES_OVERRIDE}" \
RAG_CONFIDENCE_THRESHOLD_OVERRIDE="${RAG_CONFIDENCE_THRESHOLD_OVERRIDE}" \
RAG_CONFIDENCE_THRESHOLD_MODE_OVERRIDE="${RAG_CONFIDENCE_THRESHOLD_MODE_OVERRIDE}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}" \
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE}" \
BACKUP_PARTIAL_RUNS="${BACKUP_PARTIAL_RUNS}" \
RESUME_MODE="${RESUME_MODE}" \
bash "${BASE_SIMULEVAL_SCRIPT}"

# Post-evaluate with StreamLAAL + term accuracy
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
echo "[INFO] Activated conda env: ${CONDA_ENV_NAME}"

export MWERSEGMENTER_ROOT
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

RAG_CONFIDENCE_THRESHOLD="${RAG_CONFIDENCE_THRESHOLD_DEFAULT}"
if [[ -n "${RAG_CONFIDENCE_THRESHOLD_OVERRIDE}" ]]; then
  RAG_CONFIDENCE_THRESHOLD="${RAG_CONFIDENCE_THRESHOLD_OVERRIDE}"
fi
THRESHOLD_TAG="${RAG_CONFIDENCE_THRESHOLD//./p}"

LATENCY_MULTIPLIERS=("1" "2" "3" "4")
RAG_K2_VALUES=("5" "10" "15" "20")
if [[ -n "${LATENCY_MULTIPLIERS_OVERRIDE}" ]]; then
  # shellcheck disable=SC2206
  LATENCY_MULTIPLIERS=(${LATENCY_MULTIPLIERS_OVERRIDE})
fi
if [[ -n "${RAG_K2_VALUES_OVERRIDE}" ]]; then
  # shellcheck disable=SC2206
  RAG_K2_VALUES=(${RAG_K2_VALUES_OVERRIDE})
fi

GLOSSARY_TAG="$(basename "${MERGED_GLOSSARY_PATH}" .json)"

for LATENCY_MULTIPLIER in "${LATENCY_MULTIPLIERS[@]}"; do
  for RAG_K2 in "${RAG_K2_VALUES[@]}"; do
    set +e
    OUTPUT_DIR="$(
    ls -dt "${OUTPUT_BASE}/${LANG_CODE}"/*_g${GLOSSARY_TAG}_cs*_hs${RAG_HOP_SIZE}_lm${LATENCY_MULTIPLIER}_k2${RAG_K2}_k1${RAG_K1_FIXED}_th${THRESHOLD_TAG} 2>/dev/null | head -n 1
    )"
    LS_RC="$?"
    set -e
    if [[ "${LS_RC}" != "0" ]] || [[ -z "${OUTPUT_DIR}" ]]; then
      echo "[WARN] Output dir not found for merged run (lm=${LATENCY_MULTIPLIER} k2=${RAG_K2} th=${RAG_CONFIDENCE_THRESHOLD})" >&2
      continue
    fi

    INSTANCES_PATH="${OUTPUT_DIR}/instances.log"
    if [[ ! -f "${INSTANCES_PATH}" ]] || [[ ! -s "${INSTANCES_PATH}" ]]; then
      echo "[WARN] Missing/empty instances.log: ${INSTANCES_PATH}" >&2
      continue
    fi

    set +e
    EVAL_OUT="$(python "${STREAM_LAAL_TOOL}" \
      --simuleval-instances "${INSTANCES_PATH}" \
      --reference "${MERGED_REF_TGT}" \
      --source-reference "${MERGED_REF_SRC}" \
      --audio-yaml "${MERGED_AUDIO_YAML}" \
      --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
      --latency-unit "${LATENCY_UNIT}" \
      --glossary "${MERGED_GLOSSARY_PATH}" \
      --term-lang "${LANG_CODE}" \
      --term-mismatch-examples "${TERM_MISMATCH_EXAMPLES}" 2>&1)"
    EVAL_RC="$?"
    set -e
    if [[ "${EVAL_RC}" != "0" ]]; then
      echo "[WARN] stream_laal_term.py failed for merged run (rc=${EVAL_RC})" >&2
      echo "${EVAL_OUT}" >&2
      continue
    fi

    echo "${EVAL_OUT}" > "${OUTPUT_DIR}/post_eval_streamlaal_term.log"
    echo "[INFO] Evaluated merged: output_dir=${OUTPUT_DIR} (lm=${LATENCY_MULTIPLIER} k2=${RAG_K2})"
  done
done

echo "[INFO] Done."

