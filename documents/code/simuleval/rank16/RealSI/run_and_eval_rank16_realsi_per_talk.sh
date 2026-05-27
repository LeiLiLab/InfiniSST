#!/usr/bin/env bash
set -euo pipefail

# Run SimulEval for RealSI en->zh:
# (1) Treat each talk (01..10) as one "paper/talk" with its own glossary, run SimulEval.
# (2) Post-evaluate with StreamLAAL + term accuracy per talk, and aggregate overall TERM_ACC.
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

# RealSI glossaries (generated earlier)
REALSI_GLOSSARIES_DIR="${ROOT_DIR}/documents/data/data_pre/RealSI/extracted_glossaries_by_paper"

# Output base (override via OUTPUT_BASE_OVERRIDE)
OUTPUT_BASE_DEFAULT="/mnt/gemini/data2/jiaxuanluo/realsi_rank16_v3_k1_10_k2_sweep_per_talk"
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
if [[ ! -d "${REALSI_GLOSSARIES_DIR}" ]]; then
  echo "[ERROR] REALSI_GLOSSARIES_DIR not found: ${REALSI_GLOSSARIES_DIR}" >&2
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
  --per-talk-glossaries-dir "${REALSI_GLOSSARIES_DIR}"

MAP_JSON="${INPUTS_ROOT}/talk_inputs_map.json"
if [[ ! -f "${MAP_JSON}" ]]; then
  echo "[ERROR] Mapping JSON not found: ${MAP_JSON}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

TALKS="$(
python3 - <<PY
import json
mp="${MAP_JSON}"
obj=json.load(open(mp,'r',encoding='utf-8'))
talks=obj.get("talks",{})
items=[(v.get("talk_num", 10**9), k) for k,v in talks.items()]
items.sort()
print(" ".join([k for _,k in items]))
PY
)"
if [[ -z "${TALKS}" ]]; then
  echo "[ERROR] No talks found in mapping: ${MAP_JSON}" >&2
  exit "${EXIT_DATA_ERROR}"
fi

echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] LANG_CODE=${LANG_CODE}"
echo "[INFO] Talks: ${TALKS}"
echo "[INFO] Overrides: lm='${LATENCY_MULTIPLIERS_OVERRIDE:-<none>}' k2='${RAG_K2_VALUES_OVERRIDE:-<none>}' resume='${RESUME_MODE}' model='${MODEL_NAME_OVERRIDE:-<default>}'"

# 1) Run SimulEval per talk (each talk has its own glossary + its own src/tgt list)
for TALK_ID in ${TALKS}; do
  GLOSSARY_PATH="$(
  python3 - <<PY
import json
mp="${MAP_JSON}"
tid="${TALK_ID}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("talks",{}).get(tid,{}).get("glossary_path",""))
PY
  )"
  SRC_LIST_OVERRIDE_LOCAL="$(
  python3 - <<PY
import json
mp="${MAP_JSON}"
tid="${TALK_ID}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("talks",{}).get(tid,{}).get("src_list",""))
PY
  )"
  TGT_LIST_OVERRIDE_LOCAL="$(
  python3 - <<PY
import json
mp="${MAP_JSON}"
tid="${TALK_ID}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("talks",{}).get(tid,{}).get("tgt_list",""))
PY
  )"

  if [[ -z "${GLOSSARY_PATH}" ]] || [[ ! -f "${GLOSSARY_PATH}" ]]; then
    echo "[WARN] Skip talk_id=${TALK_ID}: glossary_path missing: ${GLOSSARY_PATH}" >&2
    continue
  fi
  if [[ -z "${SRC_LIST_OVERRIDE_LOCAL}" ]] || [[ ! -f "${SRC_LIST_OVERRIDE_LOCAL}" ]]; then
    echo "[ERROR] src_list missing: ${SRC_LIST_OVERRIDE_LOCAL}" >&2
    exit "${EXIT_DATA_ERROR}"
  fi
  if [[ -z "${TGT_LIST_OVERRIDE_LOCAL}" ]] || [[ ! -f "${TGT_LIST_OVERRIDE_LOCAL}" ]]; then
    echo "[ERROR] tgt_list missing: ${TGT_LIST_OVERRIDE_LOCAL}" >&2
    exit "${EXIT_DATA_ERROR}"
  fi

  echo "[INFO] ============================================================"
  echo "[INFO] Running talk_id=${TALK_ID}"
  echo "[INFO] GLOSSARY_PATH=${GLOSSARY_PATH}"
  echo "[INFO] SRC_LIST_OVERRIDE=${SRC_LIST_OVERRIDE_LOCAL}"
  echo "[INFO] TGT_LIST_OVERRIDE=${TGT_LIST_OVERRIDE_LOCAL}"

  GLOSSARY_PATHS_OVERRIDE="${GLOSSARY_PATH}" \
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
done

# 2) Post-evaluate term accuracy per talk using StreamLAAL tool
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

SUMMARY_TSV="${OUTPUT_BASE}/${LANG_CODE}/realsi_per_talk_streamlaal_term_summary.tsv"
echo -e "talk_id\tglossary_tag\tlm\tk2\tbleu\tstreamlaal\tstreamlaal_ca\tterm_acc\tterm_correct\tterm_total\toutput_dir" > "${SUMMARY_TSV}"

overall_correct="0"
overall_total="0"

for LATENCY_MULTIPLIER in "${LATENCY_MULTIPLIERS[@]}"; do
  for RAG_K2 in "${RAG_K2_VALUES[@]}"; do
    echo "[INFO] ============================================================"
    echo "[INFO] Evaluating per-talk: lm=${LATENCY_MULTIPLIER} k2=${RAG_K2} th=${RAG_CONFIDENCE_THRESHOLD}"

    for TALK_ID in ${TALKS}; do
      GLOSSARY_PATH="$(
      python3 - <<PY
import json
mp="${MAP_JSON}"
tid="${TALK_ID}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("talks",{}).get(tid,{}).get("glossary_path",""))
PY
      )"
      AUDIO_YAML="$(
      python3 - <<PY
import json
mp="${MAP_JSON}"
tid="${TALK_ID}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("talks",{}).get(tid,{}).get("streamlaal_audio_yaml",""))
PY
      )"
      REF_TGT="$(
      python3 - <<PY
import json
mp="${MAP_JSON}"
tid="${TALK_ID}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("talks",{}).get(tid,{}).get("streamlaal_ref",""))
PY
      )"
      REF_SRC="$(
      python3 - <<PY
import json
mp="${MAP_JSON}"
tid="${TALK_ID}"
obj=json.load(open(mp,'r',encoding='utf-8'))
print(obj.get("talks",{}).get(tid,{}).get("streamlaal_source_ref",""))
PY
      )"

      if [[ -z "${GLOSSARY_PATH}" ]] || [[ ! -f "${GLOSSARY_PATH}" ]]; then
        echo "[WARN] Skip talk_id=${TALK_ID}: glossary_path missing: ${GLOSSARY_PATH}" >&2
        continue
      fi
      if [[ ! -f "${AUDIO_YAML}" ]] || [[ ! -f "${REF_TGT}" ]]; then
        echo "[ERROR] Missing streamlaal inputs for talk_id=${TALK_ID}" >&2
        exit "${EXIT_DATA_ERROR}"
      fi

      GLOSSARY_TAG="$(basename "${GLOSSARY_PATH}" .json)"

      set +e
      OUTPUT_DIR="$(
      ls -dt "${OUTPUT_BASE}/${LANG_CODE}"/*_g${GLOSSARY_TAG}_cs*_hs${RAG_HOP_SIZE}_lm${LATENCY_MULTIPLIER}_k2${RAG_K2}_k1${RAG_K1_FIXED}_th${THRESHOLD_TAG} 2>/dev/null | head -n 1
      )"
      LS_RC="$?"
      set -e
      if [[ "${LS_RC}" != "0" ]] || [[ -z "${OUTPUT_DIR}" ]]; then
        echo "[WARN] Output dir not found for talk_id=${TALK_ID} (lm=${LATENCY_MULTIPLIER} k2=${RAG_K2} th=${RAG_CONFIDENCE_THRESHOLD})" >&2
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
        --reference "${REF_TGT}" \
        --source-reference "${REF_SRC}" \
        --audio-yaml "${AUDIO_YAML}" \
        --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
        --latency-unit "${LATENCY_UNIT}" \
        --glossary "${GLOSSARY_PATH}" \
        --term-lang "${LANG_CODE}" \
        --term-mismatch-examples "${TERM_MISMATCH_EXAMPLES}" 2>&1)"
      EVAL_RC="$?"
      set -e
      if [[ "${EVAL_RC}" != "0" ]]; then
        echo "[WARN] stream_laal_term.py failed for ${TALK_ID} (rc=${EVAL_RC})" >&2
        echo "${EVAL_OUT}" >&2
        continue
      fi

      BLEU_LINE="$(echo "${EVAL_OUT}" | awk 'BEGIN{found=0} /^BLEU[[:space:]]/{found=1; next} found {print; exit}')"
      BLEU_SCORE="$(echo "${BLEU_LINE}" | awk '{print $1}')"
      STREAM_LAAL="$(echo "${BLEU_LINE}" | awk '{print $2}')"
      STREAM_LAAL_CA="$(echo "${BLEU_LINE}" | awk '{print $3}')"

      TERM_LINE="$(echo "${EVAL_OUT}" | awk '/^TERM_ACC[[:space:]]/{print; exit}')"
      TERM_ACC="$(echo "${TERM_LINE}" | awk '{print $2}')"
      TERM_CORRECT="$(echo "${TERM_LINE}" | awk '{print $4}')"
      TERM_TOTAL="$(echo "${TERM_LINE}" | awk '{print $6}')"

      if [[ -n "${TERM_CORRECT}" ]] && [[ -n "${TERM_TOTAL}" ]]; then
        overall_correct="$((overall_correct + TERM_CORRECT))"
        overall_total="$((overall_total + TERM_TOTAL))"
      fi

      echo "${EVAL_OUT}" > "${OUTPUT_DIR}/post_eval_streamlaal_term.log"
      echo -e "${TALK_ID}\t${GLOSSARY_TAG}\t${LATENCY_MULTIPLIER}\t${RAG_K2}\t${BLEU_SCORE}\t${STREAM_LAAL}\t${STREAM_LAAL_CA}\t${TERM_ACC}\t${TERM_CORRECT}\t${TERM_TOTAL}\t${OUTPUT_DIR}" >> "${SUMMARY_TSV}"
    done
  done
done

if [[ "${overall_total}" -gt 0 ]]; then
  python3 - <<PY
correct = int("${overall_correct}")
total = int("${overall_total}")
acc = correct / total if total > 0 else 0.0
print(f"[INFO] Overall TERM_ACC = {acc:.6f} (sum_correct={correct}, total_terms={total})")
print(f"[INFO] Summary written: ${SUMMARY_TSV}")
PY
else
  echo "[WARN] Overall TERM_ACC skipped: overall_total=${overall_total}" >&2
  echo "[INFO] Summary written: ${SUMMARY_TSV}"
fi

echo "[INFO] Done."

