#!/usr/bin/env bash
set -euo pipefail

# Append-only Taurus probe: keep PSC-oriented vLLM limits, but restore the
# original no-RAG streaming cache window to isolate the quality impact of
# MAX_CACHE_SECONDS=4.0 / KEEP_CACHE_SECONDS=4.0.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
DEFAULT_CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
if [[ -n "${CONDA_PREFIX_OVERRIDE:-}" ]]; then
  CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE}"
elif [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "spaCyEnv" ]]; then
  CONDA_PREFIX="${CONDA_PREFIX}"
else
  CONDA_PREFIX="${DEFAULT_CONDA_PREFIX}"
fi

MEDICINE_LAUNCHER="${MEDICINE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__medicine_abbrev_restored_norag_streaming_batched_aries67.sh}"
OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_vllm_override_probe_20260523}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_norag_vllm_override_probe_20260523}"
ESO_TEST_ROOT="${ESO_TEST_ROOT:-/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2_abbrev_exact_match_abbrev_restored/test}"
MODEL_ZH="${MODEL_ZH:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4}"
GPU_CSV="${GPU_CSV:-6:7}"
LANG_CODE="${LANG_CODE:-zh}"
LM="${LM:-4}"
SAMPLE_ID="${SAMPLE_ID:-605000}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
STREAM_LAAL_TOOL="${STREAM_LAAL_TOOL:-${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

export PATH="${CONDA_PREFIX}/bin:${MWERSEGMENTER_ROOT}:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/gemini/data1/jiaxuanluo/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-/mnt/gemini/data1/jiaxuanluo/cache/torch}"
export TMPDIR="${TMPDIR:-/tmp/${USER:-jiaxuanluo}/medicine_vllm_probe_limit8_${$}}"
mkdir -p "${TMPDIR}"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 3
  fi
}

find_output_dir() {
  local base="$1"
  local found
  found="$(find "${base}/${LANG_CODE}" -type f -name instances.log -size +0 -printf '%h\n' | sort | tail -n 1 || true)"
  if [[ -z "${found}" ]]; then
    echo "[ERROR] No non-empty instances.log under ${base}/${LANG_CODE}" >&2
    exit 4
  fi
  printf '%s\n' "${found}"
}

require_path "${CONDA_PREFIX}/bin/python"
require_path "${MEDICINE_LAUNCHER}"
require_path "${ESO_TEST_ROOT}/sample_${SAMPLE_ID}_v2/full_sample_v2.json"
require_path "${MODEL_ZH}/config.json"
require_path "${STREAM_LAAL_TOOL}"
require_path "${MWERSEGMENTER_ROOT}"
require_path "${OUT_ROOT}/orig80"

VARIANT_NAME="psc_limit8_keep80"
VARIANT_BASE="${OUT_ROOT}/${VARIANT_NAME}"

echo "[VARIANT START] ${VARIANT_NAME}"
LANGS_OVERRIDE="${LANG_CODE}" \
TARGET_LMS_OVERRIDE="${LM}" \
TARGET_SAMPLES_OVERRIDE="${SAMPLE_ID}" \
ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX}" \
PREP_PYTHON_OVERRIDE="${CONDA_PREFIX}/bin/python" \
ESO_TEST_ROOT_OVERRIDE="${ESO_TEST_ROOT}" \
OUTPUT_BASE_OVERRIDE="${VARIANT_BASE}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${GPU_CSV}" \
MODEL_ZH_OVERRIDE="${MODEL_ZH}" \
FORCE_RERUN_OVERRIDE="1" \
GPU_MEMORY_UTILIZATION_OVERRIDE="0.8" \
VLLM_TP_SIZE_OVERRIDE="2" \
VLLM_MAX_MODEL_LEN_OVERRIDE="8192" \
VLLM_LIMIT_AUDIO_OVERRIDE="8" \
VLLM_DISABLE_CUSTOM_ALL_REDUCE="1" \
MAX_CACHE_SECONDS_OVERRIDE="80.0" \
KEEP_CACHE_SECONDS_OVERRIDE="60.0" \
bash "${MEDICINE_LAUNCHER}" 2>&1 | tee "${LOG_ROOT}/${VARIANT_NAME}.launcher.log"
echo "[VARIANT DONE] ${VARIANT_NAME}"

variant_dir="$(find_output_dir "${VARIANT_BASE}")"
combined_dir="${VARIANT_BASE}/${LANG_CODE}/__medicine_inputs__/combined"

"${CONDA_PREFIX}/bin/python" "${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py" \
  --mode acl6060 \
  --instances-log "${variant_dir}/instances.log" \
  --lang-code "${LANG_CODE}" \
  --source-file "${combined_dir}/medicine5.source_text.en.sentences.txt" \
  --ref-file "${combined_dir}/medicine5.ref.${LANG_CODE}.sentences.txt" \
  --audio-yaml "${combined_dir}/medicine5.audio.yaml" \
  --glossary-acl6060 "${VARIANT_BASE}/strict_fixed_medicine_glossary.from_outputs_v2_terms.json" \
  --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
  --term-fcr-policy source_ref_negative_sentence \
  --output-tsv "${variant_dir}/eval_results_streamlaal_term.tsv" \
  --output-log "${variant_dir}/post_eval_streamlaal_term_full.log" \
  --work-dir "${variant_dir}/work_streamlaal_term" \
  --term-mismatch-examples 20 2>&1 | tee "${LOG_ROOT}/${VARIANT_NAME}.post_eval.log"

printf '%s\n' "${variant_dir}" > "${LOG_ROOT}/${VARIANT_NAME}.output_dir"

orig_dir="$(find_output_dir "${OUT_ROOT}/orig80")"
orig_combined_dir="${OUT_ROOT}/orig80/${LANG_CODE}/__medicine_inputs__/combined"

"${CONDA_PREFIX}/bin/python" "${ROOT_DIR}/documents/code/simuleval/compare_medicine_vllm_override_sentence_diff.py" \
  --left-label orig80 \
  --right-label "${VARIANT_NAME}" \
  --left-instances "${orig_dir}/instances.log" \
  --right-instances "${variant_dir}/instances.log" \
  --reference "${orig_combined_dir}/medicine5.ref.${LANG_CODE}.sentences.txt" \
  --source-reference "${orig_combined_dir}/medicine5.source_text.en.sentences.txt" \
  --audio-yaml "${orig_combined_dir}/medicine5.audio.yaml" \
  --glossary "${OUT_ROOT}/orig80/strict_fixed_medicine_glossary.from_outputs_v2_terms.json" \
  --lang-code "${LANG_CODE}" \
  --stream-laal-tool "${STREAM_LAAL_TOOL}" \
  --mwersegmenter-root "${MWERSEGMENTER_ROOT}" \
  --output-tsv "${OUT_ROOT}/sentence_diff.${LANG_CODE}_lm${LM}_sample${SAMPLE_ID}.orig80_vs_${VARIANT_NAME}.tsv" \
  --summary-json "${OUT_ROOT}/sentence_diff.${LANG_CODE}_lm${LM}_sample${SAMPLE_ID}.orig80_vs_${VARIANT_NAME}.summary.json" \
  --output-normalized-glossary "${OUT_ROOT}/strict_fixed_medicine_glossary.streamlaal_dict.json" \
  2>&1 | tee "${LOG_ROOT}/${VARIANT_NAME}.compare_sentence_diff.log"

echo "[ALL DONE] variant=${VARIANT_NAME} outputs=${OUT_ROOT}"
