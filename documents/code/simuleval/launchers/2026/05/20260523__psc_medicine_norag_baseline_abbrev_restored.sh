#!/usr/bin/env bash
set -euo pipefail

# PSC Bridges-2 launcher for restored ESO medicine no-RAG baselines.
# One Slurm job should run one language on two V100 GPUs.

PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
ENV_DIR="${ENV_DIR:-${PSC_BASE}/envs/spaCyEnv_20260518}"
MEDICINE_LAUNCHER="${MEDICINE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__medicine_abbrev_restored_norag_streaming_batched_aries67.sh}"
SELF_SCRIPT="${SELF_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260523__psc_medicine_norag_baseline_abbrev_restored.sh}"
# PSC host glibc is too old for vLLM 0.13 native extensions to register
# reliably. Default to the Ubuntu 22.04 Apptainer userspace; callers can still
# set USE_APPTAINER=0 for a host-only diagnostic.
USE_APPTAINER="${USE_APPTAINER:-1}"
APPTAINER_SIF="${APPTAINER_SIF:-${PSC_BASE}/containers/ubuntu_22_04_gcc.sif}"
IN_APPTAINER="${IN_APPTAINER:-0}"

if [[ "${USE_APPTAINER}" == "1" && "${IN_APPTAINER}" != "1" ]]; then
  if [[ ! -f "${APPTAINER_SIF}" ]]; then
    echo "[ERROR] APPTAINER_SIF not found: ${APPTAINER_SIF}" >&2
    exit 3
  fi
  export IN_APPTAINER=1
  export APPTAINERENV_IN_APPTAINER=1
  export APPTAINERENV_PSC_BASE="${PSC_BASE}"
  export APPTAINERENV_ROOT_DIR="${ROOT_DIR}"
  export APPTAINERENV_ENV_DIR="${ENV_DIR}"
  exec apptainer exec --nv -B /ocean,/jet "${APPTAINER_SIF}" bash "${SELF_SCRIPT}" "$@"
fi

RUN_STAMP="${RUN_STAMP:-20260523T2030_psc_medicine_${SLURM_JOB_ID:-manual}}"
MED_LANG="${MED_LANG:-zh}"
TARGET_LMS="${TARGET_LMS:-1 2 3 4}"
TARGET_SAMPLES="${TARGET_SAMPLES:-404 545006 596001 605000 606}"
GPU_PAIR="${GPU_PAIR:-0,1}"
ESO_TEST_ROOT="${ESO_TEST_ROOT:-${PSC_BASE}/data/eso_medicine_abbrev_restored/test}"
OUTPUT_BASE="${OUTPUT_BASE:-${PSC_BASE}/outputs/medicine_norag_baseline_abbrev_restored/${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-${PSC_BASE}/logs/medicine_norag_baseline_abbrev_restored/${RUN_STAMP}}"
MODEL_ROOT_OWASKI="${MODEL_ROOT_OWASKI:-${PSC_BASE}/models/owaski}"
CACHE_ROOT="${CACHE_ROOT:-${PSC_BASE}/cache}"
HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER:-jluo7}/infinisst_medicine_${SLURM_JOB_ID:-manual}}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-${PSC_BASE}/tools/mwerSegmenter}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-${PSC_BASE}/tools/FBK-fairseq}"

WAIT_FOR_HF_READY="${WAIT_FOR_HF_READY:-1}"
HF_READY_TIMEOUT_SECONDS="${HF_READY_TIMEOUT_SECONDS:-7200}"
HF_READY_SLEEP_SECONDS="${HF_READY_SLEEP_SECONDS:-120}"
DELETE_MODEL_AFTER_JOB="${DELETE_MODEL_AFTER_JOB:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

case "${MED_LANG}" in
  zh)
    HF_REPO_ID="${HF_REPO_ID:-gavinlaw/infinisst-no-tmsft-origin-bsz4-zh}"
    MODEL_DIRNAME="${MODEL_DIRNAME:-gigaspeech-zh-s_origin-bsz4}"
    ;;
  de)
    HF_REPO_ID="${HF_REPO_ID:-gavinlaw/infinisst-no-tmsft-origin-bsz4-de}"
    MODEL_DIRNAME="${MODEL_DIRNAME:-gigaspeech-de-s_origin-bsz4}"
    ;;
  ja)
    HF_REPO_ID="${HF_REPO_ID:-gavinlaw/infinisst-no-tmsft-origin-bsz4-ja}"
    MODEL_DIRNAME="${MODEL_DIRNAME:-gigaspeech-ja-s_origin-bsz4}"
    ;;
  *)
    echo "[ERROR] Unsupported MED_LANG=${MED_LANG}" >&2
    exit 2
    ;;
esac

MODEL_DIR="${MODEL_DIR:-${MODEL_ROOT_OWASKI}/${MODEL_DIRNAME}}"
STREAM_LAAL_TOOL="${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
POST_EVAL_SUMMARY="${OUTPUT_BASE}/streamlaal_term_eval_summary.tsv"

mkdir -p \
  "${OUTPUT_BASE}" \
  "${LOG_ROOT}" \
  "${MODEL_ROOT_OWASKI}" \
  "${CACHE_ROOT}" \
  "${HF_HOME}" \
  "${TMP_ROOT}"

export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${ENV_DIR}"
export CONDA_DEFAULT_ENV="spaCyEnv_20260518"
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${CACHE_ROOT}/numba}"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"
export MWERSEGMENTER_ROOT
export FBK_FAIRSEQ_ROOT
export INFINISST_ROOT="${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-2}"
export VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-8192}"
export VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE:-8}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.80}"
# Keep the original no-RAG streaming cache by default.  The vLLM prompt audio
# cap above bounds the actual multimodal prompt size; forcing 4s/4s caused a
# material term-accuracy drop in the Taurus zh lm=4 sentence-level probe.
export MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_OVERRIDE:-80.0}"
export KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_OVERRIDE:-60.0}"
export FORCE_RERUN_OVERRIDE="${FORCE_RERUN}"
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"

mkdir -p \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${NUMBA_CACHE_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  for token_file in "${HOME}/.cache/huggingface/token" "${HOME}/.huggingface/token"; do
    if [[ -s "${token_file}" ]]; then
      HF_TOKEN="$(<"${token_file}")"
      export HF_TOKEN
      break
    fi
  done
fi

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 3
  fi
}

validate_model_dir() {
  local path="$1"
  if [[ ! -f "${path}/config.json" ]]; then
    return 1
  fi
  local shards
  shards="$(find "${path}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shards}" == "15" ]]
}

repo_ready() {
  "${ENV_DIR}/bin/python" - "${HF_REPO_ID}" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
try:
    files = HfApi().list_repo_files(repo_id, repo_type="model")
except Exception:
    raise SystemExit(1)
shards = [f for f in files if f.startswith("model-") and f.endswith(".safetensors")]
if "config.json" not in files or len(shards) != 15:
    raise SystemExit(1)
PY
}

wait_for_repo_ready() {
  if [[ "${WAIT_FOR_HF_READY}" != "1" ]]; then
    repo_ready
    return
  fi
  local start now
  start="$(date +%s)"
  until repo_ready; do
    now="$(date +%s)"
    if (( now - start >= HF_READY_TIMEOUT_SECONDS )); then
      echo "[ERROR] HF repo not ready after ${HF_READY_TIMEOUT_SECONDS}s: ${HF_REPO_ID}" >&2
      return 4
    fi
    echo "[INFO] Waiting for HF repo readiness: ${HF_REPO_ID}"
    sleep "${HF_READY_SLEEP_SECONDS}"
  done
}

download_model_if_needed() {
  if validate_model_dir "${MODEL_DIR}"; then
    echo "[INFO] Model already present: ${MODEL_DIR}"
    return 0
  fi

  echo "[INFO] Downloading ${HF_REPO_ID} to ${MODEL_DIR}"
  wait_for_repo_ready
  if [[ -e "${MODEL_DIR}" ]]; then
    echo "[WARN] Removing incomplete model dir before HF download: ${MODEL_DIR}"
    rm -rf "${MODEL_DIR}"
  fi
  mkdir -p "${MODEL_DIR}"
  hf download "${HF_REPO_ID}" --repo-type model --local-dir "${MODEL_DIR}"
  if ! validate_model_dir "${MODEL_DIR}"; then
    echo "[ERROR] Downloaded model failed validation: ${MODEL_DIR}" >&2
    exit 4
  fi
}

post_eval_one() {
  local lm="$1"
  local timing="${OUTPUT_BASE}/timing.tsv"
  local output_dir combined_dir eval_tsv eval_log miss_tsv miss_summary norm_glossary
  if [[ ! -s "${timing}" ]]; then
    echo "[ERROR] Missing timing TSV: ${timing}" >&2
    exit 4
  fi
  output_dir="$(
    awk -F'\t' -v lang="${MED_LANG}" -v lm="${lm}" \
      '$1 == lang && $2 == lm && $5 == "success" {path=$9} END {print path}' \
      "${timing}"
  )"
  if [[ -z "${output_dir}" || ! -s "${output_dir}/instances.log" ]]; then
    echo "[ERROR] Missing successful instances.log for lang=${MED_LANG} lm=${lm}: ${output_dir}" >&2
    exit 4
  fi

  combined_dir="${OUTPUT_BASE}/${MED_LANG}/__medicine_inputs__/combined"
  eval_tsv="${output_dir}/eval_results_streamlaal_term.tsv"
  eval_log="${output_dir}/post_eval_streamlaal_term_full.log"
  miss_tsv="${output_dir}/term_misses.${MED_LANG}_lm${lm}.tsv"
  miss_summary="${output_dir}/term_miss_summary.${MED_LANG}_lm${lm}.tsv"
  norm_glossary="${output_dir}/strict_fixed_medicine_glossary.streamlaal_dict.json"

  "${ENV_DIR}/bin/python" "${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py" \
    --mode acl6060 \
    --instances-log "${output_dir}/instances.log" \
    --lang-code "${MED_LANG}" \
    --source-file "${combined_dir}/medicine5.source_text.en.sentences.txt" \
    --ref-file "${combined_dir}/medicine5.ref.${MED_LANG}.sentences.txt" \
    --audio-yaml "${combined_dir}/medicine5.audio.yaml" \
    --glossary-acl6060 "${OUTPUT_BASE}/strict_fixed_medicine_glossary.from_outputs_v2_terms.json" \
    --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
    --term-fcr-policy source_ref_negative_sentence \
    --output-tsv "${eval_tsv}" \
    --output-log "${eval_log}" \
    --work-dir "${output_dir}/work_streamlaal_term" \
    --term-mismatch-examples 20

  "${ENV_DIR}/bin/python" "${ROOT_DIR}/documents/code/simuleval/export_streamlaal_term_misses.py" \
    --instances-log "${output_dir}/instances.log" \
    --reference "${combined_dir}/medicine5.ref.${MED_LANG}.sentences.txt" \
    --source-reference "${combined_dir}/medicine5.source_text.en.sentences.txt" \
    --audio-yaml "${combined_dir}/medicine5.audio.yaml" \
    --glossary "${OUTPUT_BASE}/strict_fixed_medicine_glossary.from_outputs_v2_terms.json" \
    --lang-code "${MED_LANG}" \
    --stream-laal-tool "${STREAM_LAAL_TOOL}" \
    --mwersegmenter-root "${MWERSEGMENTER_ROOT}" \
    --output-misses "${miss_tsv}" \
    --output-summary "${miss_summary}" \
    --output-normalized-glossary "${norm_glossary}"

  if [[ ! -s "${POST_EVAL_SUMMARY}" ]]; then
    awk -F'\t' 'BEGIN {OFS="\t"} NR == 1 {print "lang", "lm", $0}' "${eval_tsv}" > "${POST_EVAL_SUMMARY}"
  fi
  awk -F'\t' -v lang="${MED_LANG}" -v lm="${lm}" 'BEGIN {OFS="\t"} NR > 1 {print lang, lm, $0}' \
    "${eval_tsv}" >> "${POST_EVAL_SUMMARY}"
}

require_path "${ROOT_DIR}"
require_path "${ENV_DIR}/bin/python"
require_path "${MEDICINE_LAUNCHER}"
require_path "${ESO_TEST_ROOT}"
require_path "${MWERSEGMENTER_ROOT}"
require_path "${STREAM_LAAL_TOOL}"

ENABLE_VLLM_MOE_TOPK_PATCH="${ENABLE_VLLM_MOE_TOPK_PATCH:-0}"
VLLM_MOE_TOPK_PATCH="${ROOT_DIR}/documents/code/simuleval/tools/patch_vllm_moe_topk_softmax_fallback.py"
if [[ "${ENABLE_VLLM_MOE_TOPK_PATCH}" == "1" && -f "${VLLM_MOE_TOPK_PATCH}" ]]; then
  echo "[INFO] Ensuring vLLM MoE topk_softmax fallback patch"
  "${ENV_DIR}/bin/python" "${VLLM_MOE_TOPK_PATCH}"
fi

hf auth whoami >/dev/null
download_model_if_needed

MODEL_ENV_NAME=""
case "${MED_LANG}" in
  zh) MODEL_ENV_NAME="MODEL_ZH_OVERRIDE" ;;
  de) MODEL_ENV_NAME="MODEL_DE_OVERRIDE" ;;
  ja) MODEL_ENV_NAME="MODEL_JA_OVERRIDE" ;;
esac
export "${MODEL_ENV_NAME}=${MODEL_DIR}"

echo "[INFO] RUN_STAMP=${RUN_STAMP}"
echo "[INFO] MED_LANG=${MED_LANG} TARGET_LMS=${TARGET_LMS} TARGET_SAMPLES=${TARGET_SAMPLES}"
echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] GPU_PAIR=${GPU_PAIR}"

ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
CONDA_PREFIX_OVERRIDE="${ENV_DIR}" \
CONDA_ENV_NAME_OVERRIDE="$(basename "${ENV_DIR}")" \
PREP_PYTHON_OVERRIDE="${ENV_DIR}/bin/python" \
ESO_TEST_ROOT_OVERRIDE="${ESO_TEST_ROOT}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
LANGS_OVERRIDE="${MED_LANG}" \
TARGET_LMS_OVERRIDE="${TARGET_LMS}" \
TARGET_SAMPLES_OVERRIDE="${TARGET_SAMPLES}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${GPU_PAIR//,/:}" \
bash "${MEDICINE_LAUNCHER}"

for lm in ${TARGET_LMS}; do
  post_eval_one "${lm}"
done

if [[ "${DELETE_MODEL_AFTER_JOB}" == "1" ]]; then
  echo "[INFO] Removing model after successful job: ${MODEL_DIR}"
  rm -rf "${MODEL_DIR}"
fi

echo "[ALL DONE] medicine no-RAG PSC lang=${MED_LANG} outputs=${OUTPUT_BASE}"
