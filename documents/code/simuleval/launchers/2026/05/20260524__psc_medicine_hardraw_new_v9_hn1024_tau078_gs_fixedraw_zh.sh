#!/usr/bin/env bash
set -euo pipefail

# PSC Bridges-2 wrapper for medicine zh hardraw fixed-denominator RASST eval.
# Runtime glossary can be gs1k/gs10k; TERM metrics stay fixed to hardraw.

PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
ENV_DIR="${ENV_DIR:-${PSC_BASE}/envs/spaCyEnv_20260518}"
BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__medicine_hardraw_lm1to4_5samples_hn1024_tau078_new_v9_batch.sh}"
USE_APPTAINER="${USE_APPTAINER:-0}"
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
  exec apptainer exec --nv -B /ocean,/jet "${APPTAINER_SIF}" bash "$0" "$@"
fi

RUN_STAMP="${RUN_STAMP:-20260524T1226_psc_med_newv9_hn1024_tau078_${SLURM_JOB_ID:-manual}}"
GLOSSARY_KIND="${GLOSSARY_KIND:-gs1k}"
TARGET_LMS="${TARGET_LMS:-${LMS:-1}}"
TARGET_SAMPLES="${TARGET_SAMPLES:-404 545006 596001 605000 606}"
GPU_PAIR="${GPU_PAIR:-0,1,2,3}"
MODEL_LABEL="${MODEL_LABEL:-new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078}"
MODEL_DIR="${MODEL_DIR:-${PSC_BASE}/models/new_v9_termtag_delay_oldnewv3_r32a64/keep1.0_r32/v0-20260524-062743-hf}"
HN1024_CKPT="${HN1024_CKPT:-${PSC_BASE}/checkpoints/lh1b88kw_best_eval_acl6060_recallat10.pt}"
ESO_TEST_ROOT="${ESO_TEST_ROOT:-${PSC_BASE}/data/eso_medicine_abbrev_restored/test}"
FIXED_RAW_GLOSSARY="${FIXED_RAW_GLOSSARY:-${PSC_BASE}/glossaries/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json}"
GS1K_GLOSSARY="${GS1K_GLOSSARY:-${PSC_BASE}/glossaries/medicine_hardraw_plus_gtwiki_gs1000_translated_fixedraw.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY:-${PSC_BASE}/glossaries/medicine_hardraw_plus_gtwiki_gs10000_translated_fixedraw.json}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-${PSC_BASE}/tools/mwerSegmenter}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-${PSC_BASE}/tools/FBK-fairseq}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__psc_medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw_zh.md}"
VLLM_MOE_TOPK_PATCH="${ROOT_DIR}/documents/code/simuleval/tools/patch_vllm_moe_topk_softmax_fallback.py"

case "${GLOSSARY_KIND}" in
  gs1k) RUNTIME_GLOSSARY="${RUNTIME_GLOSSARY:-${GS1K_GLOSSARY}}" ;;
  gs10k) RUNTIME_GLOSSARY="${RUNTIME_GLOSSARY:-${GS10K_GLOSSARY}}" ;;
  *) echo "[ERROR] Unsupported GLOSSARY_KIND=${GLOSSARY_KIND}" >&2; exit 2 ;;
esac

OUT_ROOT="${OUT_ROOT:-${PSC_BASE}/outputs/medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP}/${GLOSSARY_KIND}}"
LOG_ROOT="${LOG_ROOT:-${PSC_BASE}/logs/medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP}/${GLOSSARY_KIND}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-${PSC_BASE}/cache/maxsim_index_cache/medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP}/${GLOSSARY_KIND}}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_medgs_${GLOSSARY_KIND}_${SLURM_JOB_ID:-manual}}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${EVAL_TMPDIR}" \
  "${PSC_BASE}/cache/hf" "${PSC_BASE}/cache/torch" "${PSC_BASE}/cache/triton" \
  "${PSC_BASE}/cache/torchinductor" "${PSC_BASE}/cache/xdg" "${PSC_BASE}/cache/vllm"

export PATH="${ENV_DIR}/bin:${MWERSEGMENTER_ROOT}:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${ENV_DIR}"
export CONDA_DEFAULT_ENV="spaCyEnv_20260518"
export HF_HOME="${HF_HOME:-${PSC_BASE}/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PSC_BASE}/cache/hf/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PSC_BASE}/cache/hf/datasets}"
export TORCH_HOME="${TORCH_HOME:-${PSC_BASE}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PSC_BASE}/cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PSC_BASE}/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PSC_BASE}/cache/torchinductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${PSC_BASE}/cache/vllm}"
export TMPDIR="${EVAL_TMPDIR}"
export TMP="${EVAL_TMPDIR}"
export TEMP="${EVAL_TMPDIR}"
export MWERSEGMENTER_ROOT
export FBK_FAIRSEQ_ROOT

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 3
  fi
}

validate_model_dir() {
  [[ -f "${MODEL_DIR}/config.json" ]] || return 1
  local shards
  shards="$(find "${MODEL_DIR}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shards}" == "15" ]]
}

for p in \
  "${ROOT_DIR}" \
  "${ENV_DIR}/bin/python" \
  "${BATCH_LAUNCHER}" \
  "${ESO_TEST_ROOT}/sample_404_v2/full_sample_v2.json" \
  "${ESO_TEST_ROOT}/sample_404_v2/404_v2.wav" \
  "${MWERSEGMENTER_ROOT}" \
  "${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py" \
  "${HN1024_CKPT}" \
  "${FIXED_RAW_GLOSSARY}" \
  "${RUNTIME_GLOSSARY}" \
  "${NOTES_FILE}"; do
  require_path "${p}"
done
validate_model_dir || { echo "[ERROR] Model dir failed validation: ${MODEL_DIR}" >&2; exit 3; }

if [[ -f "${VLLM_MOE_TOPK_PATCH}" ]]; then
  echo "[INFO] Ensuring vLLM MoE topk_softmax fallback patch"
  "${ENV_DIR}/bin/python" "${VLLM_MOE_TOPK_PATCH}"
fi

echo "[INFO] RUN_STAMP=${RUN_STAMP}"
echo "[INFO] GLOSSARY_KIND=${GLOSSARY_KIND}"
echo "[INFO] TARGET_LMS=${TARGET_LMS}"
echo "[INFO] RUNTIME_GLOSSARY=${RUNTIME_GLOSSARY}"
echo "[INFO] FIXED_RAW_GLOSSARY=${FIXED_RAW_GLOSSARY}"
echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] EVAL_TMPDIR=${EVAL_TMPDIR}"

DEFAULT_GLOSSARY_TAG_PATTERN='hard_medicine_raw_fixeddenom__medicine_{sample}'
DEFAULT_ORACLE_TERM_MAP_TAG_PATTERN='hard_medicine.oracle_term_map_fixeddenom__medicine_{sample}'
GLOSSARY_TAG_PATTERN_VALUE="${GLOSSARY_TAG_PATTERN_OVERRIDE:-${DEFAULT_GLOSSARY_TAG_PATTERN}}"
ORACLE_TERM_MAP_TAG_PATTERN_VALUE="${ORACLE_TERM_MAP_TAG_PATTERN_OVERRIDE:-${DEFAULT_ORACLE_TERM_MAP_TAG_PATTERN}}"

ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
CONDA_BASE="${PSC_BASE}/envs" \
CONDA_ENV_NAME="spaCyEnv_20260518" \
MODEL_NAME_OVERRIDE="${MODEL_DIR}" \
HN1024_CKPT_OVERRIDE="${HN1024_CKPT}" \
RUNTIME_GLOSSARY_OVERRIDE="${RUNTIME_GLOSSARY}" \
FIXED_RAW_GLOSSARY_OVERRIDE="${FIXED_RAW_GLOSSARY}" \
ESO_TEST_ROOT_OVERRIDE="${ESO_TEST_ROOT}" \
RUN_STAMP="${RUN_STAMP}" \
LANG_CODE_OVERRIDE="zh" \
TARGET_SAMPLES_OVERRIDE="${TARGET_SAMPLES}" \
TARGET_LMS_OVERRIDE="${TARGET_LMS}" \
GPU_PAIR="${GPU_PAIR}" \
RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE:-cuda:3}" \
RAG_TOP_K_OVERRIDE="${RAG_TOP_K_OVERRIDE:-10}" \
RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD_OVERRIDE:-0.78}" \
RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE:-1.92}" \
OUTPUT_BASE_OVERRIDE="${OUT_ROOT}" \
LOG_ROOT_OVERRIDE="${LOG_ROOT}" \
INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
DENSITY_TAG_OVERRIDE="${DENSITY_TAG_OVERRIDE:-medhard5_${MODEL_LABEL}_${GLOSSARY_KIND}_fixedraw_tau0p78}" \
COMBINED_PREFIX_OVERRIDE="${COMBINED_PREFIX_OVERRIDE:-medicine5_${GLOSSARY_KIND}}" \
COMBINED_GLOSSARY_TAG_OVERRIDE="${COMBINED_GLOSSARY_TAG_OVERRIDE:-hard_medicine_raw_fixeddenom__medicine5}" \
GLOSSARY_TAG_PATTERN_OVERRIDE="${GLOSSARY_TAG_PATTERN_VALUE}" \
ORACLE_TERM_MAP_TAG_PATTERN_OVERRIDE="${ORACLE_TERM_MAP_TAG_PATTERN_VALUE}" \
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.70}" \
MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_OVERRIDE:-80.0}" \
KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_OVERRIDE:-60.0}" \
VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE:-8}" \
VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-4}" \
VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-8192}" \
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}" \
VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
bash "${BATCH_LAUNCHER}"

echo "[ALL DONE] PSC medicine ${GLOSSARY_KIND} lms=${TARGET_LMS} out=${OUT_ROOT}"
