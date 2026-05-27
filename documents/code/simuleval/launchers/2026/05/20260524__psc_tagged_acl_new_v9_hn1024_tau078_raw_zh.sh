#!/usr/bin/env bash
set -euo pipefail

# PSC Bridges-2 wrapper for zh tagged ACL raw main-result readout.
# Speech LLM: new_v9 assistant term-tag delay.
# Retriever: HN1024 lh1b88kw checkpoint, tau=0.78.

PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
ENV_DIR="${ENV_DIR:-${PSC_BASE}/envs/spaCyEnv_20260518}"
QUICK="${QUICK:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__tagged_acl_speech_llm_quick_zh_lm2_raw_wait_hf.sh}"
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

RUN_STAMP="${RUN_STAMP:-20260524T0400_psc_new_v9_hn1024_tau078_${SLURM_JOB_ID:-manual}}"
MODE="${MODE:-smoke}"
LMS="${LMS:-1 2 3 4}"
GLOSSARY_KINDS="${GLOSSARY_KINDS:-raw}"
GPU_PAIR="${GPU_PAIR:-0,1,2,3}"
MODEL_LABEL="${MODEL_LABEL:-new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078}"
HF_REPO_ID="${HF_REPO_ID:-gavinlaw/infinisst-new-v9-termtag-delay-oldnewv3-r32a64-keep1p0-r32-zh}"
MODEL_ROOT="${MODEL_ROOT:-${PSC_BASE}/models/new_v9_termtag_delay_oldnewv3_r32a64/keep1.0_r32}"
MODEL_DIR="${MODEL_DIR:-${MODEL_ROOT}/v0-20260524-062743-hf}"
OUT_ROOT="${OUT_ROOT:-${PSC_BASE}/outputs/tagged_acl_new_v9_hn1024_tau078_raw/${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-${PSC_BASE}/logs/tagged_acl_new_v9_hn1024_tau078_raw/${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-${PSC_BASE}/cache/maxsim_index_cache}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER:-jluo7}/infinisst_tagacl_newv9_${SLURM_JOB_ID:-manual}}"

DATA_ROOT="${DATA_ROOT:-${PSC_BASE}/data/acl6060}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-${PSC_BASE}/tools/mwerSegmenter}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-${PSC_BASE}/tools/FBK-fairseq}"
RAG_MODEL_PATH="${RAG_MODEL_PATH:-${PSC_BASE}/checkpoints/lh1b88kw_best_eval_acl6060_recallat10.pt}"
RAW_GLOSSARY="${RAW_GLOSSARY:-${PSC_BASE}/glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
GS1K_GLOSSARY="${GS1K_GLOSSARY:-${PSC_BASE}/glossaries/acl6060_tagged_gt_union_gs1000_min_norm2_backfill.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY:-${PSC_BASE}/glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__psc_tagged_acl_new_v9_hn1024_tau078_raw_zh.md}"
VLLM_MOE_TOPK_PATCH="${ROOT_DIR}/documents/code/simuleval/tools/patch_vllm_moe_topk_softmax_fallback.py"

mkdir -p "${MODEL_ROOT}" "${OUT_ROOT}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${TMP_ROOT}" \
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
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"
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
  local path="$1"
  [[ -f "${path}/config.json" ]] || return 1
  local shards
  shards="$(find "${path}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shards}" == "15" ]]
}

download_model_if_needed() {
  if validate_model_dir "${MODEL_DIR}"; then
    echo "[INFO] Model already present: ${MODEL_DIR}"
    return 0
  fi
  echo "[INFO] Downloading ${HF_REPO_ID} -> ${MODEL_DIR}"
  rm -rf "${MODEL_DIR}.tmp"
  mkdir -p "${MODEL_DIR}.tmp"
  hf download "${HF_REPO_ID}" --repo-type model --local-dir "${MODEL_DIR}.tmp"
  if ! validate_model_dir "${MODEL_DIR}.tmp"; then
    echo "[ERROR] Downloaded model failed validation: ${MODEL_DIR}.tmp" >&2
    exit 4
  fi
  rm -rf "${MODEL_DIR}"
  mv "${MODEL_DIR}.tmp" "${MODEL_DIR}"
}

for p in \
  "${ROOT_DIR}" \
  "${ENV_DIR}/bin/python" \
  "${QUICK}" \
  "${DATA_ROOT}/dev.yaml" \
  "${MWERSEGMENTER_ROOT}" \
  "${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py" \
  "${RAG_MODEL_PATH}" \
  "${RAW_GLOSSARY}" \
  "${GS1K_GLOSSARY}" \
  "${GS10K_GLOSSARY}" \
  "${NOTES_FILE}"; do
  require_path "${p}"
done

if [[ -f "${VLLM_MOE_TOPK_PATCH}" ]]; then
  echo "[INFO] Ensuring vLLM MoE topk_softmax fallback patch"
  "${ENV_DIR}/bin/python" "${VLLM_MOE_TOPK_PATCH}"
fi

download_model_if_needed

echo "[INFO] RUN_STAMP=${RUN_STAMP}"
echo "[INFO] MODE=${MODE} LMS=${LMS} GPU_PAIR=${GPU_PAIR}"
echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"

	ROOT_DIR="${ROOT_DIR}" \
	CONDA_BASE="${PSC_BASE}" \
	CONDA_ENV_NAME="envs/spaCyEnv_20260518" \
	PYTHON_BIN="${ENV_DIR}/bin/python" \
	WANDB_PYTHON="${ENV_DIR}/bin/python" \
	WANDB_HOME="${WANDB_HOME:-${HOME}}" \
	WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${HOME}/.config/wandb}" \
	DATA_ROOT="${DATA_ROOT}" \
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT}" \
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT}" \
RAW_GLOSSARY_OVERRIDE="${RAW_GLOSSARY}" \
GS1K_GLOSSARY_OVERRIDE="${GS1K_GLOSSARY}" \
GS10K_GLOSSARY_OVERRIDE="${GS10K_GLOSSARY}" \
RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
RAG_SCORE_THRESHOLD_OVERRIDE="0.78" \
RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="1.92" \
MODEL_ROOT="${MODEL_ROOT}" \
MODEL_LABEL="${MODEL_LABEL}" \
RUN_STAMP="${RUN_STAMP}" \
MODE="${MODE}" \
LANGS="zh" \
LMS="${LMS}" \
	GLOSSARY_KINDS="${GLOSSARY_KINDS}" \
RUN_GRANULARITY="full_corpus" \
PAPERS="2022.acl-long.268 2022.acl-long.367 2022.acl-long.590 2022.acl-long.110 2022.acl-long.117" \
GPU_PAIR="${GPU_PAIR}" \
OUT_ROOT="${OUT_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
INDEX_CACHE_DIR="${INDEX_CACHE_DIR}" \
NOTES_FILE="${NOTES_FILE}" \
	EVAL_GLOSSARY_PATH_GLOBAL="${EVAL_GLOSSARY_PATH_GLOBAL:-${RAW_GLOSSARY}}" \
EVAL_GLOSSARY_FOLLOWS_KIND=0 \
EVAL_TMPDIR_OVERRIDE="${TMP_ROOT}" \
	DENSITY_TAG_OVERRIDE="${DENSITY_TAG_OVERRIDE:-tagacl_new_v9_hn1024_tau078}" \
	WANDB_RUN_PREFIX_OVERRIDE="${WANDB_RUN_PREFIX_OVERRIDE:-new_v9_hn1024_tau078}" \
	WANDB_EXPERIMENT_FAMILY_OVERRIDE="${WANDB_EXPERIMENT_FAMILY_OVERRIDE:-tagged_acl_new_v9_hn1024_tau078}" \
	WANDB_VARIANT_PREFIX_OVERRIDE="${WANDB_VARIANT_PREFIX_OVERRIDE:-new_v9_hn1024_tau078}" \
WANDB_COMPUTE_TAG_OVERRIDE="compute:psc_bridges2_v100" \
STRIP_OUTPUT_TAGS_OVERRIDE="term" \
TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.70}" \
MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_OVERRIDE:-80.0}" \
KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_OVERRIDE:-60.0}" \
VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE:-8}" \
VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-4}" \
VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-8192}" \
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}" \
VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE:-cuda:3}" \
bash "${QUICK}"

echo "[ALL DONE] PSC tagged ACL new_v9 zh raw mode=${MODE} out=${OUT_ROOT}"
