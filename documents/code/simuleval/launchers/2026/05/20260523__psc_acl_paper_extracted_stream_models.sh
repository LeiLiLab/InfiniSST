#!/usr/bin/env bash
set -euo pipefail

# PSC wrapper for ACL paper-extracted main evals when /ocean cannot hold all
# six 66G speech-LLM model directories at once.

PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
ENV_DIR="${ENV_DIR:-${PSC_BASE}/envs/spaCyEnv_20260518}"
MODEL_ROOT_OWASKI="${MODEL_ROOT_OWASKI:-${PSC_BASE}/models/owaski}"
HF_HOME="${HF_HOME:-${PSC_BASE}/cache/hf}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)_psc_stream_${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PSC_BASE}/outputs/acl_paper_extracted_main/${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-${PSC_BASE}/logs/acl_paper_extracted_main/${RUN_STAMP}}"
MAIN_LAUNCHER="${MAIN_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260523__acl_paper_extracted_main_no_tmsft_llmgen_rasst.sh}"
APPTAINER_SIF="${APPTAINER_SIF:-${PSC_BASE}/containers/ubuntu_22_04_gcc.sif}"

MODEL_CHUNKS="${MODEL_CHUNKS:-no_tmsft_zh no_tmsft_de no_tmsft_ja rasst_zh rasst_de rasst_ja}"
DELETE_MODEL_AFTER_CHUNK="${DELETE_MODEL_AFTER_CHUNK:-1}"
CLEAN_HF_CACHE_AFTER_PULL="${CLEAN_HF_CACHE_AFTER_PULL:-1}"

GPU_GROUPS_CSV="${GPU_GROUPS_CSV:-0,1,2,3}"
MAX_PARALLEL_OVERRIDE="${MAX_PARALLEL_OVERRIDE:-1}"
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.68}"
VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-8192}"
VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-4}"
RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE:-cuda:3}"
VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE:-8}"
MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_OVERRIDE:-4}"
KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_OVERRIDE:-4}"

export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${ENV_DIR}"
export CONDA_DEFAULT_ENV="$(basename "${ENV_DIR}")"
export HF_HOME

mkdir -p "${MODEL_ROOT_OWASKI}" "${HF_HOME}" "${OUTPUT_ROOT}" "${LOG_ROOT}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "[ERROR] PSC env python not found: ${ENV_DIR}/bin/python" >&2
  exit 3
fi
if [[ ! -f "${MAIN_LAUNCHER}" ]]; then
  echo "[ERROR] Main launcher not found: ${MAIN_LAUNCHER}" >&2
  exit 3
fi
if [[ ! -f "${APPTAINER_SIF}" ]]; then
  echo "[ERROR] Apptainer SIF not found: ${APPTAINER_SIF}" >&2
  exit 3
fi

model_specs() {
  cat <<'EOF'
no_tmsft	zh	gigaspeech-zh-s_origin-bsz4	gavinlaw/infinisst-no-tmsft-origin-bsz4-zh
no_tmsft	de	gigaspeech-de-s_origin-bsz4	gavinlaw/infinisst-no-tmsft-origin-bsz4-de
no_tmsft	ja	gigaspeech-ja-s_origin-bsz4	gavinlaw/infinisst-no-tmsft-origin-bsz4-ja
rasst	zh	gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4	gavinlaw/infinisst-llmgen-rasst-bsz4-zh
rasst	de	gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4	gavinlaw/infinisst-llmgen-rasst-bsz4-de
rasst	ja	gigaspeech-ja-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4	gavinlaw/infinisst-llmgen-rasst-bsz4-ja
EOF
}

want_chunk() {
  local key="$1" requested
  for requested in ${MODEL_CHUNKS}; do
    [[ "${requested}" == "${key}" ]] && return 0
  done
  return 1
}

validate_model_dir() {
  local model_dir="$1"
  if [[ ! -f "${model_dir}/config.json" ]]; then
    return 1
  fi
  local shard_count
  shard_count="$(find "${model_dir}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]]
}

repo_cache_dir() {
  local repo_id="$1"
  printf '%s/hub/models--%s\n' "${HF_HOME}" "${repo_id//\//--}"
}

pull_model_if_needed() {
  local key="$1" dirname="$2" repo_id="$3"
  local target_dir="${MODEL_ROOT_OWASKI}/${dirname}"
  if validate_model_dir "${target_dir}"; then
    echo "[MODEL] existing complete ${key}: ${target_dir}"
    return 0
  fi
  echo "[MODEL] pull ${key}: ${repo_id} -> ${target_dir}"
  mkdir -p "${target_dir}"
  hf download "${repo_id}" \
    --repo-type model \
    --local-dir "${target_dir}" \
    --local-dir-use-symlinks False
  validate_model_dir "${target_dir}"
  if [[ "${CLEAN_HF_CACHE_AFTER_PULL}" == "1" ]]; then
    rm -rf "$(repo_cache_dir "${repo_id}")"
  fi
}

delete_model_if_requested() {
  local key="$1" dirname="$2" repo_id="$3"
  [[ "${DELETE_MODEL_AFTER_CHUNK}" == "1" ]] || return 0
  echo "[MODEL] delete after successful chunk ${key}: ${MODEL_ROOT_OWASKI}/${dirname}"
  rm -rf "${MODEL_ROOT_OWASKI:?}/${dirname}"
  if [[ "${CLEAN_HF_CACHE_AFTER_PULL}" == "1" ]]; then
    rm -rf "$(repo_cache_dir "${repo_id}")"
  fi
}

run_chunk() {
  local method="$1" lang="$2" dirname="$3" repo_id="$4"
  local key="${method}_${lang}"
  pull_model_if_needed "${key}" "${dirname}" "${repo_id}"
  echo "[CHUNK] start ${key} run_stamp=${RUN_STAMP}"
  PSC_BASE="${PSC_BASE}" \
  ROOT_DIR="${ROOT_DIR}" \
  ENV_DIR="${ENV_DIR}" \
  MODEL_ROOT_OWASKI="${MODEL_ROOT_OWASKI}" \
  RUN_STAMP="${RUN_STAMP}_${key}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  LOG_ROOT="${LOG_ROOT}/${key}" \
  METHODS="${method}" \
  LANGS="${lang}" \
  LMS="${LMS:-1 2 3 4}" \
  GLOSSARY_KINDS="${GLOSSARY_KINDS:-raw gs1k gs10k}" \
  GPU_GROUPS_CSV="${GPU_GROUPS_CSV}" \
  MAX_PARALLEL_OVERRIDE="${MAX_PARALLEL_OVERRIDE}" \
  USE_APPTAINER=1 \
  APPTAINER_SIF="${APPTAINER_SIF}" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE}" \
  RAG_GPU_OVERRIDE="${RAG_GPU_OVERRIDE}" \
  VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE}" \
  MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_OVERRIDE}" \
  KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_OVERRIDE}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}" \
  VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
  VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
  WANDB_COMPUTE_TAG_OVERRIDE="${WANDB_COMPUTE_TAG_OVERRIDE:-compute:psc_bridges2_v100}" \
  bash "${MAIN_LAUNCHER}"
  echo "[CHUNK] done ${key}"
  delete_model_if_requested "${key}" "${dirname}" "${repo_id}"
}

hf auth whoami
while IFS=$'\t' read -r method lang dirname repo_id; do
  key="${method}_${lang}"
  if ! want_chunk "${key}"; then
    echo "[SKIP] chunk not requested: ${key}"
    continue
  fi
  run_chunk "${method}" "${lang}" "${dirname}" "${repo_id}"
done < <(model_specs)

echo "[ALL DONE] PSC streaming model eval wrapper complete: ${OUTPUT_ROOT}"
