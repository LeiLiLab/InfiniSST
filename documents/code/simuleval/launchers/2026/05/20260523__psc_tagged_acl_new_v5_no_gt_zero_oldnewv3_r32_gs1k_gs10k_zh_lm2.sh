#!/usr/bin/env bash
set -euo pipefail

PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
ENV_DIR="${ENV_DIR:-${PSC_BASE}/envs/spaCyEnv_20260518}"
TOP_LAUNCHER="${TOP_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260523__tagged_acl_new_v5_no_gt_zero_oldnewv3_r32_gs1k_gs10k_zh_lm2.sh}"
USE_APPTAINER="${USE_APPTAINER:-0}"
APPTAINER_SIF="${APPTAINER_SIF:-${PSC_BASE}/containers/ubuntu_22_04.sif}"
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

RUN_STAMP="${RUN_STAMP:-20260523T0214_psc_${SLURM_JOB_ID:-manual}}"
GPU_PAIR="${GPU_PAIR:-0,1}"
MODEL_ROOT="${MODEL_ROOT:-${PSC_BASE}/models/keep1.0_r32}"
OUT_ROOT="${OUT_ROOT:-${PSC_BASE}/outputs/tagged_acl_new_v5_gs1k_gs10k/${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-${PSC_BASE}/logs/tagged_acl_new_v5_gs1k_gs10k/${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-${PSC_BASE}/cache/maxsim_index_cache}"
TMP_ROOT="${TMP_ROOT:-/tmp/${USER:-jluo7}/infinisst_${SLURM_JOB_ID:-manual}}"

DATA_ROOT="${DATA_ROOT:-${PSC_BASE}/data/acl6060}"
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-${PSC_BASE}/tools/mwerSegmenter}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-${PSC_BASE}/tools/FBK-fairseq}"
RAG_MODEL_PATH="${RAG_MODEL_PATH:-${PSC_BASE}/checkpoints/lh1b88kw_best_eval_acl6060_recallat10.pt}"
RAW_DENOM="${RAW_DENOM:-${PSC_BASE}/glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
GS1K_GLOSSARY="${GS1K_GLOSSARY:-${PSC_BASE}/glossaries/acl6060_tagged_gt_union_gs1000_min_norm2_backfill.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY:-${PSC_BASE}/glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"

for p in \
  "${TOP_LAUNCHER}" \
  "${ENV_DIR}/bin/python" \
  "${MODEL_ROOT}" \
  "${DATA_ROOT}/dev.yaml" \
  "${MWERSEGMENTER_ROOT}" \
  "${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py" \
  "${RAG_MODEL_PATH}" \
  "${RAW_DENOM}" \
  "${GS1K_GLOSSARY}" \
  "${GS10K_GLOSSARY}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required PSC eval path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${TMP_ROOT}"

export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${ENV_DIR}"
export CONDA_DEFAULT_ENV="spaCyEnv_20260518"
export WANDB_HOME="${WANDB_HOME:-${HOME}}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${HOME}/.config/wandb}"
export HF_HOME="${HF_HOME:-${PSC_BASE}/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PSC_BASE}/cache/hf/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PSC_BASE}/cache/hf/datasets}"
export TORCH_HOME="${TORCH_HOME:-${PSC_BASE}/cache/torch}"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"

ROOT_DIR="${ROOT_DIR}" \
CONDA_BASE="${PSC_BASE}" \
CONDA_ENV_NAME="envs/spaCyEnv_20260518" \
PYTHON_BIN="${ENV_DIR}/bin/python" \
WANDB_PYTHON="${ENV_DIR}/bin/python" \
DATA_ROOT="${DATA_ROOT}" \
MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT}" \
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT}" \
RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
RAW_GLOSSARY_OVERRIDE="${RAW_DENOM}" \
GS1K_GLOSSARY_OVERRIDE="${GS1K_GLOSSARY}" \
GS10K_GLOSSARY_OVERRIDE="${GS10K_GLOSSARY}" \
MODEL_ROOT="${MODEL_ROOT}" \
RUN_STAMP="${RUN_STAMP}" \
GPU_PAIR="${GPU_PAIR}" \
OUT_ROOT="${OUT_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
INDEX_CACHE_DIR="${INDEX_CACHE_DIR}" \
EVAL_TMPDIR_OVERRIDE="${TMP_ROOT}" \
WANDB_COMPUTE_TAG_OVERRIDE="compute:psc_bridges2_v100" \
GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.70}" \
VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-16384}" \
bash "${TOP_LAUNCHER}"
