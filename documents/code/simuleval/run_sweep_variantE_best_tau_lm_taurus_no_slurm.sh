#!/usr/bin/env bash
# Bypass-SLURM sweep driver for the variantE_best retriever.
#
# Runs the cartesian product {tau} x {latency_multiplier} sequentially on
# taurus GPUs 2, 3, 6 (vLLM TP=2 on GPU 2,3; retriever on GPU 6).
#
# Usage (must be executed on taurus; MUST NOT go through sbatch):
#   bash documents/code/simuleval/run_sweep_variantE_best_tau_lm_taurus_no_slurm.sh \
#        |& tee /mnt/gemini/data1/jiaxuanluo/logs/sweep_variantE_best_tau_lm_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

# ======Configuration=====
ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"

# Physical GPU selection (taurus local indices).  Must have 2,3,6 idle.
PHYSICAL_GPUS="2,3,6"

# Sweep grid (from the tau=[0.50,0.85] step 0.05 offline sweep, we picked the
# three most informative cuts: low-recall/high-precision (0.75), mid (0.65),
# and low-precision/high-recall (0.55)).
TAU_VALUES=(0.55 0.65 0.75)
LM_VALUES=(1 2 3 4)

# Speech LLM + retriever checkpoints (fully qualified cross-node paths).
MODEL_NAME_OVERRIDE="/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4"
RAG_MODEL_PATH_OVERRIDE="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_tcm_ep5_best_acl6060_gs10000.pt"
RAG_LORA_R_OVERRIDE="128"
RAG_TEXT_LORA_R_OVERRIDE="128"
RAG_TOP_K_OVERRIDE="10"
EVAL_MODE_OVERRIDE="acl6060"

MODEL_SHORT="gigaspeech-zh-s_v4_ner_rate1.0_k20_bsz4"
RAG_SHORT="variantE_hardneg_tcm_ep5_best"
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/simuleval_eval/${MODEL_SHORT}__${RAG_SHORT}_tau_lm_sweep"
SWEEP_SUMMARY_TSV="${OUTPUT_BASE}/sweep_summary.tsv"

DENSITY_TAG_BASE="default"

# Env
export CONDA_BASE="/mnt/taurus/home/jiaxuanluo/miniconda3"
export CONDA_ENV_NAME="spaCyEnv"
export CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

LOCAL_TMP_DIR="/tmp/${USER}_sweep_$$/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export PYTORCH_KERNEL_CACHE_PATH="${LOCAL_TMP_DIR}"
cleanup_tmpdir() { rm -rf "/tmp/${USER}_sweep_$$" 2>/dev/null || true; }
trap cleanup_tmpdir EXIT
# ======Configuration=====

mkdir -p "${OUTPUT_BASE}"

# ---- GPU preflight: verify each physical GPU is free (<500 MiB used). ----
echo "[INFO] Preflight: checking GPUs ${PHYSICAL_GPUS}"
IFS=',' read -r -a GPU_ARR <<< "${PHYSICAL_GPUS}"
for gid in "${GPU_ARR[@]}"; do
  used=$(nvidia-smi -i "${gid}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ -z "${used}" ]] || (( used > 500 )); then
    echo "[ERROR] GPU ${gid} busy (used=${used} MiB, need <500). Abort." >&2
    exit 2
  fi
  echo "[INFO]   GPU ${gid}: ${used} MiB used -> OK"
done

echo "[INFO] Sweep grid: TAU=(${TAU_VALUES[*]})  LM=(${LM_VALUES[*]})"
echo "[INFO] Output base: ${OUTPUT_BASE}"
echo "[INFO] Summary TSV: ${SWEEP_SUMMARY_TSV}"

# Initialize summary header (one row per (tau, lm)).
if [[ ! -f "${SWEEP_SUMMARY_TSV}" ]]; then
  printf "tau\tlm\tstatus\tstart_time\tend_time\teval_tsv\n" > "${SWEEP_SUMMARY_TSV}"
fi

RUN_IDX=0
TOTAL=$(( ${#TAU_VALUES[@]} * ${#LM_VALUES[@]} ))

for tau in "${TAU_VALUES[@]}"; do
  for lm in "${LM_VALUES[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    RUN_TAG="tau${tau}_lm${lm}"
    START_TS="$(date -Iseconds)"
    echo "[INFO] =============================================================="
    echo "[INFO] [${RUN_IDX}/${TOTAL}] ${RUN_TAG}  start=${START_TS}"
    echo "[INFO] =============================================================="

    export MODEL_NAME_OVERRIDE
    export RAG_MODEL_PATH_OVERRIDE
    export RAG_LORA_R_OVERRIDE
    export RAG_TEXT_LORA_R_OVERRIDE
    export RAG_SCORE_THRESHOLD_OVERRIDE="${tau}"
    export RAG_TOP_K_OVERRIDE
    export LATENCY_MULTIPLIER_OVERRIDE="${lm}"
    export EVAL_MODE_OVERRIDE
    export CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${PHYSICAL_GPUS}"
    export OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}"
    export DENSITY_TAG="${DENSITY_TAG_BASE}"
    export CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-0}"

    RUN_STATUS="ok"
    if ! bash "${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"; then
      RUN_STATUS="failed"
      echo "[WARN] Run failed: ${RUN_TAG}" >&2
    fi

    END_TS="$(date -Iseconds)"
    # Locate output dir (constructed the same way as eval_density_unified.sh).
    GLOSSARY_TAG="glossary_acl6060"
    OUTPUT_DIR_SUFFIX="d${DENSITY_TAG_BASE}_lm${lm}_k${RAG_TOP_K_OVERRIDE}_th${tau}_g${GLOSSARY_TAG}"
    RUN_DIR="${OUTPUT_BASE}/zh/${OUTPUT_DIR_SUFFIX}"
    EVAL_TSV="${RUN_DIR}/eval_results.tsv"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${tau}" "${lm}" "${RUN_STATUS}" "${START_TS}" "${END_TS}" "${EVAL_TSV}" >> "${SWEEP_SUMMARY_TSV}"
    echo "[INFO] [${RUN_IDX}/${TOTAL}] ${RUN_TAG} ${RUN_STATUS} end=${END_TS}  dir=${RUN_DIR}"
  done
done

echo "[INFO] =============================================================="
echo "[INFO] Sweep complete.  ${RUN_IDX}/${TOTAL} runs.  Summary:"
echo "[INFO]   ${SWEEP_SUMMARY_TSV}"
echo "[INFO] =============================================================="
cat "${SWEEP_SUMMARY_TSV}"
