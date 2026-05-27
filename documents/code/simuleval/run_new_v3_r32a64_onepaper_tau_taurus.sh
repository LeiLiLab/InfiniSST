#!/usr/bin/env bash
#SBATCH --job-name=acl_newv3_tau
#SBATCH --partition=taurus
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_acl_newv3_tau.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_acl_newv3_tau.err

set -euo pipefail

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
ONEPAPER="${ROOT_DIR}/documents/code/simuleval/run_acl_onepaper_lm_raw1k10k_taurus.sh"
WANDB_LOGGER="${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py"

TAU="${RAG_SCORE_THRESHOLD_OVERRIDE:?Set RAG_SCORE_THRESHOLD_OVERRIDE to 0.0 or 0.75}"
case "${TAU}" in
  0|0.0) TAU_TAG="tau0"; TAU_VALUE="0.0"; NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes_new_v3_r32a64_onepaper_tau0.md" ;;
  0.75) TAU_TAG="tau075"; TAU_VALUE="0.75"; NOTES_FILE="${ROOT_DIR}/documents/code/simuleval/notes_new_v3_r32a64_onepaper_tau075.md" ;;
  *) echo "[ERROR] Unsupported tau: ${TAU}" >&2; exit 2 ;;
esac

HF_MODEL="/mnt/gemini/data2/jiaxuanluo/speech_llm_tcmw100kgt_tau075_new_v3_r32a64_taurus8/keep1.0_r32/v0-20260508-122348-hf"
TARGET_PAPER="${TARGET_PAPER:-2022.acl-long.110}"
TARGET_LM="${TARGET_LM:-1}"
RAG_TOP_K="${RAG_TOP_K_OVERRIDE:-10}"
RAG_STREAMING_MODE="${RAG_STREAMING_MODE_OVERRIDE:-timeline}"
RAG_MAXSIM_WINDOWS="${RAG_MAXSIM_WINDOWS_OVERRIDE:-2 3 4 5 6 7 8 10 12 16 20 24}"
RAG_MAXSIM_STRIDE="${RAG_MAXSIM_STRIDE_OVERRIDE:-2}"
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/acl_onepaper_lm1_raw1k10k_timeline_fullwin_${TAU_TAG}_new_v3_r32a64_slm"
DENSITY_TAG="aclone_newv3_r32a64_timeline_fullwin_${TAU_TAG}"
TRAINED_FROM_RUN="mazrc3id"
BASELINE_RUN_ID="5e1iu7zo"
# vLLM memory budget: unset here; run_acl_onepaper_lm_raw1k10k_taurus.sh uses GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72.

if [[ -n "${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-}" ]]; then
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV//:/,}"
fi
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-0,1,2}"

for p in "${ONEPAPER}" "${WANDB_LOGGER}" "${NOTES_FILE}" "${HF_MODEL}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

echo "[INFO] TAU=${TAU_VALUE}"
echo "[INFO] HF_MODEL=${HF_MODEL}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] DENSITY_TAG=${DENSITY_TAG}"
echo "[INFO] RAG_STREAMING_MODE=${RAG_STREAMING_MODE}"
echo "[INFO] RAG_MAXSIM_WINDOWS=${RAG_MAXSIM_WINDOWS}"
echo "[INFO] RAG_MAXSIM_STRIDE=${RAG_MAXSIM_STRIDE}"
echo "[INFO] CUDA_VISIBLE_DEVICES_PHYSICAL=${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}"

MODEL_NAME_OVERRIDE="${HF_MODEL}" \
TARGET_PAPER="${TARGET_PAPER}" \
TARGET_LM="${TARGET_LM}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
RAG_SCORE_THRESHOLD_OVERRIDE="${TAU_VALUE}" \
RAG_STREAMING_MODE_OVERRIDE="${RAG_STREAMING_MODE}" \
RAG_MAXSIM_WINDOWS_OVERRIDE="${RAG_MAXSIM_WINDOWS}" \
RAG_MAXSIM_STRIDE_OVERRIDE="${RAG_MAXSIM_STRIDE}" \
RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}" \
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
bash "${ONEPAPER}"

log_one_glossary() {
  local glossary_tag="$1"
  local run_suffix="$2"
  python3 "${WANDB_LOGGER}" \
    --project simuleval_eval \
    --run-name "new_v3_r32a64__${TARGET_PAPER}__lm${TARGET_LM}__${TAU_TAG}__${run_suffix}" \
    --experiment-family sst_new_v3_onepaper \
    --data-tag acl6060_onepaper \
    --notes-file "${NOTES_FILE}" \
    --trained-from-run "${TRAINED_FROM_RUN}" \
    --baseline-run-ids "${BASELINE_RUN_ID}" \
    --extra-tags "variant:newv3_r32a64" "compute:taurus3" "tau:${TAU_TAG}" "paper:${TARGET_PAPER}" "glossary:${run_suffix}" \
    --density "${DENSITY_TAG}" \
    --rag-top-k "${RAG_TOP_K}" \
    --rag-score-threshold "${TAU_VALUE}" \
    --paper-id "${TARGET_PAPER}" \
    --output-base "${OUTPUT_BASE}" \
    --lang-code zh \
    --latency-multipliers "${TARGET_LM}" \
    --glossary-tag "${glossary_tag}" \
    --model-name "${HF_MODEL}" \
    --verdict "Logged one-paper new_v3 rank32/alpha64 SimulEval for ${TARGET_PAPER}, lm=${TARGET_LM}, tau=${TAU_VALUE}, glossary=${run_suffix}."
}

log_one_glossary "extracted_glossary__${TARGET_PAPER}" "raw"
log_one_glossary "glossary_acl6060_gt_union_gs1000" "gs1k"
log_one_glossary "glossary_acl6060_gt_union_gs10000" "gs10k"

python3 "${ROOT_DIR}/documents/code/general/wandb_tool.py" --project simuleval_eval db-sync \
  --family sst_new_v3_onepaper --best-bundles --limit 50

echo "[ALL DONE] new_v3 one-paper eval tau=${TAU_VALUE}"
