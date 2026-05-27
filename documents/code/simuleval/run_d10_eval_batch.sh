#!/usr/bin/env bash
set -euo pipefail

# Run d=10 per-paper evaluation (default glossary only).

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
RUN_SCRIPT="${ROOT_DIR}/documents/code/simuleval/run_one_density_eval.sh"
LOG_DIR="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed/zh/__logs__/d10_batch_$(date +%Y%m%d_%H%M%S)"

D10_MODEL="/mnt/aries/data4/jiaxuanluo/speech_llm_density_ablation/d10/r16/v1-20260414-160501-hf"
RAG_MODEL="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"

OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed"
GPU_SLOT="5,6,7"
# ======Configuration=====

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo " d=10 Evaluation Batch - $(date)"
echo " Model: ${D10_MODEL}"
echo " GPU slot: ${GPU_SLOT}"
echo " Log directory: ${LOG_DIR}"
echo "============================================================"

log="${LOG_DIR}/d10_default.log"
echo "[BATCH] Starting: d10_default at $(date '+%H:%M:%S')"
if env \
  DENSITY=10 MODEL_NAME="${D10_MODEL}" RAG_TOP_K=10 \
  RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  GPU_SLOT_OVERRIDE="${GPU_SLOT}" \
  SKIP_PHASE1_TAGGED=1 \
  bash "${RUN_SCRIPT}" > "${log}" 2>&1; then
  echo "[BATCH] Completed: d10_default at $(date '+%H:%M:%S')"
else
  echo "[BATCH] FAILED: d10_default (exit code $?) - see ${log}"
fi

echo ""
echo "============================================================"
echo " d=10 batch complete at $(date)"
echo " Logs: ${LOG_DIR}/"
echo "============================================================"
