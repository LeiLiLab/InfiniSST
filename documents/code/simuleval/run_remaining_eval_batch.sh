#!/usr/bin/env bash
set -euo pipefail

# Run remaining configurations (gs1k, gs10k, old_slm) after d5_default completes.
# Waits for d5_default to finish by polling for its summary.

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
RUN_SCRIPT="${ROOT_DIR}/documents/code/simuleval/run_one_density_eval.sh"
LOG_DIR="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed/zh/__logs__/batch_remaining_$(date +%Y%m%d_%H%M%S)"

D5_MODEL="/mnt/aries/data4/jiaxuanluo/speech_llm_density_ablation/d5/r16/v0-20260414-010020-hf"
OLD_SLM_MODEL="/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4"
RAG_MODEL="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"

GS1K_INDEX="/mnt/gemini/data2/jiaxuanluo/maxsim_index_cache/q3rag_maxsim_sp07__glossary_acl6060_gt_union_gs1000__maxsim.pt"
GS10K_INDEX="/mnt/gemini/data2/jiaxuanluo/maxsim_index_cache/q3rag_maxsim_sp07__glossary_acl6060_gt_union_gs10000__maxsim.pt"
GS1K_GLOSSARY="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs1000.json"
GS10K_GLOSSARY="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"

OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed"

D5_DONE_MARKER="${OUTPUT_BASE}/zh/d5_lm4_k10_per_paper_combined/eval_results_by_paper.tsv"
# ======Configuration=====

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo " Remaining Eval Batch (gs1k, gs10k, old_slm) - $(date)"
echo " Waiting for d5_default to finish..."
echo " Marker: ${D5_DONE_MARKER}"
echo "============================================================"

# Wait for d5_default to complete (poll every 60s)
while [[ ! -f "${D5_DONE_MARKER}" ]]; do
  DONE_COUNT="$(ls "${OUTPUT_BASE}/zh/d5_lm"*"_k10_g"*"_pp"*/instances.log 2>/dev/null | wc -l)"
  echo "[WAIT] $(date '+%H:%M:%S') d5_default: ${DONE_COUNT}/20 per-paper jobs done"
  sleep 60
done

echo "[INFO] d5_default complete. Starting remaining configs at $(date '+%H:%M:%S')"

run_one() {
  local label="$1"
  local log="${LOG_DIR}/${label}.log"
  shift
  echo ""
  echo "[BATCH] Starting: ${label} at $(date '+%H:%M:%S')"
  if env "$@" > "${log}" 2>&1; then
    echo "[BATCH] Completed: ${label} at $(date '+%H:%M:%S')"
  else
    echo "[BATCH] FAILED: ${label} (exit code $?) - see ${log}"
  fi
}

# 1. d=5 gs=1k per-paper
run_one "d5_gs1k" \
  DENSITY=5_gs1k MODEL_NAME="${D5_MODEL}" RAG_TOP_K=10 \
  RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  GLOBAL_GLOSSARY_OVERRIDE="${GS1K_GLOSSARY}" \
  GLOBAL_INDEX_OVERRIDE="${GS1K_INDEX}" \
  SKIP_PHASE1_TAGGED=1 \
  bash "${RUN_SCRIPT}"

# 2. d=5 gs=10k per-paper
run_one "d5_gs10k" \
  DENSITY=5_gs10k MODEL_NAME="${D5_MODEL}" RAG_TOP_K=10 \
  RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  GLOBAL_GLOSSARY_OVERRIDE="${GS10K_GLOSSARY}" \
  GLOBAL_INDEX_OVERRIDE="${GS10K_INDEX}" \
  SKIP_PHASE1_TAGGED=1 \
  bash "${RUN_SCRIPT}"

# 3. Old SLM + new MaxSim RAG per-paper
run_one "old_slm" \
  DENSITY=old_slm MODEL_NAME="${OLD_SLM_MODEL}" RAG_TOP_K=10 \
  RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  SKIP_PHASE1_TAGGED=1 \
  bash "${RUN_SCRIPT}"

echo ""
echo "============================================================"
echo " Remaining batch complete at $(date)"
echo " Logs: ${LOG_DIR}/"
echo "============================================================"
