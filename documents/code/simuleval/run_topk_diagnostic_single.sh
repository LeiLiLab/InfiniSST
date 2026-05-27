#!/usr/bin/env bash
set -euo pipefail

# Diagnostic: run ONE paper (110) at k=10 with VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
# to verify whether the "!!!" garbage-output issue on this taurus node is caused
# by custom all-reduce over corrupted CUDA-IPC/P2P. If this single run produces
# valid Chinese (not all "!"), we can re-enable the full top-k ablation with the
# same env var set. Otherwise we move the ablation to aries.
#
# All user-facing strings are in English.

# ======Configuration=====
EXIT_CONFIG_ERROR="2"

ROOT_DIR="/home/jiaxuanluo/InfiniSST"
RUN_SCRIPT="${ROOT_DIR}/documents/code/simuleval/run_one_density_eval.sh"

D5_MODEL="/mnt/aries/data4/jiaxuanluo/speech_llm_density_ablation/d5/r16/v0-20260414-010020-hf"
RAG_MODEL="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"

# Dedicated output base so we do not collide with the main ablation's dirs and
# can easily inspect / discard after diagnosis
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_diag_custom_ar"

# GPU 7 (TP rank 0), GPU 5 (TP rank 1), GPU 6 (RAG). GPU5<->GPU7 topo=NODE (no NVLink).
GPUS="7,5,6"

LATENCY_MULTIPLIER="1"
DENSITY="5"
RAG_TOP_K="10"

# Match the 04-14 k=10 backup: no explicit stride override so the agent uses its
# default (vllm_segment_sec / 2 = 0.48s). Isolates the test from my stride change.
SINGLE_PAPER="2022.acl-long.110"

LOG_ROOT="${OUTPUT_BASE}/zh/__logs__/diagnostic_$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_ROOT}/main.log"
# ======Configuration=====

mkdir -p "${LOG_ROOT}"

echo "[INFO] ============================================================"
echo "[INFO] Diagnostic: single paper k=10, VLLM_DISABLE_CUSTOM_ALL_REDUCE=1"
echo "[INFO] MODEL=${D5_MODEL}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] GPUS=${GPUS}"
echo "[INFO] PAPER=${SINGLE_PAPER}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] ============================================================"

if [[ ! -d "${D5_MODEL}" ]]; then
  echo "[ERROR] D5 model dir not found: ${D5_MODEL}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi
if [[ ! -f "${RAG_MODEL}" ]]; then
  echo "[ERROR] RAG model not found: ${RAG_MODEL}" >&2
  exit "${EXIT_CONFIG_ERROR}"
fi

(
  export DENSITY="${DENSITY}"
  export MODEL_NAME="${D5_MODEL}"
  export RAG_TOP_K="${RAG_TOP_K}"
  export RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL}"
  export OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}"
  export GPU_SLOT_OVERRIDE="${GPUS}"
  export LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIER}"
  export SKIP_PHASE1_TAGGED="1"
  # Diagnostic knob under test
  export VLLM_DISABLE_CUSTOM_ALL_REDUCE="1"
  # Single paper only
  export RUN_PAPERS_OVERRIDE="${SINGLE_PAPER}"
  bash "${RUN_SCRIPT}"
) > "${MAIN_LOG}" 2>&1 &
PID=$!
echo "[INFO] Launched diagnostic PID=${PID}, log=${MAIN_LOG}"
wait "${PID}"
rc=$?
echo "[INFO] Diagnostic finished with code ${rc}"
echo ""
echo "[INFO] ============================================================"
echo "[INFO] Inspect llm_output from runtime JSONL:"
find "${OUTPUT_BASE}/zh/" -name "runtime_omni_vllm_maxsim_rag_*.jsonl" -print0 2>/dev/null \
  | xargs -0 -I{} sh -c 'echo "== {} =="; grep "\"type\": \"llm_output\"" {} | head -5'
echo "[INFO] ============================================================"
echo "[INFO] If outputs are valid Chinese (not \"!!!!\"), re-run the full ablation"
echo "[INFO] with VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 set."
echo "[INFO] Otherwise, move the ablation to aries."
echo "[INFO] ============================================================"
exit "${rc}"
