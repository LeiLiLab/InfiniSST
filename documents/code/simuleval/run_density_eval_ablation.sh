#!/usr/bin/env bash
set -euo pipefail

# Density ablation evaluation driver.
#
# Loops over 5 density values (1, 3, 5, 8, 10), calls eval_density_unified.sh
# for each trained Speech LLM, and collects results into a summary TSV.
#
# RAG_TOP_K per density: density * 2 (for 1.92s vLLM segment = 2 units of 0.96s)
#   d=1 -> TOP_K=2, d=3 -> TOP_K=6, d=5 -> TOP_K=10, d=8 -> TOP_K=16, d=10 -> TOP_K=20
#
# All user-facing strings are in English.

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
UNIFIED_EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"

DENSITY_VALUES=(1 3 5 8 10)

MODEL_BASE="/mnt/aries/data4/jiaxuanluo/speech_llm_density_ablation"
MODEL_SUFFIX="r16/v0-20260414-010020-hf"

RAG_MODEL_PATH="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"

OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim"

CUDA_VISIBLE_DEVICES_PHYSICAL="0,2,3"

EVAL_MODE="acl6060"
GLOSSARY_SIZE="0"
LATENCY_MULTIPLIER="1"
# ======Configuration=====

# Override env vars
MODEL_BASE_OVERRIDE="${MODEL_BASE_OVERRIDE:-}"
RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH_OVERRIDE:-}"
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE_OVERRIDE:-}"
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE:-}"
EVAL_MODE_OVERRIDE="${EVAL_MODE_OVERRIDE:-}"
DENSITY_VALUES_OVERRIDE="${DENSITY_VALUES_OVERRIDE:-}"
LATENCY_MULTIPLIER_OVERRIDE="${LATENCY_MULTIPLIER_OVERRIDE:-}"

if [[ -n "${MODEL_BASE_OVERRIDE}" ]]; then
  MODEL_BASE="${MODEL_BASE_OVERRIDE}"
fi
if [[ -n "${RAG_MODEL_PATH_OVERRIDE}" ]]; then
  RAG_MODEL_PATH="${RAG_MODEL_PATH_OVERRIDE}"
fi
if [[ -n "${OUTPUT_BASE_OVERRIDE}" ]]; then
  OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE}"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}" ]]; then
  CUDA_VISIBLE_DEVICES_PHYSICAL="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}"
fi
if [[ -n "${EVAL_MODE_OVERRIDE}" ]]; then
  EVAL_MODE="${EVAL_MODE_OVERRIDE}"
fi
if [[ -n "${DENSITY_VALUES_OVERRIDE}" ]]; then
  # shellcheck disable=SC2206
  DENSITY_VALUES=(${DENSITY_VALUES_OVERRIDE})
fi
if [[ -n "${LATENCY_MULTIPLIER_OVERRIDE}" ]]; then
  LATENCY_MULTIPLIER="${LATENCY_MULTIPLIER_OVERRIDE}"
fi

if [[ ! -f "${UNIFIED_EVAL_SCRIPT}" ]]; then
  echo "[ERROR] Unified eval script not found: ${UNIFIED_EVAL_SCRIPT}" >&2
  exit 2
fi

SUMMARY_TSV="${OUTPUT_BASE}/density_eval_summary.tsv"
mkdir -p "${OUTPUT_BASE}"

echo "[INFO] ============================================================"
echo "[INFO] Density Ablation Evaluation Driver"
echo "[INFO] DENSITY_VALUES=${DENSITY_VALUES[*]}"
echo "[INFO] MODEL_BASE=${MODEL_BASE}"
echo "[INFO] RAG_MODEL_PATH=${RAG_MODEL_PATH}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[INFO] EVAL_MODE=${EVAL_MODE}"
echo "[INFO] SUMMARY_TSV=${SUMMARY_TSV}"
echo "[INFO] ============================================================"

HEADER_WRITTEN=0

for D in "${DENSITY_VALUES[@]}"; do
  MODEL_NAME="${MODEL_BASE}/d${D}/${MODEL_SUFFIX}"
  RAG_TOP_K=$((D * 2))

  echo ""
  echo "[INFO] ======== Density=${D} TOP_K=${RAG_TOP_K} ========"
  echo "[INFO] MODEL_NAME=${MODEL_NAME}"

  if [[ ! -d "${MODEL_NAME}" ]]; then
    echo "[WARN] Model dir not found, skipping d=${D}: ${MODEL_NAME}" >&2
    continue
  fi

  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  EVAL_MODE_OVERRIDE="${EVAL_MODE}" \
  GLOSSARY_SIZE_OVERRIDE="${GLOSSARY_SIZE}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL}" \
  LATENCY_MULTIPLIER_OVERRIDE="${LATENCY_MULTIPLIER}" \
  DENSITY_TAG="${D}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="0" \
  bash "${UNIFIED_EVAL_SCRIPT}" || {
    echo "[WARN] eval_density_unified.sh failed for d=${D}" >&2
    continue
  }

  # Collect results
  MODEL_SHORT="$(basename "${MODEL_NAME}")"
  GLOSSARY_TAG="glossary_acl6060"
  RESULT_TSV_DIR="${OUTPUT_BASE}/zh/d${D}_${MODEL_SHORT}_g${GLOSSARY_TAG}_k${RAG_TOP_K}_lm${LATENCY_MULTIPLIER}"
  RESULT_TSV="${RESULT_TSV_DIR}/eval_results.tsv"

  if [[ -f "${RESULT_TSV}" ]]; then
    if [[ "${HEADER_WRITTEN}" -eq 0 ]]; then
      echo -e "density\trag_top_k\t$(head -1 "${RESULT_TSV}")" > "${SUMMARY_TSV}"
      HEADER_WRITTEN=1
    fi
    echo -e "${D}\t${RAG_TOP_K}\t$(tail -1 "${RESULT_TSV}")" >> "${SUMMARY_TSV}"
    echo "[INFO] Collected d=${D} results."
  else
    echo "[WARN] No eval_results.tsv for d=${D}: ${RESULT_TSV}" >&2
  fi
done

echo ""
echo "[INFO] ============================================================"
echo "[INFO] Density evaluation complete."
if [[ -f "${SUMMARY_TSV}" ]]; then
  echo "[INFO] Summary TSV: ${SUMMARY_TSV}"
  echo "[INFO] Contents:"
  cat "${SUMMARY_TSV}"
fi
echo "[INFO] ============================================================"
