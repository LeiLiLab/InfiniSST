#!/usr/bin/env bash
# Launch SimulEval on ACL6060 dev using:
#   - Retriever : variantE_hardneg_tcm_ep5 best checkpoint
#   - Speech LLM: gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4
#   - Threshold : tau = 0.5 (midpoint of TCM pos=0.85 / neg=0.25 region;
#                           filters out noisy candidates before they hit the LLM)
#
# Run on a node with >=3 visible GPUs (2 for vLLM TP + 1 for retriever).
# Usage:
#   bash documents/code/simuleval/run_eval_variantE_best_tau0p5_gigaspeech_zh.sh
#
# Override examples:
#   RAG_SCORE_THRESHOLD=0.45 bash .../run_eval_...sh   # sweep tau
#   RAG_TOP_K=5            bash .../run_eval_...sh     # tighter top-k

set -euo pipefail

# ======Configuration=====
ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"

# Speech LLM checkpoint (fully qualified cross-node path).
MODEL_NAME_OVERRIDE="/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4"

# Retriever: ep5 best variantE (before LR-shock resume, still the best we have).
RAG_MODEL_PATH_OVERRIDE="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_tcm_ep5_best_acl6060_gs10000.pt"
RAG_LORA_R_OVERRIDE="128"
RAG_TEXT_LORA_R_OVERRIDE="128"

# MaxSim score threshold.  Chosen as the midpoint between tcm_pos_threshold
# (0.85) and an "expected wrong-term score" (~0.15).  At training the GT
# positives sit around 0.88 and random negatives around -0.05, so tau=0.5
# keeps essentially all high-confidence positives while rejecting noise.
RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD:-0.5}"

RAG_TOP_K_OVERRIDE="${RAG_TOP_K:-10}"
LATENCY_MULTIPLIER_OVERRIDE="${LATENCY_MULTIPLIER:-1}"
EVAL_MODE_OVERRIDE="${EVAL_MODE:-acl6060}"

# GPU selection: 2 for vLLM TP, 1 for retriever.
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_VISIBLE_DEVICES_PHYSICAL:-0,1,2}"

# Output dir root tagged with the retriever + model short name.
MODEL_SHORT="gigaspeech-zh-s_v4_ner_rate1.0_k20_bsz4"
RAG_SHORT="variantE_hardneg_tcm_ep5_best"
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE:-/mnt/gemini/data2/jiaxuanluo/simuleval_eval/${MODEL_SHORT}__${RAG_SHORT}}"

DENSITY_TAG="${DENSITY_TAG:-default}"
CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT:-0}"
# ======Configuration=====

export MODEL_NAME_OVERRIDE
export RAG_MODEL_PATH_OVERRIDE
export RAG_LORA_R_OVERRIDE
export RAG_TEXT_LORA_R_OVERRIDE
export RAG_SCORE_THRESHOLD_OVERRIDE
export RAG_TOP_K_OVERRIDE
export LATENCY_MULTIPLIER_OVERRIDE
export EVAL_MODE_OVERRIDE
export CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE
export OUTPUT_BASE_OVERRIDE
export DENSITY_TAG
export CLEAN_OUTPUT_DIR_OVERRIDE

echo "[INFO] =============================================================="
echo "[INFO] SimulEval launcher (variantE best + tau=${RAG_SCORE_THRESHOLD_OVERRIDE})"
echo "[INFO] MODEL_NAME=${MODEL_NAME_OVERRIDE}"
echo "[INFO] RAG_MODEL_PATH=${RAG_MODEL_PATH_OVERRIDE}"
echo "[INFO] RAG_SCORE_THRESHOLD=${RAG_SCORE_THRESHOLD_OVERRIDE}"
echo "[INFO] RAG_TOP_K=${RAG_TOP_K_OVERRIDE}"
echo "[INFO] LATENCY_MULTIPLIER=${LATENCY_MULTIPLIER_OVERRIDE}"
echo "[INFO] CUDA_VISIBLE_DEVICES_PHYSICAL=${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE}"
echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE_OVERRIDE}"
echo "[INFO] =============================================================="

exec bash "${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
