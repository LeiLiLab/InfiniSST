#!/usr/bin/env bash
set -euo pipefail

# Delayed text-only baseline runner.
# Waits for dual-encoder runs to finish (sleep), then runs text-only RAG
# with gs1000 expanded glossaries, k2=10, latency 1 2 3 4.
# After completion, aggregates results.

# ======Configuration=====
DELAY_SECONDS="${DELAY_SECONDS:-7200}"

MODEL_NAME="/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4"
RAG_MODEL_PATH="/mnt/gemini/data/jiaxuanluo/q3rag_scale_lora-r32-tr64_bs4k_t=0.03_v1_best.pt"
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_text_only_gs1000_owaski_bsz4"

BYPASS_SCRIPT="/home/jiaxuanluo/InfiniSST/documents/code/simuleval/rank16/tts/paper_extract/bypass_simuleval_rank16_tts_extracted_glossary_by_paper.sh"
AGGREGATE_SCRIPT="/home/jiaxuanluo/InfiniSST/documents/code/simuleval/rank16/aggregate_streamlaal_term_acc_per_paper_rank16.sh"

CUDA_DEVICES="0,1,2,3"
GLOSSARY_SIZE="1000"
RAG_K2="10"
RAG_K1="10"
LATENCY_MULTIPLIERS="1 2 3 4"
# ======Configuration=====

echo "[$(date)] Waiting ${DELAY_SECONDS}s before starting text-only baseline..."
sleep "${DELAY_SECONDS}"
echo "[$(date)] Delay finished. Starting text-only baseline runs."

echo "[$(date)] Checking GPU availability..."
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${CUDA_DEVICES}" \
LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIERS}" \
RAG_K2_VALUES_OVERRIDE="${RAG_K2}" \
GLOSSARY_SIZE_OVERRIDE="${GLOSSARY_SIZE}" \
RESUME_MODE="0" \
AUTO_BUILD_RAG_INDEX="1" \
RAG_EVAL_MODE_OVERRIDE="text" \
MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
bash "${BYPASS_SCRIPT}"

echo "[$(date)] Text-only baseline runs complete. Starting aggregation..."

OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
LATENCY_MULTIPLIERS_OVERRIDE="${LATENCY_MULTIPLIERS}" \
RAG_K2_VALUES_OVERRIDE="${RAG_K2}" \
RAG_K1_FIXED="${RAG_K1}" \
GLOSSARY_SIZE_OVERRIDE="${GLOSSARY_SIZE}" \
bash "${AGGREGATE_SCRIPT}"

echo "[$(date)] All done. Results aggregated."
