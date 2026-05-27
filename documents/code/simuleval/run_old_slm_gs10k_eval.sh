#!/usr/bin/env bash
# Run old_slm + gs10k per-paper evaluation (lm=1, sliding window disabled via stride=window)
set -euo pipefail

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
OLD_SLM_MODEL="/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4"
GS10K_GLOSSARY="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"
GS10K_INDEX="/mnt/gemini/data2/jiaxuanluo/maxsim_index_cache/q3rag_maxsim_sp07__glossary_acl6060_gt_union_gs10000__maxsim.pt"
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed"
PAPER_INPUTS="${OUTPUT_BASE}/zh/__paper_inputs__/lists"
GPUS="${OLD_SLM_GS10K_GPUS:-6,7,5}"
PAPERS=("2022.acl-long.110" "2022.acl-long.117" "2022.acl-long.268" "2022.acl-long.367" "2022.acl-long.590")
LATENCY_MULTIPLIER="1"
VLLM_SEGMENT_SEC="0.96"
# ======Configuration=====

EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"

for PAPER_ID in "${PAPERS[@]}"; do
    SRC_LIST="${PAPER_INPUTS}/dev.source__${PAPER_ID}.txt"
    TGT_LIST="${PAPER_INPUTS}/dev.target.zh__${PAPER_ID}.txt"

    if [[ ! -f "${SRC_LIST}" ]] || [[ ! -f "${TGT_LIST}" ]]; then
        echo "[ERROR] Missing source/target for ${PAPER_ID}" >&2
        continue
    fi

    OUTPUT_DIR="${OUTPUT_BASE}/zh/dold_slm_gs10k_lm${LATENCY_MULTIPLIER}_k10_gglossary_acl6060_gt_union_gs10000_pp${PAPER_ID}"
    INSTANCES_LOG="${OUTPUT_DIR}/instances.log"

    if [[ -f "${INSTANCES_LOG}" ]] && [[ -s "${INSTANCES_LOG}" ]]; then
        echo "[SKIP] Already done: ${PAPER_ID}"
        continue
    fi

    echo "[RUN] Paper: ${PAPER_ID} on GPUs ${GPUS}"

    MODEL_NAME_OVERRIDE="${OLD_SLM_MODEL}" \
    EVAL_MODE_OVERRIDE="extracted_by_paper" \
    OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPUS}" \
    LATENCY_MULTIPLIER_OVERRIDE="${LATENCY_MULTIPLIER}" \
    DENSITY_TAG="old_slm_gs10k" \
    SKIP_OFFLINE_EVAL="1" \
    GLOSSARY_PATH_OVERRIDE="${GS10K_GLOSSARY}" \
    INDEX_PATH_OVERRIDE="${GS10K_INDEX}" \
    SRC_LIST_OVERRIDE="${SRC_LIST}" \
    TGT_LIST_OVERRIDE="${TGT_LIST}" \
    PAPER_ID_TAG="${PAPER_ID}" \
    RAG_RETRIEVE_STRIDE_SEC_OVERRIDE="${VLLM_SEGMENT_SEC}" \
    bash "${EVAL_SCRIPT}"

    echo "[DONE] Paper: ${PAPER_ID}"
done

echo "[ALL DONE] old_slm + gs10k evaluation complete."
