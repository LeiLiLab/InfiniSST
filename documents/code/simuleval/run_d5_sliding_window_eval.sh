#!/usr/bin/env bash
set -euo pipefail

# Phase 1: d5 sliding window eval (stride=half) across 4 lm x 3 glossary x 5 papers.
# Compares against existing d5 results that used stride=full (no overlap).

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
D5_MODEL="/mnt/aries/data4/jiaxuanluo/speech_llm_density_ablation/d5/r16/v0-20260414-010020-hf"
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed"
PAPER_INPUTS="${OUTPUT_BASE}/zh/__paper_inputs__/lists"
GPUS="6,7,5"
RAG_TOP_K="10"

LATENCY_MULTIPLIERS=(1 2 3 4)
PAPERS=("2022.acl-long.110" "2022.acl-long.117" "2022.acl-long.268" "2022.acl-long.367" "2022.acl-long.590")

GLOSSARY_DEFAULT_DIR="${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper"
GLOSSARY_GS1K="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs1000.json"
GLOSSARY_GS10K="${ROOT_DIR}/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json"
INDEX_GS1K="/mnt/gemini/data2/jiaxuanluo/maxsim_index_cache/q3rag_maxsim_sp07__glossary_acl6060_gt_union_gs1000__maxsim.pt"
INDEX_GS10K="/mnt/gemini/data2/jiaxuanluo/maxsim_index_cache/q3rag_maxsim_sp07__glossary_acl6060_gt_union_gs10000__maxsim.pt"
# ======Configuration=====

DENSITY_TAG="5_sw"

run_one() {
    local LM="$1" GLOSSARY_PATH="$2" GLOSSARY_TAG="$3" INDEX_OVERRIDE="$4" PAPER_ID="$5"

    local SRC_LIST="${PAPER_INPUTS}/dev.source__${PAPER_ID}.txt"
    local TGT_LIST="${PAPER_INPUTS}/dev.target.zh__${PAPER_ID}.txt"
    local OUTPUT_DIR="${OUTPUT_BASE}/zh/d${DENSITY_TAG}_lm${LM}_k${RAG_TOP_K}_g${GLOSSARY_TAG}_pp${PAPER_ID}"
    local INSTANCES_LOG="${OUTPUT_DIR}/instances.log"

    if [[ -f "${INSTANCES_LOG}" ]] && [[ -s "${INSTANCES_LOG}" ]]; then
        echo "[SKIP] lm=${LM} gs=${GLOSSARY_TAG} paper=${PAPER_ID}"
        return 0
    fi

    echo "[RUN] lm=${LM} gs=${GLOSSARY_TAG} paper=${PAPER_ID}"

    export INDEX_PATH_OVERRIDE="${INDEX_OVERRIDE}"

    MODEL_NAME_OVERRIDE="${D5_MODEL}" \
    EVAL_MODE_OVERRIDE="extracted_by_paper" \
    OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPUS}" \
    LATENCY_MULTIPLIER_OVERRIDE="${LM}" \
    RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
    DENSITY_TAG="${DENSITY_TAG}" \
    SKIP_OFFLINE_EVAL="1" \
    GLOSSARY_PATH_OVERRIDE="${GLOSSARY_PATH}" \
    SRC_LIST_OVERRIDE="${SRC_LIST}" \
    TGT_LIST_OVERRIDE="${TGT_LIST}" \
    PAPER_ID_TAG="${PAPER_ID}" \
    bash "${EVAL_SCRIPT}"

    unset INDEX_PATH_OVERRIDE

    echo "[DONE] lm=${LM} gs=${GLOSSARY_TAG} paper=${PAPER_ID}"
}

for LM in "${LATENCY_MULTIPLIERS[@]}"; do
    for PAPER_ID in "${PAPERS[@]}"; do
        # default (per-paper extracted glossary)
        GLOSSARY_PATH="${GLOSSARY_DEFAULT_DIR}/extracted_glossary__${PAPER_ID}.json"
        run_one "${LM}" "${GLOSSARY_PATH}" "extracted_glossary__${PAPER_ID}" "" "${PAPER_ID}"

        # gs1k
        run_one "${LM}" "${GLOSSARY_GS1K}" "glossary_acl6060_gt_union_gs1000" "${INDEX_GS1K}" "${PAPER_ID}"

        # gs10k
        run_one "${LM}" "${GLOSSARY_GS10K}" "glossary_acl6060_gt_union_gs10000" "${INDEX_GS10K}" "${PAPER_ID}"
    done
done

echo "[ALL DONE] d5 sliding window evaluation complete."
