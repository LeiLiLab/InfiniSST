#!/usr/bin/env bash
set -euo pipefail

# Dense ablation: d3 vs d5 vs d8 at lm=2, default glossary, per-paper eval.
# d5 lm=2 already exists -> skip.

# ======Configuration=====
ROOT_DIR="/home/jiaxuanluo/InfiniSST"
EVAL_SCRIPT="${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh"
MODEL_BASE="/mnt/aries/data4/jiaxuanluo/speech_llm_density_ablation"
MODEL_SUFFIX="r16/v0-20260414-010020-hf"
OUTPUT_BASE="/mnt/gemini/data2/jiaxuanluo/density_eval_maxsim_fixed"
PAPER_INPUTS="${OUTPUT_BASE}/zh/__paper_inputs__/lists"
GPUS="6,7,5"
LATENCY_MULTIPLIER="2"
RAG_TOP_K="10"
DENSITIES=(3 5 8)
PAPERS=("2022.acl-long.110" "2022.acl-long.117" "2022.acl-long.268" "2022.acl-long.367" "2022.acl-long.590")
# ======Configuration=====

for D in "${DENSITIES[@]}"; do
    MODEL_NAME="${MODEL_BASE}/d${D}/${MODEL_SUFFIX}"
    if [[ ! -d "${MODEL_NAME}" ]]; then
        echo "[ERROR] Model not found: ${MODEL_NAME}" >&2
        continue
    fi

    for PAPER_ID in "${PAPERS[@]}"; do
        SRC_LIST="${PAPER_INPUTS}/dev.source__${PAPER_ID}.txt"
        TGT_LIST="${PAPER_INPUTS}/dev.target.zh__${PAPER_ID}.txt"

        OUTPUT_DIR="${OUTPUT_BASE}/zh/d${D}_lm${LATENCY_MULTIPLIER}_k${RAG_TOP_K}_gextracted_glossary__${PAPER_ID}_pp${PAPER_ID}"
        INSTANCES_LOG="${OUTPUT_DIR}/instances.log"

        if [[ -f "${INSTANCES_LOG}" ]] && [[ -s "${INSTANCES_LOG}" ]]; then
            echo "[SKIP] d=${D} paper=${PAPER_ID} (already done)"
            continue
        fi

        echo "[RUN] d=${D} paper=${PAPER_ID}"

        GLOSSARY_PATH="${ROOT_DIR}/documents/data/data_pre/extracted_glossaries_by_paper/extracted_glossary__${PAPER_ID}.json"

        MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
        EVAL_MODE_OVERRIDE="extracted_by_paper" \
        OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
        CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPUS}" \
        LATENCY_MULTIPLIER_OVERRIDE="${LATENCY_MULTIPLIER}" \
        RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
        DENSITY_TAG="${D}" \
        SKIP_OFFLINE_EVAL="1" \
        GLOSSARY_PATH_OVERRIDE="${GLOSSARY_PATH}" \
        SRC_LIST_OVERRIDE="${SRC_LIST}" \
        TGT_LIST_OVERRIDE="${TGT_LIST}" \
        PAPER_ID_TAG="${PAPER_ID}" \
        RAG_RETRIEVE_STRIDE_SEC_OVERRIDE="1.92" \
        bash "${EVAL_SCRIPT}"

        echo "[DONE] d=${D} paper=${PAPER_ID}"
    done
done

echo "[ALL DONE] Dense ablation lm=2 complete."
