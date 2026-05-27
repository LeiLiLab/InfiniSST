#!/usr/bin/env bash
# ACL6060 offline evaluation using the *extracted paper glossary* (95 GT terms).
#
# Sweeps across glossary sizes (baseline, 1000, 10000) with the text recall model.
# Baseline here is the 95 extracted-paper-glossary terms (matching run_glossary_scale_eval.sh).
#
# Usage:
#   bash run_acl6060_extracted_paper_eval.sh                           # full sweep with TTS
#   SKIP_TTS=1 bash run_acl6060_extracted_paper_eval.sh                # text-only (fast)
#   GLOSSARY_SIZES="0 1000" bash run_acl6060_extracted_paper_eval.sh   # subset
#   CUDA_GPU_ID=2 bash run_acl6060_extracted_paper_eval.sh             # specify GPU
set -euo pipefail

# ======Configuration=====
REPO_DIR="/home/jiaxuanluo/InfiniSST"
PY_SCRIPT="${REPO_DIR}/documents/code/offline_evaluation/tts/glossary_scale/acl6060_extracted_paper_glossary_eval.py"

CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

CUDA_GPU_ID="${CUDA_GPU_ID:-0}"
TOP_K="${TOP_K:-10}"
INTERSECTION_TOP_K="${INTERSECTION_TOP_K:-20}"
SKIP_TTS="${SKIP_TTS:-0}"

GLOSSARY_SIZES="${GLOSSARY_SIZES:-0 1000 10000}"

WIKI_GLOSSARY_PATH="${WIKI_GLOSSARY_PATH:-${REPO_DIR}/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json}"
ADDITIONAL_TTS_MAPPING="${ADDITIONAL_TTS_MAPPING:-/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval/acl6060_combined_tts_mapping.jsonl}"

LOG_DIR="/tmp/acl6060_extracted_paper_eval_logs"
# ======Configuration=====

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export NLTK_DATA="/mnt/gemini/data/jiaxuanluo/nltk_data"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"

echo "================================================================"
echo " ACL6060 Extracted-Paper-Glossary Evaluation (95 GT terms)"
echo "   Qwen3-Omni (text) + XEUS (TTS)"
echo "================================================================"
echo "  CUDA_GPU_ID:        ${CUDA_GPU_ID}"
echo "  TOP_K:              ${TOP_K}"
echo "  INTERSECTION_TOP_K: ${INTERSECTION_TOP_K}"
echo "  SKIP_TTS:           ${SKIP_TTS}"
echo "  GLOSSARY_SIZES:     ${GLOSSARY_SIZES}"
echo "  ADDITIONAL_TTS:     ${ADDITIONAL_TTS_MAPPING}"
echo "  PYTHON_BIN:         ${PYTHON_BIN}"
echo "  LOG_DIR:            ${LOG_DIR}"
echo "================================================================"

read -ra GS_ARRAY <<< "${GLOSSARY_SIZES}"
FAILED=0

for gs in "${GS_ARRAY[@]}"; do
    gs_label="${gs}"
    if [[ "${gs}" == "0" ]]; then gs_label="baseline"; fi
    label="gs${gs_label}"
    log="${LOG_DIR}/${label}.log"

    echo ""
    echo "=============================================="
    echo "  Running: ${label} (GLOSSARY_SIZE=${gs})"
    echo "  Log: ${log}"
    echo "=============================================="

    if CUDA_VISIBLE_DEVICES="${CUDA_GPU_ID}" \
        OFFLINE_EVAL_DEVICE="cuda:0" \
        OFFLINE_EVAL_TOP_K="${TOP_K}" \
        INTERSECTION_TOP_K="${INTERSECTION_TOP_K}" \
        GLOSSARY_SIZE="${gs}" \
        WIKI_GLOSSARY_PATH="${WIKI_GLOSSARY_PATH}" \
        ADDITIONAL_TTS_MAPPING="${ADDITIONAL_TTS_MAPPING}" \
        SKIP_TTS="${SKIP_TTS}" \
        "${PYTHON_BIN}" "${PY_SCRIPT}" > "${log}" 2>&1; then
        echo "[OK]   ${label} finished."
    else
        echo "[FAIL] ${label} exited with error! Check ${log}" >&2
        FAILED=1
    fi
done

if [[ ${FAILED} -ne 0 ]]; then
    echo ""
    echo "ERROR: One or more runs failed. Check logs in ${LOG_DIR}/" >&2
    exit 1
fi

# ---- Print comparison summary ----
echo ""
echo "================================================================"
echo " COMPARISON SUMMARY (Extracted Paper Glossary, 95 GT terms)"
echo " Text@${TOP_K} vs Intersection@${INTERSECTION_TOP_K}"
echo "================================================================"

# Extract metric from the multi-K table in the log.
# The multi-K table has lines like:
#   Text@10              0.9364   0.0998   0.1804    604   6050     10.00
#   Intersection@20      0.9302   0.1039   0.1869    600   5774      9.55
_extract_mk() {
    local log="$1" pattern="$2" col="$3"
    grep -P "^\s+${pattern}\s+[0-9]" "${log}" \
        | tail -1 \
        | grep -oP '[0-9]+\.[0-9]+' \
        | sed -n "${col}p" \
        || echo "n/a"
}

# Extract avg inter terms per no-term chunk from "Avg inter terms per no-term chunk: X.XXXX"
_extract_nt_noise() {
    local log="$1"
    grep -oP 'Avg inter terms per no-term chunk:\s+\K[0-9]+\.[0-9]+' "${log}" \
        | tail -1 \
        || echo "n/a"
}

printf "%-15s  %8s  %8s  %8s  %8s  %8s  %8s  %9s  %9s\n" \
    "Run" "TextR" "TextP" "TextF1" "InterR" "InterP" "InterF1" "AvgInter" "NTNoise"
printf "%-15s  %8s  %8s  %8s  %8s  %8s  %8s  %9s  %9s\n" \
    "---------------" "--------" "--------" "--------" "--------" "--------" "--------" "---------" "---------"

for gs in "${GS_ARRAY[@]}"; do
    gs_label="${gs}"
    if [[ "${gs}" == "0" ]]; then gs_label="baseline"; fi
    label="gs${gs_label}"
    log="${LOG_DIR}/${label}.log"
    if [[ ! -f "${log}" ]]; then
        printf "%-15s  %8s  %8s  %8s  %8s  %8s  %8s  %9s  %9s\n" \
            "${label}" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a"
        continue
    fi
    text_r=$(_extract_mk "${log}" "Text@${TOP_K}" 1)
    text_p=$(_extract_mk "${log}" "Text@${TOP_K}" 2)
    text_f=$(_extract_mk "${log}" "Text@${TOP_K}" 3)
    int_r=$(_extract_mk "${log}" "Intersection@${INTERSECTION_TOP_K}" 1)
    int_p=$(_extract_mk "${log}" "Intersection@${INTERSECTION_TOP_K}" 2)
    int_f=$(_extract_mk "${log}" "Intersection@${INTERSECTION_TOP_K}" 3)
    avg_inter=$(_extract_mk "${log}" "Intersection@${INTERSECTION_TOP_K}" 4)
    nt_noise=$(_extract_nt_noise "${log}")
    printf "%-15s  %8s  %8s  %8s  %8s  %8s  %8s  %9s  %9s\n" \
        "${label}" "${text_r}" "${text_p}" "${text_f}" "${int_r}" "${int_p}" "${int_f}" "${avg_inter}" "${nt_noise}"
done

echo ""
echo "================================================================"
echo " All logs: ${LOG_DIR}/"
echo "================================================================"
