#!/usr/bin/env bash
# ACL6060 offline evaluation with glossary-scale ablation.
#
# Sweeps across glossary sizes (baseline, 1000, 10000) with the text recall model.
# For each glossary size, runs:
#   - Text recall (always)
#   - TTS recall (optional, requires ADDITIONAL_TTS_MAPPING for expanded glossaries)
#
# Usage:
#   bash run_acl6060_eval.sh                         # baseline only, with TTS
#   GLOSSARY_SIZES="0 1000" bash run_acl6060_eval.sh # sweep baseline + 1000
#   SKIP_TTS=1 bash run_acl6060_eval.sh              # text-only (fast)
#   CUDA_GPU_ID=2 bash run_acl6060_eval.sh           # specify GPU
set -euo pipefail

# ======Configuration=====
REPO_DIR="/home/jiaxuanluo/InfiniSST"
PY_SCRIPT="${REPO_DIR}/documents/code/offline_evaluation/tts/acl6060_xeus_tts_text_eval.py"

CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

CUDA_GPU_ID="${CUDA_GPU_ID:-0}"
TOP_K="${TOP_K:-10}"
SKIP_TTS="${SKIP_TTS:-0}"

GLOSSARY_SIZES="${GLOSSARY_SIZES:-0 1000 10000}"

WIKI_GLOSSARY_PATH="${WIKI_GLOSSARY_PATH:-${REPO_DIR}/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json}"
ADDITIONAL_TTS_MAPPING="${ADDITIONAL_TTS_MAPPING:-/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval/acl6060_combined_tts_mapping.jsonl}"

LOG_DIR="/tmp/acl6060_eval_logs"
# ======Configuration=====

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export NLTK_DATA="/mnt/gemini/data/jiaxuanluo/nltk_data"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"

echo "================================================================"
echo " ACL6060 Dual-Encoder Evaluation: Qwen3-Omni (text) + XEUS (TTS)"
echo "================================================================"
echo "  CUDA_GPU_ID:   ${CUDA_GPU_ID}"
echo "  TOP_K:         ${TOP_K}"
echo "  SKIP_TTS:      ${SKIP_TTS}"
echo "  GLOSSARY_SIZES: ${GLOSSARY_SIZES}"
echo "  PYTHON_BIN:    ${PYTHON_BIN}"
echo "  LOG_DIR:       ${LOG_DIR}"
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
echo " COMPARISON SUMMARY"
echo "================================================================"

_extract_metric() {
    local log="$1" pattern="$2" col="$3"
    grep -P "^\s+${pattern}\s+[0-9]" "${log}" \
        | head -1 \
        | grep -oP '[0-9]+\.[0-9]+' \
        | sed -n "${col}p" \
        || echo "n/a"
}

printf "%-20s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n" \
    "Run" "TextR" "TextP" "TextF1" "TTSR" "TTSP" "TTSF1" "InterR" "InterP" "InterF1"
printf "%-20s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n" \
    "--------------------" "----------" "----------" "----------" "----------" "----------" "----------" "----------" "----------" "----------"

for gs in "${GS_ARRAY[@]}"; do
    gs_label="${gs}"
    if [[ "${gs}" == "0" ]]; then gs_label="baseline"; fi
    label="gs${gs_label}"
    log="${LOG_DIR}/${label}.log"
    if [[ ! -f "${log}" ]]; then
        printf "%-20s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n" \
            "${label}" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a" "n/a"
        continue
    fi
    text_r=$(_extract_metric "${log}" "Text \(Qwen3-Omni, semantic\)" 1)
    text_p=$(_extract_metric "${log}" "Text \(Qwen3-Omni, semantic\)" 2)
    text_f=$(_extract_metric "${log}" "Text \(Qwen3-Omni, semantic\)" 3)
    tts_r=$(_extract_metric "${log}" "TTS \(XEUS, acoustic\)" 1)
    tts_p=$(_extract_metric "${log}" "TTS \(XEUS, acoustic\)" 2)
    tts_f=$(_extract_metric "${log}" "TTS \(XEUS, acoustic\)" 3)
    int_r=$(_extract_metric "${log}" "Intersection" 1)
    int_p=$(_extract_metric "${log}" "Intersection" 2)
    int_f=$(_extract_metric "${log}" "Intersection" 3)
    printf "%-20s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n" \
        "${label}" "${text_r}" "${text_p}" "${text_f}" "${tts_r}" "${tts_p}" "${tts_f}" "${int_r}" "${int_p}" "${int_f}"
done

echo ""
echo "================================================================"
echo " All logs: ${LOG_DIR}/"
echo "================================================================"
