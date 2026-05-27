#!/usr/bin/env bash
# Run glossary-scale ablation across multiple text models and glossary sizes.
# Each model runs on a separate GPU in parallel; glossary sizes run sequentially per model.
#
# Usage:
#   bash run_glossary_scale_eval.sh
#   CUDA_GPU_IDS="2,3" bash run_glossary_scale_eval.sh
set -euo pipefail

# ======Configuration=====
REPO_DIR="/home/jiaxuanluo/InfiniSST"
EVAL_SCRIPT="${REPO_DIR}/documents/code/offline_evaluation/tts/glossary_scale/acl6060_extracted_paper_glossary_eval.py"
PYTHON_BIN="/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python"

CUDA_GPU_IDS="${CUDA_GPU_IDS:-0,1,2}"
LOG_DIR="/tmp/glossary_scale_eval_logs"

MODEL_DIR="/mnt/gemini/data/jiaxuanluo"
TEXT_MODEL_PATHS=(
    "${MODEL_DIR}/q3rag_scale_lora-r32-tr64_bs4k_t=0.03_v1_best.pt"
    "${MODEL_DIR}/q3rag_scale_lora-r32-tr16_bs4092_t=0.03_v1_best.pt"
    "/mnt/gemini/data/jiaxuanluo/q3rag_scale_lora-r32-tr128_bs4k_t=0.03_v1_best.pt"
    #"${MODEL_DIR}/q3rag_masked_neg_lora-r32-tr16_bs4k_ttm=query key value_t=0.03_nb4096_v1_epoch_3.pt"
)
MODEL_LABELS=(
    "scale_lora-r64-tr64"
    "scale_lora-r16-tr16"
    "scale_lora-r256-tr128"
    #"nb4096_ep3"
)

GLOSSARY_SIZES=("0" "1000" "10000")
# ======Configuration=====

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}"

IFS=',' read -ra GPU_LIST <<< "${CUDA_GPU_IDS}"
num_gpus=${#GPU_LIST[@]}
num_models=${#TEXT_MODEL_PATHS[@]}

if [[ ${#MODEL_LABELS[@]} -ne ${num_models} ]]; then
    echo "ERROR: MODEL_LABELS length (${#MODEL_LABELS[@]}) != TEXT_MODEL_PATHS length (${num_models})" >&2
    exit 1
fi
if [[ ${num_gpus} -ne ${num_models} ]]; then
    echo "ERROR: CUDA_GPU_IDS has ${num_gpus} GPUs but there are ${num_models} models. Must match." >&2
    exit 1
fi

run_model_sweep() {
    local gpu_id="$1"
    local model_path="$2"
    local model_label="$3"

    if [[ ! -f "${model_path}" ]]; then
        echo "ERROR: Model not found: ${model_path}" >&2
        return 1
    fi

    for gs in "${GLOSSARY_SIZES[@]}"; do
        local gs_label="${gs}"
        if [[ "${gs}" == "0" ]]; then gs_label="baseline"; fi
        local label="${model_label}_gs${gs_label}"
        local log="${LOG_DIR}/${label}.log"

        echo ""
        echo "[GPU ${gpu_id}] =============================================="
        echo "[GPU ${gpu_id}]  Running: ${label}"
        echo "[GPU ${gpu_id}]    MODEL: $(basename "${model_path}")"
        echo "[GPU ${gpu_id}]    GLOSSARY_SIZE=${gs}"
        echo "[GPU ${gpu_id}] =============================================="

        CUDA_VISIBLE_DEVICES="${gpu_id}" \
            GLOSSARY_SIZE="${gs}" \
            TEXT_MODEL_PATH="${model_path}" \
            SKIP_TTS=1 \
            "${PYTHON_BIN}" "${EVAL_SCRIPT}" > "${log}" 2>&1

        echo "[GPU ${gpu_id}] [done] ${label} -> ${log}"
    done
}

# Launch each model on its own GPU in background
PIDS=()
for (( m=0; m<num_models; m++ )); do
    gpu_id="${GPU_LIST[$m]}"
    model_path="${TEXT_MODEL_PATHS[$m]}"
    model_label="${MODEL_LABELS[$m]}"

    echo "Launching ${model_label} on GPU ${gpu_id} (background)"
    run_model_sweep "${gpu_id}" "${model_path}" "${model_label}" &
    PIDS+=($!)
done

echo ""
echo "All ${num_models} model sweeps launched. Waiting..."
echo "  PIDs: ${PIDS[*]}"
echo ""

FAILED=0
for (( m=0; m<num_models; m++ )); do
    pid="${PIDS[$m]}"
    label="${MODEL_LABELS[$m]}"
    if wait "${pid}"; then
        echo "[OK]   ${label} (PID ${pid}) finished."
    else
        echo "[FAIL] ${label} (PID ${pid}) exited with error!" >&2
        FAILED=1
    fi
done

if [[ ${FAILED} -ne 0 ]]; then
    echo ""
    echo "ERROR: One or more models failed. Check logs in ${LOG_DIR}/" >&2
    exit 1
fi

# ---- Print comparison summary ----
echo ""
echo "================================================================"
echo " COMPARISON SUMMARY"
echo "================================================================"
printf "%-30s  %12s\n" "Run" "TextRecall"
printf "%-30s  %12s\n" "------------------------------" "------------"

for (( m=0; m<num_models; m++ )); do
    model_label="${MODEL_LABELS[$m]}"
    for gs in "${GLOSSARY_SIZES[@]}"; do
        gs_label="${gs}"
        if [[ "${gs}" == "0" ]]; then gs_label="baseline"; fi
        label="${model_label}_gs${gs_label}"
        log="${LOG_DIR}/${label}.log"
        if [[ ! -f "${log}" ]]; then
            printf "%-30s  %12s\n" "${label}" "(no log)"
            continue
        fi
        recall=$(grep -oP "Text \(Qwen3-Omni, semantic\)\s+\K[0-9]+\.[0-9]+" "${log}" | head -1 || echo "n/a")
        printf "%-30s  %12s\n" "${label}" "${recall}"
    done
done

echo ""
echo "================================================================"
echo " All logs: ${LOG_DIR}/"
echo "================================================================"
