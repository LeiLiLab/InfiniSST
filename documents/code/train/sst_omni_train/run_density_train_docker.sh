#!/usr/bin/env bash
set -euo pipefail

# Direct docker training on taurus (bypass drained slurm).
# 8 GPUs total: run 4 jobs in parallel, then 5th uses the first freed pair.
#
# Usage: nohup bash run_density_train_docker.sh &

# ======Configuration=====
DOCKER_IMAGE="modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1"
TRAIN_SCRIPT="/workspace/InfiniSST/documents/code/train/sst_omni_train/run_speech_llm_4gpu_maxsim.sh"

BASE_MODEL_HOST="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct"
BASE_MODEL_DOCKER="/workspace/Qwen3-Omni-30B-A3B-Instruct"

LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs"

DENSITIES=(1 3 5 8 10)
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
BASE_PORT=29519
# ======Configuration=====

mkdir -p "${LOG_DIR}"

launch_train() {
    local d=$1
    local gpus=$2
    local port=$3
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log="${LOG_DIR}/docker_train_d${d}_${ts}.log"

    echo "[$(date '+%H:%M:%S')] Launching d=${d} on GPUs ${gpus}, port=${port}, log=${log}"

    docker run --rm \
        --gpus "\"device=${gpus}\"" \
        --shm-size=32g \
        --ipc=host \
        --name "train_d${d}" \
        -e CUDA_VISIBLE_DEVICES="0,1" \
        -e NCCL_P2P_DISABLE=1 \
        -e NCCL_IB_DISABLE=1 \
        -v /home/jiaxuanluo/InfiniSST:/workspace/InfiniSST \
        -v "${BASE_MODEL_HOST}":"${BASE_MODEL_DOCKER}":ro \
        -v /mnt/gemini/data:/mnt/gemini/data \
        -v /mnt/gemini/data1:/mnt/gemini/data1 \
        -v /mnt/gemini/data2:/mnt/gemini/data2 \
        -v /mnt/aries/data4:/mnt/aries/data4 \
        "${DOCKER_IMAGE}" \
        bash "${TRAIN_SCRIPT}" "${d}" "${port}" \
        > "${log}" 2>&1

    local rc=$?
    echo "[$(date '+%H:%M:%S')] d=${d} finished with exit code ${rc}"
    return ${rc}
}

echo "=============================================="
echo " Direct Docker Training on taurus"
echo " Densities: ${DENSITIES[*]}"
echo " GPU pairs: ${GPU_PAIRS[*]}"
echo "=============================================="

declare -A PID_MAP
declare -A GPU_MAP

for i in 0 1 2 3; do
    d="${DENSITIES[$i]}"
    gpus="${GPU_PAIRS[$i]}"
    port=$((BASE_PORT + i))

    launch_train "${d}" "${gpus}" "${port}" &
    PID_MAP["${d}"]=$!
    GPU_MAP["${d}"]="${gpus}"
done

echo ""
echo "Launched 4 parallel jobs. Waiting for one to finish to start d=${DENSITIES[4]}..."
echo "PIDs: d=1=${PID_MAP[1]}, d=3=${PID_MAP[3]}, d=5=${PID_MAP[5]}, d=8=${PID_MAP[8]}"
echo ""

FIFTH_D="${DENSITIES[4]}"
FREED_GPUS=""

while true; do
    for d in 1 3 5 8; do
        pid="${PID_MAP[$d]}"
        if ! kill -0 "${pid}" 2>/dev/null; then
            wait "${pid}" || true
            FREED_GPUS="${GPU_MAP[$d]}"
            echo "[$(date '+%H:%M:%S')] d=${d} (PID ${pid}) done -> freed GPUs ${FREED_GPUS}"
            unset PID_MAP["${d}"]
            break 2
        fi
    done
    sleep 30
done

port=$((BASE_PORT + 4))
echo "[$(date '+%H:%M:%S')] Starting d=${FIFTH_D} on freed GPUs ${FREED_GPUS}, port=${port}"
launch_train "${FIFTH_D}" "${FREED_GPUS}" "${port}" &
PID_MAP["${FIFTH_D}"]=$!

echo ""
echo "Waiting for all remaining jobs to finish..."

FAIL_COUNT=0
for d in "${!PID_MAP[@]}"; do
    pid="${PID_MAP[$d]}"
    echo "  Waiting d=${d} (PID ${pid})..."
    if ! wait "${pid}"; then
        echo "  [WARN] d=${d} exited with error"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        echo "  d=${d} completed successfully"
    fi
done

echo ""
echo "=============================================="
if [ "${FAIL_COUNT}" -gt 0 ]; then
    echo " DONE with ${FAIL_COUNT} failures. Check logs in ${LOG_DIR}/"
else
    echo " ALL 5 DENSITY JOBS COMPLETED SUCCESSFULLY"
fi
echo " Logs: ${LOG_DIR}/docker_train_d*"
echo "=============================================="
