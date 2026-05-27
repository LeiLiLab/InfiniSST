#!/bin/bash
set -euo pipefail

# Submit 5 density ablation training jobs across taurus + aries.
# Each job uses 2 GPUs (EP=2) inside docker.
#
# Usage: bash run_density_train_sbatch.sh

# ======Configuration=====
DOCKER_IMAGE="modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1"
TRAIN_SCRIPT="/workspace/InfiniSST/documents/code/train/sst_omni_train/run_speech_llm_4gpu_maxsim.sh"
LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs"

BASE_MODEL_HOST="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct"
BASE_MODEL_DOCKER="/workspace/Qwen3-Omni-30B-A3B-Instruct"

DENSITY_PARTITION_MAP="1:taurus 3:taurus 5:taurus 8:aries 10:aries"
BASE_PORT=29519
# ======Configuration=====

mkdir -p "${LOG_DIR}"

echo "=============================================="
echo " Submitting 5 density training jobs (2 GPU each)"
echo " Distribution: ${DENSITY_PARTITION_MAP}"
echo "=============================================="

job_idx=0
for entry in ${DENSITY_PARTITION_MAP}; do
    d="${entry%%:*}"
    partition="${entry##*:}"
    port=$((BASE_PORT + job_idx))
    job_idx=$((job_idx + 1))

    DATASET="/mnt/gemini/data1/jiaxuanluo/density_ablation/train_maxsim_varlen_d${d}.jsonl"
    if [ ! -f "${DATASET}" ]; then
        echo "[ERROR] Missing dataset for d=${d}: ${DATASET}"
        continue
    fi

    SBATCH_SCRIPT=$(mktemp /tmp/train_d${d}_XXXXXX.sh)
    cat > "${SBATCH_SCRIPT}" << INNER_EOF
#!/bin/bash
#SBATCH --job-name=train_d${d}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
set -euo pipefail

echo "[TRAIN d=${d}] Starting on \$(hostname), partition=${partition}"
echo "[TRAIN d=${d}] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-not set}"

ALLOCATED_GPUS="\${CUDA_VISIBLE_DEVICES:-0,1}"

docker run --rm \\
    --gpus all \\
    --shm-size=32g \\
    --ipc=host \\
    -e CUDA_VISIBLE_DEVICES="\${ALLOCATED_GPUS}" \\
    -e NCCL_P2P_DISABLE=1 \\
    -e NCCL_IB_DISABLE=1 \\
    -v /mnt/taurus/home/jiaxuanluo/InfiniSST:/workspace/InfiniSST \\
    -v ${BASE_MODEL_HOST}:${BASE_MODEL_DOCKER}:ro \\
    -v /mnt/gemini/data:/mnt/gemini/data \\
    -v /mnt/gemini/data1:/mnt/gemini/data1 \\
    -v /mnt/gemini/data2:/mnt/gemini/data2 \\
    -v /mnt/aries/data4:/mnt/aries/data4 \\
    "${DOCKER_IMAGE}" \\
    bash "${TRAIN_SCRIPT}" "${d}" "${port}"

echo "[TRAIN d=${d}] Finished."
INNER_EOF

    JOB_ID=$(sbatch --parsable \
        -p "${partition}" \
        -o "${LOG_DIR}/%j_train_d${d}.out" \
        -e "${LOG_DIR}/%j_train_d${d}.err" \
        "${SBATCH_SCRIPT}")

    echo "  d=${d} -> partition=${partition}, port=${port}, job=${JOB_ID}"
done

echo ""
echo "=============================================="
echo " All jobs submitted. Monitor: squeue -u \$(whoami)"
echo " Logs: ${LOG_DIR}/"
echo "=============================================="
