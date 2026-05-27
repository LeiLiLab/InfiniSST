#!/bin/bash
set -euo pipefail

# Phase 6 training submission: d5_cap_adv_B (= d5_cap_adv + no-GT term_map cap).
# Single-variant wrapper mirroring run_adversarial_train_sbatch.sh.

# ======Configuration=====
DOCKER_IMAGE="modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1"
TRAIN_SCRIPT="/workspace/InfiniSST/documents/code/train/sst_omni_train/run_speech_llm_4gpu_maxsim.sh"
LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs"

BASE_MODEL_HOST="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct"
BASE_MODEL_DOCKER="/workspace/Qwen3-Omni-30B-A3B-Instruct"

TAG="phase6"
DENSITY_ARG="5_cap_adv_B"
DATASET="${DATASET_PATH_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap_adv_B.jsonl}"
SAVE_BASE="${SAVE_BASE_OVERRIDE:-/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap_adv_B}"
PARTITION="${PARTITION_OVERRIDE:-taurus}"
PORT="${PORT_OVERRIDE:-29553}"
LORA_RANK="16"
# ======Configuration=====

mkdir -p "${LOG_DIR}"

if [[ ! -f "${DATASET}" ]]; then
  echo "[ERROR] Missing dataset: ${DATASET}" >&2
  exit 2
fi

case "${PARTITION}" in
  taurus) TAURUS_DATA2_HOST="/mnt/data2" ;;
  aries)  TAURUS_DATA2_HOST="/mnt/taurus/data2" ;;
  *) echo "[ERROR] Unsupported partition: ${PARTITION}" >&2; exit 2 ;;
esac

SBATCH_SCRIPT="$(mktemp "/tmp/train_${DENSITY_ARG}_XXXXXX.sh")"
cat > "${SBATCH_SCRIPT}" <<INNER_EOF
#!/bin/bash
#SBATCH --job-name=train_${DENSITY_ARG}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
set -euo pipefail

echo "[TRAIN ${TAG}] Starting on \$(hostname), partition=${PARTITION}"
echo "[TRAIN ${TAG}] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-not set}"

ALLOCATED_GPUS="\${CUDA_VISIBLE_DEVICES:-0,1}"

docker run --rm \\
    --gpus "\"device=\${ALLOCATED_GPUS}\"" \\
    --shm-size=32g \\
    --ipc=host \\
    -e CUDA_VISIBLE_DEVICES="0,1" \\
    -e NCCL_P2P_DISABLE=1 \\
    -e NCCL_IB_DISABLE=1 \\
    -e DATASET_PATH_OVERRIDE="${DATASET}" \\
    -e SAVE_BASE_OVERRIDE="${SAVE_BASE}" \\
    -e LORA_RANK_OVERRIDE="${LORA_RANK}" \\
    -v /mnt/taurus/home/jiaxuanluo/InfiniSST:/workspace/InfiniSST \\
    -v ${BASE_MODEL_HOST}:${BASE_MODEL_DOCKER}:ro \\
    -v /mnt/gemini/data:/mnt/gemini/data \\
    -v /mnt/gemini/data1:/mnt/gemini/data1 \\
    -v /mnt/gemini/data2:/mnt/gemini/data2 \\
    -v ${TAURUS_DATA2_HOST}:/mnt/taurus/data2 \\
    "${DOCKER_IMAGE}" \\
    bash "${TRAIN_SCRIPT}" "${DENSITY_ARG}" "${PORT}"

echo "[TRAIN ${TAG}] Finished."
INNER_EOF

JOB_ID=$(sbatch --parsable \
    -p "${PARTITION}" \
    -o "${LOG_DIR}/%j_train_${DENSITY_ARG}.out" \
    -e "${LOG_DIR}/%j_train_${DENSITY_ARG}.err" \
    "${SBATCH_SCRIPT}")

echo "variant=${TAG} density_arg=${DENSITY_ARG} partition=${PARTITION}"
echo "port=${PORT}  job=${JOB_ID}"
