#!/bin/bash
set -euo pipefail

# Submit a single rank-ablation training run for the 8h audit-rank-neg
# experiment. Clones the structure of run_adversarial_train_sbatch.sh but
# only does ONE variant at a time (selected by env vars) so the orchestrator
# can run them sequentially on GPUs 5,6 with LORA rank override.
#
# Required env:
#   RUN_TAG          e.g. "d5_r32"   (used for sbatch job name and log prefix)
#   DENSITY_ARG      e.g. "5_r32"    (wandb experiment tag, derived: omni-maxsim-varlen-d${DENSITY_ARG}-r${LORA_RANK}-2gpu)
#   DATASET_PATH     absolute path to training JSONL (must exist)
#   SAVE_BASE        absolute path for SAVE_BASE_OVERRIDE
#   LORA_RANK        integer (e.g. 32, 64)
#   LORA_ALPHA       integer (we pass 32 to match old SLM args.json)
#   PARTITION        "taurus" or "aries"
#   BASE_PORT        integer (e.g. 29561)
#
# Optional env:
#   CUDA_VISIBLE_DEVICES_HINT  only for documentation; actual allocation comes from slurm.
#
# Writes stdout/stderr to /mnt/gemini/data1/jiaxuanluo/logs/${JOB_ID}_${RUN_TAG}.{out,err}.
# Returns the slurm JOB_ID to stdout.

# ======Configuration=====
DOCKER_IMAGE="modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1"
TRAIN_SCRIPT="/workspace/InfiniSST/documents/code/train/sst_omni_train/run_speech_llm_4gpu_maxsim.sh"
LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs"

BASE_MODEL_HOST="/mnt/gemini/data2/jiaxuanluo/Qwen3-Omni-30B-A3B-Instruct"
BASE_MODEL_DOCKER="/workspace/Qwen3-Omni-30B-A3B-Instruct"
# ======Configuration=====

RUN_TAG="${RUN_TAG:?RUN_TAG required}"
DENSITY_ARG="${DENSITY_ARG:?DENSITY_ARG required}"
DATASET_PATH="${DATASET_PATH:?DATASET_PATH required}"
SAVE_BASE="${SAVE_BASE:?SAVE_BASE required}"
LORA_RANK="${LORA_RANK:?LORA_RANK required}"
LORA_ALPHA="${LORA_ALPHA:?LORA_ALPHA required}"
PARTITION="${PARTITION:?PARTITION required (taurus|aries)}"
BASE_PORT="${BASE_PORT:?BASE_PORT required}"

# Parallelism knobs (forwarded to the training script inside docker). Default
# matches the historical 2-GPU EP=2 TP=1 recipe so callers that don't set them
# get byte-identical behaviour.
GRES_GPU="${GRES_GPU:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GRES_GPU}}"
EP_SIZE="${EP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"

if (( NPROC_PER_NODE != GRES_GPU )); then
  echo "[ERROR] NPROC_PER_NODE (${NPROC_PER_NODE}) must equal GRES_GPU (${GRES_GPU})" >&2
  exit 2
fi
if (( NPROC_PER_NODE % TP_SIZE != 0 )); then
  echo "[ERROR] NPROC_PER_NODE (${NPROC_PER_NODE}) not divisible by TP_SIZE (${TP_SIZE})" >&2
  exit 2
fi

# Build CUDA_VISIBLE_DEVICES inside docker as 0,1,...,NPROC-1
DOCKER_CVD=""
for i in $(seq 0 $((NPROC_PER_NODE - 1))); do
  DOCKER_CVD+="${DOCKER_CVD:+,}${i}"
done

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[ERROR] Missing dataset: ${DATASET_PATH}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
mkdir -p "${SAVE_BASE}"

case "${PARTITION}" in
  taurus) TAURUS_DATA2_HOST="/mnt/data2" ;;
  aries)  TAURUS_DATA2_HOST="/mnt/taurus/data2" ;;
  *) echo "[ERROR] Unsupported PARTITION=${PARTITION}" >&2; exit 2 ;;
esac

SBATCH_SCRIPT="$(mktemp "/tmp/train_${RUN_TAG}_XXXXXX.sh")"
cat > "${SBATCH_SCRIPT}" << INNER_EOF
#!/bin/bash
#SBATCH --job-name=${RUN_TAG}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:${GRES_GPU}
#SBATCH --time=1-00:00:00
# Send SIGTERM 60s before the hard kill so our trap can docker-kill cleanly.
#SBATCH --signal=B:TERM@60
set -euo pipefail

echo "[TRAIN ${RUN_TAG}] host=\$(hostname) partition=${PARTITION}"
echo "[TRAIN ${RUN_TAG}] slurm CVD=\${CUDA_VISIBLE_DEVICES:-not set}"
echo "[TRAIN ${RUN_TAG}] parallel: NPROC=${NPROC_PER_NODE} EP=${EP_SIZE} TP=${TP_SIZE}"

ALLOCATED_GPUS="\${CUDA_VISIBLE_DEVICES:-${DOCKER_CVD}}"

# Unique container name so scancel can reach the docker process.
CONTAINER_NAME="train_${RUN_TAG}_\${SLURM_JOB_ID}"
echo "[TRAIN ${RUN_TAG}] container=\${CONTAINER_NAME}"

# Forward slurm-induced termination to the docker container. Without this,
# docker daemon keeps the container alive after scancel kills the sbatch shell,
# because docker's PID is containerd-shim / daemon-managed, not our bash child.
_cleanup() {
  rc=\$?
  echo "[TRAIN ${RUN_TAG}] cleanup triggered (rc=\$rc), killing container \${CONTAINER_NAME}" >&2
  docker kill "\${CONTAINER_NAME}" 2>/dev/null || true
  # Give docker a moment to reap, then SIGKILL if still alive.
  sleep 3
  docker kill --signal=KILL "\${CONTAINER_NAME}" 2>/dev/null || true
  exit \$rc
}
trap _cleanup TERM INT HUP

docker run --rm \\
    --name "\${CONTAINER_NAME}" \\
    --init \\
    --gpus "\"device=\${ALLOCATED_GPUS}\"" \\
    --shm-size=32g \\
    --ipc=host \\
    -e CUDA_VISIBLE_DEVICES="${DOCKER_CVD}" \\
    -e NCCL_P2P_DISABLE=1 \\
    -e NCCL_IB_DISABLE=1 \\
    -e WANDB_MODE=offline \\
    -e DATASET_PATH_OVERRIDE="${DATASET_PATH}" \\
    -e SAVE_BASE_OVERRIDE="${SAVE_BASE}" \\
    -e LORA_RANK_OVERRIDE="${LORA_RANK}" \\
    -e LORA_ALPHA_OVERRIDE="${LORA_ALPHA}" \\
    -e MAX_LENGTH_OVERRIDE="${MAX_LENGTH_OVERRIDE:-4096}" \\
    -e NPROC_PER_NODE_OVERRIDE="${NPROC_PER_NODE}" \\
    -e EP_OVERRIDE="${EP_SIZE}" \\
    -e TP_OVERRIDE="${TP_SIZE}" \\
    -v /mnt/taurus/home/jiaxuanluo/InfiniSST:/workspace/InfiniSST \\
    -v ${BASE_MODEL_HOST}:${BASE_MODEL_DOCKER}:ro \\
    -v /mnt/gemini/data:/mnt/gemini/data \\
    -v /mnt/gemini/data1:/mnt/gemini/data1 \\
    -v /mnt/gemini/data2:/mnt/gemini/data2 \\
    -v ${TAURUS_DATA2_HOST}:/mnt/taurus/data2 \\
    "${DOCKER_IMAGE}" \\
    bash "${TRAIN_SCRIPT}" "${DENSITY_ARG}" "${BASE_PORT}" &
DOCKER_PID=\$!
# wait -n so signals are handled promptly (bash would otherwise block wait).
wait "\${DOCKER_PID}"
DOCKER_RC=\$?
trap - TERM INT HUP
echo "[TRAIN ${RUN_TAG}] Finished (rc=\${DOCKER_RC})."
exit "\${DOCKER_RC}"
INNER_EOF

JOB_ID=$(sbatch --parsable \
    -p "${PARTITION}" \
    -o "${LOG_DIR}/%j_${RUN_TAG}.out" \
    -e "${LOG_DIR}/%j_${RUN_TAG}.err" \
    "${SBATCH_SCRIPT}")

echo "${JOB_ID}"
echo "[submitted] RUN_TAG=${RUN_TAG} DENSITY_ARG=${DENSITY_ARG} rank=${LORA_RANK} alpha=${LORA_ALPHA} partition=${PARTITION} job=${JOB_ID}" >&2
echo "[submitted] dataset=${DATASET_PATH}" >&2
echo "[submitted] save_base=${SAVE_BASE}" >&2
echo "[submitted] parallel: NPROC=${NPROC_PER_NODE} EP=${EP_SIZE} TP=${TP_SIZE} gres=gpu:${GRES_GPU}" >&2
