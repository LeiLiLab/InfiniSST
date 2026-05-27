#!/bin/bash
# Launch W&B Sweep agents on taurus.
# Creates the sweep (if needed) and submits Slurm jobs as agents.
#
# Usage:
#   bash launch_sweep.sh                 # create sweep + launch agents
#   bash launch_sweep.sh <sweep_id>      # reuse existing sweep, launch more agents

set -euo pipefail

# ======Configuration=====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWEEP_CONFIG="${SCRIPT_DIR}/sweep_config.yaml"
SWEEP_TRAIN="${SCRIPT_DIR}/sweep_train.py"
WANDB_PROJECT="qwen3_rag_autoresearch"
LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs/autoresearch"

GPUS_PER_AGENT=2
NUM_AGENTS=2
SLURM_TIME="6:00:00"

CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
# ======Configuration=====

mkdir -p "${LOG_DIR}"

assert_file_exists() { [ -f "$1" ] || { echo "ERROR: $1 missing" >&2; exit 1; }; }
assert_file_exists "${SWEEP_CONFIG}"
assert_file_exists "${SWEEP_TRAIN}"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export WANDB_API_KEY=${WANDB_API_KEY:-}

if [ $# -ge 1 ]; then
    SWEEP_ID="$1"
    echo "[SWEEP] Reusing existing sweep: ${SWEEP_ID}"
else
    echo "[SWEEP] Creating new sweep..."
    SWEEP_ID=$(wandb sweep --project "${WANDB_PROJECT}" "${SWEEP_CONFIG}" 2>&1 | grep -oP '[\w-]+/[\w-]+/[\w]+$' | tail -1)
    assert_file_exists /dev/null  # dummy, just to make shellcheck happy
    echo "[SWEEP] Created sweep: ${SWEEP_ID}"
fi

echo "[SWEEP] Launching ${NUM_AGENTS} agents (${GPUS_PER_AGENT} GPUs each)..."

for i in $(seq 1 ${NUM_AGENTS}); do
    JOB=$(sbatch --parsable \
        --partition=taurus \
        --job-name="sweep_a${i}" \
        --nodes=1 --ntasks-per-node=1 \
        --cpus-per-task=16 --mem=80G \
        --gres=gpu:${GPUS_PER_AGENT} \
        --time=${SLURM_TIME} \
        --output="${LOG_DIR}/%j_sweep_agent${i}.out" \
        --error="${LOG_DIR}/%j_sweep_agent${i}.err" \
        --export=ALL,SWEEP_NUM_GPUS=${GPUS_PER_AGENT},WANDB_API_KEY=${WANDB_API_KEY} \
        --wrap="
export PATH=${CONDA_PREFIX}/bin:\$PATH
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/taurus/home/jiaxuanluo/InfiniSST:\${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export SWEEP_NUM_GPUS=${GPUS_PER_AGENT}
export WANDB_API_KEY=${WANDB_API_KEY}
mkdir -p /dev/shm/\${USER}/pytorch_tmp
export TMPDIR=/dev/shm/\${USER}/pytorch_tmp

cd ${SCRIPT_DIR}
echo '[AGENT${i}] Starting wandb agent for sweep ${SWEEP_ID}'
wandb agent --count 10 ${SWEEP_ID}
echo '[AGENT${i}] Done'
")
    echo "[SWEEP] Agent ${i} → job ${JOB}"
done

echo ""
echo "=== Sweep launched ==="
echo "Sweep ID: ${SWEEP_ID}"
echo "Dashboard: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/${WANDB_PROJECT}/sweeps"
echo "Monitor:   squeue -u \$USER -p taurus"
