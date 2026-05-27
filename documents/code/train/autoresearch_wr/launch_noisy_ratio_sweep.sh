#!/bin/bash
set -euo pipefail

# Launch noisy_ratio sweep on taurus.
# 5 configs × ~25 min each = sequential on 2 GPUs.
# Uses grid search (all 5 values will be tried).

# ======Configuration=====
WANDB_PROJECT="qwen3_rag_autoresearch"
SWEEP_CONFIG="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/autoresearch_wr/sweep_noisy_ratio_config.yaml"
SCRIPT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/autoresearch_wr"
CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"

GPUS_PER_AGENT=2
SLURM_TIME="3:00:00"
# ======Configuration=====

export PATH="${CONDA_PREFIX}/bin:${PATH}"
export WANDB_API_KEY=${WANDB_API_KEY:-}

echo "[SWEEP] Creating W&B sweep..."
SWEEP_ID=$(wandb sweep --project "${WANDB_PROJECT}" "${SWEEP_CONFIG}" 2>&1 | grep -oP '[\w-]+/[\w-]+/[\w]+$' | tail -1)

if [ -z "${SWEEP_ID}" ]; then
    echo "[ERROR] Failed to create sweep"
    exit 1
fi
echo "[SWEEP] Sweep ID: ${SWEEP_ID}"

JOB=$(sbatch --parsable \
    --partition=taurus \
    --job-name="nr_sweep" \
    --gres=gpu:${GPUS_PER_AGENT} \
    --cpus-per-task=16 \
    --mem=80G \
    --time=${SLURM_TIME} \
    --output=/mnt/gemini/data1/jiaxuanluo/logs/autoresearch/%j_nr_sweep.out \
    --error=/mnt/gemini/data1/jiaxuanluo/logs/autoresearch/%j_nr_sweep.err \
    --wrap="
export PATH=${CONDA_PREFIX}/bin:\${PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/taurus/home/jiaxuanluo/InfiniSST:\${PYTHONPATH:-}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export SWEEP_NUM_GPUS=${GPUS_PER_AGENT}
export SWEEP_AGENT_IDX=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200
export NCCL_P2P_DISABLE=1
cd ${SCRIPT_DIR}
wandb agent --count 5 ${SWEEP_ID}
")

echo "[SWEEP] Submitted Slurm job: ${JOB}"
echo "[SWEEP] W&B sweep: ${SWEEP_ID}"
echo "[SWEEP] Logs: /mnt/gemini/data1/jiaxuanluo/logs/autoresearch/${JOB}_nr_sweep.{out,err}"
