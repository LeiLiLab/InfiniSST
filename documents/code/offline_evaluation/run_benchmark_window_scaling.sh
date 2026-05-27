#!/bin/bash
#SBATCH --job-name=win_ablation
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_window_ablation.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_window_ablation.err

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0}}"
export PYTHONUNBUFFERED=1

SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/benchmark_maxsim_window_scaling.py"
OUTPUT_DIR="/mnt/gemini/data2/jiaxuanluo/tcr_fcr_eval"
OUTPUT_TSV="${OUTPUT_DIR}/window_ablation_results.tsv"
# ======Configuration=====

mkdir -p "${OUTPUT_DIR}"

echo "[BENCH] Starting MaxSim window ablation at $(date)"
echo "[BENCH] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python3 "${SCRIPT}" \
    --device cuda:0 \
    --output "${OUTPUT_TSV}"

echo ""
echo "[BENCH] Done at $(date)"
echo "[BENCH] Results:"
cat "${OUTPUT_TSV}"
