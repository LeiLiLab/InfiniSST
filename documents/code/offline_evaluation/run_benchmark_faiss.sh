#!/bin/bash
#SBATCH --job-name=faiss_bench
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_faiss_bench.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_faiss_bench.err

set -euo pipefail

export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="0"

PYTHONUNBUFFERED=1 python /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/benchmark_faiss_retrieval.py
