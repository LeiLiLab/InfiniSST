#!/bin/bash
#SBATCH --job-name=win_align
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_win_align.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_win_align.err

set -euo pipefail

export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="/mnt/data/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/data/jiaxuanluo/cache"

python /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/validate_maxsim_window_alignment.py
