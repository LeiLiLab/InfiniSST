#!/bin/bash
#SBATCH --job-name=prep_shards
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_prep_shards.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_prep_shards.err

set -euo pipefail

export PYTHONPATH="/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

python /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/prepare_split_transfer_tar.py

echo "All shards prepared!"

















