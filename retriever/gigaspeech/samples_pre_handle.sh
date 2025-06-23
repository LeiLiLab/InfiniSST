#!/bin/bash

# 总量 8282989，按 100000 一份切分，共 82 个任务
# 根据总数切分

#SBATCH --job-name=preprocess
#SBATCH --partition=taurus
#SBATCH --array=0-23%3
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=logs/samples_%A_%a.out
#SBATCH --error=logs/samples_%A_%a.err

name=$1
text_field=$2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

START=$((SLURM_ARRAY_TASK_ID * 100000))
echo "[INFO] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, START=${START}"

PYTHONUNBUFFERED=1 python3 train_samples_pre_handle.py --start=${START} --limit=100000 --name=$name --text_field=$text_field