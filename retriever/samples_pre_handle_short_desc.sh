#!/bin/bash

# 总量 230000，按 10000 一份切分，共 23 个任务
# 根据总数切分

#SBATCH --job-name=preprocess
#SBATCH --partition=taurus
#SBATCH --array=0-23%4               # 共 9 个任务，每次最多跑 6 个并发（可调整）
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --dependency=afterok:36469
#SBATCH --output=logs/samples_%A_%a.out
#SBATCH --error=logs/samples_%A_%a.err

name=$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

START=$((SLURM_ARRAY_TASK_ID * 100000))
echo "[INFO] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, START=${START}"

PYTHONUNBUFFERED=1 python3 new_giga_speech.py --start=${START} --limit=100000 --name=$name --text_field=short_description