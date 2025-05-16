#!/bin/bash

#SBATCH --job-name=test_samples
#SBATCH --partition=taurus
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --output=logs/test_samples%j.out
#SBATCH --error=logs/test_samples%j.err

text_field=$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

START=$((SLURM_ARRAY_TASK_ID * 100000))
echo "[INFO] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, START=${START}"

PYTHONUNBUFFERED=1 python3 new_giga_speech.py --text_field=$text_field --name=dev