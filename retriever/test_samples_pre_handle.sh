#!/bin/bash

# 总量 230000，按 10000 一份切分，共 23 个任务
# 根据总数切分

#SBATCH --job-name=preprocess
#SBATCH --partition=taurus
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=logs/samples_%A_%a.out
#SBATCH --error=logs/samples_%A_%a.err

text_field=$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

PYTHONUNBUFFERED=1 python3 new_giga_speech.py --name=dev --split=validation --text_field=$text_field