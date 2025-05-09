#!/bin/bash
#SBATCH --job-name=yago_batch
#SBATCH --output=logs/yago_%A_%a.out
#SBATCH --error=logs/yago_%A_%a.err
#SBATCH --mem=1G
#SBATCH --partition=taurus
#SBATCH --cpus-per-task=1

mapfile -t FILES < <(find data/yago -type f | sort)

TASK_INDEX=$((OFFSET + SLURM_ARRAY_TASK_ID))

echo "[INFO] Task ID: $SLURM_ARRAY_TASK_ID (absolute index $TASK_INDEX)"
echo "[INFO] Processing file: ${FILES[$TASK_INDEX]}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

python YAGO.py "${FILES[$TASK_INDEX]}"