#!/bin/bash
#SBATCH --job-name=embed_all
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --array=0-3
#SBATCH --output=logs/embed_array_%A_%a.out
#SBATCH --error=logs/embed_array_%A_%a.err

if [ -z "$1" ]; then
  echo "[ERROR] Missing text_field argument. Usage: sbatch glossary_embedding.sh <text_field>"
  exit 1
fi

text_field=$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

echo "[INFO] Running SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "[INFO] Using chunk_dir = data/glossary_chunks/$text_field"

python3 term_embedding_pre_handle.py \
    --chunk_id=$SLURM_ARRAY_TASK_ID \
    --gpu_id=0 \
    --chunk_dir=data/glossary_chunks/$text_field \
    --output_dir=data/$text_field/text_embedding_cache_batch