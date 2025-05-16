#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=96GB
#SBATCH --output=logs/evalute%j.out
#SBATCH --error=logs/evalute%j.err



if [ -z "$1" ]; then
  echo "[ERROR] Missing text_field argument. Usage: sbatch glossary_embedding.sh <text_field>"
  exit 1
fi

text_field=$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

python3 new_retrieve.py --text_field=$text_field