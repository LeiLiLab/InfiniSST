#!/bin/bash
#SBATCH --job-name=build_glossary_index
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=96GB
#SBATCH --output=logs/build_glossary_index%j.out
#SBATCH --error=logs/build_glossary_index%j.err

if [ -z "$1" ]; then
  echo "[ERROR] Missing text_field argument. Usage: sbatch glossary_embedding.sh <text_field>"
  exit 1
fi

text_field=$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

python3 build_glossary_index.py --text_field=$text_field