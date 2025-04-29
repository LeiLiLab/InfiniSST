#!/bin/bash
#SBATCH --job-name=group_glossary
#SBATCH --output=logs/group_glossary_%j.out
#SBATCH --error=logs/group_glossary_%j.err
#SBATCH --partition=gemini
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

python group_glossary.py

#16351