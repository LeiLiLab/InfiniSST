#!/bin/bash

#SBATCH --job-name=wiki_term
#SBATCH --output=logs/wiki/wiki_term_%j.out
#SBATCH --error=logs/wiki/wiki_term_%j.err
#SBATCH --partition=taurus
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# job id = wiki_term_17400.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

INPUT_FILE="/home/jiaxuanluo/InfiniSST/latest-truthy.nt.bz2"
OUTPUT_DIR="data/final_split_terms"
CHUNK_SIZE=5000000

python3 wiki_term.py --input "$INPUT_FILE" --output_dir "$OUTPUT_DIR" --chunk_size "$CHUNK_SIZE"
