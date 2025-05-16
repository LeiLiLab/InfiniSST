#!/bin/bash
#SBATCH --job-name=split_glossary
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --output=logs/split_glossary%j.out
#SBATCH --error=logs/split_glossary%j.err

echo "[INFO] Step 2: split the glossary files into 4 chunk files to speed glossary embeddings"

text_field=$1

python3 glossary_chunk_handle.py --chunks=4 --text_field=$text_field
