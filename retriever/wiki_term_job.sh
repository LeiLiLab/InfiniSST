#!/bin/bash

#SBATCH --job-name=wiki_term
#SBATCH --output=logs/wiki_term_%j.out
#SBATCH --error=logs/wiki_term_%j.err
#SBATCH --time=144:00:00
#SBATCH --partition=taurus
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

INPUT_FILE="/home/jiaxuanluo/InfiniSST/latest-truthy.nt.bz2"
OUTPUT_DIR="split_terms"
CHUNK_SIZE=10000

MAX_JOBS=200   # 预估总量，切200个子任务，每个子任务处理大约5000万行

START=0
JOB_IDX=0
LINES_PER_JOB=50000000  # 每个子任务 5000万行

while [ $JOB_IDX -lt $MAX_JOBS ]; do
    END=$((START + LINES_PER_JOB))
    sbatch --job-name=wiki_term_${JOB_IDX} --cpus-per-task=8 --mem=32G --partition=taurus --output=logs/wiki_term_${JOB_IDX}.out --error=logs/wiki_term_${JOB_IDX}.err --wrap="bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate infinisst && python3 wiki_term.py --input $INPUT_FILE --output_dir $OUTPUT_DIR --chunk_size $CHUNK_SIZE --start $START --end $END'"
    START=$END
    JOB_IDX=$((JOB_IDX + 1))
done
