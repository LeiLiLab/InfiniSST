#!/bin/bash

# 总量 8282989，按 1000000 一份切分，共 9 个任务
# 根据总数切分

#SBATCH --job-name=extract_ner_cache
#SBATCH --partition=taurus
#SBATCH --array=0-8%3
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/extract_ner_cache_%A_%a.out
#SBATCH --error=logs/extract_ner_cache_%A_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate spaCyEnv


# Preprocessing: split train_xl.tsv into per-slice files, only if needed
SPLIT_DIR="data/split_tsv"
mkdir -p "$SPLIT_DIR"

INPUT_TSV="/mnt/data/siqiouyang/datasets/gigaspeech/manifests/train_xl.tsv"
SPLIT_TSV="$SPLIT_DIR/train_xl_split_${SLURM_ARRAY_TASK_ID}.tsv"

if [[ ! -f "$SPLIT_TSV" ]]; then
    echo "[INFO] Generating split TSV for task ID $SLURM_ARRAY_TASK_ID..."

    TOTAL_LINES=$(wc -l < "$INPUT_TSV")
    START_LINE=$((SLURM_ARRAY_TASK_ID * 1000000 + 1))  # +1 to skip header
    END_LINE=$((START_LINE + 1000000 - 1))
    if (( END_LINE > TOTAL_LINES )); then
        END_LINE=$TOTAL_LINES
    fi

    # Retain header, then slice the appropriate lines
    (head -n 1 "$INPUT_TSV" && sed -n "${START_LINE},${END_LINE}p" "$INPUT_TSV") > "$SPLIT_TSV"
else
    echo "[INFO] Split TSV for task ID $SLURM_ARRAY_TASK_ID already exists."
fi

PYTHONUNBUFFERED=1 python3 extract_ner_cache.py --tsv_path=${SPLIT_TSV}