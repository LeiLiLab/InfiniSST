#!/bin/bash

# 分任务执行，每个任务读取 split_tsv 文件夹下的一个分片 TSV，
# 并使用外层提供的命名实体 JSON 文件进行预处理。
#SBATCH --job-name=preprocess_split
#SBATCH --partition=taurus
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=logs/samples_split_test.out
#SBATCH --error=logs/samples_split_test.err



text_field=${1:-term}
ner_json=${2:-data/named_entities_train_xl_split_0.json}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

SPLIT_TSV="data/split_tsv/train_xl_split_0.tsv"

echo "[INFO] SLURM_ARRAY_TASK_ID=0 (fixed)"
echo "[INFO] Using text_field=${text_field}"
echo "[INFO] Reading TSV file: ${SPLIT_TSV}"
echo "[INFO] Using named entity file: ${ner_json}"

if [[ ! -f "$SPLIT_TSV" ]]; then
    echo "[ERROR] TSV split file not found: $SPLIT_TSV"
    exit 1
fi

if [[ ! -f "$ner_json" ]]; then
    echo "[ERROR] Named entity file not found: $ner_json"
    exit 1
fi

PYTHONUNBUFFERED=1 python3 train_samples_pre_handle.py \
    --tsv_path="${SPLIT_TSV}" \
    --split_id=0 \
    --text_field=${text_field} \
    --ner_json="${ner_json}"

echo "[INFO] Task 0 (fixed) completed successfully"