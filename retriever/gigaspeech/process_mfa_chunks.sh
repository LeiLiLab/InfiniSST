#!/bin/bash

#SBATCH --job-name=mfa_chunks
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=logs/mfa_chunks_%j.out
#SBATCH --error=logs/mfa_chunks_%j.err

# 参数说明:
# $1: 输入JSON文件路径
# $2: 输出JSON文件路径
# $3: chunk数量 (默认3)
# $4: chunk长度 (默认0.96秒)

input_json=${1:-"data/samples/xl/term_preprocessed_samples_0_500000.json"}
output_json=${2:-"data/samples/xl/mfa_3chunks_samples_0_500000.json"}
n_chunks=${3:-3}
chunk_len=${4:-0.96}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

echo "[INFO] Processing MFA chunks with parameters:"
echo "  Input: $input_json"
echo "  Output: $output_json"
echo "  N chunks: $n_chunks"
echo "  Chunk length: $chunk_len seconds"

# 确保输出目录存在
mkdir -p $(dirname "$output_json")
mkdir -p logs

PYTHONUNBUFFERED=1 python3 handle_MFA_n_chunk_samples.py \
    --input_json="$input_json" \
    --output_json="$output_json" \
    --n=$n_chunks \
    --chunk_len=$chunk_len \
    --textgrid_dir="/mnt/data/siqiouyang/datasets/gigaspeech/textgrids"

echo "[INFO] MFA chunk processing completed" 