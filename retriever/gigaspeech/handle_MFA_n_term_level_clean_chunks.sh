#!/bin/bash

# SLURM脚本：并行处理MFA chunk样本
# 支持处理多个分片文件：term_preprocessed_samples_0_500000.json 到 term_preprocessed_samples_8000000_end.json

#SBATCH --job-name=mfa_chunks
#SBATCH --partition=taurus
#SBATCH --array=0-16%4
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --output=logs/mfa_chunks_%A_%a.out
#SBATCH --error=logs/mfa_chunks_%A_%a.err

# 参数说明:
# $1: chunk数量 (默认3)
# $2: chunk长度 (默认0.96秒)
# $3: 文件后缀模式 (默认"term_preprocessed_samples")

n_chunks=${1:-3}
chunk_len=${2:-0.96}
file_pattern=${3:-"term_preprocessed_samples"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 确保日志目录存在
mkdir -p logs

# 根据SLURM_ARRAY_TASK_ID计算文件索引
start_idx=$((SLURM_ARRAY_TASK_ID * 500000))

# 构建输入输出文件路径
if [ $SLURM_ARRAY_TASK_ID -eq 16 ]; then
    # 最后一个任务处理剩余数据
    input_json="data/samples/xl/${file_pattern}_${start_idx}_end.json"
    output_json="data/samples/xl/mfa_${n_chunks}chunks_samples_${start_idx}_end.json"
else
    end_idx=$((start_idx + 500000))
    input_json="data/samples/xl/${file_pattern}_${start_idx}_${end_idx}.json"
    output_json="data/samples/xl/mfa_${n_chunks}chunks_samples_${start_idx}_${end_idx}.json"
fi

echo "[INFO] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] Processing MFA chunks with parameters:"
echo "  Input: $input_json"
echo "  Output: $output_json"
echo "  N chunks: $n_chunks"
echo "  Chunk length: $chunk_len seconds"

# 检查输入文件是否存在
if [[ ! -f "$input_json" ]]; then
    echo "[ERROR] Input file not found: $input_json"
    exit 1
fi

# 确保输出目录存在
mkdir -p $(dirname "$output_json")
mkdir -p /mnt/gemini/data1/jiaxuanluo/audio_chunks

# 运行MFA chunk处理
PYTHONUNBUFFERED=1 python3 handle_MFA_n_term_level_clean_chunks.py \
    --input_json="$input_json" \
    --output_json="$output_json" \
    --n=$n_chunks \
    --chunk_len=$chunk_len \
    --textgrid_dir="/mnt/data/siqiouyang/datasets/gigaspeech/textgrids"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "[INFO] Task ${SLURM_ARRAY_TASK_ID} completed successfully"
    echo "[INFO] Output saved to: $output_json"
else
    echo "[ERROR] Task ${SLURM_ARRAY_TASK_ID} failed with exit code $exit_code"
    exit $exit_code
fi
