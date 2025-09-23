#!/bin/bash

# SLURM脚本：并行处理MFA term-level chunk样本
# 为每个ground truth term生成单独的音频chunk
# 支持处理多个分片文件：term_preprocessed_samples_0_500000.json 到 term_preprocessed_samples_8000000_end.json

#SBATCH --job-name=mfa_term_chunks
#SBATCH --partition=taurus
#SBATCH --array=0-16%4
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --output=logs/mfa_term_chunks_%A_%a.out
#SBATCH --error=logs/mfa_term_chunks_%A_%a.err

# 参数说明:
# $1: 文件后缀模式 (默认"term_preprocessed_samples")
# $2: 输出音频目录 (默认"/mnt/gemini/data1/jiaxuanluo/term_chunks")

file_pattern=${1:-"term_preprocessed_samples"}
output_audio_dir=${2:-"/mnt/gemini/data1/jiaxuanluo/term_chunks"}

# 尝试多个conda路径
if [ -f "/mnt/taurus/home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh" ]; then
    source /mnt/taurus/home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
else
    # 直接使用conda可执行文件
    export PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/bin:$PATH"
fi
conda activate infinisst

# 确保日志目录存在
mkdir -p logs

# 根据SLURM_ARRAY_TASK_ID计算文件索引
start_idx=$((SLURM_ARRAY_TASK_ID * 500000))

if [ $SLURM_ARRAY_TASK_ID -eq 16 ]; then
    # 最后一个分片：8000000_end
    input_file="data/samples/xl_cleaned/${file_pattern}_${start_idx}_end.json"
    output_file="data/samples/xl_cleaned/term_level_chunks_${start_idx}_end.json"
else
    # 其他分片：0_500000, 500000_1000000, ..., 7500000_8000000
    end_idx=$((start_idx + 500000))
    input_file="data/samples/xl_cleaned/${file_pattern}_${start_idx}_${end_idx}.json"
    output_file="data/samples/xl_cleaned/term_level_chunks_${start_idx}_${end_idx}.json"
fi

echo "===== Task $SLURM_ARRAY_TASK_ID ====="
echo "Processing: $input_file"
echo "Output: $output_file"
echo "Audio output dir: $output_audio_dir"
echo "Start time: $(date)"

# 检查输入文件是否存在
if [ ! -f "$input_file" ]; then
    echo "ERROR: Input file not found: $input_file"
    exit 1
fi

# 运行term-level chunk处理
python3 handle_MFA_term_level_chunks.py \
    --input_json="$input_file" \
    --output_json="$output_file" \
    --textgrid_dir=/mnt/data/siqiouyang/datasets/gigaspeech/textgrids \
    --output_audio_dir="$output_audio_dir"

if [ $? -eq 0 ]; then
    echo "SUCCESS: Term-level chunk processing completed for task $SLURM_ARRAY_TASK_ID"
    echo "Output saved to: $output_file"
else
    echo "ERROR: Term-level chunk processing failed for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "End time: $(date)"
echo "============================="
