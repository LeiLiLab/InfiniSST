#!/bin/bash

# 单独测试MFA chunk处理功能，只处理第一个分片文件
# 用于调试和验证处理流程

#SBATCH --job-name=test_mfa_chunks
#SBATCH --partition=taurus
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=logs/test_mfa_chunks.out
#SBATCH --error=logs/test_mfa_chunks.err

# 参数设置
n_chunks=${1:-3}
chunk_len=${2:-0.96}
file_pattern=${3:-"term_preprocessed_samples"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 确保日志目录存在
mkdir -p logs

# 只处理第一个分片文件
input_json="data/samples/xl/${file_pattern}_0_500000.json"
output_json="data/samples/xl/test_mfa_${n_chunks}chunks_samples_0_500000.json"

echo "[INFO] Testing MFA chunk processing with parameters:"
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

echo "[INFO] Starting MFA chunk processing..."

# 运行MFA chunk处理，增加详细输出
PYTHONUNBUFFERED=1 python3 handle_MFA_n_chunk_samples.py \
    --input_json="$input_json" \
    --output_json="$output_json" \
    --n=$n_chunks \
    --chunk_len=$chunk_len \
    --textgrid_dir="/mnt/data/siqiouyang/datasets/gigaspeech/textgrids"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "[INFO] Test completed successfully"
    echo "[INFO] Output saved to: $output_json"
    
    # 显示处理结果统计
    if [[ -f "$output_json" ]]; then
        result_count=$(python3 -c "import json; print(len(json.load(open('$output_json'))))")
        input_count=$(python3 -c "import json; print(len(json.load(open('$input_json'))))")
        echo "[INFO] Processing results:"
        echo "  Input samples: $input_count"
        echo "  Output samples: $result_count"
        echo "  Success rate: $(python3 -c "print(f'{$result_count/$input_count*100:.2f}%')")"
    fi
else
    echo "[ERROR] Test failed with exit code $exit_code"
    exit $exit_code
fi 