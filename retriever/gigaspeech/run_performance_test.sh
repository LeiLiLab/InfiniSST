#!/bin/bash

# 运行性能测试脚本

echo "=== Qwen2-Audio 性能测试 ==="

cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech

# 激活conda环境
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

echo "Environment activated"
echo "Current directory: $(pwd)"

# 显示GPU信息
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
fi

# 显示CPU信息
echo "CPU信息:"
echo "CPU核心数: $(nproc)"
echo "内存信息:"
free -h
echo ""

# 运行性能测试
echo "开始性能测试..."
python3 test_performance.py

echo "性能测试完成!"
