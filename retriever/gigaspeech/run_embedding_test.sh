#!/bin/bash

# 运行Qwen2-Audio embedding测试脚本

echo "=== Qwen2-Audio Embedding Test ==="

cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech

# 激活conda环境
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

echo "Environment activated"
echo "Current directory: $(pwd)"

# 运行快速测试
echo "Running quick embedding test..."
python3 quick_test_embedding.py

echo "Test completed!"


