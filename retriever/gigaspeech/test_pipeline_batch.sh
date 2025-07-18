#!/bin/bash

# 批量运行测试流水线，支持多个n值
# 使用方法: bash test_pipeline_batch.sh [n1,n2,n3...] [text_field]
# 例如: bash test_pipeline_batch.sh "1,3,5" term

# 解析参数
n_values=${1:-"1,2,3,5"}  # 默认测试n=1,2,3,5
text_field=${2:-term}   # 默认使用term字段

# 将n_values字符串转换为数组
IFS=',' read -ra N_ARRAY <<< "$n_values"

echo "[INFO] Starting batch test pipeline"
echo "[INFO] N values: ${n_values}"
echo "[INFO] Text field: ${text_field}"
echo ""

# 为每个n值运行流水线
for n in "${N_ARRAY[@]}"; do
    echo "[INFO] ===== Starting pipeline for n=${n} ====="
    
    # 运行测试流水线
    bash test_samples_generate_pipeline.sh "$n" "$text_field"
    
    echo "[INFO] Pipeline for n=${n} submitted"
    echo ""
    
    # 等待一小段时间避免同时提交太多任务
    sleep 5
done

echo "[INFO] All test pipelines submitted successfully!"
echo ""
echo "Monitor all jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  ls -la logs/test_pipeline_n*" 