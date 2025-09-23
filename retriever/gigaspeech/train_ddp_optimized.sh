#!/bin/bash

# DDP优化训练脚本 - 专为8个GPU优化
# 使用方法: ./train_ddp_optimized.sh

# ===== 环境变量优化 =====
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# NCCL优化配置
export NCCL_DEBUG=WARN  # 减少调试信息
export NCCL_IB_DISABLE=1  # 禁用InfiniBand
export NCCL_P2P_DISABLE=0  # 启用P2P通信以提高性能
export NCCL_TREE_THRESHOLD=0  # 强制使用tree算法
export NCCL_SOCKET_IFNAME=lo  # 使用本地接口

# PyTorch优化
export OMP_NUM_THREADS=4  # 每个进程使用4个CPU线程
export TORCH_CUDNN_V8_API_ENABLED=1  # 启用CuDNN v8 API
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"  # 支持的CUDA架构

# 内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 限制内存分割大小

# ===== 训练参数优化 =====
TRAIN_SAMPLES_PATH="data/xl_cleaned_term_level_chunks_merged.json"
TEST_SAMPLES_PATH="data/samples/test_cleaned/term_preprocessed_samples_test.json"
EPOCHS=20
BATCH_SIZE=4096  # 8个GPU总batch size (每GPU 512)
LR=5e-5
SAVE_PATH="data/clap_sonar_term_level_full_ddp_optimized.pt"
BEST_MODEL_PATH="data/full_dataset_sonar_term_level_best.pt"
AUDIO_TEXT_LOSS_RATIO=0.3
AUDIO_TERM_LOSS_RATIO=0.7
GLOSSARY_PATH="data/terms/glossary_merged.json"
UNFREEZE_LAYERS=10
GPU_IDS="0,1,2,3,4,5,6,7"  # 使用全部8个GPU

# 日志文件
LOG_FILE="sonar_train_ddp_optimized_$(date +%Y%m%d_%H%M%S).log"

echo "=== DDP Optimized Training Configuration ==="
echo "GPU IDs: $GPU_IDS"
echo "Total Batch Size: $BATCH_SIZE"
echo "Per-GPU Batch Size: $((BATCH_SIZE / 8))"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "OMP Threads per process: $OMP_NUM_THREADS"
echo "Log File: $LOG_FILE"
echo "============================================="

# 检查GPU状态
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "Starting DDP training..."

# 启动优化的DDP训练
python3 SONAR_term_level_train_glossary_ddp.py \
    --train_samples_path=$TRAIN_SAMPLES_PATH \
    --test_samples_path=$TEST_SAMPLES_PATH \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --lr=$LR \
    --save_path=$SAVE_PATH \
    --best_model_path=$BEST_MODEL_PATH \
    --audio_text_loss_ratio=$AUDIO_TEXT_LOSS_RATIO \
    --audio_term_loss_ratio=$AUDIO_TERM_LOSS_RATIO \
    --glossary_path=$GLOSSARY_PATH \
    --unfreeze_layers=$UNFREEZE_LAYERS \
    --filter_no_term \
    --gpu_ids=$GPU_IDS \
    > $LOG_FILE 2>&1 &

# 获取进程ID
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Log file: $LOG_FILE"
echo ""

# 实时监控脚本
echo "Monitoring training progress..."
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo ""

# 监控循环
while kill -0 $TRAIN_PID 2>/dev/null; do
    echo "=== $(date) ==="
    echo "Training PID: $TRAIN_PID (running)"
    
    # 显示GPU使用情况
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F, '{printf "GPU %s: %s%% util, %s/%s MB mem, %s°C\n", $1, $2, $3, $4, $5}'
    
    # 显示最新的训练日志（最后几行）
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "Latest training logs:"
        tail -n 3 "$LOG_FILE" | grep -E "\[INFO\]|\[EVAL\]" | tail -n 2
    fi
    
    echo "----------------------------------------"
    sleep 60  # 每分钟更新一次
done

echo ""
echo "Training completed!"
echo "Final log file: $LOG_FILE"

# 显示最终结果
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "Final training results:"
    grep -E "Best Recall@10|Training completed" "$LOG_FILE" | tail -n 2
fi
