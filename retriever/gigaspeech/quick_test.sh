#!/bin/bash

# 快速测试脚本 - 避免与主训练冲突
# 使用方法: ./quick_test.sh

echo "=== Quick Test Script ==="
echo "开始时间: $(date)"
echo ""

# ===== CUDA环境设置 =====
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ===== NCCL配置优化 =====
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_TIMEOUT=1800

# ===== PyTorch优化 =====
export OMP_NUM_THREADS=4
export TORCH_CUDNN_V8_API_ENABLED=1

# ===== 快速测试参数 =====
TRAIN_SAMPLES_PATH="data/samples/xl_cleaned/term_level_chunks_0_500000.json"
TEST_SAMPLES_PATH="data/samples/xl_cleaned/term_level_chunks_500000_1000000.json"
EPOCHS=2  # 只训练2个epoch进行快速测试
BATCH_SIZE=512  # 较小的batch size
LR=5e-5
SAVE_PATH="data/quick_test_model.pt"
BEST_MODEL_PATH=""  # 不使用预训练模型
AUDIO_TEXT_LOSS_RATIO=0.3
AUDIO_TERM_LOSS_RATIO=0.7
GLOSSARY_PATH="data/terms/glossary_merged.json"
UNFREEZE_LAYERS=10
GPU_IDS="0,2"  # 使用不同的GPU避免冲突
MIN_UNSEEN_RATIO=0.20
FORCE_UNSEEN_RATIO=true

# 日志文件
LOG_FILE="quick_test_$(date +%Y%m%d_%H%M%S).log"

echo "=== 环境检查 ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "GPU IDs: $GPU_IDS"
echo "Total Batch Size: $BATCH_SIZE"
echo "Per-GPU Batch Size: $((BATCH_SIZE / 2))"
echo "Epochs: $EPOCHS (quick test)"

# 激活conda环境
source /home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

echo "=== PyTorch CUDA检查 ==="
python3 -c "
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(min(2, torch.cuda.device_count())):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "CUDA检查失败，退出"
    exit 1
fi

echo ""
echo "=== GPU状态 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "=== 启动快速测试 ==="
echo "开始时间: $(date)"
echo ""

# 启动训练
python3 SONAR_term_level_train_glossary_ddp.py \
    --train_samples_path=$TRAIN_SAMPLES_PATH \
    --test_samples_path=$TEST_SAMPLES_PATH \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --lr=$LR \
    --save_path=$SAVE_PATH \
    --audio_text_loss_ratio=$AUDIO_TEXT_LOSS_RATIO \
    --audio_term_loss_ratio=$AUDIO_TERM_LOSS_RATIO \
    --glossary_path=$GLOSSARY_PATH \
    --unfreeze_layers=$UNFREEZE_LAYERS \
    --filter_no_term \
    --gpu_ids=$GPU_IDS \
    --min_unseen_ratio=$MIN_UNSEEN_RATIO \
    --force_unseen_ratio \
    2>&1 | tee $LOG_FILE

echo ""
echo "=== 快速测试完成 ==="
echo "结束时间: $(date)"
echo "日志文件: $LOG_FILE"

# 显示最终结果
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "最终结果:"
    grep -E "Best Recall@10|Training completed|ERROR|Exception" "$LOG_FILE" | tail -n 5
    
    echo ""
    echo "=== 时间统计 ==="
    echo "初始化时间:"
    grep "Initialization completed" "$LOG_FILE" | tail -n 1
    echo "训练时间统计:"
    grep "Epoch.*completed in" "$LOG_FILE" | tail -n 3
    echo "总训练时间:"
    grep "Training completed in" "$LOG_FILE" | tail -n 1
fi

