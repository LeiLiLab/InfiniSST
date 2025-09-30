#!/bin/bash

# DDP训练脚本 - 使用平衡测试集
# 使用方法: ./train_balanced_ddp.sh

echo "=== DDP Training with Balanced Test Set ==="

# ===== CUDA环境设置 =====
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ===== fairseq2缓存设置 =====
export FAIRSEQ2_CACHE_DIR=/mnt/data2/jiaxuanluo/.cache/fairseq2
export HF_HOME=/mnt/data2/jiaxuanluo/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/data2/jiaxuanluo/.cache/huggingface/transformers

# 确保缓存目录存在
mkdir -p $FAIRSEQ2_CACHE_DIR
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE

# ===== NCCL配置优化 =====
export NCCL_DEBUG=WARN  # 降低日志噪音
export NCCL_IB_DISABLE=1  # 禁用InfiniBand
export NCCL_P2P_DISABLE=0  # 启用P2P，提升单机多卡带宽
export NCCL_SHM_DISABLE=0  # 启用共享内存
export NCCL_SOCKET_IFNAME=lo  # 使用loopback接口
export NCCL_TIMEOUT=1800  # 保持长超时时间

# ===== PyTorch优化 =====
export OMP_NUM_THREADS=4
export TORCH_CUDNN_V8_API_ENABLED=1

# ===== 检查平衡数据集是否存在 =====
BALANCED_TRAIN_PATH="data/balanced_train_set.json"
BALANCED_TEST_PATH="data/balanced_test_set.json"

if [ ! -f "$BALANCED_TRAIN_PATH" ] || [ ! -f "$BALANCED_TEST_PATH" ]; then
    echo "ERROR: Balanced datasets not found!"
    echo "Expected files:"
    echo "  - $BALANCED_TRAIN_PATH"
    echo "  - $BALANCED_TEST_PATH"
    echo ""
    echo "Please run the extraction script first:"
    echo "  ./run_extract_test_set.sh"
    echo ""
    exit 1
fi

# ===== 训练参数 =====
TRAIN_SAMPLES_PATH="$BALANCED_TRAIN_PATH"
TEST_SAMPLES_PATH="$BALANCED_TEST_PATH"
EPOCHS=20
BATCH_SIZE=2048
LR=5e-5
SAVE_PATH="data/clap_sonar_balanced_ddp.pt"
BEST_MODEL_PATH="data/full_dataset_sonar_term_level_best.pt"  # 可选：从之前的模型继续训练
AUDIO_TEXT_LOSS_RATIO=0.3
AUDIO_TERM_LOSS_RATIO=0.7
GLOSSARY_PATH="data/terms/glossary_merged.json"
UNFREEZE_LAYERS=10
GPU_IDS="0,1,2,3,4"

# 日志文件
LOG_FILE="sonar_balanced_train_ddp_$(date +%Y%m%d_%H%M%S).log"

echo "=== 数据集信息 ==="
echo "Training set: $TRAIN_SAMPLES_PATH"
echo "Test set: $TEST_SAMPLES_PATH"

# 显示数据集统计
if command -v python3 &> /dev/null; then
    echo "Dataset statistics:"
    python3 -c "
import json
try:
    with open('$TRAIN_SAMPLES_PATH') as f:
        train_data = json.load(f)
    print(f'  Training samples: {len(train_data)}')
    
    with open('$TEST_SAMPLES_PATH') as f:
        test_data = json.load(f)
    print(f'  Test samples: {len(test_data)}')
    
    # 显示术语信息
    terms_info_path = '$TEST_SAMPLES_PATH'.replace('.json', '_terms_info.json')
    try:
        with open(terms_info_path) as f:
            terms_info = json.load(f)
            stats = terms_info['stats']
            print(f'  Test unseen terms: {stats[\"unseen_terms_count\"]} ({stats[\"unseen_ratio\"]:.1%})')
            print(f'  Test seen terms: {stats[\"seen_terms_count\"]}')
    except:
        print('  Terms info not available')
        
except Exception as e:
    print(f'  Error reading dataset info: {e}')
"
fi

echo ""
echo "=== 环境检查 ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version: $(/usr/local/cuda/bin/nvcc --version | grep release)"
echo "GPU IDs: $GPU_IDS"
echo "Total Batch Size: $BATCH_SIZE"
echo "Per-GPU Batch Size: $((BATCH_SIZE / 5))"

# 激活conda环境
source /home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
CUDA_VISIBLE_DEVICES=$GPU_IDS
CUDA_DEVICE_ORDER="PCI_BUS_ID"

echo "=== PyTorch CUDA检查 ==="
python3 -c "
import os
# 确保在导入torch之前设置CUDA环境
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(min(8, torch.cuda.device_count())):
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
echo "=== 启动DDP训练（平衡数据集）==="
echo "开始时间: $(date)"
echo "日志文件: $LOG_FILE"
echo ""

# 启动训练 - 使用独立的测试集
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
    2>&1 | tee $LOG_FILE &

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
echo "=== 训练完成 ==="
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
    
    echo ""
    echo "=== Unseen Terms Recall ==="
    echo "Seen/Unseen recall statistics:"
    grep -E "Seen Recall@|Unseen Recall@|Unseen Term Percentage" "$LOG_FILE" | tail -n 6
fi

