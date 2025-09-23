#!/bin/bash

# Qwen2-Audio DDP修复版训练脚本 - 解决CUDA环境问题
# 使用方法: ./qwen2_audio_train_ddp_fixed.sh

echo "=== Qwen2-Audio DDP Fixed Training Script ==="

# ===== CUDA环境设置 =====
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID

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

# ===== 训练参数 =====
TRAIN_SAMPLES_PATH="data/xl_term_level_chunks_merged.json"
TEST_SAMPLES_PATH="data/samples/xl/term_level_chunks_500000_1000000.json"
EPOCHS=20
BATCH_SIZE=128  # Qwen2-Audio 7B需要较小的batch size
LR=1e-4
SAVE_PATH="data/qwen2_audio_term_level_full_ddp_fixed.pt"
BEST_MODEL_PATH="data/qwen2_audio_term_level_best.pt"
AUDIO_TEXT_LOSS_RATIO=0.3
AUDIO_TERM_LOSS_RATIO=0.7
GLOSSARY_PATH="data/terms/glossary_filtered.json"
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
GPU_IDS="0,1,2,3,4,5,6,7"

# LoRA参数
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Hard negative mining参数
ENABLE_HARD_NEG=true
HARD_NEG_SOURCE="used"  # 使用训练数据中的术语作为hard negative源
HARD_NEG_K=10
HARD_NEG_WEIGHT=0.2
HARD_NEG_MARGIN=0.1
HARD_NEG_CANDIDATES=100

# No-term相关参数
ENABLE_NO_TERM=false
FILTER_NO_TERM=true
USE_NO_TERM_LOSS=false
NO_TERM_MARGIN=0.15
LAMBDA_NO_TERM=0.5
NO_TERM_TOP_M=100

# 日志文件
LOG_FILE="qwen2_audio_train_ddp_fixed_$(date +%Y%m%d_%H%M%S).log"

echo "=== 环境检查 ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version: $(/usr/local/cuda/bin/nvcc --version | grep release)"
echo "GPU IDs: $GPU_IDS"
echo "Total Batch Size: $BATCH_SIZE"
echo "Per-GPU Batch Size: $((BATCH_SIZE / 8))"
echo "Model: $MODEL_NAME"
echo "LoRA Config: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"

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
        props = torch.cuda.get_device_properties(i)
        print(f'  Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'  Compute Capability: {props.major}.{props.minor}')
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
echo "=== 启动Qwen2-Audio DDP训练 ==="

# 构建训练命令
TRAIN_CMD="python3 Qwen2_Audio_term_level_train_ddp.py \
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
    --model_name=$MODEL_NAME \
    --lora_r=$LORA_R \
    --lora_alpha=$LORA_ALPHA \
    --lora_dropout=$LORA_DROPOUT \
    --gpu_ids=$GPU_IDS"

# 添加hard negative mining参数
if [ "$ENABLE_HARD_NEG" = "true" ]; then
    TRAIN_CMD+=" --enable_hard_neg"
    TRAIN_CMD+=" --hard_neg_source=$HARD_NEG_SOURCE"
    TRAIN_CMD+=" --hard_neg_k=$HARD_NEG_K"
    TRAIN_CMD+=" --hard_neg_weight=$HARD_NEG_WEIGHT"
    TRAIN_CMD+=" --hard_neg_margin=$HARD_NEG_MARGIN"
    TRAIN_CMD+=" --hard_neg_candidates=$HARD_NEG_CANDIDATES"
fi

# 添加no-term参数
if [ "$ENABLE_NO_TERM" = "true" ]; then
    TRAIN_CMD+=" --enable_no_term"
fi

if [ "$FILTER_NO_TERM" = "true" ]; then
    TRAIN_CMD+=" --filter_no_term"
fi

if [ "$USE_NO_TERM_LOSS" = "true" ]; then
    TRAIN_CMD+=" --use_no_term_loss"
    TRAIN_CMD+=" --no_term_margin=$NO_TERM_MARGIN"
    TRAIN_CMD+=" --lambda_no_term=$LAMBDA_NO_TERM"
    TRAIN_CMD+=" --no_term_top_m=$NO_TERM_TOP_M"
fi

echo "执行命令: $TRAIN_CMD"
echo ""

# 启动训练
eval "$TRAIN_CMD" 2>&1 | tee $LOG_FILE

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
echo "日志文件: $LOG_FILE"

# 显示最终结果
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "最终结果:"
    grep -E "Best Recall@10|Training completed|ERROR|Exception" "$LOG_FILE" | tail -n 5
fi

# 显示模型文件
echo ""
echo "生成的模型文件:"
ls -la data/qwen2_audio_term_level*.pt 2>/dev/null || echo "未找到模型文件"
