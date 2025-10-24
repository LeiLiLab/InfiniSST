#!/bin/bash

# Qwen3-Omni 30B Sharded Training for 4x A6000
# 单进程 + device_map="auto" 自动分片到多卡
# 梯度检查点 + 梯度累积 + 4bit量化

echo "=== Qwen3-Omni Sharded Training (4x A6000) ==="

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 设置CUDA环境变量
TORCH_LIB_PATH=$(python -c "import torch; print(torch.__file__.replace('__init__.py', 'lib'))" 2>/dev/null)
TRITON_LIB_PATH=$(python -c "import torch; print(torch.__file__.replace('torch/__init__.py', 'triton/backends/nvidia/lib'))" 2>/dev/null)
export LD_LIBRARY_PATH="${TORCH_LIB_PATH}:${TRITON_LIB_PATH}:${LD_LIBRARY_PATH}"

# 设置HuggingFace缓存
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface"

# CUDA内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# 指定使用的GPU（4张A6000）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# AuT模型加载配置（会被脚本内部设置，这里备份说明）
# export AUT_DEVICE_MAP=auto           # 自动分片
# export AUT_LOAD_IN_4BIT=1            # 4bit量化
# export AUT_MAX_MEMORY=46GiB          # 每卡上限
# export AUT_NO_FLASH_ATTENTION=1      # 禁用FA2
# export AUT_DTYPE=bfloat16            # 数据类型

# 验证CUDA
echo "=== CUDA Status ==="
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)')
"

# 数据路径配置
TRAIN_SAMPLES="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_train_set.json"
TEST_SAMPLES="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_test_set.json"
MMAP_DIR="/mnt/gemini/data1/jiaxuanluo/mmap_shards"
SAVE_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen3_aut_sharded.pt"
OFFLOAD_DIR="/mnt/data2/jiaxuanluo/offload_qwen3"  # SSD路径用于CPU溢出
SCRIPT_PATH="./modal/Qwen3_AuT_train_sharded.py"

# 创建目录
mkdir -p /mnt/gemini/data2/jiaxuanluo/models
mkdir -p ./logs
mkdir -p "$OFFLOAD_DIR"

# 训练参数
EPOCHS=20
BATCH_SIZE=8              # 小batch避免OOM
GRAD_ACCUM_STEPS=16       # 累积16步 = 有效batch 128
LR=1e-4

echo "=== Training Configuration ==="
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Physical batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM_STEPS"
echo "Learning rate: $LR"
echo "Epochs: $EPOCHS"
echo "Offload folder: $OFFLOAD_DIR"

# 启动训练（单进程，模型自动分片）
echo "=== Starting Sharded Training ==="

python "$SCRIPT_PATH" \
    --train_samples_path "$TRAIN_SAMPLES" \
    --test_samples_path "$TEST_SAMPLES" \
    --mmap_shard_dir "$MMAP_DIR" \
    --save_path "$SAVE_PATH" \
    --offload_folder "$OFFLOAD_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --lr $LR \
    --aut_model_name "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
    --text_model_name "Qwen/Qwen2-Audio-7B-Instruct" \
    --enable_speech_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --audio_text_loss_ratio 0.3 \
    --audio_term_loss_ratio 0.7 \
    2>&1 | tee ./logs/qwen3_sharded_$(date +%Y%m%d_%H%M%S).log

echo "=== Training Completed ==="
echo "模型保存位置: $SAVE_PATH"
echo "最佳模型: ${SAVE_PATH/.pt/_best.pt}"

