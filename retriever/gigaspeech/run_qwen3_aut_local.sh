#!/bin/bash

# Qwen3-AuT本地训练脚本 (无需Slurm)
# 使用torchrun进行多GPU DDP训练

echo "=== Qwen3-AuT Term-Level Training (Local) ==="

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
SAVE_PATH="./models/qwen3_aut_term_level.pt"
SCRIPT_PATH="./modal/Qwen3_AuT_term_level_train_ddp.py"

# 创建模型保存目录
mkdir -p ./models

# 创建日志目录
mkdir -p ./logs

# 检查GPU数量
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "=== 检测到 $NUM_GPUS 个GPU ==="

# 启动训练
echo "=== Starting Training ==="

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    "$SCRIPT_PATH" \
    --train_samples_path "$TRAIN_SAMPLES" \
    --test_samples_path "$TEST_SAMPLES" \
    --mmap_shard_dir "$MMAP_DIR" \
    --save_path "$SAVE_PATH" \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --aut_model_name "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
    --text_model_name "Qwen/Qwen2-Audio-7B-Instruct" \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --audio_text_loss_ratio 0.3 \
    --audio_term_loss_ratio 0.7 \
    2>&1 | tee ./logs/qwen3_aut_training_$(date +%Y%m%d_%H%M%S).log

echo "=== Training Completed ==="
echo "模型保存位置: $SAVE_PATH"
echo "最佳模型保存位置: ${SAVE_PATH/.pt/_best.pt}"












