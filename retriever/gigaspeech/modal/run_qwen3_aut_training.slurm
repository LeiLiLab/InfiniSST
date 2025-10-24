#!/bin/bash
#SBATCH --job-name=qwen3_aut_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:2
#SBATCH --partition=taurus
#SBATCH --mem=512G
#SBATCH --output=logs/qwen3_aut_training_%j.out
#SBATCH --error=logs/qwen3_aut_training_%j.err

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# 创建日志目录
mkdir -p logs

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 设置CUDA环境变量
TORCH_LIB_PATH=$(python -c "import torch; print(torch.__file__.replace('__init__.py', 'lib'))" 2>/dev/null)
TRITON_LIB_PATH=$(python -c "import torch; print(torch.__file__.replace('torch/__init__.py', 'triton/backends/nvidia/lib'))" 2>/dev/null)
export LD_LIBRARY_PATH="${TORCH_LIB_PATH}:${TRITON_LIB_PATH}:${LD_LIBRARY_PATH}"

# 设置HuggingFace缓存目录
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface"

# 设置DDP环境变量
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
export WORLD_SIZE=8

# 设置NCCL环境变量
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0

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
SAVE_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/models/qwen3_aut_term_level.pt"

# 训练脚本路径
SCRIPT_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/Qwen3_AuT_term_level_train_ddp.py"

# 检查文件是否存在
echo "=== Checking Files ==="
[ -f "$TRAIN_SAMPLES" ] && echo "✓ Train samples found" || echo "✗ Train samples NOT found: $TRAIN_SAMPLES"
[ -f "$TEST_SAMPLES" ] && echo "✓ Test samples found" || echo "✗ Test samples NOT found: $TEST_SAMPLES"
[ -d "$MMAP_DIR" ] && echo "✓ MMap directory found" || echo "✗ MMap directory NOT found: $MMAP_DIR"
[ -f "$SCRIPT_PATH" ] && echo "✓ Training script found" || echo "✗ Training script NOT found: $SCRIPT_PATH"

# 创建模型保存目录
mkdir -p $(dirname "$SAVE_PATH")

echo "=== Starting Training ==="

# 使用torchrun进行多GPU训练
torchrun \
    --nproc_per_node=8 \
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
    --audio_term_loss_ratio 0.7

echo "=== Training Completed ==="
echo "End time: $(date)"
echo "Best model saved to: ${SAVE_PATH/.pt/_best.pt}"

