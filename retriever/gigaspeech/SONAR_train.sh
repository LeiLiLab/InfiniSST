#!/bin/bash
#SBATCH --job-name=train_mfa_chunks
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --output=logs/train_mfa_%j.out
#SBATCH --error=logs/train_mfa_%j.err

# 参数说明:
# $1: 训练样本路径 (可选，默认使用测试样本)
# $2: epochs数量 (可选，默认30)
# $3: batch_size (可选，默认32)

train_samples_path=${1:-"data/samples/xl/test_mfa_3chunks_samples_0_500000.json"}
epochs=${2:-30}
batch_size=${3:-32}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
PYTHONUNBUFFERED=1

echo "[INFO] Training MFA chunk model with:"
echo "[INFO] Train samples: $train_samples_path (99% train, 1% test split)"
echo "[INFO] Epochs: $epochs"
echo "[INFO] Batch size: $batch_size"

# 检查训练样本文件是否存在
if [[ ! -f "$train_samples_path" ]]; then
    echo "[ERROR] Training samples file not found: $train_samples_path"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 执行训练
python3 SONAR_train.py \
    --train_samples_path="$train_samples_path" \
    --epochs=$epochs \
    --batch_size=$batch_size \
    --lr=1e-4 \
    --patience=5 \
    --save_path="data/clap_mfa_chunks.pt"

echo "[INFO] Training completed!"