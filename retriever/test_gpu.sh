#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err
#SBATCH --partition=taurus    # 看你服务器分区名称，改成你的
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# 加载conda环境（根据你自己的配置）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 执行python命令
python3 -c "import torch; print(torch.cuda.is_available())"