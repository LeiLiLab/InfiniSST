#!/bin/bash
#SBATCH --job-name=sonar_ddp_fixed_2
#SBATCH --partition=taurus           # 如需改分区，改这里
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # 你的脚本是单进程驱动（内部自行多卡/多进程）
#SBATCH --cpus-per-task=1           # 给 DataLoader / OMP 用
#SBATCH --gres=gpu:3                 # 申请 3 张 GPU（与脚本里的 GPU_IDS 对应）
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


echo "训练结束，但保持任务占用..."
while true; do
    date
    sleep 600   # 每 10 分钟输出一次时间
done