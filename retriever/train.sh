#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=96GB
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err


name=$1  # ✅ 从命令行获取传入的第一个参数
text_field=$2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
PYTHONUNBUFFERED=1

echo "[INFO] Using dataset: $name"
python3 SONAR_train.py --samples_path="data/${name}_preprocessed_samples_merged.json" --text_field=$text_field

#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate infinisst
#PYTHONUNBUFFERED=1
#
#python3 SONAR_train.py --samples_path="data/preprocessed_samples_merged.json" --text_field="short_description"