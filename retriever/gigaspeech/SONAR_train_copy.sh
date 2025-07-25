#!/bin/bash

# SONAR训练流水线
# 参数: $1 = n (chunk数量), $2 = text_field (可选，默认为term), $3 = single_slice (可选，用于快速验证)

# 设置参数
n=${1:-3}  # 默认n=3
text_field=${2:-term}  # 默认使用term字段
single_slice=${3:-false}  # 默认使用完整数据集

final_samples="data/xl_mfa_${n}chunks_samples_merged.json"
model_save_path="data/clap_sonar_full_n${n}.pt"

# 根据模式设置模型保存路径
if [[ "$single_slice" == "true" ]]; then
    model_save_path="data/clap_sonar_single_n${n}.pt"
    job_name="sonar_train_single_n${n}"
else
    model_save_path="data/clap_sonar_full_n${n}.pt"
    job_name="sonar_train_full_n${n}"
fi

train_job=$(sbatch \
    --job-name=$job_name \
    --partition=taurus \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gres=gpu:2 \
    --mem=64GB \
    --output=logs/${job_name}_%j.out \
    --error=logs/${job_name}_%j.err \
    --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 SONAR_train.py --train_samples_path=${final_samples} --epochs=20 --batch_size=512 --save_path=${model_save_path}" | awk '{print $4}')

echo "sonar_train: $train_job"