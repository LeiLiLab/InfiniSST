#!/bin/bash

# SONAR完整评估脚本
# 使用训练好的模型在完整glossary上进行评估
# 参数: $1 = n, $2 = text_field, $3 = single_slice, $4 = test_samples_path
# 设置参数
n=${1:-2}  # 默认n=2
text_field=${2:-term}  # 默认使用term字段
single_slice=${3:-true}  # 默认使用完整数据集
test_samples_path=${4:-"data/samples/xl/term_level_chunks_500000_1000000.json"}  # 默认测试数据集路径
model_save_path="data/clap_sonar_term_level_single.pt"

# 根据模式设置模型保存路径
if [[ "$single_slice" == "true" ]]; then
    model_save_path="data/clap_sonar_term_level_single_best.pt"
    job_name="sonar_train_term_level_single"
else
    model_save_path="data/clap_sonar_term_level_full_best.pt"
    job_name="sonar_train_term_level_full"
fi


eval_job=$(sbatch \
    --job-name=$job_name \
    --partition=taurus \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gres=gpu:2 \
    --mem=96GB \
    --output=logs/${job_name}_%j.out \
    --error=logs/${job_name}_%j.err \
    --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 SONAR_full_evaluate.py --model_path=$model_save_path  --build_offline_assets --asset_out_dir data   --index_type ivfpq    --use_ip --nlist 4096  --pq_m 64  --pq_bits 8  --nprobe 16  --test_samples_path=$test_samples_path --max_eval=1000" | awk '{print $4}')

echo "Full evaluation job submitted: $eval_job"