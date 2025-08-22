#!/bin/bash

# SONAR训练流水线
# 参数: $1 = n (chunk数量), $2 = text_field (可选，默认为term), $3 = single_slice (可选，用于快速验证), $4 = loss_ratios (可选，用于loss sweep)

# 设置参数
n=${1:-3}  # 默认n=3
text_field=${2:-term}  # 默认使用term字段
single_slice=${3:-false}  # 默认使用完整数据集
loss_ratios=${4:-""}  # 可选的loss比例sweep参数

# 设置数据路径
final_samples="data/xl_mfa_${n}chunks_samples_merged.json"

# 根据模式设置数据路径
if [[ "$single_slice" == "true" ]]; then
    final_samples="data/samples/xl/mfa_${n}chunks_samples_single_0_500000.json"
    # 单分片MFA处理
    if [[ "$text_field" == "term" ]]; then
        final_samples="data/samples/xl/mfa_${n}chunks_samples_single_0_500000.json"
    else
        final_samples="data/samples/xl/mfa_${n}chunks_samples_single_0_500000.json"
    fi
fi

# 函数：提交单个训练任务
submit_training_job() {
    local audio_text_ratio=$1
    local audio_term_ratio=$2
    local job_suffix=$3
    
    # 设置模型保存路径和任务名称
    if [[ "$single_slice" == "true" ]]; then
        model_save_path="data/temp_clap_sonar_single_n${n}${job_suffix}.pt"
        job_name="sonar_train_single_n${n}${job_suffix}"
    else
        model_save_path="data/clap_sonar_full_n${n}${job_suffix}.pt"
        job_name="sonar_train_full_n${n}${job_suffix}"
    fi
    
    echo "[INFO] Submitting job: $job_name with ratios ${audio_text_ratio}:${audio_term_ratio}"
    
    train_job=$(sbatch \
        --job-name=$job_name \
        --partition=taurus \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --gres=gpu:1 \
        --mem=64GB \
        --output=logs/${job_name}_%j.out \
        --error=logs/${job_name}_%j.err \
        --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 SONAR_train.py \
    --train_samples_path=${final_samples} \
    --epochs=20 \
    --batch_size=512 \
    --save_path=${model_save_path} \
    --audio_text_loss_ratio=${audio_text_ratio} \
    --audio_term_loss_ratio=${audio_term_ratio}" | awk '{print $4}')
    
    echo "sonar_train${job_suffix}: $train_job"
}

# 检查是否进行loss sweep
if [[ -n "$loss_ratios" ]]; then
    echo "[INFO] Starting loss ratio sweep with ratios: $loss_ratios"
    
    # 分割loss_ratios字符串，用分号分隔
    IFS=';' read -ra RATIO_PAIRS <<< "$loss_ratios"
    
    for ratio_pair in "${RATIO_PAIRS[@]}"; do
        # 分割每对比例，用逗号分隔
        IFS=',' read -ra RATIOS <<< "$ratio_pair"
        
        if [[ ${#RATIOS[@]} -eq 2 ]]; then
            audio_text_ratio=${RATIOS[0]}
            audio_term_ratio=${RATIOS[1]}
            
            # 创建任务后缀，用于区分不同的比例
            job_suffix="_r${audio_text_ratio}_${audio_term_ratio}"
            
            # 提交训练任务
            submit_training_job "$audio_text_ratio" "$audio_term_ratio" "$job_suffix"
        else
            echo "[ERROR] Invalid ratio pair format: $ratio_pair (expected format: audio_text_ratio,audio_term_ratio)"
        fi
    done
    
    echo "[INFO] Loss ratio sweep completed. Submitted ${#RATIO_PAIRS[@]} jobs."
else
    # 默认行为：使用默认比例提交单个任务
    echo "[INFO] Using default loss ratios (0.3:0.7)"
    submit_training_job "0.3" "0.7" ""
fi