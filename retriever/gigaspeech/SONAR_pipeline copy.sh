#!/bin/bash

# SONAR训练流水线
# 参数: $1 = n (chunk数量), $2 = text_field (可选，默认为term), $3 = single_slice (可选，用于快速验证)

# 设置参数
n=${1:-3}  # 默认n=3
text_field=${2:-term}  # 默认使用term字段
single_slice=${3:-false}  # 默认使用完整数据集

$mfa_job = 3
# 训练数据集路径
TRAIN_TSV="/mnt/data/siqiouyang/datasets/gigaspeech/manifests/train_xl.tsv"

# 创建日志文件
LOG_FILE="logs/sonar_pipeline_n${n}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# === 4. Merge samples (如果不是单分片模式) ===
if [[ "$single_slice" != "true" ]]; then
    echo "[INFO] Step 4: Merging MFA processed samples..." | tee -a "$LOG_FILE"
    
    merge_job=$(sbatch \
        --dependency=afterok:$mfa_job \
        --job-name=merge_mfa_n${n} \
        --partition=taurus \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=96GB \
        --output=logs/merge_mfa_n${n}_%j.out \
        --error=logs/merge_mfa_n${n}_%j.err \
        --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 -c \"
import json, glob
files = sorted(glob.glob('data/samples/xl/mfa_${n}chunks_samples_*.json'))
merged = []
for f in files:
    with open(f, encoding='utf-8') as j:
        merged.extend(json.load(j))
print(f'Merged total {len(merged)} MFA samples with n=${n}')
with open('data/xl_mfa_${n}chunks_samples_merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
\"" | awk '{print $4}')
    
    echo "merge_mfa_samples: $merge_job" | tee -a "$LOG_FILE"
    final_samples="data/xl_mfa_${n}chunks_samples_merged.json"
    dependency_job=$merge_job
else
    # 单分片模式直接使用单个文件
    final_samples=$output_samples
    dependency_job=$mfa_job
fi

# === 5. Train SONAR model ===
echo "[INFO] Step 5: Training SONAR model..." | tee -a "$LOG_FILE"

# 根据模式设置模型保存路径
if [[ "$single_slice" == "true" ]]; then
    model_save_path="data/clap_sonar_single_n${n}.pt"
    job_name="sonar_train_single_n${n}"
else
    model_save_path="data/clap_sonar_full_n${n}.pt"
    job_name="sonar_train_full_n${n}"
fi

train_job=$(sbatch \
    --dependency=afterok:$dependency_job \
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

echo "sonar_train: $train_job" | tee -a "$LOG_FILE"

# === 总结 ===
echo "" | tee -a "$LOG_FILE"
echo "=== SONAR Pipeline Summary ===" | tee -a "$LOG_FILE"
echo "n (chunk count): ${n}" | tee -a "$LOG_FILE"
echo "text_field: ${text_field}" | tee -a "$LOG_FILE"
echo "single_slice: ${single_slice}" | tee -a "$LOG_FILE"
echo "Input TSV: ${TRAIN_TSV}" | tee -a "$LOG_FILE"
echo "Final samples: ${final_samples}" | tee -a "$LOG_FILE"
echo "Model save path: ${model_save_path}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Job IDs:" | tee -a "$LOG_FILE"
if [[ "$single_slice" == "true" ]]; then
    echo "  - NER extraction (single): $ner_job" | tee -a "$LOG_FILE"
    echo "  - Preprocessing (single): $preprocess_job" | tee -a "$LOG_FILE"
    echo "  - MFA chunks (single): $mfa_job" | tee -a "$LOG_FILE"
    echo "  - Training (single): $train_job" | tee -a "$LOG_FILE"
else
    echo "  - NER extraction (full): $ner_job" | tee -a "$LOG_FILE"
    echo "  - Preprocessing (full): $preprocess_job" | tee -a "$LOG_FILE"
    echo "  - MFA chunks (full): $mfa_job" | tee -a "$LOG_FILE"
    echo "  - Merge samples: $merge_job" | tee -a "$LOG_FILE" 
    echo "  - Training (full): $train_job" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Monitor progress with:" | tee -a "$LOG_FILE"
echo "  squeue -u \$USER" | tee -a "$LOG_FILE"
echo "  tail -f ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [[ "$single_slice" == "true" ]]; then
    echo "✅ SONAR single-slice pipeline submitted successfully!" | tee -a "$LOG_FILE"
    echo "📝 Quick validation mode: using only first 500K samples" | tee -a "$LOG_FILE"
else
    echo "✅ SONAR full pipeline submitted successfully!" | tee -a "$LOG_FILE"
    echo "📝 Full dataset mode: processing all samples" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Usage examples:" | tee -a "$LOG_FILE"
echo "  # Full pipeline with n=3 chunks" | tee -a "$LOG_FILE"
echo "  bash SONAR_pipeline.sh 3 term" | tee -a "$LOG_FILE"
echo "  # Single slice quick validation with n=5 chunks" | tee -a "$LOG_FILE"
echo "  bash SONAR_pipeline.sh 5 term true" | tee -a "$LOG_FILE"
