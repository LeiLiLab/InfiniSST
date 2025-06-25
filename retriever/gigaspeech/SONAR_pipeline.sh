#!/bin/bash

# SONAR训练流水线
# 参数: $1 = n (chunk数量), $2 = text_field (可选，默认为term), $3 = single_slice (可选，用于快速验证)

# 设置参数
n=${1:-3}  # 默认n=3
text_field=${2:-term}  # 默认使用term字段
single_slice=${3:-false}  # 默认使用完整数据集

# 训练数据集路径
TRAIN_TSV="/mnt/data/siqiouyang/datasets/gigaspeech/manifests/train_xl.tsv"

# 创建日志文件
LOG_FILE="logs/sonar_pipeline_n${n}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "[INFO] Starting SONAR pipeline with n=${n}, text_field=${text_field}, single_slice=${single_slice}" | tee -a "$LOG_FILE"
echo "[INFO] Train TSV: ${TRAIN_TSV}" | tee -a "$LOG_FILE"

# === 1. Extract NER cache for training data ===
echo "[INFO] Step 1: Extracting NER cache for training data..." | tee -a "$LOG_FILE"

if [[ "$single_slice" == "true" ]]; then
    # 单分片快速验证模式
    echo "[INFO] Using single slice mode for quick validation" | tee -a "$LOG_FILE"
    
    # 只处理第0分片的NER提取
    ner_job=$(sbatch \
        --job-name=train_ner_single \
        --partition=taurus \
        --mem=64GB \
        --cpus-per-task=1 \
        --ntasks=1 \
        --gres=gpu:1 \
        --output=logs/train_ner_single_%j.out \
        --error=logs/train_ner_single_%j.err \
        --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate spaCyEnv
# 创建分片目录
mkdir -p data/split_tsv
# 只处理前500000行作为快速验证
SPLIT_TSV=\"data/split_tsv/train_xl_split_0.tsv\"
if [[ ! -f \"\$SPLIT_TSV\" ]]; then
    echo \"[INFO] Creating single slice for quick validation...\"
    (head -n 1 ${TRAIN_TSV} && sed -n '2,500001p' ${TRAIN_TSV}) > \"\$SPLIT_TSV\"
fi
python3 extract_ner_cache.py --tsv_path=\$SPLIT_TSV" | awk '{print $4}')
    
    echo "train_ner_single: $ner_job" | tee -a "$LOG_FILE"
else
    # 完整数据集模式
    ner_job=$(sbatch extract_ner_cache.sh | awk '{print $4}')
    echo "train_ner_cache: $ner_job" | tee -a "$LOG_FILE"
fi

# === 2. Preprocess training samples ===
echo "[INFO] Step 2: Preprocessing training samples..." | tee -a "$LOG_FILE"

if [[ "$single_slice" == "true" ]]; then
    # 单分片预处理
    preprocess_job=$(sbatch \
        --dependency=afterok:$ner_job \
        --job-name=train_preprocess_single \
        --partition=taurus \
        --mem=32GB \
        --cpus-per-task=1 \
        --ntasks=1 \
        --output=logs/train_preprocess_single_%j.out \
        --error=logs/train_preprocess_single_%j.err \
        --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 train_samples_pre_handle.py --tsv_path=data/split_tsv/train_xl_split_0.tsv --split_id=0 --text_field=${text_field} --ner_json=data/named_entities_train_xl_split_0.json" | awk '{print $4}')
    
    echo "train_preprocess_single: $preprocess_job" | tee -a "$LOG_FILE"
else
    # 完整数据集预处理
    preprocess_job=$(sbatch --dependency=afterok:$ner_job samples_pre_handle_split.sh ${text_field} | awk '{print $4}')
    echo "train_preprocess: $preprocess_job" | tee -a "$LOG_FILE"
fi

# === 3. Handle MFA n-chunk samples ===
echo "[INFO] Step 3: Handling MFA ${n}-chunk samples..." | tee -a "$LOG_FILE"

if [[ "$single_slice" == "true" ]]; then
    # 单分片MFA处理
    if [[ "$text_field" == "term" ]]; then
        input_samples="data/samples/xl/term_preprocessed_samples_0_500000.json"
        output_samples="data/samples/xl/mfa_${n}chunks_samples_single_0_500000.json"
    else
        input_samples="data/samples/xl/preprocessed_samples_0_500000.json"
        output_samples="data/samples/xl/mfa_${n}chunks_samples_single_0_500000.json"
    fi
    
    mfa_job=$(sbatch \
        --dependency=afterok:$preprocess_job \
        --job-name=train_mfa_single_n${n} \
        --partition=taurus \
        --mem=32GB \
        --cpus-per-task=4 \
        --ntasks=1 \
        --output=logs/train_mfa_single_n${n}_%j.out \
        --error=logs/train_mfa_single_n${n}_%j.err \
        --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 handle_MFA_n_chunk_samples.py --input_json=${input_samples} --output_json=${output_samples} --n=${n} --chunk_len=0.96 --textgrid_dir=/mnt/data/siqiouyang/datasets/gigaspeech/textgrids" | awk '{print $4}')
    
    echo "train_mfa_single: $mfa_job" | tee -a "$LOG_FILE"
else
    # 完整数据集MFA处理
    mfa_job=$(sbatch --dependency=afterok:$preprocess_job handle_MFA_n_chunk_samples.sh ${n} 0.96 ${text_field}_preprocessed_samples | awk '{print $4}')
    echo "train_mfa_chunks: $mfa_job" | tee -a "$LOG_FILE"
fi

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
python3 SONAR_train.py --train_samples_path=${final_samples} --epochs=50 --batch_size=32 --save_path=${model_save_path}" | awk '{print $4}')

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
