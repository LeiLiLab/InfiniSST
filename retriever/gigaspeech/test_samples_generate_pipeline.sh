#!/bin/bash

# 测试数据集处理流水线
# 参数: $1 = n (chunk数量), $2 = text_field (可选，默认为term)

# 设置参数
n=${1:-3}  # 默认n=3
text_field=${2:-term}  # 默认使用term字段

# 测试数据集路径（远程服务器路径）
TEST_TSV="/mnt/data/siqiouyang/datasets/gigaspeech/manifests/test.tsv"

# 创建日志文件
LOG_FILE="logs/test_pipeline_n${n}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "[INFO] Starting test pipeline with n=${n}, text_field=${text_field}" | tee -a "$LOG_FILE"
echo "[INFO] Test TSV: ${TEST_TSV}" | tee -a "$LOG_FILE"

# 注意：数据集路径检查在远程服务器上，这里跳过本地检查

# === 1. Extract NER cache for test data ===
echo "[INFO] Step 1: Extracting NER cache for test data..." | tee -a "$LOG_FILE"

# 提交NER提取任务
ner_job=$(sbatch \
    --job-name=test_ner_cache \
    --partition=taurus \
    --mem=32GB \
    --cpus-per-task=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --output=logs/test_ner_cache_%j.out \
    --error=logs/test_ner_cache_%j.err \
    --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate spaCyEnv
python3 extract_ner_cache_test.py --tsv_path=${TEST_TSV}" | awk '{print $4}')

echo "test_ner_cache: $ner_job" | tee -a "$LOG_FILE"

# === 2. Preprocess test samples ===
echo "[INFO] Step 2: Preprocessing test samples..." | tee -a "$LOG_FILE"

# 等待NER任务完成后执行样本预处理
preprocess_job=$(sbatch \
    --dependency=afterok:$ner_job \
    --job-name=test_preprocess \
    --partition=taurus \
    --mem=32GB \
    --cpus-per-task=1 \
    --ntasks=1 \
    --output=logs/test_preprocess_%j.out \
    --error=logs/test_preprocess_%j.err \
    --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 train_samples_pre_handle_test.py --tsv_path=${TEST_TSV} --ner_json=data/named_entities_test.json --text_field=${text_field}" | awk '{print $4}')

echo "test_preprocess: $preprocess_job" | tee -a "$LOG_FILE"

# === 3. Handle MFA n-chunk samples ===
echo "[INFO] Step 3: Handling MFA ${n}-chunk samples..." | tee -a "$LOG_FILE"

# 设置输入和输出路径
if [[ "$text_field" == "term" ]]; then
    input_samples="data/samples/test/term_preprocessed_samples_test.json"
else
    input_samples="data/samples/test/preprocessed_samples_test.json"
fi

output_samples="data/samples/test/test_mfa_${n}chunks_samples.json"

# 等待预处理完成后执行MFA chunk处理
mfa_job=$(sbatch \
    --dependency=afterok:$preprocess_job \
    --job-name=test_mfa_chunks \
    --partition=taurus \
    --mem=64GB \
    --cpus-per-task=8 \
    --ntasks=1 \
    --output=logs/test_mfa_chunks_n${n}_%j.out \
    --error=logs/test_mfa_chunks_n${n}_%j.err \
    --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 handle_MFA_n_chunk_samples_test.py --input_json=${input_samples} --output_json=${output_samples} --n=${n}" | awk '{print $4}')

echo "test_mfa_chunks: $mfa_job" | tee -a "$LOG_FILE"

# === 4. Optional: Train model with test data (for evaluation) ===
echo "[INFO] Step 4: Training model with test data (optional)..." | tee -a "$LOG_FILE"

# 等待MFA处理完成后可选择训练模型进行评估
train_job=$(sbatch \
    --dependency=afterok:$mfa_job \
    --job-name=test_train_n${n} \
    --partition=taurus \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gres=gpu:2 \
    --mem=64GB \
    --output=logs/test_train_n${n}_%j.out \
    --error=logs/test_train_n${n}_%j.err \
    --wrap="#!/bin/bash
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 SONAR_train.py --train_samples_path=${output_samples} --test_samples_path=${output_samples} --epochs=10 --batch_size=16 --save_path=data/clap_test_n${n}.pt" | awk '{print $4}')

echo "test_train: $train_job" | tee -a "$LOG_FILE"

# === 总结 ===
echo "" | tee -a "$LOG_FILE"
echo "=== Test Pipeline Summary ===" | tee -a "$LOG_FILE"
echo "n (chunk count): ${n}" | tee -a "$LOG_FILE"
echo "text_field: ${text_field}" | tee -a "$LOG_FILE"
echo "Input TSV: ${TEST_TSV}" | tee -a "$LOG_FILE"
echo "Output samples: ${output_samples}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Job IDs:" | tee -a "$LOG_FILE"
echo "  - NER extraction: $ner_job" | tee -a "$LOG_FILE"
echo "  - Preprocessing: $preprocess_job" | tee -a "$LOG_FILE"
echo "  - MFA chunks: $mfa_job" | tee -a "$LOG_FILE"
echo "  - Training: $train_job" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Monitor progress with:" | tee -a "$LOG_FILE"
echo "  squeue -u \$USER" | tee -a "$LOG_FILE"
echo "  tail -f ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "✅ Test pipeline submitted successfully!" | tee -a "$LOG_FILE"
