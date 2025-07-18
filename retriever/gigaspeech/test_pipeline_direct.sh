#!/bin/bash

# 直接执行版本的测试流水线（不使用SLURM）
# 适用于小数据集或本地测试
# 参数: $1 = n (chunk数量), $2 = text_field (可选，默认为term)

set -e  # 遇到错误立即退出

# 设置参数
n=${1:-3}  # 默认n=3
text_field=${2:-term}  # 默认使用term字段

# 测试数据集路径（远程服务器路径）
TEST_TSV="/mnt/data/siqiouyang/datasets/gigaspeech/manifests/test.tsv"

# 创建日志文件
LOG_FILE="logs/test_pipeline_direct_n${n}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "[INFO] Starting direct test pipeline with n=${n}, text_field=${text_field}" | tee -a "$LOG_FILE"
echo "[INFO] Test TSV: ${TEST_TSV}" | tee -a "$LOG_FILE"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh

# 注意：这个脚本应该在远程服务器上运行，数据集路径检查在运行时进行

# === 1. Extract NER cache for test data ===
echo "[INFO] Step 1: Extracting NER cache for test data..." | tee -a "$LOG_FILE"

conda activate spaCyEnv

# 检查数据集是否存在（在远程服务器上）
if [[ ! -f "$TEST_TSV" ]]; then
    echo "[ERROR] Test TSV file not found: $TEST_TSV" | tee -a "$LOG_FILE"
    echo "[INFO] Please ensure this script is running on the remote server where the data is located." | tee -a "$LOG_FILE"
    exit 1
fi

python3 extract_ner_cache_test.py --tsv_path="${TEST_TSV}" 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "[ERROR] NER extraction failed" | tee -a "$LOG_FILE"
    exit 1
fi

# === 2. Preprocess test samples ===
echo "[INFO] Step 2: Preprocessing test samples..." | tee -a "$LOG_FILE"

conda activate infinisst
python3 train_samples_pre_handle_test.py \
    --tsv_path="${TEST_TSV}" \
    --ner_json="data/named_entities_test.json" \
    --text_field="${text_field}" 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "[ERROR] Sample preprocessing failed" | tee -a "$LOG_FILE"
    exit 1
fi

# === 3. Handle MFA n-chunk samples ===
echo "[INFO] Step 3: Handling MFA ${n}-chunk samples..." | tee -a "$LOG_FILE"

# 设置输入和输出路径
if [[ "$text_field" == "term" ]]; then
    input_samples="data/samples/test/term_preprocessed_samples_test.json"
else
    input_samples="data/samples/test/preprocessed_samples_test.json"
fi

output_samples="data/samples/test/test_mfa_${n}chunks_samples.json"

# 检查输入文件是否存在
if [[ ! -f "$input_samples" ]]; then
    echo "[ERROR] Input samples file not found: $input_samples" | tee -a "$LOG_FILE"
    exit 1
fi

python3 handle_MFA_n_chunk_samples_test.py \
    --input_json="${input_samples}" \
    --output_json="${output_samples}" \
    --n="${n}" 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "[ERROR] MFA chunk processing failed" | tee -a "$LOG_FILE"
    exit 1
fi

# === 4. Optional: Quick validation ===
echo "[INFO] Step 4: Validating output..." | tee -a "$LOG_FILE"

if [[ -f "$output_samples" ]]; then
    sample_count=$(python3 -c "import json; data=json.load(open('$output_samples')); print(len(data))")
    echo "[INFO] Generated ${sample_count} chunk samples" | tee -a "$LOG_FILE"
    
    # 显示第一个样本的结构
    echo "[INFO] Sample structure:" | tee -a "$LOG_FILE"
    python3 -c "
import json
data = json.load(open('$output_samples'))
if data:
    sample = data[0]
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 100:
            print(f'  {key}: {value[:100]}...')
        else:
            print(f'  {key}: {value}')
" 2>&1 | tee -a "$LOG_FILE"
else
    echo "[ERROR] Output file not created: $output_samples" | tee -a "$LOG_FILE"
    exit 1
fi

# === 总结 ===
echo "" | tee -a "$LOG_FILE"
echo "=== Test Pipeline Summary ===" | tee -a "$LOG_FILE"
echo "n (chunk count): ${n}" | tee -a "$LOG_FILE"
echo "text_field: ${text_field}" | tee -a "$LOG_FILE"
echo "Input TSV: ${TEST_TSV}" | tee -a "$LOG_FILE"
echo "Output samples: ${output_samples}" | tee -a "$LOG_FILE"
echo "Sample count: ${sample_count}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "✅ Test pipeline completed successfully!" | tee -a "$LOG_FILE"

# 可选：运行训练进行快速验证
read -p "Do you want to run a quick training validation? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[INFO] Running quick training validation..." | tee -a "$LOG_FILE"
    python3 SONAR_train.py \
        --train_samples_path="${output_samples}" \
        --test_samples_path="${output_samples}" \
        --epochs=2 \
        --batch_size=8 \
        --save_path="data/clap_test_validation_n${n}.pt" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "[INFO] Training validation completed successfully!" | tee -a "$LOG_FILE"
    else
        echo "[WARNING] Training validation failed, but pipeline data is ready" | tee -a "$LOG_FILE"
    fi
fi 