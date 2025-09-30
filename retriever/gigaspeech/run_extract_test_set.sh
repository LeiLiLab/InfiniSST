#!/bin/bash

# 平衡测试集提取脚本
# 从训练集中提取1000个unique terms的样本作为测试集，确保20%的terms是unseen的

echo "=== Balanced Test Set Extraction ==="

# 激活conda环境
source /home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 设置参数
INPUT_PATH="data/xl_cleaned_term_level_chunks_merged.json"
OUTPUT_TRAIN_PATH="data/balanced_train_set.json"
OUTPUT_TEST_PATH="data/balanced_test_set.json"
TEST_SIZE=1000
UNSEEN_RATIO=0.20
SEED=42

echo "Input file: $INPUT_PATH"
echo "Output train file: $OUTPUT_TRAIN_PATH"
echo "Output test file: $OUTPUT_TEST_PATH"
echo "Test size: $TEST_SIZE samples"
echo "Unseen ratio: ${UNSEEN_RATIO} (${UNSEEN_RATIO}0%)"
echo "Random seed: $SEED"
echo ""

# 检查输入文件是否存在
if [ ! -f "$INPUT_PATH" ]; then
    echo "ERROR: Input file not found: $INPUT_PATH"
    echo "Please check the file path and try again."
    exit 1
fi

# 创建输出目录
mkdir -p $(dirname "$OUTPUT_TRAIN_PATH")
mkdir -p $(dirname "$OUTPUT_TEST_PATH")

echo "Starting extraction..."
echo "$(date): Extraction started"

# 运行提取脚本
python3 extract_balanced_test_set.py \
    --input_path="$INPUT_PATH" \
    --output_train_path="$OUTPUT_TRAIN_PATH" \
    --output_test_path="$OUTPUT_TEST_PATH" \
    --test_size=$TEST_SIZE \
    --unseen_ratio=$UNSEEN_RATIO \
    --seed=$SEED

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "$(date): Extraction completed successfully!"
    echo ""
    
    # 显示输出文件信息
    if [ -f "$OUTPUT_TRAIN_PATH" ]; then
        TRAIN_SIZE=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_TRAIN_PATH'))))")
        echo "Training set: $TRAIN_SIZE samples -> $OUTPUT_TRAIN_PATH"
    fi
    
    if [ -f "$OUTPUT_TEST_PATH" ]; then
        TEST_SIZE_ACTUAL=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_TEST_PATH'))))")
        echo "Test set: $TEST_SIZE_ACTUAL samples -> $OUTPUT_TEST_PATH"
    fi
    
    # 显示术语信息文件
    TERMS_INFO_PATH="${OUTPUT_TEST_PATH%.*}_terms_info.json"
    if [ -f "$TERMS_INFO_PATH" ]; then
        echo "Terms info: $TERMS_INFO_PATH"
        echo ""
        echo "Terms statistics:"
        python3 -c "
import json
with open('$TERMS_INFO_PATH') as f:
    info = json.load(f)
    stats = info['stats']
    print(f\"  Total test samples: {stats['total_test_samples']}\")
    print(f\"  Seen terms: {stats['seen_terms_count']}\")
    print(f\"  Unseen terms: {stats['unseen_terms_count']}\")
    print(f\"  Unseen ratio: {stats['unseen_ratio']:.1%}\")
"
    fi
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Use the balanced datasets for training:"
    echo "   - Training set: $OUTPUT_TRAIN_PATH"
    echo "   - Test set: $OUTPUT_TEST_PATH"
    echo ""
    echo "2. Update your training script to use these files:"
    echo "   TRAIN_SAMPLES_PATH=\"$OUTPUT_TRAIN_PATH\""
    echo "   TEST_SAMPLES_PATH=\"$OUTPUT_TEST_PATH\""
    echo ""
    echo "3. Run training with the new datasets:"
    echo "   ./train_ddp_fixed.sh"
    
else
    echo ""
    echo "$(date): Extraction failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi

