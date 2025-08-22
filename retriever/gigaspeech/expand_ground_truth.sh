#!/bin/bash

# 扩充ground truth terms脚本

echo "[INFO] Starting ground truth terms expansion..."

# 默认参数
INPUT_FILE=${1:-"data/xl_mfa_2chunks_samples_merged.json"}
OUTPUT_FILE=${2:-"data/xl_mfa_2chunks_samples_expanded.json"}
STRATEGY=${3:-"moderate"}
MAX_TERMS=${4:-6}

echo "[INFO] Input file: $INPUT_FILE"
echo "[INFO] Output file: $OUTPUT_FILE"
echo "[INFO] Strategy: $STRATEGY"
echo "[INFO] Max additional terms: $MAX_TERMS"

# 运行扩充脚本
python3 expand_ground_truth_terms.py \
    --input_file="$INPUT_FILE" \
    --output_file="$OUTPUT_FILE" \
    --strategy="$STRATEGY" \
    --max_additional_terms="$MAX_TERMS"

echo "[INFO] Expansion completed!"
echo ""
echo "=== Usage Examples ==="
echo "# 使用扩充后的数据训练："
echo "python3 SONAR_train.py --train_samples_path=$OUTPUT_FILE --batch_size=256 --save_path=data/clap_sonar_expanded.pt"
echo ""
echo "# 不同扩充策略："
echo "bash expand_ground_truth.sh $INPUT_FILE data/xl_mfa_conservative.json conservative 3"
echo "bash expand_ground_truth.sh $INPUT_FILE data/xl_mfa_aggressive.json aggressive 10" 