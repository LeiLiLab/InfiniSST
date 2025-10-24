#!/bin/bash
#SBATCH --job-name=eval_local
#SBATCH --output=eval_local.out
#SBATCH --error=eval_local.err
#SBATCH --partition=taurus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
# 本地评估脚本 - 单GPU版本（7B模型fp16约14GB，单卡足够）

# ===================== 配置参数 =====================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 模型路径（修改为你的模型路径）
MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"

# 数据路径
TEST_SAMPLES="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/balanced_test_set.json"
GLOSSARY_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json"

# mmap音频分片目录（如果使用）
MMAP_SHARD_DIR="/mnt/gemini/data1/jiaxuanluo/mmap_shards"

# 模型配置
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# 评估配置
MAX_EVAL=1000           # 最大评估样本数
NUM_FAILED_CASES=10     # 打印的失败案例数量
DEVICE="cuda:0"         # 使用的GPU设备（单GPU足够）

# 预生成的索引（如果有的话）
PREBUILT_INDEX="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index.pkl"

# ===================== 运行评估 =====================

# 切换到脚本所在目录
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal

echo "=== 本地模型评估 ==="
echo "模型路径: $MODEL_PATH"
echo "测试数据: $TEST_SAMPLES"
echo "词汇表: $GLOSSARY_PATH"
echo "预生成索引: $PREBUILT_INDEX"
echo "最大评估样本: $MAX_EVAL"
echo "设备: $DEVICE"
echo "可用GPU数量: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo ""

# 检查索引是否存在
if [ -f "$PREBUILT_INDEX" ]; then
    echo "✅ 找到预生成索引，将跳过term编码步骤"
    INDEX_ARG="--prebuilt_index $PREBUILT_INDEX"
else
    echo "⚠️  未找到预生成索引，将现场生成（较慢）"
    INDEX_ARG=""
fi
echo ""

# 运行评估
python eval_local.py \
    --model_path "$MODEL_PATH" \
    --test_samples_path "$TEST_SAMPLES" \
    --glossary_path "$GLOSSARY_PATH" \
    --mmap_shard_dir "$MMAP_SHARD_DIR" \
    --model_name "$MODEL_NAME" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --max_eval "$MAX_EVAL" \
    --num_failed_cases "$NUM_FAILED_CASES" \
    --device "$DEVICE" \
    $INDEX_ARG

echo ""
echo "=== 评估完成 ==="

