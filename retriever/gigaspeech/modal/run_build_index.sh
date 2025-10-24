#!/bin/bash
#SBATCH --job-name=build_index
#SBATCH --output=build_index.out
#SBATCH --error=build_index.err
#SBATCH --partition=taurus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
# 多GPU生成索引

# ===================== 配置参数 =====================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 模型路径
MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"

# 词汇表路径
GLOSSARY_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json"

# 输出路径
OUTPUT_PATH="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index.pkl"

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_PATH")"

# 模型配置
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"

# LoRA配置（必须与训练时一致）
LORA_R=16
LORA_ALPHA=32

# 多GPU配置  
NUM_GPUS=1              # 使用1个GPU（慢但100%稳定）
BATCH_SIZE=64           # 单GPU可以用更大的batch_size

# 如果愿意等更久但想用多GPU，可以改为：
# NUM_GPUS=6
# BATCH_SIZE=8  # 多GPU必须用小batch

# ===================== 内存优化设置 =====================

# 设置PyTorch内存分配器，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 禁用tokenizers并行警告
export TOKENIZERS_PARALLELISM=false

# ===================== 运行索引生成 =====================

# 切换到脚本所在目录
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal

echo "=== 多GPU生成文本索引（with LoRA）==="
echo "模型路径: $MODEL_PATH"
echo "词汇表: $GLOSSARY_PATH"
echo "输出路径: $OUTPUT_PATH"
echo "使用GPU数量: $NUM_GPUS"
echo "每GPU batch size: $BATCH_SIZE"
echo "LoRA配置: r=$LORA_R, alpha=$LORA_ALPHA"
echo ""

# 运行索引生成
python build_index_multi_gpu.py \
    --model_path "$MODEL_PATH" \
    --glossary_path "$GLOSSARY_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --num_gpus "$NUM_GPUS" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "=== 索引生成完成 ==="
echo "索引文件: $OUTPUT_PATH"
echo ""

# 显示文件信息
if [ -f "$OUTPUT_PATH" ]; then
    ls -lh "$OUTPUT_PATH"
fi

