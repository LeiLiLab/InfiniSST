#!/bin/bash
#SBATCH --job-name=build_index
#SBATCH --output=logs/%j_build_index.out
#SBATCH --error=logs/%j_build_index.err
#SBATCH --partition=taurus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB

# sbatch --export=USE_IMPORT=1 run_build_index.sh

# ===================== 配置参数 =====================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"

# 默认 glossary
GLOSSARY_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json"
OUTPUT_PATH="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index.pkl"

# imported glossary（ACL6060）
IMPORT_GLOSSARY_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060.json"
IMPORT_OUTPUT_PATH="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_acl6060.pkl"

# ✅ 控制参数 (默认=0)
USE_IMPORT=${USE_IMPORT:-0}

if [[ $USE_IMPORT -eq 1 ]]; then
    echo ">>> [INFO] USE_IMPORT=1，启用专用 glossary (ACL6060)"
    GLOSSARY_PATH="$IMPORT_GLOSSARY_PATH"
    OUTPUT_PATH="$IMPORT_OUTPUT_PATH"
else
    echo ">>> [INFO] USE_IMPORT=0，使用默认 glossary"
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
LORA_R=16
LORA_ALPHA=32

NUM_GPUS=1
BATCH_SIZE=64

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal

echo "=== 多GPU生成文本索引（with LoRA）==="
echo "模型路径:                 $MODEL_PATH"
echo "Glossary:                 $GLOSSARY_PATH"
echo "输出索引路径:             $OUTPUT_PATH"
echo "GPU数:                     $NUM_GPUS"
echo "Batch size:                $BATCH_SIZE"
echo "LoRA配置:                  r=$LORA_R, alpha=$LORA_ALPHA"
echo ""

python build_index_multi_gpu.py \
    --model_path "$MODEL_PATH" \
    --glossary_path "$GLOSSARY_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --num_gpus "$NUM_GPUS" \
    --batch_size "$BATCH_SIZE" \
    --exclude_confused

echo ""
echo "=== 索引生成完成 ==="
echo "索引文件: $OUTPUT_PATH"
echo ""

if [ -f "$OUTPUT_PATH" ]; then
    ls -lh "$OUTPUT_PATH"
fi