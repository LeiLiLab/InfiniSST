#!/bin/bash

# 并行数（建议根据 CPU 核心数设置，例如 16 或 32）
NUM_SHARDS=16

PYTHON_SCRIPT="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/fix_missing_audio_chunks.py"
INPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_dataset_final.jsonl"
INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
OUTPUT_DIR="/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks"

echo "Starting parallel audio chunk fix with $NUM_SHARDS shards..."

for ((i=0; i<NUM_SHARDS; i++)); do
    python3 "$PYTHON_SCRIPT" \
        --input-jsonl "$INPUT_JSONL" \
        --input-tsv "$INPUT_TSV" \
        --output-dir "$OUTPUT_DIR" \
        --shard-id $i \
        --total-shards $NUM_SHARDS &
done

# 等待所有进程结束
wait

echo "All shards completed!"