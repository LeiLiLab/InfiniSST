#!/bin/bash

# DDP训练启动脚本
# 使用方法: ./train_ddp.sh

# 设置环境变量
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO  # 可选：用于调试NCCL通信问题
export NCCL_IB_DISABLE=1  # 如果没有InfiniBand，禁用IB
export NCCL_P2P_DISABLE=1  # 如果P2P通信有问题，可以禁用

# 训练参数配置
TRAIN_SAMPLES_PATH="data/xl_cleaned_term_level_chunks_merged.json"
TEST_SAMPLES_PATH="data/samples/test_cleaned/term_preprocessed_samples_test.json"
EPOCHS=20
BATCH_SIZE=4096  # 8个GPU的总batch size，每个GPU分到512
LR=5e-5
SAVE_PATH="data/clap_sonar_term_level_full_ddp.pt"
BEST_MODEL_PATH="data/full_dataset_sonar_term_level_best.pt"
AUDIO_TEXT_LOSS_RATIO=0.3
AUDIO_TERM_LOSS_RATIO=0.7
GLOSSARY_PATH="data/terms/glossary_merged.json"
UNFREEZE_LAYERS=10
GPU_IDS="0,1,2,3,4,5,6,7"  # 使用8个GPU

# 日志文件
LOG_FILE="sonar_train_ddp_full.log"

echo "=== DDP Training Configuration ==="
echo "GPU IDs: $GPU_IDS"
echo "Total Batch Size: $BATCH_SIZE"
echo "Per-GPU Batch Size: $((BATCH_SIZE / 8))"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Log File: $LOG_FILE"
echo "================================="

# 启动DDP训练
python3 SONAR_term_level_train_glossary_ddp.py \
    --train_samples_path=$TRAIN_SAMPLES_PATH \
    --test_samples_path=$TEST_SAMPLES_PATH \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --lr=$LR \
    --save_path=$SAVE_PATH \
    --best_model_path=$BEST_MODEL_PATH \
    --audio_text_loss_ratio=$AUDIO_TEXT_LOSS_RATIO \
    --audio_term_loss_ratio=$AUDIO_TERM_LOSS_RATIO \
    --glossary_path=$GLOSSARY_PATH \
    --unfreeze_layers=$UNFREEZE_LAYERS \
    --filter_no_term \
    --gpu_ids=$GPU_IDS \
    > $LOG_FILE 2>&1

echo "Training completed. Check log file: $LOG_FILE"
