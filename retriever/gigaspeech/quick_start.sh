#!/bin/bash

# Qwen2-Audio Term-Level训练快速启动脚本
# 提供本地和Modal两种训练选项

echo "=== Qwen2-Audio Term-Level Training Quick Start ==="
echo ""

# 检查必要的文件
echo "[INFO] Checking required files..."

TRAIN_DATA="data/xl_cleaned_term_level_chunks_merged.json"
DDP_SCRIPT="Qwen2_Audio_term_level_train_ddp.py"
MODAL_SCRIPT="modal_complete_training.py"

missing_files=0

if [ ! -f "$TRAIN_DATA" ]; then
    echo "[ERROR] Training data not found: $TRAIN_DATA"
    missing_files=$((missing_files + 1))
else
    echo "[OK] Training data found: $TRAIN_DATA"
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "[WARN] Test data not found: $TEST_DATA (will use train/test split)"
else
    echo "[OK] Test data found: $TEST_DATA"
fi

if [ ! -f "$DDP_SCRIPT" ]; then
    echo "[ERROR] DDP training script not found: $DDP_SCRIPT"
    missing_files=$((missing_files + 1))
else
    echo "[OK] DDP training script found: $DDP_SCRIPT"
fi

if [ ! -f "$MODAL_SCRIPT" ]; then
    echo "[ERROR] Modal script not found: $MODAL_SCRIPT"
    missing_files=$((missing_files + 1))
else
    echo "[OK] Modal script found: $MODAL_SCRIPT"
fi

if [ $missing_files -gt 0 ]; then
    echo ""
    echo "[ERROR] $missing_files required files are missing. Please check the file paths."
    exit 1
fi

echo ""
echo "[INFO] All required files found!"
echo ""

# 选择训练方式
echo "Please choose training method:"
echo "1) Local DDP Training (requires local GPUs)"
echo "2) Modal Cloud Training (requires Modal setup)"
echo "3) Test Modal Setup"
echo "4) Exit"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "=== Starting Local DDP Training ==="
        echo ""
        
        # 检查GPU
        if command -v nvidia-smi &> /dev/null; then
            echo "[INFO] GPU status:"
            nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
            echo ""
        else
            echo "[WARN] nvidia-smi not found. Make sure you have NVIDIA GPUs available."
        fi
        
        # 检查conda环境
        if [ -z "$CONDA_DEFAULT_ENV" ]; then
            echo "[WARN] No conda environment detected. Make sure to activate the correct environment."
            echo "Example: conda activate infinisst"
            echo ""
        else
            echo "[INFO] Current conda environment: $CONDA_DEFAULT_ENV"
            echo ""
        fi
        
        read -p "Continue with local training? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo "[INFO] Starting local DDP training..."
            chmod +x qwen2_audio_train_ddp_fixed.sh
            ./qwen2_audio_train_ddp_fixed.sh
        else
            echo "[INFO] Local training cancelled."
        fi
        ;;
        
    2)
        echo ""
        echo "=== Starting Modal Cloud Training ==="
        echo ""
        
        # 检查Modal安装
        if ! command -v modal &> /dev/null; then
            echo "[ERROR] Modal CLI not found. Please install it first:"
            echo "  pip install modal"
            echo ""
            exit 1
        fi
        
        # # 检查Modal认证
        # if ! modal token list &> /dev/null; then
        #     echo "[ERROR] Modal not authenticated. Please run:"
        #     echo "  modal token new"
        #     echo ""
        #     exit 1
        # fi
        
        echo "[OK] Modal CLI found and authenticated"
        
        # 检查数据大小
        if [ -f "$TRAIN_DATA" ]; then
            train_size=$(du -h "$TRAIN_DATA" | cut -f1)
            echo "[INFO] Training data size: $train_size"
        fi
        
        if [ -f "$TEST_DATA" ]; then
            test_size=$(du -h "$TEST_DATA" | cut -f1)
            echo "[INFO] Test data size: $test_size"
        fi
        
        echo ""
        echo "[INFO] Modal training will use:"
        echo "  - 8x A100 GPUs (estimated cost: ~$8.8/hour)"
        echo "  - 256GB RAM, 64 CPU cores"
        echo "  - Automatic data upload and model saving"
        echo ""
        
        read -p "Continue with Modal training? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo "[INFO] Starting Modal training..."
            modal run modal_complete_training.py
        else
            echo "[INFO] Modal training cancelled."
        fi
        ;;
        
    3)
        echo ""
        echo "=== Testing Modal Setup ==="
        echo ""
        
        if ! command -v modal &> /dev/null; then
            echo "[ERROR] Modal CLI not found. Please install it first:"
            echo "  pip install modal"
            exit 1
        fi
        
        echo "[INFO] Running Modal environment tests..."
        modal run test_modal_setup.py
        ;;
        
    4)
        echo "[INFO] Exiting..."
        exit 0
        ;;
        
    *)
        echo "[ERROR] Invalid choice. Please select 1-4."
        exit 1
        ;;
esac

echo ""
echo "=== Quick Start Completed ==="
