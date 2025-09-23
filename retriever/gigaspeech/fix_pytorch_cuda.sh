#!/bin/bash

# PyTorch CUDA版本修复脚本

echo "=== PyTorch CUDA版本修复 ==="

# 激活环境
source /home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

echo "当前PyTorch信息:"
python3 -c "import torch; print('PyTorch版本:', torch.__version__); print('编译CUDA版本:', torch.version.cuda); print('CUDA可用:', torch.cuda.is_available())"

echo ""
echo "系统CUDA版本:"
/usr/local/cuda/bin/nvcc --version | grep release

echo ""
echo "解决方案选项:"
echo "1. 重新安装匹配CUDA 11.8的PyTorch"
echo "2. 使用CPU版本进行训练（较慢）"
echo "3. 强制使用现有PyTorch（可能不稳定）"

read -p "选择解决方案 (1/2/3): " choice

case $choice in
    1)
        echo "正在安装CUDA 11.8版本的PyTorch..."
        pip uninstall torch torchvision torchaudio -y
        pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
        echo "安装完成，测试CUDA支持..."
        python3 -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU数量:', torch.cuda.device_count())"
        ;;
    2)
        echo "将使用CPU版本训练..."
        echo "注意：CPU训练会非常慢，建议修复CUDA问题"
        ;;
    3)
        echo "尝试强制使用现有PyTorch..."
        export CUDA_HOME=/usr/local/cuda
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        export PATH=/usr/local/cuda/bin:$PATH
        # 尝试重新导入
        python3 -c "
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
import torch
print('强制设置后CUDA可用:', torch.cuda.is_available())
"
        ;;
    *)
        echo "无效选择"
        ;;
esac
