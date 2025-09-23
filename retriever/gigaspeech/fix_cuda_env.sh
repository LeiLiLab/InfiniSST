#!/bin/bash

# CUDA环境修复脚本

echo "=== CUDA环境诊断与修复 ==="

# 1. 查找CUDA安装路径
echo "1. 查找CUDA安装路径..."
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-12.4"
    "/usr/local/cuda-12"
    "/opt/cuda"
    "/usr/cuda"
)

CUDA_PATH=""
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
        CUDA_PATH="$path"
        echo "Found CUDA at: $CUDA_PATH"
        break
    fi
done

if [ -z "$CUDA_PATH" ]; then
    echo "CUDA not found in standard locations. Checking system nvcc..."
    SYSTEM_NVCC=$(which nvcc 2>/dev/null)
    if [ -n "$SYSTEM_NVCC" ]; then
        CUDA_PATH=$(dirname $(dirname $SYSTEM_NVCC))
        echo "Found CUDA via system nvcc at: $CUDA_PATH"
    else
        echo "ERROR: CUDA not found!"
        exit 1
    fi
fi

# 2. 设置环境变量
echo "2. 设置CUDA环境变量..."
export CUDA_HOME="$CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version: $($CUDA_PATH/bin/nvcc --version | grep release)"

# 3. 激活conda环境并测试PyTorch
echo "3. 测试PyTorch CUDA支持..."
source /home/jiaxuanluo/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 4. 测试简单的CUDA操作
echo "4. 测试CUDA张量操作..."
python3 -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = x + y
    print('CUDA tensor operation successful!')
    print('Result device:', z.device)
else:
    print('CUDA not available for tensor operations')
"

echo "=== 诊断完成 ==="
