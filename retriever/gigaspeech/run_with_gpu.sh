#!/bin/bash

# GPU训练启动脚本
# 设置正确的CUDA环境变量以确保PyTorch能够使用GPU

echo "=== 设置CUDA环境变量 ==="

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 获取PyTorch安装路径并设置库路径
TORCH_LIB_PATH=$(python -c "import torch; print(torch.__file__.replace('__init__.py', 'lib'))" 2>/dev/null)
TRITON_LIB_PATH=$(python -c "import torch; print(torch.__file__.replace('torch/__init__.py', 'triton/backends/nvidia/lib'))" 2>/dev/null)

echo "PyTorch库路径: $TORCH_LIB_PATH"
echo "Triton库路径: $TRITON_LIB_PATH"

# 设置LD_LIBRARY_PATH（确保在所有子进程中生效）
export LD_LIBRARY_PATH="${TORCH_LIB_PATH}:${TRITON_LIB_PATH}:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH已设置"

# 验证CUDA是否可用
echo "=== 验证CUDA状态 ==="
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA编译版本: {torch.version.cuda}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)')
    print('✅ CUDA验证成功')
else:
    print('❌ CUDA不可用，将使用CPU')
"

echo "=== 开始训练 ==="

# 运行训练脚本，传递所有参数，确保环境变量传递
python3 SONAR_term_level_train_glossary.py "$@"
