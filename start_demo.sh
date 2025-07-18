#!/bin/bash

echo "🚀 启动 InfiniSST 整合演示系统"
echo "=================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 未找到Python，请先安装Python 3.8+"
    exit 1
fi

# 检查基本依赖
echo "📦 检查依赖..."
python -c "import flask, torch, numpy" 2>/dev/null || {
    echo "⚠️ 正在安装基本依赖..."
    pip install flask flask-cors torch numpy requests
}

# 设置环境变量
export PYTHONPATH=$PWD:$PYTHONPATH

echo "🏗️ 启动整合服务器..."
echo "   - 模式: 模拟推理"
echo "   - GPU: 0"
echo "   - 语言对: English -> Chinese"
echo "   - 地址: http://localhost:8000"
echo ""

# 启动服务器
python serve/start_integrated_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --gpus "0" \
    --languages "English -> Chinese" \
    --max-batch-size 32 \
    --batch-timeout 0.1

echo ""
echo "🛑 服务器已停止" 