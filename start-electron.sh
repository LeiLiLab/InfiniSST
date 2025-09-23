#!/bin/bash

echo "=== 启动 InfiniSST Electron 应用 ==="

# 检查是否在正确目录
if [ ! -f "package.json" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source env/bin/activate

# 启动API服务器
echo "启动本地API服务器..."
cd serve
python3 api-local.py --port 8001 &
API_PID=$!
cd ..

# 等待服务器启动
echo "等待API服务器启动..."
sleep 3

# 检查服务器
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✓ API服务器运行正常"
else
    echo "✗ API服务器启动失败"
    kill $API_PID 2>/dev/null
    exit 1
fi

# 启动Electron应用
echo "启动Electron应用..."
npm run electron-dev

# 清理
echo "清理进程..."
kill $API_PID 2>/dev/null
echo "完成" 