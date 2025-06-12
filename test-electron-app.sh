#!/bin/bash

echo "=== InfiniSST Electron Test Script ==="

# 检查是否在正确的目录
if [ ! -f "package.json" ]; then
    echo "Error: Must run from project root directory"
    exit 1
fi

# 启动本地API服务器
echo "Starting local API server..."
cd serve
python api-local.py --port 8001 &
API_PID=$!
cd ..

# 等待服务器启动
echo "Waiting for server to start..."
sleep 3

# 检查服务器是否运行
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✓ API server is running on port 8001"
else
    echo "✗ Failed to start API server"
    kill $API_PID 2>/dev/null
    exit 1
fi

# 启动Electron应用
echo "Starting Electron application..."
npm run electron-dev

# 清理
echo "Cleaning up..."
kill $API_PID 2>/dev/null
echo "Done." 