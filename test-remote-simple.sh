#!/bin/bash

echo "=== 简化远程连接测试 ==="
echo ""

# 检查ngrok URL
if [ -z "$1" ]; then
    echo "❌ 请提供ngrok URL"
    echo "用法: $0 <ngrok-url>"
    exit 1
fi

REMOTE_URL="$1"
echo "🌐 测试远程URL: $REMOTE_URL"

# 测试连接
echo "🔍 测试连接..."
if curl -s "$REMOTE_URL" > /dev/null; then
    echo "✅ 远程服务器可访问"
else
    echo "❌ 无法连接到远程服务器"
    exit 1
fi

echo ""
echo "🚀 启动Electron远程连接（无开发者工具）..."
echo "观察是否还有 electronAPI 重复声明错误"
echo ""

# 设置环境变量并启动
export ELECTRON_IS_DEV=true
export REMOTE_SERVER_URL="$REMOTE_URL"

# 启动Electron应用
./node_modules/.bin/electron electron/main-simple.js

echo ""
echo "✅ 测试完成" 