#!/bin/bash

echo "=== 远程连接测试 ==="
echo ""

# 检查ngrok URL
if [ -z "$1" ]; then
    echo "❌ 请提供ngrok URL"
    echo "用法: $0 <ngrok-url>"
    echo "例如: $0 https://abc123.ngrok.io"
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
echo "🚀 启动Electron远程连接..."
echo "注意观察控制台中的错误信息"

# 设置环境变量并启动
REMOTE_SERVER_URL="$REMOTE_URL" ELECTRON_IS_DEV=true electron electron/main-simple.js 