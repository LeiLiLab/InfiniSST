#!/bin/bash

echo "=== InfiniSST Electron Remote Connection ==="
echo ""

# 检查是否提供了远程URL
if [ -z "$1" ]; then
    echo "❌ Please provide the ngrok URL as an argument"
    echo ""
    echo "Usage: $0 <ngrok-url>"
    echo "Example: $0 https://abc123.ngrok.io"
    echo ""
    echo "Steps to get ngrok URL:"
    echo "1. Run: ./start-ngrok-test.sh"
    echo "2. Copy the HTTPS URL from ngrok output"
    echo "3. Run: $0 <copied-url>"
    exit 1
fi

REMOTE_URL="$1"

# 验证URL格式
if [[ ! "$REMOTE_URL" =~ ^https?:// ]]; then
    echo "❌ Invalid URL format. Please provide a complete URL starting with http:// or https://"
    echo "Example: https://abc123.ngrok.io"
    exit 1
fi

echo "🌐 Remote server URL: $REMOTE_URL"
echo ""

# 测试远程连接
echo "🔍 Testing connection to remote server..."
if curl -s "$REMOTE_URL" > /dev/null; then
    echo "✅ Remote server is accessible"
else
    echo "❌ Cannot connect to remote server. Please check:"
    echo "   - The URL is correct"
    echo "   - The ngrok tunnel is running"
    echo "   - The backend server is running"
    exit 1
fi

echo ""
echo "🚀 Starting Electron with remote server connection..."
echo "📡 Connecting to: $REMOTE_URL"
echo ""

# 设置环境变量并启动Electron
export ELECTRON_IS_DEV=true
export REMOTE_SERVER_URL="$REMOTE_URL"

# 启动Electron应用
./node_modules/.bin/electron electron/main-simple.js

echo ""
echo "✅ Electron application closed" 