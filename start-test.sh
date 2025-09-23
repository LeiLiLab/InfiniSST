#!/bin/bash

echo "🚀 Starting InfiniSST Local Test Environment"
echo "=========================================="

# 检查是否已经有API服务器在运行
if curl -s http://localhost:8001 > /dev/null; then
    echo "✅ API server is already running on port 8001"
else
    echo "🔄 Starting local API server..."
    python3 serve/api-local.py --host 0.0.0.0 --port 8001 &
    API_PID=$!
    echo "📝 API server started with PID: $API_PID"
    
    # 等待API服务器启动
    echo "⏳ Waiting for API server to start..."
    for i in {1..10}; do
        if curl -s http://localhost:8001 > /dev/null; then
            echo "✅ API server is ready!"
            break
        fi
        echo "   Attempt $i/10..."
        sleep 1
    done
fi

echo ""
echo "🖥️  Starting Electron debug application..."
npm run electron-simple

# 清理：如果我们启动了API服务器，关闭它
if [ ! -z "$API_PID" ]; then
    echo ""
    echo "🧹 Cleaning up..."
    kill $API_PID 2>/dev/null
    echo "✅ API server stopped"
fi

echo "👋 Test session ended" 