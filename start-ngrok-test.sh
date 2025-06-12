#!/bin/bash

echo "=== InfiniSST Ngrok Remote Testing Setup ==="
echo ""

# 检查ngrok是否安装
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Please install ngrok first:"
    echo "   brew install ngrok/ngrok/ngrok"
    exit 1
fi

# 检查后端服务器是否运行
echo "🔍 Checking if backend server is running on port 8001..."
if ! curl -s http://localhost:8001 > /dev/null; then
    echo "❌ Backend server not running. Starting backend server..."
    
    # 启动后端服务器
    echo "📡 Starting backend server..."
    cd serve
    source env/bin/activate
    python api.py --host 0.0.0.0 --port 8001 &
    SERVER_PID=$!
    cd ..
    
    # 等待服务器启动
    echo "⏳ Waiting for server to start..."
    sleep 5
    
    # 再次检查
    if curl -s http://localhost:8001 > /dev/null; then
        echo "✅ Backend server started successfully"
    else
        echo "❌ Failed to start backend server"
        exit 1
    fi
else
    echo "✅ Backend server is already running"
fi

echo ""
echo "🌐 Starting ngrok tunnel..."
echo "📝 This will create a public URL for your local server"
echo ""

# 启动ngrok
echo "🚀 Starting ngrok on port 8001..."
echo "📋 Copy the HTTPS URL from ngrok and use it to access your app remotely"
echo ""
echo "⚠️  Important notes:"
echo "   - Use the HTTPS URL (not HTTP) for better security"
echo "   - The URL will change each time you restart ngrok (unless you have a paid plan)"
echo "   - Share this URL with others to let them test your app"
echo ""
echo "🛑 Press Ctrl+C to stop ngrok and the server"
echo ""

# 启动ngrok（这会阻塞直到用户按Ctrl+C）
ngrok http 8001

# 清理（当用户按Ctrl+C时执行）
echo ""
echo "🧹 Cleaning up..."
if [ ! -z "$SERVER_PID" ]; then
    echo "🛑 Stopping backend server..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
fi

echo "✅ Cleanup completed!" 