#!/bin/bash

echo "🔧 InfiniSST 快速测试 - 验证修复"
echo

# 激活虚拟环境
echo "0. 激活虚拟环境..."
source env/bin/activate

# 停止可能运行的进程
pkill -f "api-local.py" 2>/dev/null || true
sleep 1

# 启动API服务器
echo "1. 启动API服务器..."
cd serve
python3 api-local.py --port 8001 > /tmp/api.log 2>&1 &
API_PID=$!
cd ..

sleep 3

# 测试API
echo "2. 测试API端点..."
if curl -s http://localhost:8001/ > /dev/null; then
    echo "   ✅ API服务器正常"
    
    # 测试初始化
    RESPONSE=$(curl -s -X POST "http://localhost:8001/init?agent_type=InfiniSST&language_pair=English%20-%3E%20Chinese&latency_multiplier=2&client_id=test")
    if echo "$RESPONSE" | grep -q "session_id"; then
        echo "   ✅ 会话初始化正常"
        echo "   📝 响应: $RESPONSE"
    else
        echo "   ❌ 会话初始化失败"
    fi
else
    echo "   ❌ API服务器无法访问"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo
echo "3. 启动Electron应用..."
echo "   🎯 现在应该看到："
echo "   - 点击'Load Model'时立即显示翻译窗口"
echo "   - 可以看到模型加载状态反馈"
echo "   - 麦克风功能应该正常工作（无404错误）"
echo

# 启动Electron
npm run electron-dev &
ELECTRON_PID=$!

echo "按 Ctrl+C 停止测试"

# 等待用户中断
trap 'echo; echo "正在停止..."; kill $API_PID $ELECTRON_PID 2>/dev/null; exit 0' INT

wait $ELECTRON_PID

# 清理
kill $API_PID 2>/dev/null
echo "测试完成" 