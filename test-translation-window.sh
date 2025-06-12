#!/bin/bash

echo "🔧 测试翻译窗口启动和状态更新"
echo "================================"

# 激活虚拟环境
echo "1. 激活虚拟环境..."
source env/bin/activate || {
    echo "❌ 错误: 无法激活虚拟环境"
    exit 1
}

# 停止现有进程
echo "2. 停止现有进程..."
pkill -f "api-local.py" 2>/dev/null || true
pkill -f "electron" 2>/dev/null || true
sleep 2

# 启动API服务器
echo "3. 启动API服务器..."
cd serve
python3 api-local.py --port 8001 &
API_PID=$!
cd ..

# 等待服务器启动
echo "4. 等待API服务器启动..."
sleep 3

# 测试API连接
echo "5. 测试API连接..."
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✅ API服务器运行正常"
else
    echo "❌ API服务器启动失败"
    kill $API_PID 2>/dev/null
    exit 1
fi

# 启动Electron应用
echo "6. 启动Electron应用..."
echo ""
echo "🎯 重点测试项目:"
echo "   1. 应用启动后独立翻译窗口应自动显示"
echo "   2. 翻译窗口标题应显示 '翻译窗口已准备就绪，请加载模型开始翻译'"
echo "   3. 点击 'Load Model' 按钮"
echo "   4. 观察翻译窗口状态是否正确更新："
echo "      - Loading model... (processing状态)"
echo "      - Model loaded successfully (ready状态)"
echo "   5. 测试麦克风功能是否正常，窗口不白屏"
echo ""
echo "🔍 调试信息："
echo "   - 主进程日志会显示IPC通信"
echo "   - 翻译窗口控制台会显示接收到的状态更新"
echo "   - 如果状态不更新，检查控制台错误信息"
echo ""

# 启动Electron并显示详细日志
ELECTRON_ENABLE_LOGGING=true npm run electron-dev

# 清理
echo "7. 清理进程..."
kill $API_PID 2>/dev/null
echo "✅ 测试完成" 