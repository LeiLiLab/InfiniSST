#!/bin/bash

echo "🔧 测试翻译窗口修复效果"
echo "============================="

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
echo "   1. 翻译窗口应该立即显示，没有JavaScript错误"
echo "   2. 翻译窗口控制台应该显示:"
echo "      - 'electronAPI available: true'"
echo "      - 'electronAPI methods: [...]'"
echo "      - 'Setting up translation update listener...'"
echo "      - 'Translation update listener set up successfully'"
echo "      - 'Setting up status update listener...'"
echo "      - 'Status update listener set up successfully'"
echo "   3. 点击 'Load Model' 后，翻译窗口状态应该正确更新"
echo "   4. 主进程日志应该显示 IPC 通信成功"
echo ""
echo "📝 调试步骤:"
echo "   1. 右键点击翻译窗口 → 检查元素"
echo "   2. 查看控制台是否有错误"
echo "   3. 检查 electronAPI 是否正确加载"
echo ""

# 运行Electron应用
npm run electron-dev

# 清理
echo ""
echo "7. 清理进程..."
kill $API_PID 2>/dev/null || true
pkill -f "electron" 2>/dev/null || true

echo "✅ 测试完成" 