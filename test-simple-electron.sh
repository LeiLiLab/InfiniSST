#!/bin/bash

echo "=== 简化Electron音频稳定性测试 ==="
echo ""

# 检查后端服务
if ! curl -s http://localhost:8001/ > /dev/null; then
    echo "❌ 后端服务未运行，请先启动：cd serve && python api.py"
    exit 1
fi

echo "✅ 后端服务正常"
echo ""
echo "🎯 测试重点：音频处理稳定性"
echo "📋 测试步骤："
echo "1. 加载模型"
echo "2. 开始录音"
echo "3. 说话5-10秒"
echo "4. 停止录音"
echo "5. 检查是否崩溃"
echo ""

# 启动Electron
cd electron
echo "🚀 启动Electron..."
npm start &
ELECTRON_PID=$!

echo "Electron PID: $ELECTRON_PID"
echo ""
echo "请按照上述步骤测试，完成后按Ctrl+C退出"

# 等待用户中断
trap "echo ''; echo '停止测试...'; kill $ELECTRON_PID 2>/dev/null; exit 0" INT

# 监控进程
while kill -0 $ELECTRON_PID 2>/dev/null; do
    sleep 1
done

echo "Electron进程已退出" 