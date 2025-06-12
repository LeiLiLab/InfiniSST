#!/bin/bash

echo "=== Electron渲染进程崩溃修复测试 ==="
echo ""

# 检查后端服务
echo "🔍 检查本地后端服务..."
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✅ 本地后端服务正在运行"
else
    echo "❌ 本地后端服务未运行，请先启动：cd serve && python api.py"
    exit 1
fi

echo ""
echo "🎯 测试目标："
echo "1. 验证音频缓冲区优化"
echo "2. 测试渲染进程稳定性"
echo "3. 确认麦克风翻译功能"
echo ""

echo "🚀 启动Electron崩溃修复测试..."
echo "请按以下步骤测试："
echo "1. 点击'Load Model'加载模型"
echo "2. 点击'Record Audio'开始录音"
echo "3. 说话测试翻译功能"
echo "4. 观察是否出现渲染进程崩溃"
echo ""

# 启动Electron并捕获输出
cd electron
npm start 2>&1 | tee ../electron-crash-fix-test.log &
ELECTRON_PID=$!

echo "Electron进程ID: $ELECTRON_PID"
echo "日志文件: electron-crash-fix-test.log"
echo ""
echo "测试运行中... 按Ctrl+C停止测试"

# 等待用户中断
trap "echo ''; echo '停止测试...'; kill $ELECTRON_PID 2>/dev/null; exit 0" INT

# 监控进程
while kill -0 $ELECTRON_PID 2>/dev/null; do
    sleep 1
done

echo "Electron进程已退出" 