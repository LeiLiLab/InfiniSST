#!/bin/bash

echo "=== Electron音频处理修复测试 ==="
echo ""

# 检查后端服务
if ! curl -s http://localhost:8001/ > /dev/null; then
    echo "❌ 后端服务未运行，请先启动：cd serve && python api-local.py"
    exit 1
fi

echo "✅ 后端服务正常"
echo ""

echo "🔧 修复内容："
echo "1. 创建Electron专用音频处理器"
echo "2. 使用异步队列处理音频数据"
echo "3. 最小化缓冲区大小 (512)"
echo "4. 限制处理频率 (100ms间隔)"
echo "5. 移除destination连接以避免反馈"
echo ""

echo "🧪 测试步骤："
echo "1. 加载模型"
echo "2. 开始录音"
echo "3. 说话10-15秒"
echo "4. 观察是否还会崩溃"
echo "5. 检查翻译是否正常输出"
echo ""

# 启动Electron
cd electron
echo "🚀 启动Electron (使用音频修复)..."
npm start &
ELECTRON_PID=$!

echo "Electron PID: $ELECTRON_PID"
echo ""
echo "请按照上述步骤测试，观察是否还会出现白屏崩溃"
echo "完成后按Ctrl+C退出"

# 等待用户中断
trap "echo ''; echo '停止测试...'; kill $ELECTRON_PID 2>/dev/null; exit 0" INT

# 监控进程
while kill -0 $ELECTRON_PID 2>/dev/null; do
    sleep 1
done

echo "Electron进程已退出" 