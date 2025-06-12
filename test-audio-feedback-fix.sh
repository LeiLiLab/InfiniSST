#!/bin/bash

echo "=== 音频反馈修复测试 ==="
echo ""

# 确保后端服务正在运行
echo "🔍 检查本地后端服务..."
if curl -s "http://localhost:8001" | head -n 1 | grep -q "<!DOCTYPE html"; then
    echo "✅ 本地后端服务正在运行"
else
    echo "❌ 本地后端服务未运行，请先启动 api-local.py"
    exit 1
fi

echo ""
echo "🎯 测试目标："
echo "1. 验证麦克风模式不再导致音频反馈"
echo "2. 确认不再出现 SyncReader::Read 超时错误"
echo "3. 验证渲染进程不再崩溃"
echo ""

echo "🚀 启动本地Electron连接测试..."

# 启动Electron应用，重点观察音频处理
export ELECTRON_IS_DEV=true
unset REMOTE_SERVER_URL

# 使用较长的超时时间，观察是否还会崩溃
timeout 60s ./node_modules/.bin/electron electron/main-simple.js \
    --enable-logging \
    --log-level=0 \
    2>&1 | tee audio-feedback-fix-test.log

echo ""
echo "✅ 测试完成"

echo ""
echo "📊 关键问题分析："

echo ""
echo "--- 音频反馈相关错误 ---"
grep -n -A2 -B2 "SyncReader\|audio glitch\|Renderer process crashed" audio-feedback-fix-test.log || echo "✅ 未发现音频反馈错误"

echo ""
echo "--- 音频连接日志 ---"
grep -n "Audio nodes connected\|microphone mode\|no feedback" audio-feedback-fix-test.log || echo "❌ 未找到音频连接日志"

echo ""
echo "--- WebSocket连接状态 ---"
grep -n "WebSocket connected\|WebSocket closed" audio-feedback-fix-test.log || echo "❌ 未找到WebSocket连接日志"

echo ""
if grep -q "SyncReader::Read timed out" audio-feedback-fix-test.log; then
    echo "❌ 仍然存在音频读取超时问题"
    echo "需要进一步调试"
else
    echo "✅ 音频反馈问题已修复！"
    echo "不再出现 SyncReader::Read 超时错误"
fi

echo ""
echo "📄 完整测试日志已保存到: audio-feedback-fix-test.log" 