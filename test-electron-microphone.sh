#!/bin/bash

echo "=== Electron麦克风权限测试 ==="
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
echo "1. 验证Electron麦克风权限配置"
echo "2. 测试媒体设备访问"
echo "3. 确认音频处理流程"
echo ""

echo "🚀 启动Electron麦克风权限测试..."

# 启动Electron应用，重点关注权限相关日志
export ELECTRON_IS_DEV=true
unset REMOTE_SERVER_URL

timeout 60s ./node_modules/.bin/electron electron/main-simple.js \
    --enable-logging \
    --log-level=0 \
    2>&1 | tee electron-microphone-test.log

echo ""
echo "✅ Electron麦克风测试完成"

echo ""
echo "=== 权限测试结果分析 ==="

echo ""
echo "--- 媒体权限请求 ---"
if grep -q "Media access permission requested" electron-microphone-test.log; then
    echo "✅ 检测到媒体权限请求"
    grep -n "Media access permission\|Granting.*permission" electron-microphone-test.log
else
    echo "❌ 未检测到媒体权限请求"
    echo "可能原因：页面未尝试访问麦克风，或权限处理有问题"
fi

echo ""
echo "--- 麦克风访问结果 ---"
if grep -q "Microphone access granted" electron-microphone-test.log; then
    echo "✅ 麦克风访问成功"
    grep -n "Microphone access granted" electron-microphone-test.log
elif grep -q "Permission denied\|NotAllowedError" electron-microphone-test.log; then
    echo "❌ 麦克风访问被拒绝"
    grep -n "Permission denied\|NotAllowedError" electron-microphone-test.log
else
    echo "⚠️  未检测到明确的麦克风访问结果"
fi

echo ""
echo "--- 音频处理状态 ---"
if grep -q "AudioContext created" electron-microphone-test.log; then
    echo "✅ AudioContext创建成功"
else
    echo "❌ AudioContext创建失败"
fi

if grep -q "Audio nodes connected" electron-microphone-test.log; then
    echo "✅ 音频节点连接成功"
else
    echo "❌ 音频节点连接失败"
fi

echo ""
echo "--- 错误信息 ---"
echo "权限相关错误："
grep -n "Permission\|NotAllowed\|Denied" electron-microphone-test.log | head -3

echo ""
echo "音频相关错误："
grep -n "AudioContext.*error\|getUserMedia.*error" electron-microphone-test.log | head -3

echo ""
echo "📄 完整测试日志已保存到: electron-microphone-test.log"
echo ""
echo "🔍 下一步建议："
echo "1. 如果权限请求未触发 → 检查页面是否正确调用getUserMedia"
echo "2. 如果权限被拒绝 → 检查系统麦克风权限设置"
echo "3. 如果AudioContext失败 → 检查浏览器兼容性和安全策略" 