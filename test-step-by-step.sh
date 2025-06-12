#!/bin/bash

echo "=== 逐步音频测试 ==="
echo ""

# 检查后端服务
if ! curl -s http://localhost:8001/ > /dev/null; then
    echo "❌ 后端服务未运行，请先启动：cd serve && python api.py"
    exit 1
fi

echo "✅ 后端服务正常"
echo ""

echo "📋 测试步骤："
echo "1. 基础麦克风权限测试"
echo "2. AudioContext创建测试"  
echo "3. MediaRecorder API测试"
echo "4. Electron环境测试"
echo ""

echo "🔗 测试链接："
echo "1. 基础麦克风: http://localhost:8001/test-mic-basic.html"
echo "2. AudioContext: http://localhost:8001/test-audiocontext.html"
echo "3. MediaRecorder: http://localhost:8001/test-mediarecorder.html"
echo ""

echo "请按顺序在浏览器中测试这些链接，确认每一步都正常工作"
echo "然后我们再在Electron中测试"

# 将测试文件复制到静态目录
cp test-mic-basic.html serve/static/
cp test-audiocontext.html serve/static/
cp test-mediarecorder.html serve/static/

echo ""
echo "✅ 测试文件已复制到 serve/static/ 目录"
echo "现在可以通过上述链接访问测试页面" 