#!/bin/bash

echo "=== 静态文件处理修复测试 ==="
echo ""

# 检查ngrok URL
if [ -z "$1" ]; then
    echo "❌ 请提供ngrok URL"
    echo "用法: $0 <ngrok-url>"
    exit 1
fi

REMOTE_URL="$1"
echo "🌐 远程URL: $REMOTE_URL"

# 测试连接
echo "🔍 测试连接..."
if curl -s --head "$REMOTE_URL" | head -n 1 | grep -q "200 OK"; then
    echo "✅ 远程服务器可访问"
else
    echo "❌ 远程服务器不可访问"
    exit 1
fi

echo ""
echo "🚀 启动Electron远程连接测试..."
echo "观察是否还有 electronAPI 重复声明错误"
echo ""

# 启动Electron应用并捕获输出
export ELECTRON_IS_DEV=true
export REMOTE_SERVER_URL="$REMOTE_URL"

timeout 30s ./node_modules/.bin/electron electron/main-simple.js 2>&1 | tee static-fix-test.log

echo ""
echo "✅ 测试完成"

# 检查是否还有electronAPI错误
if grep -q "electronAPI.*already.*declared" static-fix-test.log; then
    echo "❌ 仍然存在 electronAPI 重复声明错误"
    echo "错误详情："
    grep "electronAPI.*already.*declared" static-fix-test.log
    exit 1
else
    echo "✅ 没有发现 electronAPI 重复声明错误"
    echo "🎉 静态文件处理修复成功！"
fi 