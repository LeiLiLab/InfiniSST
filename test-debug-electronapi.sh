#!/bin/bash

echo "=== ElectronAPI 错误调试 ==="
echo ""

# 检查ngrok URL
if [ -z "$1" ]; then
    echo "❌ 请提供ngrok URL"
    echo "用法: $0 <ngrok-url>"
    exit 1
fi

REMOTE_URL="$1"
echo "🌐 远程URL: $REMOTE_URL"

# 下载远程页面并保存
echo "📥 下载远程页面内容..."
curl -s "$REMOTE_URL" > remote-page.html

# 检查第642行附近的内容
echo "🔍 检查第642行附近的内容:"
echo "--- 第640-650行 ---"
sed -n '640,650p' remote-page.html | nl -v640

echo ""
echo "🔍 搜索所有可能的electronAPI声明:"
grep -n "electronAPI" remote-page.html | head -10

echo ""
echo "🔍 搜索可能的变量声明:"
grep -n "const.*=\|let.*=\|var.*=" remote-page.html | grep -i electron

echo ""
echo "🔍 检查是否有重复的script标签:"
grep -n "<script>" remote-page.html

echo ""
echo "🚀 启动Electron并捕获详细错误..."

# 创建临时的preload文件，添加更多调试信息
cp electron/preload.js electron/preload-debug.js

# 在preload文件中添加更多调试
cat >> electron/preload-debug.js << 'EOF'

// 额外的调试信息
console.log('=== Additional Debug Info ===');
console.log('Global electronAPI exists:', typeof window !== 'undefined' && 'electronAPI' in window);
console.log('Window object keys:', typeof window !== 'undefined' ? Object.keys(window).filter(k => k.includes('electron')) : 'window not available');

// 监听错误事件
if (typeof window !== 'undefined') {
  window.addEventListener('error', (event) => {
    console.error('Window error caught:', event.error);
    console.error('Error message:', event.message);
    console.error('Error filename:', event.filename);
    console.error('Error line:', event.lineno);
    console.error('Error column:', event.colno);
  });
}
EOF

# 修改main-simple.js使用调试版本的preload
sed 's/preload.js/preload-debug.js/g' electron/main-simple.js > electron/main-simple-debug.js

echo "启动命令: REMOTE_SERVER_URL=$REMOTE_URL ELECTRON_IS_DEV=true electron electron/main-simple-debug.js"
echo ""

# 设置环境变量并启动
export ELECTRON_IS_DEV=true
export REMOTE_SERVER_URL="$REMOTE_URL"

# 启动Electron应用
./node_modules/.bin/electron electron/main-simple-debug.js 2>&1 | tee electronapi-debug.log

echo ""
echo "✅ 调试完成"
echo "📄 远程页面已保存到: remote-page.html"
echo "📄 调试日志已保存到: electronapi-debug.log"

# 清理临时文件
rm -f electron/preload-debug.js electron/main-simple-debug.js 