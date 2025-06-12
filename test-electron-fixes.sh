#!/bin/bash

echo "🔧 测试Electron翻译窗口修复效果"
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

# 检查依赖
echo "3. 检查Python依赖..."
python3 -c "import fastapi, uvicorn, websockets" 2>/dev/null || {
    echo "❌ 缺少Python依赖，正在安装..."
    pip install fastapi uvicorn[standard] websockets
}

# 启动API服务器
echo "4. 启动API服务器..."
cd serve
python3 api-local.py --port 8001 &
API_PID=$!
cd ..

# 等待服务器启动
echo "5. 等待API服务器启动..."
sleep 3

# 测试API连接
echo "6. 测试API连接..."
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✅ API服务器运行正常"
else
    echo "❌ API服务器启动失败"
    kill $API_PID 2>/dev/null
    exit 1
fi

# 启动Electron应用
echo "7. 启动Electron应用..."
echo ""
echo "📝 测试步骤:"
echo "   1. 点击 'Load Model' 按钮"
echo "   2. 查看独立翻译窗口是否立即出现"
echo "   3. 观察状态是否从 'Ready' 变为 'Loading model...'"
echo "   4. 等待模型加载完成，状态应变为 'Model loaded successfully'"
echo "   5. 点击麦克风按钮测试语音输入"
echo "   6. 确认翻译窗口不会白屏，能正常显示翻译结果"
echo ""
echo "🔍 如果遇到问题，请检查开发者工具的控制台输出"
echo ""

npm run electron-dev

# 清理
echo "8. 清理进程..."
kill $API_PID 2>/dev/null
echo "✅ 测试完成" 