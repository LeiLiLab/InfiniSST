#!/bin/bash

echo "=== InfiniSST Electron 完整测试脚本 ==="
echo

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 输出带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 检查命令是否存在
check_command() {
    if command -v $1 >/dev/null 2>&1; then
        print_message $GREEN "✓ $1 已安装"
        return 0
    else
        print_message $RED "✗ $1 未安装"
        return 1
    fi
}

# 检查是否在正确的目录
if [ ! -f "package.json" ]; then
    print_message $RED "错误: 必须在项目根目录运行此脚本"
    exit 1
fi

print_message $BLUE "1. 检查系统环境..."
echo

# 检查必要的命令
MISSING_DEPS=false

if ! check_command "python3"; then
    print_message $YELLOW "提示: 请安装 Python 3"
    MISSING_DEPS=true
fi

if ! check_command "npm"; then
    print_message $YELLOW "提示: 请安装 Node.js 和 npm"
    MISSING_DEPS=true
fi

if ! check_command "curl"; then
    print_message $YELLOW "提示: 请安装 curl"
    MISSING_DEPS=true
fi

if [ "$MISSING_DEPS" = true ]; then
    print_message $RED "请安装缺失的依赖后重新运行"
    exit 1
fi

print_message $BLUE "2. 检查Python依赖..."
echo

# 检查Python依赖
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    print_message $YELLOW "正在安装Python依赖..."
    pip3 install fastapi uvicorn python-multipart
fi

print_message $BLUE "3. 检查Node.js依赖..."
echo

# 检查Node.js依赖
if [ ! -d "node_modules" ]; then
    print_message $YELLOW "正在安装Node.js依赖..."
    npm install
fi

print_message $BLUE "4. 启动本地API服务器..."
echo

# 启动API服务器
cd serve
print_message $GREEN "启动API服务器在端口8001..."
python3 api-local.py --port 8001 &
API_PID=$!
cd ..

# 等待服务器启动
print_message $YELLOW "等待API服务器启动..."
sleep 3

# 检查服务器是否运行
MAX_ATTEMPTS=10
ATTEMPT=1
SERVER_RUNNING=false

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8001/ > /dev/null 2>&1; then
        SERVER_RUNNING=true
        break
    fi
    print_message $YELLOW "尝试 $ATTEMPT/$MAX_ATTEMPTS: 等待服务器响应..."
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
done

if [ "$SERVER_RUNNING" = true ]; then
    print_message $GREEN "✓ API服务器运行正常 (http://localhost:8001)"
else
    print_message $RED "✗ API服务器启动失败"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo
print_message $BLUE "5. 测试API端点..."
echo

# 测试关键API端点
echo "测试 /init 端点:"
INIT_RESPONSE=$(curl -s -X POST "http://localhost:8001/init?agent_type=InfiniSST&language_pair=English%20-%3E%20Chinese&latency_multiplier=2&client_id=test_client")
echo "响应: $INIT_RESPONSE"

if echo "$INIT_RESPONSE" | grep -q "session_id"; then
    print_message $GREEN "✓ /init 端点正常工作"
    
    # 提取session_id用于测试WebSocket
    SESSION_ID=$(echo "$INIT_RESPONSE" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
    echo "会话ID: $SESSION_ID"
    
    # 测试WebSocket连接（使用简单的模拟）
    echo
    echo "测试 WebSocket 端点:"
    if curl -s "http://localhost:8001/wss/$SESSION_ID" | grep -q "404"; then
        print_message $YELLOW "⚠ WebSocket端点返回404，这是正常的（需要WebSocket连接）"
    else
        print_message $GREEN "✓ WebSocket端点可访问"
    fi
else
    print_message $YELLOW "⚠ /init 端点响应异常，但服务器正在运行"
fi

echo
print_message $BLUE "6. 启动Electron应用..."
echo

# 创建临时启动脚本
cat > temp_electron_start.js << 'EOF'
const { app, BrowserWindow } = require('electron');
const path = require('path');

app.whenReady().then(() => {
    const win = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'electron', 'preload.js')
        }
    });

    // 加载本地服务器
    win.loadURL('http://localhost:8001');
    
    // 打开开发者工具
    win.webContents.openDevTools();
    
    console.log('Electron app started with local API server');
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
EOF

# 启动Electron
print_message $GREEN "启动Electron应用..."
npm start &
ELECTRON_PID=$!

echo
print_message $GREEN "=== 测试环境已准备就绪 ==="
echo
print_message $BLUE "API服务器: http://localhost:8001"
print_message $BLUE "Electron应用已启动"
echo
print_message $YELLOW "测试步骤:"
echo "1. 在Electron应用中点击 'Load Model'"
echo "2. 选择音频文件或使用麦克风"
echo "3. 测试翻译功能和独立翻译窗口"
echo "4. 测试重置翻译功能"
echo
print_message $YELLOW "按 Ctrl+C 停止所有服务"

# 等待用户中断
trap 'echo; print_message $YELLOW "正在关闭服务..."; kill $API_PID $ELECTRON_PID 2>/dev/null; rm -f temp_electron_start.js; print_message $GREEN "所有服务已关闭"; exit 0' INT

# 保持脚本运行
while true; do
    sleep 1
    
    # 检查进程是否还在运行
    if ! kill -0 $API_PID 2>/dev/null; then
        print_message $RED "API服务器进程已停止"
        break
    fi
    
    if ! kill -0 $ELECTRON_PID 2>/dev/null; then
        print_message $YELLOW "Electron应用已关闭"
        break
    fi
done

# 清理
kill $API_PID $ELECTRON_PID 2>/dev/null
rm -f temp_electron_start.js
print_message $GREEN "测试完成" 