#!/bin/bash

# Simple Ray API Startup Script
# 简化的Ray API启动脚本

echo "🚀 Starting Ray-based InfiniSST API Server..."

# 确保在正确的目录
PROJECT_ROOT="/home/jiaxuanluo/new-infinisst"
cd "$PROJECT_ROOT"

echo "[INFO] Working directory: $(pwd)"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_DISABLE_IMPORT_WARNING=1

echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] PYTHONPATH: $PYTHONPATH"

# 创建必要的目录
mkdir -p logs
mkdir -p serve/static

# 检查必要文件
if [ ! -f "serve/ray_api.py" ]; then
    echo "❌ [ERROR] ray_api.py not found in serve/ directory"
    exit 1
fi

if [ ! -f "serve/ray_config.py" ]; then
    echo "❌ [ERROR] ray_config.py not found in serve/ directory"
    exit 1
fi

# 激活conda环境
echo "[INFO] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 检查Ray安装
echo "[INFO] Checking Ray installation..."
python -c "import ray; print(f'Ray version: {ray.__version__}')" || {
    echo "[ERROR] Ray not available. Installing..."
    pip install ray[default]
}

# 停止现有Ray集群
echo "[INFO] Stopping existing Ray cluster..."
ray stop --force 2>/dev/null || true

# 清理端口
echo "[INFO] Cleaning up ports..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8265 | xargs kill -9 2>/dev/null || true

# 创建默认配置文件
if [ ! -f "serve/ray_config.json" ]; then
    echo "[INFO] Creating default Ray configuration..."
    python serve/ray_config.py --create-default --config serve/ray_config.json || {
        echo "[WARNING] Failed to create config, continuing anyway..."
    }
fi

# 启动Ray集群
echo "[INFO] Starting Ray cluster..."
ray start --head \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-cpus=8 \
    --num-gpus=2 \
    --disable-usage-stats || {
    echo "[ERROR] Failed to start Ray cluster"
    exit 1
}

# 等待Ray集群启动
sleep 3

# 验证Ray集群
ray status || {
    echo "[ERROR] Ray cluster not healthy"
    ray stop --force
    exit 1
}

echo "[INFO] Ray cluster started successfully"

# 启动API服务器
echo "[INFO] Starting Ray API server..."
export MAX_BATCH_SIZE=32
export BATCH_TIMEOUT_MS=100.0

python serve/ray_api.py \
    --host 0.0.0.0 \
    --port 8000 \
    --max-batch-size $MAX_BATCH_SIZE \
    --batch-timeout-ms $BATCH_TIMEOUT_MS \
    --enable-dynamic-batching &

SERVER_PID=$!

# 等待服务器启动
echo "[INFO] Waiting for API server to start..."
for i in {1..30}; do
    if lsof -i:8000 &>/dev/null; then
        echo "✅ [INFO] Ray API server started successfully on port 8000"
        break
    fi
    echo "⏳ Waiting for API server to bind on port 8000... (attempt $i/30)"
    sleep 2
done

# 检查服务器状态
if lsof -i:8000 &>/dev/null; then
    echo ""
    echo "🎉 Ray-based InfiniSST API Server is running!"
    echo "📊 Ray Dashboard: http://localhost:8265"
    echo "🌐 API Server: http://localhost:8000"
    echo ""
    echo "📋 Useful commands:"
    echo "  - Test API: python serve/test_ray_api.py"
    echo "  - Health check: curl http://localhost:8000/health"
    echo "  - Ray stats: curl http://localhost:8000/ray/stats"
    echo "  - Stop server: kill $SERVER_PID && ray stop --force"
    echo ""
    
    # 运行基本测试
    echo "[INFO] Running basic API test..."
    sleep 3
    python serve/test_ray_api.py --url http://localhost:8000 || echo "[WARNING] API test failed, but server is running"
    
    echo ""
    echo "🔄 Server is running in background (PID: $SERVER_PID)"
    echo "📝 Press Ctrl+C to stop the server"
    
    # 监控服务器
    trap "echo 'Stopping server...'; kill $SERVER_PID 2>/dev/null; ray stop --force; exit 0" SIGINT SIGTERM
    
    while kill -0 $SERVER_PID 2>/dev/null; do
        sleep 10
        echo "[INFO] $(date): Server still running (PID: $SERVER_PID)"
    done
    
    echo "[INFO] Server process ended"
else
    echo "❌ [ERROR] Failed to start Ray API server"
    kill $SERVER_PID 2>/dev/null || true
    ray stop --force
    exit 1
fi 