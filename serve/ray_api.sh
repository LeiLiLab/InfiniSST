#!/bin/bash
#SBATCH --job-name=ray_infinisst_api
#SBATCH --output=logs/ray_infinisst_api_%j.log
#SBATCH --error=logs/ray_infinisst_api_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=taurus
#SBATCH --mem=128GB

# 确保logs目录存在
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 设置PYTHONPATH
export PYTHONPATH=/home/jiaxuanluo/new-infinisst

# 检查并创建必要的目录
echo "[INFO] Checking and creating necessary directories..."

# 检查当前工作目录并设置正确的路径
if [ -f "ray_api.py" ]; then
    echo "[INFO] Executing from serve directory"
    EXECUTION_DIR="serve"
elif [ -f "serve/ray_api.py" ]; then
    echo "[INFO] Executing from project root directory"
    EXECUTION_DIR="root"
else
    echo "[INFO] ray_api.py not found, assuming project root..."
    cd /home/jiaxuanluo/new-infinisst
    EXECUTION_DIR="root"
fi

# 创建数据目录
mkdir -p /mnt/data/jiaxuanluo/infinisst/ray_temp
mkdir -p /mnt/data/jiaxuanluo/infinisst/static

# 创建本地logs目录
mkdir -p logs

# 创建静态文件目录软链接
if [ "$EXECUTION_DIR" = "serve" ]; then
    # 从serve目录执行，创建static目录软链接
    if [ ! -d "static" ] && [ ! -L "static" ]; then
        echo "[INFO] Creating static directory symlink..."
        ln -sf /mnt/data/jiaxuanluo/infinisst/static static
    fi
    # 如果父目录有static文件，复制到数据目录
    if [ -d "../static" ] && [ ! "$(ls -A /mnt/data/jiaxuanluo/infinisst/static 2>/dev/null)" ]; then
        echo "[INFO] Copying static files from parent directory..."
        cp -r ../static/* /mnt/data/jiaxuanluo/infinisst/static/ 2>/dev/null || echo "[INFO] No static files to copy from parent"
    fi
else
    # 从项目根目录执行，创建serve/static目录软链接
    if [ ! -d "serve/static" ] && [ ! -L "serve/static" ]; then
        echo "[INFO] Creating serve/static directory symlink..."
        ln -sf /mnt/data/jiaxuanluo/infinisst/static serve/static
    fi
    # 如果有static文件，复制到数据目录
    if [ -d "static" ] && [ ! "$(ls -A /mnt/data/jiaxuanluo/infinisst/static 2>/dev/null)" ]; then
        echo "[INFO] Copying static files to data directory..."
        cp -r static/* /mnt/data/jiaxuanluo/infinisst/static/ 2>/dev/null || echo "[INFO] No static files to copy"
    fi
fi

# 检查ray_config.py文件是否存在
if [ -f "serve/ray_config.py" ]; then
    # 从项目根目录执行
    RAY_CONFIG_PATH="serve/ray_config.py"
    echo "[INFO] Found ray_config.py at serve/ray_config.py"
elif [ -f "ray_config.py" ]; then
    # 从serve目录执行
    RAY_CONFIG_PATH="ray_config.py"
    echo "[INFO] Found ray_config.py in current directory"
else
    echo "[ERROR] ray_config.py not found"
    echo "[INFO] Current directory: $(pwd)"
    echo "[INFO] Files in current directory:"
    ls -la | head -10
    if [ -d "serve" ]; then
        echo "[INFO] Files in serve/ directory:"
        ls -la serve/ | head -10
    fi
    exit 1
fi

# 显示SLURM分配的GPU信息
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] SLURM GPU devices: $CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM GPU count: $SLURM_GPUS"
echo "[INFO] SLURM GPU list: $SLURM_GPU_BIND"

# CUDA_VISIBLE_DEVICES由SLURM自动设置
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "[WARNING] No GPU allocated by SLURM, this might be a direct run"
    echo "[INFO] Using default GPUs for testing: 4,5"
    export CUDA_VISIBLE_DEVICES="4,5"
else
    echo "[INFO] SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"
fi

# 显示最终的GPU配置
echo "[INFO] Final CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L | head -4

# 清理之前的进程
echo "[INFO] Killing existing ngrok..."
pkill -f ngrok || true

echo "[INFO] Killing any process using port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

echo "[INFO] Cleaning up any existing Ray processes for current user..."
ray stop --force 2>/dev/null || true

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 检查Ray安装
echo "[INFO] Checking Ray installation..."
python -c "import ray; print(f'Ray version: {ray.__version__}')" || {
    echo "[ERROR] Ray not installed. Installing Ray..."
    pip install ray[default]
}

# Ray配置环境变量
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_LOG_TO_DRIVER=1
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0

# 批处理配置
export MAX_BATCH_SIZE=32
export BATCH_TIMEOUT_MS=100.0
export ENABLE_DYNAMIC_BATCHING=true

# 创建默认配置文件（如果不存在）
if [ -f "serve/ray_config.json" ] || [ -f "ray_config.json" ]; then
    echo "[INFO] Ray configuration file already exists"
else
    echo "[INFO] Creating default Ray configuration..."
    
    # 根据ray_config.py的位置创建配置文件
    if [ -f "serve/ray_config.py" ]; then
        # 从项目根目录运行
        python serve/ray_config.py --create-default --config serve/ray_config.json
    elif [ -f "ray_config.py" ]; then
        # 如果我们在serve目录中
        python ray_config.py --create-default --config ray_config.json
    else
        echo "[WARNING] ray_config.py not found, creating basic config directory"
        mkdir -p serve 2>/dev/null || true
    fi
fi

echo "[INFO] Starting Ray-based InfiniSST API server..."
echo "[INFO] Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "[INFO] Ray configuration:"
echo "  - Max batch size: $MAX_BATCH_SIZE"
echo "  - Batch timeout: $BATCH_TIMEOUT_MS ms"
echo "  - Dynamic batching: $ENABLE_DYNAMIC_BATCHING"

# 启动独立的Ray集群（使用SLURM分配的资源）
echo "[INFO] Starting independent Ray cluster with SLURM allocated resources..."
echo "[INFO] SLURM allocated GPUs: $CUDA_VISIBLE_DEVICES"

# 停止任何现有的Ray进程（只针对当前用户）
ray stop --force 2>/dev/null || true

# 启动新的Ray集群
ray start --head \
    --port=6380 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8266 \
    --disable-usage-stats \
    --temp-dir=/mnt/data/jiaxuanluo/infinisst/ray_temp || {
    echo "[ERROR] Failed to start Ray cluster"
    echo "[INFO] Available ports:"
    lsof -i:6380 || echo "Port 6380 is free"
    lsof -i:8266 || echo "Port 8266 is free"
    exit 1
}

# 等待Ray集群启动
sleep 5

# 验证Ray集群状态
ray status || {
    echo "[ERROR] Ray cluster failed to start properly"
    ray stop --force
    exit 1
}

echo "[INFO] Ray cluster started successfully"

# 启动Ray-based API服务器
echo "[INFO] Starting Ray-based API server..."

# 根据当前位置选择正确的路径
if [ -f "serve/ray_api.py" ]; then
    # 从项目根目录启动
    echo "[INFO] Starting from project root directory"
    RAY_API_PATH="serve/ray_api.py"
    PYTHONUNBUFFERED=1 python serve/ray_api.py \
        --host 0.0.0.0 \
        --port 8000 \
        --max-batch-size $MAX_BATCH_SIZE \
        --batch-timeout-ms $BATCH_TIMEOUT_MS \
        --enable-dynamic-batching &
elif [ -f "ray_api.py" ]; then
    # 从serve目录启动
    echo "[INFO] Starting from serve directory"
    RAY_API_PATH="ray_api.py"
    PYTHONUNBUFFERED=1 python ray_api.py \
        --host 0.0.0.0 \
        --port 8000 \
        --max-batch-size $MAX_BATCH_SIZE \
        --batch-timeout-ms $BATCH_TIMEOUT_MS \
        --enable-dynamic-batching &
else
    echo "[ERROR] ray_api.py not found in expected locations"
    echo "[INFO] Current directory: $(pwd)"
    echo "[INFO] Files in current directory:"
    ls -la | head -10
    if [ -d "serve" ]; then
        echo "[INFO] Files in serve/ directory:"
        ls -la serve/ | head -10
    fi
    exit 1
fi

SERVER_PID=$!

# 等待端口8000启动
echo "[INFO] Waiting for Ray API server to start..."
for i in {1..60}; do
    if lsof -i:8000 &>/dev/null; then
        echo "[INFO] Ray API server started successfully on port 8000"
        break
    fi
    echo "Waiting for Ray API to bind on port 8000... (attempt $i/60)"
    sleep 2
done

# 检查服务器是否启动成功
if lsof -i:8000 &>/dev/null; then
    echo "[INFO] Ray API Server is running (PID: $SERVER_PID)"
    
    # 测试健康检查
    sleep 5
    echo "[INFO] Testing health check..."
    curl -s http://localhost:8000/health | python3 -m json.tool || echo "[WARNING] Health check failed"
    
    # 测试Ray统计信息
    echo "[INFO] Testing Ray stats..."
    curl -s http://localhost:8000/ray/stats | python3 -m json.tool || echo "[WARNING] Ray stats failed"
    
else
    echo "[ERROR] Ray API Server failed to start"
    kill $SERVER_PID 2>/dev/null || true
    ray stop --force
    exit 1
fi

# 启动 ngrok tunnel
echo "[INFO] Starting ngrok tunnel..."
/mnt/aries/data6/jiaxuanluo/bin/ngrok http --url=infinisst.ngrok.app 8000 &
NGROK_PID=$!

# 等待一下让ngrok启动
sleep 3

# 显示连接信息
echo ""
echo "🎉 Ray-based InfiniSST API Server is running!"
echo "📊 Ray Dashboard: http://localhost:8266"
echo "🌐 API Server: http://localhost:8000"
echo "🔗 Public URL: https://infinisst.ngrok.app"
echo "🎯 SLURM Job ID: $SLURM_JOB_ID"
echo "🎮 Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo ""
echo "📋 Useful commands:"
echo "  - Ray status: ray status"
echo "  - Ray logs: ray logs"
echo "  - API health: curl http://localhost:8000/health"
echo "  - Ray stats: curl http://localhost:8000/ray/stats"
echo ""

# 设置cleanup函数
cleanup() {
    echo ""
    echo "[INFO] Shutting down InfiniSST API server..."
    
    # 停止ngrok
    if [ ! -z "$NGROK_PID" ]; then
        kill $NGROK_PID 2>/dev/null || true
        echo "[INFO] Ngrok stopped"
    fi
    
    # 停止API服务器
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        echo "[INFO] API server stopped"
    fi
    
    # 停止我们的Ray集群
    ray stop --force
    echo "[INFO] Ray cluster stopped"
    
    # 清理临时文件
    rm -rf /mnt/data/jiaxuanluo/infinisst/ray_temp/* 2>/dev/null || true
    
    echo "[INFO] Cleanup completed"
    exit 0
}

# 注册cleanup函数
trap cleanup SIGINT SIGTERM EXIT

# 监控服务器进程
echo "[INFO] Monitoring server processes..."
echo "[INFO] Press Ctrl+C to stop all services"

while true; do
    # 检查API服务器是否还在运行
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[ERROR] API server process died unexpectedly"
        break
    fi
    
    # 检查Ray集群是否健康
    if ! ray status &>/dev/null; then
        echo "[ERROR] Ray cluster became unhealthy"
        break
    fi
    
    # 每30秒显示一次状态
    echo "[INFO] $(date): Services running normally"
    sleep 30
done

# 如果到达这里，说明出现了问题
echo "[ERROR] Service monitoring detected an issue, initiating cleanup..."
cleanup 