#!/bin/bash

echo "[INFO] Starting comprehensive Ray cleanup..."

# 1. 强制停止Ray集群
echo "[INFO] Force stopping Ray cluster..."
ray stop --force 2>/dev/null || true

# 2. 杀死所有Ray相关进程
echo "[INFO] Killing all Ray processes..."
pkill -f "ray::" || true
pkill -f "gcs_server" || true
pkill -f "raylet" || true
pkill -f "ray_" || true
pkill -f "dashboard" || true

# 3. 清理Ray临时文件
echo "[INFO] Cleaning Ray temporary files..."
rm -rf /tmp/ray* 2>/dev/null || true
rm -rf ~/.ray 2>/dev/null || true

# 4. 清理特定端口
echo "[INFO] Cleaning up ports..."
for port in 8000 8265 6379 6380 10001 10002; do
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
    echo "  - Cleared port $port"
done

# 5. 等待一下让进程完全结束
echo "[INFO] Waiting for processes to terminate..."
sleep 3

# 6. 验证清理结果
echo "[INFO] Verification:"
echo "  - Ray processes: $(pgrep -f ray | wc -l)"
echo "  - Port 8000: $(lsof -ti:8000 | wc -l)"
echo "  - Port 8265: $(lsof -ti:8265 | wc -l)"

echo "[INFO] Ray cleanup completed!"