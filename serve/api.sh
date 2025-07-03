#!/bin/bash
#SBATCH --job-name=infinisst_api
#SBATCH --output=logs/infinisst_api_%j.log
#SBATCH --error=logs/infinisst_api_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=taurus
#SBATCH --mem=128GB

# 确保logs目录存在

export DEBUG_MODE=true
export DEBUG_PORT=5678

mkdir -p logs

# 设置PYTHONPATH
export PYTHONPATH=/home/jiaxuanluo/new-infinisst

# 显示SLURM分配的GPU信息
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"
echo "[INFO] SLURM GPU devices: $CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM GPU count: $SLURM_GPUS"
echo "[INFO] SLURM GPU list: $SLURM_GPU_BIND"

# 如果CUDA_VISIBLE_DEVICES没有设置，尝试从其他SLURM变量获取
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # 尝试从SLURM变量中获取GPU信息
    if [ ! -z "$SLURM_GPUS" ]; then
        echo "[INFO] Using SLURM_GPUS: $SLURM_GPUS"
        export CUDA_VISIBLE_DEVICES="0,1"  # 假设分配了2个GPU，映射为0,1
    elif [ ! -z "$SLURM_LOCALID" ]; then
        echo "[INFO] Using SLURM_LOCALID to set GPU"
        export CUDA_VISIBLE_DEVICES="0,1"  # 默认使用前2个GPU
    else
        echo "[WARNING] No SLURM GPU info found, using default GPUs 0,1"
        export CUDA_VISIBLE_DEVICES="0,1"
    fi
else
    echo "[INFO] CUDA_VISIBLE_DEVICES already set: $CUDA_VISIBLE_DEVICES"
fi

# 显示最终的GPU配置
echo "[INFO] Final CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L | head -4

echo "[INFO] Killing existing ngrok..."
pkill -f ngrok || true

echo "[INFO] Killing any process using port 8000..."
# 修复fuser命令
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

echo "[INFO] Starting InfiniSST API server with scheduler system..."
echo "[INFO] Available GPUs: $CUDA_VISIBLE_DEVICES"

PYTHONUNBUFFERED=1 python api.py \
    --latency-multiplier 2 \
    --min-start-sec 0 \
    --w2v2-path /mnt/aries/data6/xixu/demo/wav2_vec_vox_960h_pl.pt \
    --w2v2-type w2v2 \
    --ctc-finetuned True \
    \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
    --block-size 48 \
    --max-cache-size 576 \
    --model-type w2v2_qwen25 \
    --rope 1 \
    --audio-normalize 0 \
    \
    --max-llm-cache-size 1000 \
    --always-cache-system-prompt \
    \
    --max-len-a 10 \
    --max-len-b 20 \
    --max-new-tokens 10 \
    --beam 4 \
    --no-repeat-ngram-lookback 100 \
    --no-repeat-ngram-size 5 \
    --repetition-penalty 1.2 \
    --suppress-non-language \
    \
    --model-name /mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct \
    --lora-rank 32 \
    --host 0.0.0.0 \
    --port 8000 &

# 等待端口8000启动
echo "[INFO] Waiting for FastAPI server to start..."
for i in {1..60}; do
    if lsof -i:8000 &>/dev/null; then
        echo "[INFO] FastAPI server started successfully on port 8000"
        break
    fi
    echo "Waiting for FastAPI to bind on port 8000... (attempt $i/60)"
    sleep 2
done

# 检查服务器是否启动成功
if lsof -i:8000 &>/dev/null; then
    echo "[INFO] API Server is running"
    # 测试健康检查
    sleep 5
    curl -s http://localhost:8000/health | python3 -m json.tool || echo "[WARNING] Health check failed"
else
    echo "[ERROR] API Server failed to start"
    exit 1
fi

# 启动 ngrok tunnel
echo "[INFO] Starting ngrok tunnel..."
/mnt/aries/data6/jiaxuanluo/bin/ngrok http --url=infinisst.ngrok.app 8000