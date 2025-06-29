#!/bin/bash
#SBATCH --job-name=infinisst_api
#SBATCH --output=logs/infinisst_api_%j.log
#SBATCH --error=logs/infinisst_api_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=taurus
#SBATCH --mem=64GB

export PYTHONPATH=/home/jiaxuanluo/infinisst-demo-v2

echo "[INFO] Killing existing ngrok..."
pkill -f ngrok || true

echo "Killing any process using port 8000..."
fuser -k 8000/tcp || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

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

# 等待端口8000启动，最多等待10秒
for i in {1..100}; do
    if lsof -i:8000 &>/dev/null; then
        break
    fi
    echo "Waiting for FastAPI to bind on port 8000..."
    sleep 1
done
# 启动 ngrok tunnel
echo "Starting ngrok tunnel..."
/mnt/aries/data6/jiaxuanluo/bin/ngrok http --url=infinisst.ngrok.app 8000