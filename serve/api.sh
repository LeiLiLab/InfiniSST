#export PYTHONPATH=/home/xixu/work/InfiniSST
export PYTHONPATH=/home/jiaxuanluo/InfiniSST

echo "Killing any process using port 8001..."
fuser -k 8001/tcp || true

#conda activate /mnt/aries/data6/jiaxuanluo/infinisst

#/usr/bin/python3 api.py \
/mnt/aries/data6/jiaxuanluo/infinisst/bin/python api.py \
    --latency-multiplier 2 \
    --min-start-sec 0 \
    --w2v2-path /mnt/data6/xixu/demo/wav2_vec_vox_960h_pl.pt \
    --w2v2-type w2v2 \
    --ctc-finetuned True \
    \
    --length-shrink-cfg "[(1024,2,2)] * 2" \
    --block-size 48 \
    --max-cache-size 576 \
    --xpos 0 \
    --rope 1 \
    --audio-normalize 1 \
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
    --model-name /mnt/data6/xixu/demo/llama3.1-8b-instruct-hf \
    --lora-rank 32 > backend.log 2>&1 &

# 等待端口8001启动，最多等待10秒
for i in {1..100}; do
    if lsof -i:8001 &>/dev/null; then
        break
    fi
    echo "Waiting for FastAPI to bind on port 8001..."
    sleep 1
done
# 启动 ngrok
echo "Starting ngrok tunnel..."
/mnt/aries/data6/jiaxuanluo/bin/ngrok http --url=radically-mutual-sailfish.ngrok-free.app 8001