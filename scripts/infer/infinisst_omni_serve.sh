#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
##SBATCH --dependency=afterok:job_id
##SBATCH --array=1
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/omni-serve.err
#SBATCH -o slurm_logs/omni-serve.out

# Requeue the job when SLURM sends USR1 (5 min before time limit)
requeue_handler() {
    echo "$(date): Received USR1 signal, requeuing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
}
trap 'requeue_handler' USR1

source /home/siqiouya/miniconda3/bin/activate omni_inference

VLLM_PORT=8199
MODEL_PATH=/data/user_data/siqiouya/ckpts/infinisst-omni/gigaspeech-zh-s_origin-bsz4/v1-20260122-055820-hf

# Find a free port starting from VLLM_PORT
while ss -tlnp | grep -q ":$VLLM_PORT "; do
    echo "Port $VLLM_PORT is in use, trying $((VLLM_PORT + 1))..."
    VLLM_PORT=$((VLLM_PORT + 1))
done
echo "Using port $VLLM_PORT"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

vllm serve \
    $MODEL_PATH \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --limit-mm-per-prompt '{"audio": 60}' \
    --max-model-len 4096 \
    --port "$VLLM_PORT" &

VLLM_PID=$!

# Wait for vllm to be ready, then start ngrok
(
    echo "Waiting for vllm to start on port $VLLM_PORT..."
    while ! curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
        sleep 5
    done
    # Warmup request with instruction + audio to trigger full multimodal pipeline compilation
    echo "$(date): vllm is ready, sending warmup request..."
    WARMUP_AUDIO_BASE64=$(python3 -c "
import base64, io, struct
sr, duration = 16000, 0.96
n_samples = int(sr * duration)
samples = b'\x00\x00' * n_samples
buf = io.BytesIO()
buf.write(b'RIFF')
buf.write(struct.pack('<I', 36 + len(samples)))
buf.write(b'WAVEfmt ')
buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
buf.write(b'data')
buf.write(struct.pack('<I', len(samples)))
buf.write(samples)
print(base64.b64encode(buf.getvalue()).decode())
")
    curl -s "http://localhost:$VLLM_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_PATH"'",
            "messages": [
                {"role": "system", "content": "You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text."},
                {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,'"$WARMUP_AUDIO_BASE64"'"}}]}
            ],
            "max_tokens": 1
        }' > /dev/null 2>&1
    echo "$(date): warmup done, starting ngrok tunnel on port $VLLM_PORT"
    /home/siqiouya/download/ngrok http "$VLLM_PORT" --log=stdout > ngrok.log 2>&1
) &
NGROK_PID=$!

wait $VLLM_PID