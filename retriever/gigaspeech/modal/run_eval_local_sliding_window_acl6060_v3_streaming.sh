#!/bin/bash
#SBATCH --job-name=eval_acl6060_v3_streaming
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/eval_acl6060_v3_streaming_%A_%a.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/eval_acl6060_v3_streaming_%A_%a.err
#SBATCH --partition=aries
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --array=0-0

# Streaming Simulation Evaluation for ACL6060 Dataset
#
# This script simulates the streaming behavior of infinisst_omni_vllm_rag.py:
# - Process audio in fixed-duration cycles (default: 1.92s per vLLM call)
# - Within each cycle: sliding window -> max pooling -> top-N filtering
# - Each cycle is INDEPENDENT (terms are NOT accumulated across cycles)
# - top-N = ceil(cycle_duration * terms_per_second) = ceil(1.92 * 2.5) = 5
#
# This differs from v2 which does top-N filtering on the ENTIRE segment.

# ===================== Configuration =====================

export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/mnt/taurus/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# Parameter arrays for grid search
# Note: cycle_duration is the key new parameter (simulates vLLM call interval)
CYCLE_DURATIONS=(1.92)
CHUNK_SIZES=(2.5)
HOP_SIZES=(0.1)

# Get parameters for this array task
CYCLE_DURATION=${CYCLE_DURATIONS[$SLURM_ARRAY_TASK_ID]}
CHUNK_SIZE=${CHUNK_SIZES[$SLURM_ARRAY_TASK_ID]}
HOP_SIZE=${HOP_SIZES[$SLURM_ARRAY_TASK_ID]}

# Model path
MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"

# Prebuilt index (built from acl6060 glossary)
PREBUILT_INDEX="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_acl6060.pkl"

# Glossary for term retrieval
#GLOSSARY_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/acl_terminology_glossary.json"
GLOSSARY_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060.json"
# ACL6060 data paths
WAV_DIR="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/segmented_wavs/gold"
TXT_PATH="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt"

# Model config
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.0

# Evaluation config
MAX_SAMPLES=0           # 0 = use all samples
TOP_K=5                 # Top-k terms to retrieve per chunk
BATCH_SIZE=32           # Batch size for audio encoding
DEVICE="cuda"
MIN_WORDS=1             # Min word count for GT terms
MIN_CHARS=3             # Min character count for GT terms
SEED=42                 # Random seed

# Streaming config
TERMS_PER_SECOND=2.5    # top-N = ceil(cycle_duration * terms_per_second)
THRESHOLD=0.0           # Score threshold to filter terms (avoid poison output)

# ===================== Run Evaluation =====================

cd /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal

echo "=== ACL6060 Streaming Simulation Evaluation (v3) ==="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo ""
echo ">>> Streaming Parameters for this run:"
echo "    Cycle duration: ${CYCLE_DURATION}s (simulates vLLM call interval)"
echo "    Chunk size: ${CHUNK_SIZE}s"
echo "    Hop size: ${HOP_SIZE}s"
echo "    Terms per second: ${TERMS_PER_SECOND}"
echo "    Score threshold: ${THRESHOLD}"
echo "    Fixed top-N per cycle: ceil(${CYCLE_DURATION} * ${TERMS_PER_SECOND}) = $(python3 -c "import math; print(math.ceil(${CYCLE_DURATION} * ${TERMS_PER_SECOND}))")"
echo ""
echo "Model: $MODEL_PATH"
echo "Index: $PREBUILT_INDEX"
echo "Glossary: $GLOSSARY_PATH"
echo "Wav Dir: $WAV_DIR"
echo "Text Path: $TXT_PATH"
echo "Max samples: $MAX_SAMPLES (0=all)"
echo "Top-k per chunk: $TOP_K"
echo "Device: $DEVICE"
echo ""

# Check files exist
if [ ! -f "$PREBUILT_INDEX" ]; then
    echo "ERROR: Prebuilt index not found: $PREBUILT_INDEX"
    exit 1
fi

if [ ! -f "$GLOSSARY_PATH" ]; then
    echo "ERROR: Glossary not found: $GLOSSARY_PATH"
    exit 1
fi

if [ ! -d "$WAV_DIR" ]; then
    echo "ERROR: Wav directory not found: $WAV_DIR"
    exit 1
fi

if [ ! -f "$TXT_PATH" ]; then
    echo "ERROR: Text file not found: $TXT_PATH"
    exit 1
fi

echo "All required files found. Starting evaluation..."
echo ""

# Run streaming evaluation
python eval_local_sliding_window_acl6060_v3_streaming.py \
    --model_path $MODEL_PATH \
    --prebuilt_index $PREBUILT_INDEX \
    --glossary_path $GLOSSARY_PATH \
    --wav_dir $WAV_DIR \
    --txt_path $TXT_PATH \
    --model_name $MODEL_NAME \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --max_samples $MAX_SAMPLES \
    --cycle_duration $CYCLE_DURATION \
    --chunk_size $CHUNK_SIZE \
    --hop_size $HOP_SIZE \
    --top_k $TOP_K \
    --terms_per_second $TERMS_PER_SECOND \
    --threshold $THRESHOLD \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --min_words $MIN_WORDS \
    --min_chars $MIN_CHARS \
    --restrict_index_to_eval_terms \
    --seed $SEED \
    --save_plot_dir /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/plots

echo ""
echo "=== Evaluation Complete (Task $SLURM_ARRAY_TASK_ID: cycle=${CYCLE_DURATION}s, chunk=${CHUNK_SIZE}s, hop=${HOP_SIZE}s) ==="

