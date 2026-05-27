#!/bin/bash
#SBATCH --job-name=eval_acl6060
#SBATCH --output=/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/eval_acl6060_%A_%a.out
#SBATCH --error=/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/eval_acl6060_%A_%a.err
#SBATCH --partition=taurus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --array=0-0

# Sliding Window Evaluation for ACL6060 Dataset
# Data source:
#   - wav_dir: segmented gold wavs (one wav per sentence, sorted by sent_id numerically)
#   - tagged_txt: each line has [term] markers for GT terms
#   - glossary: used to build the prebuilt index

# ===================== Configuration =====================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# Parameter arrays (for grid search via SLURM array jobs)
CHUNK_SIZES=(1.4)
HOP_SIZES=(0.1)

# Get parameters for this array task
CHUNK_SIZE=${CHUNK_SIZES[$SLURM_ARRAY_TASK_ID]}
HOP_SIZE=${HOP_SIZES[$SLURM_ARRAY_TASK_ID]}

# Model path
MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"

# Prebuilt index (built from acl6060 glossary)
PREBUILT_INDEX="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_acl6060.pkl"

# Glossary for term retrieval
GLOSSARY_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060.json"

# ACL6060 data paths
WAV_DIR="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/segmented_wavs/gold"
TAGGED_TXT_PATH="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/tagged_terminology/ACL.6060.dev.tagged.en-xx.en.txt"

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
MIN_WORDS=2             # Min word count for GT terms
MIN_CHARS=3             # Min character count for GT terms
SEED=42                 # Random seed for sampling

# ===================== Run Evaluation =====================

cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal

echo "=== ACL6060 Sliding Window Evaluation ==="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo ""
echo ">>> Parameters for this run:"
echo "    Chunk size: ${CHUNK_SIZE}s"
echo "    Hop size: ${HOP_SIZE}s"
echo "    Min words for GT: ${MIN_WORDS}"
echo ""
echo "Model: $MODEL_PATH"
echo "Index: $PREBUILT_INDEX"
echo "Glossary: $GLOSSARY_PATH"
echo "Wav Dir: $WAV_DIR"
echo "Tagged Text: $TAGGED_TXT_PATH"
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

if [ ! -f "$TAGGED_TXT_PATH" ]; then
    echo "ERROR: Tagged text file not found: $TAGGED_TXT_PATH"
    exit 1
fi

echo "All required files found. Starting evaluation..."
echo ""

python eval_local_sliding_window_acl6060.py \
    --model_path "$MODEL_PATH" \
    --prebuilt_index "$PREBUILT_INDEX" \
    --glossary_path "$GLOSSARY_PATH" \
    --wav_dir "$WAV_DIR" \
    --tagged_txt_path "$TAGGED_TXT_PATH" \
    --model_name "$MODEL_NAME" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --max_samples "$MAX_SAMPLES" \
    --chunk_size "$CHUNK_SIZE" \
    --hop_size "$HOP_SIZE" \
    --top_k "$TOP_K" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --min_words "$MIN_WORDS" \
    --min_chars "$MIN_CHARS" \
    --restrict_index_to_eval_terms \
    --seed "$SEED" \
    --save_plot_dir "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/plots"

echo ""
echo "=== Evaluation Complete (Task $SLURM_ARRAY_TASK_ID: chunk=${CHUNK_SIZE}s, hop=${HOP_SIZE}s) ==="

