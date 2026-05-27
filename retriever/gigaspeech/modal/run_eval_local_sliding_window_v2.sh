#!/bin/bash
#SBATCH --job-name=eval_sliding
#SBATCH --output=/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/eval_sliding_%A_%a.out
#SBATCH --error=/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/eval_sliding_%A_%a.err
#SBATCH --partition=taurus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --array=0-0
# Sliding Window Evaluation - Array job for parameter sweep
# CHUNK_SIZE: 2.0, 2.5, 3.0 (3 values)
# HOP_SIZE: 0.2, 0.4, 0.6, 0.8, 1.0 (5 values)
# Total: 15 combinations

# ===================== Configuration =====================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# Parameter arrays
CHUNK_SIZES=(1.4)
HOP_SIZES=(0.1)

# Get parameters for this array task
CHUNK_SIZE=${CHUNK_SIZES[$SLURM_ARRAY_TASK_ID]}
HOP_SIZE=${HOP_SIZES[$SLURM_ARRAY_TASK_ID]}

# Model path
MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"

# Prebuilt index (required)
PREBUILT_INDEX="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_used_terms.pkl"

# Glossary for GT term matching
GLOSSARY_PATH="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_used.json"

# TSV test data
TSV_PATH="/mnt/taurus/data/siqiouyang/datasets/gigaspeech/manifests/dev_case.tsv"

# Model config
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.0

# Evaluation config
MAX_SAMPLES=1000        # Number of samples to evaluate
TOP_K=5                 # Top-k terms to retrieve per chunk
BATCH_SIZE=32           # Batch size for audio encoding
DEVICE="cuda"
MIN_WORDS=1             # Min word count for GT terms (2=multi-word only, filters "sense","lot","got")
MIN_CHARS=3             # Min character count for GT terms
SEED=42                 # Random seed for TSV sampling

# ===================== Run Evaluation =====================

cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal

echo "=== Sliding Window Evaluation (Array Job) ==="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID / 14"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo ""
echo ">>> Parameters for this run:"
echo "    Chunk size: ${CHUNK_SIZE}s"
echo "    Hop size: ${HOP_SIZE}s"
echo "    Min words for GT: ${MIN_WORDS} (2=multi-word only)"
echo "    Random sample: true, seed: ${SEED}"
echo ""
echo "Model: $MODEL_PATH"
echo "Index: $PREBUILT_INDEX"
echo "Glossary: $GLOSSARY_PATH"
echo "TSV: $TSV_PATH"
echo "Max samples: $MAX_SAMPLES"
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

if [ ! -f "$TSV_PATH" ]; then
    echo "ERROR: TSV file not found: $TSV_PATH"
    exit 1
fi

echo "All required files found. Starting evaluation..."
echo ""

python eval_local_sliding_window.py \
    --model_path "$MODEL_PATH" \
    --prebuilt_index "$PREBUILT_INDEX" \
    --glossary_path "$GLOSSARY_PATH" \
    --tsv_path "$TSV_PATH" \
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
    --random_sample \
    --seed "$SEED" \
    --save_plot_dir "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/plots" \

echo ""
echo "=== Evaluation Complete (Task $SLURM_ARRAY_TASK_ID: chunk=${CHUNK_SIZE}s, hop=${HOP_SIZE}s) ==="
