#!/bin/bash
#SBATCH --job-name=build_index
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_build_index.out
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/logs/%j_build_index.err
##SBATCH --partition=aries
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB

# PY_BIN_OVERRIDE=/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python \
# sbatch --partition=aries /mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/modal/run_build_index.sh

# sbatch --export=USE_IMPORT=1 run_build_index.sh
# sbatch --partition=aries run_build_index.sh   # override partition when needed

set -euo pipefail

SHARED_HOME="/mnt/taurus/home/jiaxuanluo"
LOCAL_HOME="/home/jiaxuanluo"
if [[ -d "${SHARED_HOME}" ]]; then
    BASE_HOME="${SHARED_HOME}"
else
    BASE_HOME="${LOCAL_HOME}"
fi

REPO_ROOT="${BASE_HOME}/InfiniSST"
SCRIPT_DIR="${REPO_ROOT}/retriever/gigaspeech/modal"
DATA_DIR="${REPO_ROOT}/retriever/gigaspeech/data"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PY_BIN="${PY_BIN_OVERRIDE:-${PY_BIN:-}}"
ENV_BIN_DIR=""

if [[ -n "${PY_BIN}" && -x "${PY_BIN}" ]]; then
    ENV_BIN_DIR="$(dirname "${PY_BIN}")"
else
    CANDIDATE_PY_BINS=(
        "/mnt/data6/jiaxuanluo/conda_envs/infinisst/bin/python"
        "${HOME:-}/conda_envs/infinisst/bin/python"
        "/mnt/gemini/data2/jiaxuanluo/conda_envs/infinisst/bin/python"
        "${HOME:-}/miniconda3/envs/infinisst/bin/python"
        "/mnt/aries/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python"
        "/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python"
        "/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python"
        "${BASE_HOME}/miniconda3/envs/infinisst/bin/python"
        "${HOME:-}/miniconda3/bin/python"
        "/mnt/aries/home/jiaxuanluo/miniconda3/bin/python"
        "/home/jiaxuanluo/miniconda3/bin/python"
        "/mnt/taurus/home/jiaxuanluo/miniconda3/bin/python"
    )
    for candidate in "${CANDIDATE_PY_BINS[@]}"; do
        if [[ -x "${candidate}" ]]; then
            PY_BIN="${candidate}"
            ENV_BIN_DIR="$(dirname "${candidate}")"
            break
        fi
    done
fi

if [[ -z "${PY_BIN}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PY_BIN="$(command -v python)"
        ENV_BIN_DIR="$(dirname "${PY_BIN}")"
    fi
fi

if [[ -z "${PY_BIN}" || ! -x "${PY_BIN}" ]]; then
    echo "[ERROR] Unable to locate a usable python binary. Set PY_BIN_OVERRIDE to override." >&2
    exit 1
fi

export PATH="${ENV_BIN_DIR}:${PATH}"
export PYTHONNOUSERSITE=1

MODEL_PATH="/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt"

GLOSSARY_PATH="${DATA_DIR}/terms/glossary_cleaned.json"
OUTPUT_PATH="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index.pkl"

IMPORT_GLOSSARY_PATH="${DATA_DIR}/terms/glossary_acl6060.json"
IMPORT_OUTPUT_PATH="/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_acl6060.pkl"

USE_IMPORT=${USE_IMPORT:-0}

if [[ $USE_IMPORT -eq 1 ]]; then
    echo ">>> [INFO] USE_IMPORT=1, using ACL6060 glossary"
    GLOSSARY_PATH="$IMPORT_GLOSSARY_PATH"
    OUTPUT_PATH="$IMPORT_OUTPUT_PATH"
else
    echo ">>> [INFO] USE_IMPORT=0, using default glossary"
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct"
LORA_R=16
LORA_ALPHA=32

NUM_GPUS=1
BATCH_SIZE=64

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd "${SCRIPT_DIR}"

echo "=== Multi-GPU text index generation (LoRA enabled) ==="
echo "Model path:               $MODEL_PATH"
echo "Glossary:                 $GLOSSARY_PATH"
echo "Index output path:        $OUTPUT_PATH"
echo "GPU count:                $NUM_GPUS"
echo "Batch size:               $BATCH_SIZE"
echo "LoRA config:              r=$LORA_R, alpha=$LORA_ALPHA"
echo ""

"${PY_BIN}" build_index_multi_gpu.py \
    --model_path "$MODEL_PATH" \
    --glossary_path "$GLOSSARY_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --num_gpus "$NUM_GPUS" \
    --batch_size "$BATCH_SIZE" \
    --exclude_confused

echo ""
echo "=== Index generation finished ==="
echo "Index file: $OUTPUT_PATH"
echo ""

if [ -f "$OUTPUT_PATH" ]; then
    ls -lh "$OUTPUT_PATH"
fi