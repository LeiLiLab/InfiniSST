#!/bin/bash
#SBATCH --job-name=extract_all_terms
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_extract_all_terms.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_extract_all_terms.err

set -euo pipefail

# 环境变量设置
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spacy_gpu_env"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="/mnt/gemini/data1/jiaxuanluo/xdg_cache"
mkdir -p "$XDG_CACHE_HOME"

# 验证 Python 位置
which python
python --version

export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/.local/lib/python3.10/site-packages:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

export HF_HOME="/mnt/gemini/data1/jiaxuanluo/huggingface_cache"
mkdir -p "$HF_HOME"

# --- 配置区 ---
MODE=${1:-"all"}
MULTIPLIER=${2:-2}
# --------------

INPUT_TSV="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
NER_JSONL="/mnt/gemini/data1/jiaxuanluo/ner_candidates_merged_v4.jsonl"
M1_DATASET="/mnt/gemini/data1/jiaxuanluo/term_train_dataset_m1.jsonl"

OUTPUT_DIR="/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks_m${MULTIPLIER}"
OUTPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_dataset_m${MULTIPLIER}.jsonl"

cd /mnt/taurus/home/jiaxuanluo/InfiniSST

# --- Step 0: Prepare NER Candidates if needed ---
if [ ! -f "${NER_JSONL}" ] || [ ! -s "${NER_JSONL}" ]; then
    echo "[INFO] ${NER_JSONL} is missing or empty. Extracting from ${M1_DATASET}..."
    if [ -f "${M1_DATASET}" ]; then
        python retriever/gigaspeech/prepare_ner_candidates.py
    else
        echo "[ERROR] ${M1_DATASET} not found. Cannot prepare ${NER_JSONL}."
        exit 1
    fi
fi

# --- RUN: Term Extraction & Audio Chunking (CPU only) ---
if [[ "$MODE" == "all" || "$MODE" == "run" ]]; then
    echo "[RUN] Extracting terms and chunking audio using CPU workers..."
    python retriever/gigaspeech/extract_all_terms_from_tsv.py \
      --input-tsv "${INPUT_TSV}" \
      --output-dir "${OUTPUT_DIR}" \
      --output-jsonl "${OUTPUT_JSONL}" \
      --multiplier-merge "${MULTIPLIER}" \
      --ner-candidates-jsonl "${NER_JSONL}" \
      --num-workers 64
    echo "[RUN] Completed."
fi

# Note: No merge step needed as extract_all_terms_from_tsv.py now handles full dataset via multiprocessing
