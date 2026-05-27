#!/bin/bash
#SBATCH --job-name=gemini_p31
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_gemini_p31.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_gemini_p31.err

set -euo pipefail

# ======Configuration=====
CONDA_PREFIX_PATH="/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
SCRIPT_DIR="/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth"

TERMS_JSON="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/wiki_synth_terms_p31_balanced_1000k.json"
OUTPUT_DIR="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/p31_balanced"
OUTPUT_JSONL="${OUTPUT_DIR}/wiki_synth_utterances_p31.jsonl"

VARIANTS_PER_TERM=1
BATCH_SIZE=40
CONCURRENCY=15
# ======Configuration=====

export PATH="${CONDA_PREFIX_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"
mkdir -p /mnt/gemini/data1/jiaxuanluo/logs

echo "================================================================"
echo "[GEMINI] P31-balanced utterance generation"
echo "[GEMINI] Terms: ${TERMS_JSON}"
echo "[GEMINI] Output: ${OUTPUT_JSONL}"
echo "[GEMINI] Variants/term: ${VARIANTS_PER_TERM}"
echo "[GEMINI] Batch size: ${BATCH_SIZE}, Concurrency: ${CONCURRENCY}"
echo "================================================================"

python "${SCRIPT_DIR}/generate_term_utterances.py" \
    --terms "${TERMS_JSON}" \
    --output "${OUTPUT_JSONL}" \
    --variants_per_term "${VARIANTS_PER_TERM}" \
    --batch_size "${BATCH_SIZE}" \
    --concurrency "${CONCURRENCY}"

UTT_COUNT=$(wc -l < "${OUTPUT_JSONL}")
echo "================================================================"
echo "[GEMINI] Done. Total utterances: ${UTT_COUNT}"
echo "================================================================"
