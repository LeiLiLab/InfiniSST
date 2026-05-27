#!/bin/bash
#SBATCH --job-name=gemini_3var
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_gemini_3var.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_gemini_3var.err

set -euo pipefail

# ======Configuration=====
CONDA_PREFIX_PATH="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
SCRIPT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth"

TERMS_JSON="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant/top_1000k_terms.json"
OUTPUT_DIR="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant"
OUTPUT_JSONL="${OUTPUT_DIR}/wiki_synth_utterances_3variant.jsonl"

VARIANTS_PER_TERM=3
BATCH_SIZE=40
CONCURRENCY=15
# ======Configuration=====

export PATH="${CONDA_PREFIX_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"

echo "================================================================"
echo "[GEMINI] 3-variant utterance generation"
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
echo "[GEMINI] Expected: ~$((1000000 * VARIANTS_PER_TERM))"
echo "================================================================"
