#!/bin/bash
#SBATCH --job-name=enrich_mfa
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_enrich_mfa.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_enrich_mfa.err

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONUNBUFFERED=1

SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/enrich_jsonl_with_mfa_timestamps.py"
INPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m.jsonl"
OUTPUT_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
SQLITE_INDEX="/mnt/gemini/data1/jiaxuanluo/gigaspeech_mfa_index/gigaspeech_mfa_index.sqlite"
GS_TEXTGRID_DIR="/mnt/taurus/data/siqiouyang/datasets/gigaspeech/textgrids"
# ======Configuration=====

echo "[ENRICH] Starting MFA enrichment at $(date)"
echo "[ENRICH] Input: ${INPUT_JSONL}"
echo "[ENRICH] Output: ${OUTPUT_JSONL}"
echo "[ENRICH] SQLite: ${SQLITE_INDEX}"

python3 "${SCRIPT}" \
    --input "${INPUT_JSONL}" \
    --output "${OUTPUT_JSONL}" \
    --sqlite_index "${SQLITE_INDEX}" \
    --gs_textgrid_dir "${GS_TEXTGRID_DIR}"

echo ""
echo "[ENRICH] Done at $(date)"
echo "[ENRICH] Output lines: $(wc -l < ${OUTPUT_JSONL})"
