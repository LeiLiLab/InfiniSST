#!/bin/bash
#SBATCH --job-name=sample_p31full
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_sample_p31_full_jsonl.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_sample_p31_full_jsonl.err

set -euo pipefail

# ======Configuration=====
CONDA_ENV="spaCyEnv"
CONDA_PREFIX_PATH="${CONDA_PREFIX_PATH:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/${CONDA_ENV}}"

SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/sample_wiki_terms_by_domain.py"
# Default output (overridden by first argument if provided)
OUT_DEFAULT="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/wiki_synth_terms_p31_balanced_full.jsonl"
# ======Configuration=====

export PATH="${CONDA_PREFIX_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

OUTPUT_PATH="${1:-${OUT_DEFAULT}}"

echo "================================================================"
echo "[sample_p31] Full P31-balanced sample -> JSONL"
echo "[sample_p31] Output: ${OUTPUT_PATH}"
echo "[sample_p31] Host: $(hostname)"
echo "[sample_p31] Start: $(date)"
echo "================================================================"

python "${SCRIPT}" \
    --full_pool \
    --format jsonl \
    --output "${OUTPUT_PATH}"

echo "[sample_p31] Done at $(date)"
dou b le a