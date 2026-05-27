#!/usr/bin/env bash
#SBATCH --job-name=extract_p31
#SBATCH --partition=taurus
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/logs/extract_p31_%j.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/logs/extract_p31_%j.err
set -euo pipefail

# ======Configuration=====
SCRIPT_DIR="/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth"
RDF_INPUT="/mnt/gemini/data1/jiaxuanluo/glossary/latest-truthy.nt"
OUTPUT_DIR="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data"
CONDA_ENV="/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
# ======Configuration=====

mkdir -p "${OUTPUT_DIR}/logs"

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

echo "===== extract_rdf_terms_with_p31.py ====="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Input: ${RDF_INPUT}"
echo "Output: ${OUTPUT_DIR}/wiki_rdf_terms_with_p31.jsonl"
echo ""

python3 "${SCRIPT_DIR}/extract_rdf_terms_with_p31.py" \
    --input "${RDF_INPUT}" \
    --output "${OUTPUT_DIR}/wiki_rdf_terms_with_p31.jsonl"

echo ""
echo "===== Done ====="
echo "Date: $(date)"
