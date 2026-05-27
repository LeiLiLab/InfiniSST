#!/bin/bash
#SBATCH --job-name=density_ablation
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_density_ablation.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_density_ablation.err

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

REBUILD_SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/hard_negative_jsonl_for_speech_llm/rebuild_termmap.py"

RETRIEVER_RESULTS_JSONL="/mnt/gemini/data1/jiaxuanluo/train_cleaned_with_retriever_results_varlen.jsonl"
OUTPUT_BASE="/mnt/gemini/data1/jiaxuanluo/density_ablation"

DENSITY_VALUES="1 3 5 8 10"
SEED=42
MAX_CONVERSATIONS="${MAX_CONVERSATIONS:-0}"
# ======Configuration=====

if [ ! -f "${RETRIEVER_RESULTS_JSONL}" ]; then
    echo "[FATAL] Retriever results not found: ${RETRIEVER_RESULTS_JSONL}"
    echo "[HINT] Run run_generate_termmap_maxsim.sh first."
    exit 1
fi

mkdir -p "${OUTPUT_BASE}"
mkdir -p /mnt/gemini/data1/jiaxuanluo/logs

echo "============================================"
echo " Density Ablation: rebuild_termmap for d={${DENSITY_VALUES}}"
echo " Input: ${RETRIEVER_RESULTS_JSONL}"
echo " Output base: ${OUTPUT_BASE}"
echo "============================================"

for d in ${DENSITY_VALUES}; do
    OUTPUT_JSONL="${OUTPUT_BASE}/train_maxsim_varlen_d${d}.jsonl"
    echo ""
    echo "[ABLATION] density_coeff=${d} -> ${OUTPUT_JSONL}"

    python3 "${REBUILD_SCRIPT}" \
        --input_jsonl "${RETRIEVER_RESULTS_JSONL}" \
        --output_jsonl "${OUTPUT_JSONL}" \
        --density_coeff "${d}" \
        --seed "${SEED}" \
        --max_conversations "${MAX_CONVERSATIONS}"

    echo "[ABLATION] density_coeff=${d} done."
done

echo ""
echo "============================================"
echo " All density variants generated."
echo " Files:"
for d in ${DENSITY_VALUES}; do
    f="${OUTPUT_BASE}/train_maxsim_varlen_d${d}.jsonl"
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "   d=${d}: ${f} (${lines} lines)"
    else
        echo "   d=${d}: MISSING"
    fi
done
echo "============================================"
