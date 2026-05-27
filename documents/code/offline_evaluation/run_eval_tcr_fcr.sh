#!/bin/bash
# ===========================================================================
# DEPRECATED: This script computes chunk-level TCR/FCR which is incorrect for
# streaming inference evaluation. Use the new sentence-level pipeline instead:
#
#   documents/code/simuleval/eval_density_unified.sh
#   documents/code/simuleval/run_density_eval_ablation.sh
#
# The new pipeline computes sentence-level TCR (=TERM_ACC) and FCR via
# stream_laal_term.py after mWER resegmentation, integrated with SimulEval.
# ===========================================================================
#SBATCH --job-name=tcr_fcr
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_tcr_fcr.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_tcr_fcr.err

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

export CUDA_VISIBLE_DEVICES="6,7"
export PYTHONUNBUFFERED=1

MODEL_PATH="/mnt/aries/data4/jiaxuanluo/speech_llm_maxsim_enriched/keep1.0_r16/v1-20260411-182618-hf"
EVAL_BASE="/mnt/gemini/data2/jiaxuanluo/tcr_fcr_eval"
SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/eval_tcr_fcr.py"

STRATEGIES="baseline chunk_gate term_filter"
GLOSSARY_SIZES=(100 1000 10000)
# ======Configuration=====

echo "[SBATCH] Starting TCR/FCR evaluation at $(date)"
echo "[SBATCH] GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[SBATCH] Model: ${MODEL_PATH}"
echo "[SBATCH] vLLM version: $(python3 -c 'import vllm; print(vllm.__version__)')"

# --- Dev evaluation for each glossary size ---
for GS in "${GLOSSARY_SIZES[@]}"; do
    echo ""
    echo "================================================================"
    echo "======== Dev Evaluation gs=${GS} ========"
    echo "================================================================"
    python3 "${SCRIPT}" \
        --eval_jsonl "${EVAL_BASE}/dev_gs${GS}_retriever_results.jsonl" \
        --output_dir "${EVAL_BASE}/dev_gs${GS}_eval" \
        --strategies ${STRATEGIES} \
        --run_inference \
        --model_path "${MODEL_PATH}" \
        --tp_size 2
done

# --- ACL evaluation for each glossary size ---
for GS in "${GLOSSARY_SIZES[@]}"; do
    echo ""
    echo "================================================================"
    echo "======== ACL Evaluation gs=${GS} ========"
    echo "================================================================"
    python3 "${SCRIPT}" \
        --eval_jsonl "${EVAL_BASE}/acl_gs${GS}_retriever_results.jsonl" \
        --output_dir "${EVAL_BASE}/acl_gs${GS}_eval" \
        --strategies ${STRATEGIES} \
        --run_inference \
        --model_path "${MODEL_PATH}" \
        --tp_size 2
done

echo ""
echo "[SBATCH] All done at $(date)"
echo ""
echo "================================================================"
echo "Summary of all results:"
echo "================================================================"
for f in "${EVAL_BASE}"/*/tcr_fcr_results.json; do
    if [ -f "$f" ]; then
        dir=$(dirname "$f")
        name=$(basename "$dir")
        echo "--- ${name} ---"
        python3 -c "
import json
results = json.load(open('${f}'))
print(f\"{'Strategy':15s} | {'TCR':>8s} | {'FCR':>8s} | {'GT_hit':>8s} | {'GT_total':>8s} | {'NEG_fc':>8s} | {'NEG_total':>8s}\")
for r in results:
    print(f\"{r['strategy']:15s} | {r['TCR']:>8.4f} | {r['FCR']:>8.4f} | {r['gt_correct']:>8d} | {r['gt_total']:>8d} | {r['neg_false_copy']:>8d} | {r['neg_total']:>8d}\")
"
    fi
done
