#!/bin/bash
#SBATCH --job-name=thr_sweep
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_thr_sweep.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_thr_sweep.err

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

export CUDA_VISIBLE_DEVICES="0"

MODEL_PATH="/mnt/taurus/data/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs10752_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_sp07_best_acl6060_gs10000.pt"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
OUTPUT_DIR="/mnt/gemini/data2/jiaxuanluo/threshold_sweep_maxsim_sp07"

SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/threshold_sweep_maxsim.py"
# ======Configuration=====

mkdir -p "${OUTPUT_DIR}"

echo "[EVAL] Threshold sweep for MaxSim + sparse=0.7 model"
echo "[EVAL] Model: ${MODEL_PATH}"
echo "[EVAL] Output: ${OUTPUT_DIR}"

python3 "${SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --dev_jsonl "${DEV_JSONL}" \
    --acl_jsonl "${ACL_JSONL}" \
    --wiki_glossary "${WIKI_GLOSSARY}" \
    --glossary_sizes 1000 10000 \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda:0" \
    --target_dim 1024 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --pooling_type transformer \
    --temperature 0.03 \
    --use_maxsim \
    --maxsim_windows "6 10 16 24" \
    --maxsim_stride 2 \
    --text_lora_rank 128 \
    --text_lora_alpha 256 \
    --text_pooling cls \
    --sparse_weight 0.7

echo "[EVAL] Done at $(date)"
