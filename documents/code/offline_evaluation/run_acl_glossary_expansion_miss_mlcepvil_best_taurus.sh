#!/bin/bash
#SBATCH --job-name=acl_miss_mlcepvil
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_acl_miss_mlcepvil.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_acl_miss_mlcepvil.err

set -euo pipefail

export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

export HF_HOME="/mnt/taurus/data/jiaxuanluo/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/mnt/taurus/data/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/taurus/data/jiaxuanluo/cache"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs6k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k4096_tcmoff_l4best_c5e5_resume3000_aries8_smoke3000_best.pt"
ACL_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
OUTPUT_DIR="/mnt/gemini/data2/jiaxuanluo/acl_glossary_expansion_miss/mlcepvil_best_gs10000_tau0p70_top30"
SCRIPT="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/offline_evaluation/audit_acl_glossary_expansion_misses.py"

mkdir -p "${OUTPUT_DIR}"

echo "[ACL_MISS] run_id          = mlcepvil"
echo "[ACL_MISS] checkpoint      = ${MODEL_PATH}"
echo "[ACL_MISS] output_dir      = ${OUTPUT_DIR}"
echo "[ACL_MISS] started_at      = $(date)"

python3 "${SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --acl_jsonl "${ACL_JSONL}" \
    --wiki_glossary "${WIKI_GLOSSARY}" \
    --output_dir "${OUTPUT_DIR}" \
    --gs_size 10000 \
    --top_k 30 \
    --eval_topk 10 \
    --focus_tau 0.70 \
    --compare_tau 0.80 \
    --device "cuda:0" \
    --target_dim 1024 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --pooling_type "transformer" \
    --temperature 0.07 \
    --use_maxsim \
    --maxsim_windows "2 3 4 5 6 7 8 10 12 16 20 24" \
    --maxsim_stride 2 \
    --text_lora_rank 128 \
    --text_lora_alpha 256 \
    --text_pooling "cls" \
    --sparse_weight 0.0 \
    --sim_batch_rows 64 \
    --report_examples 16

echo "[ACL_MISS] finished_at     = $(date)"
