#!/bin/bash
#SBATCH --job-name=lr_finder
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/autoresearch/%j_lr_finder.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/autoresearch/%j_lr_finder.err

set -euo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_TIMEOUT=7200
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online

LOCAL_TMP_DIR="/dev/shm/${USER}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"

NUM_GPUS=2
WIKI_RANK=1000000
LR_FINDER_STEPS=300
LR_FINDER_START=1e-7
LR_FINDER_END=1e-1
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/autoresearch_wr/train.py"
# ======Configuration=====

echo "[LR_FINDER] wiki_rank=${WIKI_RANK} steps=${LR_FINDER_STEPS} range=[${LR_FINDER_START}, ${LR_FINDER_END}]"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="127.0.0.1" \
    --master_port=29923 \
    "${SCRIPT_PATH}" \
    --train_jsonl /mnt/gemini/data1/jiaxuanluo/term_train_v1_0.jsonl \
    --dev_jsonl /mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl \
    --save_path /mnt/data/jiaxuanluo/autoresearch_wr/lr_finder_tmp.pt \
    --lr 1e-4 \
    --text_lr 0 \
    --batch_size 1024 \
    --epochs 1 \
    --num_workers 8 \
    --temperature 0.03 \
    --target_dim 1024 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --text_lora_rank 128 \
    --text_lora_alpha 256 \
    --lora_target_modules q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2 \
    --text_lora_target_modules query key value dense \
    --glossary_neg_path "" \
    --glossary_neg_refresh_steps 0 \
    --neg_bank_size 0 \
    --neg_bank_refresh_steps 0 \
    --hard_neg_k 0 \
    --hard_neg_glossary "" \
    --save_steps 99999 \
    --eval_steps_sample 99999 \
    --keep_checkpoints 1 \
    --eval_topk 10 \
    --acl_dev_jsonl /mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl \
    --eval_wiki_glossary /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json \
    --eval_glossary_sizes 1000 10000 \
    --best_metric "eval_acl6060/recall@10_gs1000" \
    --best_metric_secondary "eval_acl6060/recall@10_gs10000" \
    --use_lora \
    --wiki_rank ${WIKI_RANK} \
    --lr_finder_steps ${LR_FINDER_STEPS} \
    --lr_finder_start ${LR_FINDER_START} \
    --lr_finder_end ${LR_FINDER_END} \
    --enable_wandb \
    --wandb_project qwen3_rag_autoresearch \
    --wandb_exp_name "lr_finder_wr${WIKI_RANK}"

echo "[LR_FINDER] Done at $(date)"
