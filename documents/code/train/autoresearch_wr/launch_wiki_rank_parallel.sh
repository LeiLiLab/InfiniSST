#!/bin/bash
set -euo pipefail

# Parallel wiki_rank search on Taurus.
# 1-variant data, clean-only (noisy_ratio=0.0).
# Previously tested: 1M, 2M, 3.5M, 4.5M → 2M best.
# Now refine around 2M: 1.25M, 1.5M, 1.75M, 2M, 2.25M, 2.5M, 3M

# ======Configuration=====
WIKI_RANKS=(1250000 1500000 1750000 2000000 2250000 2500000 3000000)

TRAIN_PY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/autoresearch_wr/train.py"
CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"

TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_v1_0.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"

SAVE_DIR="/mnt/data/jiaxuanluo/autoresearch_wiki_rank_clean"
LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs/autoresearch"

GPUS_PER_RUN=2
TIME_BUDGET=1500
NOISY_RATIO=0.0
LR="1e-4"
TEMPERATURE="0.03"
BATCH_SIZE=1024

WANDB_PROJECT="qwen3_rag_autoresearch"
WANDB_API="${WANDB_API_KEY:-}"
# ======Configuration=====

mkdir -p "${LOG_DIR}"

port=29930
for wr in "${WIKI_RANKS[@]}"; do
    wr_tag="wr$(( wr / 1000 ))k"
    save_path="${SAVE_DIR}/${wr_tag}_best.pt"

    job_id=$(sbatch --parsable \
        --partition=taurus \
        --job-name="${wr_tag}" \
        --gres="gpu:${GPUS_PER_RUN}" \
        --cpus-per-task=16 \
        --mem=80G \
        --time=1:00:00 \
        --output="${LOG_DIR}/%j_${wr_tag}.out" \
        --error="${LOG_DIR}/%j_${wr_tag}.err" \
        --wrap="
set -eu
export CONDA_PREFIX=${CONDA_PREFIX}
export PATH=\${CONDA_PREFIX}/bin:\${PATH}
export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/taurus/home/jiaxuanluo/InfiniSST:\${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200
export NCCL_P2P_DISABLE=1
export WANDB_MODE=disabled
mkdir -p ${SAVE_DIR}

torchrun \\
    --nproc_per_node=${GPUS_PER_RUN} \\
    --master_addr=127.0.0.1 \\
    --master_port=${port} \\
    ${TRAIN_PY} \\
    --train_jsonl ${TRAIN_JSONL} \\
    --dev_jsonl ${DEV_JSONL} \\
    --save_path ${save_path} \\
    --noisy_ratio ${NOISY_RATIO} \\
    --wiki_rank ${wr} \\
    --lr ${LR} \\
    --temperature ${TEMPERATURE} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs 1 \\
    --num_workers 8 \\
    --target_dim 1024 \\
    --lora_rank 32 \\
    --lora_alpha 64 \\
    --text_lora_rank 128 \\
    --text_lora_alpha 256 \\
    --text_lr 0 \\
    --lora_target_modules q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2 \\
    --text_lora_target_modules query key value dense \\
    --glossary_neg_path '' \\
    --glossary_neg_refresh_steps 0 \\
    --neg_bank_size 0 \\
    --neg_bank_refresh_steps 0 \\
    --hard_neg_k 0 \\
    --hard_neg_glossary '' \\
    --save_steps 99999 \\
    --eval_steps_sample 50 \\
    --keep_checkpoints 1 \\
    --eval_topk 10 \\
    --eval_glossary_sizes 1000 10000 \\
    --best_metric eval_acl6060/recall@10_gs1000 \\
    --best_metric_secondary eval_acl6060/recall@10_gs10000 \\
    --warmup_steps 200 \\
    --time_budget ${TIME_BUDGET} \\
    --acl_dev_jsonl ${ACL_DEV_JSONL} \\
    --eval_wiki_glossary ${EVAL_WIKI_GLOSSARY} \\
    --use_lora
")
    echo "[SUBMIT] taurus | wiki_rank=${wr} (${wr_tag}) | job=${job_id} | port=${port}"
    port=$((port + 1))
done

echo "========== All ${#WIKI_RANKS[@]} wiki_rank jobs submitted =========="
echo "Monitor: squeue -u \${USER}"
echo "Logs: ${LOG_DIR}/"
