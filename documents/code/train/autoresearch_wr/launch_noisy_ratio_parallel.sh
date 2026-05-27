#!/bin/bash
set -euo pipefail

# Launch noisy_ratio sweep on Taurus (7 GPUs per run, 1-hour each).
# Jobs queue sequentially since 7 GPUs ≈ full node.

# ======Configuration=====
NOISY_RATIOS=(0.0 0.5 1.0)

TRAIN_PY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/autoresearch_wr/train.py"
CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"

TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"

SAVE_DIR="/mnt/data/jiaxuanluo/autoresearch_noisy_ratio"
LOG_DIR="/mnt/gemini/data1/jiaxuanluo/logs/autoresearch"

GPUS_PER_RUN=7
TIME_BUDGET=3600
LR="1e-4"
TEMPERATURE="0.03"
WIKI_RANK=1000000
BATCH_SIZE=3584
WARMUP_STEPS=25

LORA_RANK=128
LORA_ALPHA=256
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256

WANDB_API="${WANDB_API_KEY:-}"
# ======Configuration=====

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

submit_job() {
    local noisy_ratio="$1"
    local master_port="$2"

    local tag
    tag=$(printf "nr%.2f" "${noisy_ratio}")
    local save_path="${SAVE_DIR}/${tag}_best.pt"

    local job_id
    job_id=$(sbatch --parsable \
        --partition=taurus \
        --job-name="${tag}" \
        --gres="gpu:${GPUS_PER_RUN}" \
        --cpus-per-task=48 \
        --mem=200G \
        --time=1:30:00 \
        --output="${LOG_DIR}/%j_${tag}.out" \
        --error="${LOG_DIR}/%j_${tag}.err" \
        --wrap="
set -eu
export CONDA_PREFIX=${CONDA_PREFIX}
export PATH=\${CONDA_PREFIX}/bin:\${PATH}
export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/taurus/home/jiaxuanluo/InfiniSST:\${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200
export NCCL_P2P_DISABLE=1
export WANDB_API_KEY=${WANDB_API}
export WANDB_MODE=disabled

torchrun \\
    --nproc_per_node=${GPUS_PER_RUN} \\
    --master_addr=127.0.0.1 \\
    --master_port=${master_port} \\
    ${TRAIN_PY} \\
    --train_jsonl ${TRAIN_JSONL} \\
    --dev_jsonl ${DEV_JSONL} \\
    --save_path ${save_path} \\
    --noisy_ratio ${noisy_ratio} \\
    --lr ${LR} \\
    --temperature ${TEMPERATURE} \\
    --wiki_rank ${WIKI_RANK} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs 1 \\
    --num_workers 8 \\
    --target_dim 1024 \\
    --lora_rank ${LORA_RANK} \\
    --lora_alpha ${LORA_ALPHA} \\
    --text_lora_rank ${TEXT_LORA_RANK} \\
    --text_lora_alpha ${TEXT_LORA_ALPHA} \\
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
    --warmup_steps ${WARMUP_STEPS} \\
    --time_budget ${TIME_BUDGET} \\
    --acl_dev_jsonl ${ACL_DEV_JSONL} \\
    --eval_wiki_glossary ${EVAL_WIKI_GLOSSARY} \\
    --use_lora
")
    echo "[SUBMIT] taurus | noisy_ratio=${noisy_ratio} | job=${job_id} | port=${master_port}"
}

echo "========== Launching noisy_ratio sweep on taurus (7 GPUs, 1h each) =========="

port=29920
for nr in "${NOISY_RATIOS[@]}"; do
    submit_job "${nr}" "${port}"
    port=$((port + 1))
done

echo "========== All ${#NOISY_RATIOS[@]} jobs submitted (will queue sequentially) =========="
echo "Monitor: squeue -u \${USER}"
echo "Logs: ${LOG_DIR}/"
