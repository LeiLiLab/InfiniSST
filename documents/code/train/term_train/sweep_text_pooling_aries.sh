#!/bin/bash
#SBATCH --job-name=sw_tpool
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_sw_tpool.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_sw_tpool.err

# Sweep text_pooling strategy: sequential 1-hour runs per setting.
# Compare: cls (baseline), mean, max, cls_mean, cls_max, gated
# All use Transformer Pooling (audio side), no sparse, no phoneme.
# Aries 8 GPUs, GradCache 12K.

set -uo pipefail

# ======Configuration=====
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

LOCAL_TMP_DIR="/dev/shm/${USER}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=OFF
CUDA_VISIBLE_GPU_LIST="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_GPU_LIST}"
NUM_GPUS=8
MASTER_ADDR="127.0.0.1"

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

export HF_HOME="/mnt/data4/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/data4/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/data4/jiaxuanluo/cache"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
SAVE_DIR="/mnt/data4/jiaxuanluo/train_outputs/sweep_text_pooling"

ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"

TEXT_POOLINGS="cls mean max cls_mean cls_max gated"
RUN_SECONDS=3600
RESULT_FILE="/mnt/gemini/data1/jiaxuanluo/sweep_text_pooling_result.txt"
# ======Configuration=====

mkdir -p "${SAVE_DIR}"
> "${RESULT_FILE}"

PORT_BASE=29920
RUN_IDX=0

for TP in ${TEXT_POOLINGS}; do
    RUN_IDX=$((RUN_IDX + 1))
    MASTER_PORT=$((PORT_BASE + RUN_IDX))

    PER_GPU_BATCH=1536
    BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
    BS_ABBR=$((BATCH_SIZE / 1024))k

    # Adaptive chunk_size: cls/mean/max can use 512; gated variants need smaller chunks
    case "${TP}" in
        cls|mean|max) GC_CHUNK=512 ;;
        cls_mean|cls_max) GC_CHUNK=256 ;;
        gated) GC_CHUNK=128 ;;
    esac

    SAVE_NAME="sweep_tp_${TP}_tfpool_bs${BS_ABBR}"
    SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
    WANDB_EXP_NAME="sweep_tpool_${SAVE_NAME}"

    echo ""
    echo "================================================================"
    echo "[SWEEP ${RUN_IDX}] text_pooling=${TP}"
    echo "[SWEEP ${RUN_IDX}] Save: ${SAVE_PATH}"
    echo "[SWEEP ${RUN_IDX}] WandB: ${WANDB_EXP_NAME}"
    echo "[SWEEP ${RUN_IDX}] Timeout: ${RUN_SECONDS}s"
    echo "================================================================"

    timeout "${RUN_SECONDS}" torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        "${SCRIPT_PATH}" \
        --train_jsonl "${TRAIN_JSONL}" \
        --dev_jsonl "${DEV_JSONL}" \
        --save_path "${SAVE_PATH}" \
        --lr 1.7e-4 \
        --text_lr 0 \
        --batch_size "${BATCH_SIZE}" \
        --epochs 5 \
        --num_workers 8 \
        --temperature 0.03 \
        --target_dim 1024 \
        --pooling_type transformer \
        --text_pooling "${TP}" \
        --sparse_weight 0.0 \
        --lora_rank 128 \
        --lora_alpha 256 \
        --text_lora_rank 128 \
        --text_lora_alpha 256 \
        --lora_target_modules q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2 \
        --text_lora_target_modules query key value dense \
        --glossary_neg_path "" \
        --glossary_neg_refresh_steps 0 \
        --neg_bank_size 0 \
        --neg_bank_refresh_steps 0 \
        --hard_neg_k 0 \
        --noisy_ratio 0.0 \
        --margin 0.1 \
        --online_hard_neg_k 0 \
        --grad_cache_chunk_size "${GC_CHUNK}" \
        --wiki_rank 1000000 \
        --save_steps 100 \
        --eval_steps_sample 33 \
        --eval_topk 10 \
        --keep_checkpoints 2 \
        --acl_dev_jsonl "${ACL_DEV_JSONL}" \
        --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
        --eval_glossary_sizes 1000 10000 \
        --best_metric "eval_acl6060/recall@10_gs1000" \
        --best_metric_secondary "eval_acl6060/recall@10_gs10000" \
        --eval_top100_samples 3 \
        --use_lora \
        --enable_wandb \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_exp_name "${WANDB_EXP_NAME}" \
    || true

    echo "[SWEEP ${RUN_IDX}] text_pooling=${TP} finished"
    echo "[SWEEP ${RUN_IDX}] Cleaning up GPU memory..."
    sleep 10
done

echo ""
echo "================================================================"
echo "[SWEEP] All text pooling runs completed at $(date)"
echo "================================================================"

# ---- Parse best metric from each run's best checkpoint ----
echo "[SWEEP] Parsing results..."
BEST_TP=""
BEST_VAL="-1"
BS_ABBR=$((NUM_GPUS * 1536 / 1024))k

for TP in ${TEXT_POOLINGS}; do
    BEST_PT="${SAVE_DIR}/sweep_tp_${TP}_tfpool_bs${BS_ABBR}_best.pt"
    if [ -f "${BEST_PT}" ]; then
        VAL=$("${CONDA_PREFIX}/bin/python3" -c "
import torch
ckpt = torch.load('${BEST_PT}', map_location='cpu')
v1 = ckpt.get('best_metric_value', -1)
v2 = ckpt.get('best_metric_secondary_value', -1)
print(f'{v1:.4f} {v2:.4f}')
" 2>/dev/null)
        GS1K=$(echo "${VAL}" | awk '{print $1}')
        GS10K=$(echo "${VAL}" | awk '{print $2}')
        echo "  text_pooling=${TP}  gs1000=${GS1K}  gs10000=${GS10K}"
        echo "${TP} ${GS1K} ${GS10K}" >> "${RESULT_FILE}"
        BETTER=$(echo "${GS1K} > ${BEST_VAL}" | bc -l 2>/dev/null || echo "0")
        if [ "${BETTER}" = "1" ]; then
            BEST_VAL="${GS1K}"
            BEST_TP="${TP}"
        fi
    else
        echo "  text_pooling=${TP}  not found (skipping)"
    fi
done

echo ""
echo "================================================================"
echo "[SWEEP RESULT] Best text_pooling = ${BEST_TP} (gs1000=${BEST_VAL})"
echo "================================================================"
echo "BEST_TEXT_POOLING=${BEST_TP}" >> "${RESULT_FILE}"
