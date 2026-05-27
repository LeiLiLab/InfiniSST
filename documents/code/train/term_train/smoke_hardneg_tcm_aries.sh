#!/bin/bash
#SBATCH --job-name=q3_smoke_E
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=0:40:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_smoke_E_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_smoke_E_%x.err

# Single-GPU smoke test for variant E = hard_neg_k bank mining + TCM only.
# HCL is OFF: with genuine hard negs in the batch, InfoNCE already upweights
# them via the softmax; adding the Robinson importance reweighting on top is
# redundant and just adds gradient variance.
#
# Purpose: verify the pipeline end-to-end before committing 8 GPUs to the
# full run, and measure neg_bank.refresh() wall-clock at high frequency
# (every 10 steps) so we can budget the 1.4M-term full-scale refresh cost.
#
# Expected observations (first ~50 steps):
#   - [NEG_BANK] line prints non-empty train_terms + wiki_terms
#   - Bank refresh triggers every 10 steps, logged via [NEG_BANK] Refreshing
#   - mine_hard_negatives appends hard_negs > 0 per step (log line)
#   - train/neg_sim_mean may fluctuate (dilution by 50k appended negs) but
#     train/tcm_neg > 0 every step (T_alpha=0.25 biting the hard tail)
#   - InfoNCE drops from ~5.8 to <4 within 40 steps
#   - Wall-clock between consecutive "[NEG_BANK] Refreshing" lines - 10
#     steps of normal training time = bank encode cost at 26k terms,
#     extrapolate linearly to 1.4M for the full run.

set -euo pipefail

# ======Configuration=====
# Small-batch single-GPU smoke; keep everything else identical to the full
# launcher so the pipeline under test matches what will be submitted.
NUM_GPUS=1
PER_GPU_BATCH=512        # Small enough to fit 1 GPU incl. bank sim matrix
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
GRAD_CACHE_CHUNK_SIZE=128
MAX_TRAIN_SECONDS=900    # 15 min runtime cap
EPOCHS=1
TRAIN_LIMIT=28000        # ~55 optimizer steps at bs=512 (enough for 5 refreshes at refresh=10)
NUM_WORKERS=4
LR="1.7e-4"
TEMPERATURE="0.07"
WIKI_RANK=1000000
NOISY_RATIO=0.0
ONLINE_HARD_NEG_K=0
MARGIN="0.0"

# --- Data-side hard-neg bank (the whole point of this smoke) ---
HARD_NEG_K=64
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=10     # high-frequency sync refresh; measure wall-clock cost

# --- Loss flags (variant E = TCM only, HCL removed) ---
# With bank-mined genuine hard negs in the batch, InfoNCE's softmax already
# upweights them - Robinson HCL on top is redundant and just adds variance.
HCL_BETA="0.0"
TCM_LOSS_WEIGHT="1.0"
TCM_POS_THRESHOLD="0.85"
TCM_NEG_THRESHOLD="0.25"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"

# --- Always-on full-glossary negatives (not used for smoke) ---
GLOSSARY_NEG_PATH=""
GLOSSARY_NEG_REFRESH_STEPS=0

# --- Env / paths ---
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

export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEBUG=INFO

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

export HF_HOME="/mnt/aries/data4/jiaxuanluo/cache/huggingface"
export TORCH_HOME="/mnt/aries/data4/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/aries/data4/jiaxuanluo/cache"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
SAVE_DIR="/mnt/gemini/home/jiaxuanluo/train_outputs"
mkdir -p "${SAVE_DIR}"

USE_LORA="true"
LORA_RANK=128
LORA_ALPHA=256
TARGET_DIM=1024
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
POOLING_TYPE="transformer"

USE_MAXSIM="true"
MAXSIM_WINDOWS="6 10 16 24"
MAXSIM_STRIDE=2
MFA_SUPERVISED="true"

TEXT_LR="0"
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
TEXT_TARGET_MODULES="query key value dense"
TEXT_POOLING="cls"
SPARSE_WEIGHT="0.0"

# Minimal eval for smoke (we only care that the training step works; eval is
# noise here).  EVAL_STEPS_SAMPLE is set to a value larger than the total
# steps so the expensive full-glossary eval never fires.
SAVE_STEPS=999999
EVAL_STEPS_SAMPLE=999999
KEEP_CHECKPOINTS=1
EVAL_TOPK=10

ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
EVAL_GLOSSARY_SIZES="1000"
BEST_METRIC="eval_acl6060/recall@10_gs1000"
BEST_METRIC_SECONDARY="eval_dev/recall@10_gs1000"

MASTER_ADDR="127.0.0.1"
MASTER_PORT=29971
# ======Configuration=====

# Pre-flight: pick 1 clean GPU.
PREFLIGHT_OUT="$(python3 - "$NUM_GPUS" <<'PYEOF'
import os, subprocess, sys, time
needed = int(sys.argv[1])
threshold_mib = 500
max_retry = 6
sleep_s = 10
for attempt in range(1, max_retry + 1):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = []
    busy = []
    for line in out.strip().splitlines():
        idx, used = [x.strip() for x in line.split(",")]
        if int(used) <= threshold_mib:
            free.append(idx)
        else:
            busy.append((idx, used))
    print(f"[PREFLIGHT] attempt={attempt} free={free} busy={busy}", file=sys.stderr)
    if len(free) >= needed:
        chosen = free[:needed]
        print(",".join(chosen))
        sys.exit(0)
    time.sleep(sleep_s)
print(
    f"[PREFLIGHT] only {len(free)}/{needed} clean GPUs after "
    f"{max_retry * sleep_s}s, aborting.",
    file=sys.stderr,
)
sys.exit(1)
PYEOF
)"
if [ -z "${PREFLIGHT_OUT}" ]; then
    echo "[PREFLIGHT] failed to pick ${NUM_GPUS} clean GPUs" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${PREFLIGHT_OUT}"
echo "[PREFLIGHT] selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

TEXT_TAG="tr${TEXT_LORA_RANK}"
MODE_NAME="scale_lora-r${LORA_RANK}-${TEXT_TAG}"
VERSION="smoke_E_hardneg_hcl_tcm"
SAVE_NAME="q3rag_${MODE_NAME}_bs${BATCH_SIZE}_t=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="smoke_${SAVE_NAME}"

echo "[SMOKE] Single-GPU pipeline verification for variant E"
echo "[SMOKE] HARD_NEG_K=${HARD_NEG_K} NEG_BANK_REFRESH_STEPS=${NEG_BANK_REFRESH_STEPS}"
echo "[SMOKE] HCL_BETA=${HCL_BETA} TCM_LOSS_WEIGHT=${TCM_LOSS_WEIGHT} T_beta=${TCM_POS_THRESHOLD} T_alpha=${TCM_NEG_THRESHOLD}"
echo "[SMOKE] max_train_seconds=${MAX_TRAIN_SECONDS} batch=${BATCH_SIZE} train_limit=${TRAIN_LIMIT}"

OPTS=""
if [ "${USE_LORA}" = "true" ]; then OPTS="${OPTS} --use_lora"; fi
if [ "${USE_MAXSIM}" = "true" ]; then OPTS="${OPTS} --use_maxsim"; fi
if [ "${MFA_SUPERVISED}" = "true" ]; then OPTS="${OPTS} --mfa_supervised_maxsim"; fi
if [ "${WIKI_RANK}" -gt 0 ]; then OPTS="${OPTS} --wiki_rank ${WIKI_RANK}"; fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_PATH}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --dev_jsonl "${DEV_JSONL}" \
    --save_path "${SAVE_PATH}" \
    --lr "${LR}" \
    --text_lr "${TEXT_LR}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --train_limit "${TRAIN_LIMIT}" \
    --num_workers "${NUM_WORKERS}" \
    --temperature "${TEMPERATURE}" \
    --target_dim "${TARGET_DIM}" \
    --pooling_type "${POOLING_TYPE}" \
    --maxsim_windows ${MAXSIM_WINDOWS} \
    --maxsim_stride "${MAXSIM_STRIDE}" \
    --text_pooling "${TEXT_POOLING}" \
    --sparse_weight "${SPARSE_WEIGHT}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --text_lora_rank "${TEXT_LORA_RANK}" \
    --text_lora_alpha "${TEXT_LORA_ALPHA}" \
    --lora_target_modules ${TARGET_MODULES} \
    --text_lora_target_modules ${TEXT_TARGET_MODULES} \
    --glossary_neg_path "${GLOSSARY_NEG_PATH}" \
    --glossary_neg_refresh_steps "${GLOSSARY_NEG_REFRESH_STEPS}" \
    --neg_bank_size "${NEG_BANK_SIZE}" \
    --neg_bank_refresh_steps "${NEG_BANK_REFRESH_STEPS}" \
    --hard_neg_k "${HARD_NEG_K}" \
    --noisy_ratio "${NOISY_RATIO}" \
    --margin "${MARGIN}" \
    --online_hard_neg_k "${ONLINE_HARD_NEG_K}" \
    --grad_cache_chunk_size "${GRAD_CACHE_CHUNK_SIZE}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_steps_sample "${EVAL_STEPS_SAMPLE}" \
    --eval_topk "${EVAL_TOPK}" \
    --keep_checkpoints "${KEEP_CHECKPOINTS}" \
    --acl_dev_jsonl "${ACL_DEV_JSONL}" \
    --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
    --eval_glossary_sizes ${EVAL_GLOSSARY_SIZES} \
    --best_metric "${BEST_METRIC}" \
    --best_metric_secondary "${BEST_METRIC_SECONDARY}" \
    --eval_top100_samples 1 \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${WANDB_EXP_NAME}" \
    --tcm_loss_weight "${TCM_LOSS_WEIGHT}" \
    --tcm_pos_threshold "${TCM_POS_THRESHOLD}" \
    --tcm_neg_threshold "${TCM_NEG_THRESHOLD}" \
    --tcm_loss_form "${TCM_LOSS_FORM}" \
    --tcm_reduction "${TCM_REDUCTION}" \
    --hcl_beta "${HCL_BETA}" \
    --max_train_seconds "${MAX_TRAIN_SECONDS}" \
    ${OPTS}

echo "[SMOKE] smoke run completed at $(date)"
