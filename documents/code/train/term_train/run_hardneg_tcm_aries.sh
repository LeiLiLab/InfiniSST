#!/bin/bash
#SBATCH --job-name=q3_variant_E
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=14:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_variant_E_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_variant_E_%x.err

# Variant E: InfoNCE + TCM + bank-mined hard negatives (HCL dropped).
#
# Rationale (settled by the smoke A/B under identical LR/refresh conditions,
# jobs 43763 vs 43764):
#   * With genuine in-domain hard negatives from the bank already in every
#     batch (hard_neg_k=64), InfoNCE's softmax already concentrates gradient
#     on the hardest negatives.  Adding Robinson HCL on top only inflates the
#     loss magnitude by ~0.45 without moving pos_sim / neg_sim_mean (score
#     gap was actually 0.047 tighter under HCL=1 at step 20).
#   * TCM with T_beta=0.85 / T_alpha=0.25 / lambda=1.0 is kept, because the
#     TCM_neg term only activates meaningfully when the batch contains hard
#     negatives, which the bank now guarantees.
#
# NegativeTermBank.refresh() is now DDP-parallel: every rank encodes a
# shard of the bank and all_gather re-assembles the full table.  Combined
# with refresh_every=50 the amortized cost stays well under 3% wall-time
# on 8 GPUs.
#
# Budget: 5 full epochs over the 6.5M-sample training set (no wall-time
# cap).  Rough runtime estimate at ~13 s/step * 530 steps * 5 epochs
# ~ 9.5 h.  slurm --time=14:00:00 gives a safety margin for eval / ckpt IO.
#
# Resume is disabled (from scratch).  Submit with plain:
#   sbatch run_hardneg_tcm_aries.sh

set -euo pipefail

# ======Configuration=====
# --- Variant-specific loss flags (variant E: bank-mined hard-negs + TCM) ---
# MARGIN stays at 0 so TCM's positive threshold is the only pos-push signal
# (avoid double-pushing with CosFace margin which confounds TCM's effect).
MARGIN="0.0"
HCL_BETA="0.0"
TCM_LOSS_WEIGHT="1.0"
TCM_POS_THRESHOLD="0.85"
TCM_NEG_THRESHOLD="0.25"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"

# --- Data-side hard-neg bank (the core change vs ablation A/B/C/D) ---
HARD_NEG_K=64
NEG_BANK_SIZE=0
# refresh every 50 optimizer steps; DDP-parallel refresh keeps overhead flat.
NEG_BANK_REFRESH_STEPS=50

# --- Shared recipe (mirrors run_tcm_hcl_ablation_aries.sh) ---
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
export TORCH_DISTRIBUTED_DEBUG=INFO

NUM_GPUS=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29965

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

PER_GPU_BATCH=1536
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
GRAD_CACHE_CHUNK_SIZE=256
# Full 5-epoch training (no wall-time cap).  Aries has infinite partition
# time limit so we use the natural cosine LR schedule across
# total_steps = steps_per_epoch * EPOCHS.  At bs=12288 / 6.5M samples this
# is 530 steps per epoch -> ~2650 total optimizer steps; warmup at 10%
# lands at ~265 steps, giving the model a full decay curve (unlike the
# prior 2h / EPOCHS=1 wall-time-capped ablation runs).  MAX_TRAIN_SECONDS
# must stay at 0 to bypass the script's epochs-vs-walltime alignment guard.
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-0}"
EPOCHS="${EPOCHS:-5}"
NUM_WORKERS=4
LR="1.7e-4"
TEMPERATURE="0.07"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
WIKI_RANK=1000000
NOISY_RATIO=0.0
ONLINE_HARD_NEG_K=0

GLOSSARY_NEG_PATH=""
GLOSSARY_NEG_REFRESH_STEPS=0

# Relaxed eval/save for wall-time budget (matches ablation A/B/C/D cadence).
SAVE_STEPS=999999
EVAL_STEPS_SAMPLE=40
KEEP_CHECKPOINTS=2
EVAL_TOPK=10

ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
EVAL_GLOSSARY_SIZES="1000 10000"
BEST_METRIC="eval_acl6060/recall@10_gs1000"
BEST_METRIC_SECONDARY="eval_acl6060/recall@10_gs10000"
# ======Configuration=====

mkdir -p "${SAVE_DIR}"

# Pre-flight: pick NUM_GPUS clean GPUs (see ablation launcher).
PREFLIGHT_OUT="$(python3 - "$NUM_GPUS" <<'PYEOF'
import os, subprocess, sys, time
needed = int(sys.argv[1])
threshold_mib = 500
max_retry = 12
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

BS_ABBR=$((BATCH_SIZE / 1024))k
if [ $((BATCH_SIZE % 1024)) -ne 0 ]; then
    BS_ABBR="${BATCH_SIZE}"
fi

TEXT_TAG="tr${TEXT_LORA_RANK}"
MODE_NAME="scale_lora-r${LORA_RANK}-${TEXT_TAG}"
VERSION="3var_clean_gc_wr$((WIKI_RANK / 1000))k_m${MARGIN}_maxsim_mfa_variantE_hardneg_tcm_ep${EPOCHS}"
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_t=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="variantE_${SAVE_NAME}"

echo "[TRAIN] VARIANT=E (bank-mined hard-neg + TCM, HCL dropped)"
echo "[TRAIN] MAX_TRAIN_SECONDS=${MAX_TRAIN_SECONDS} EPOCHS=${EPOCHS}"
echo "[TRAIN] Save: ${SAVE_PATH}"
echo "[TRAIN] Batch: ${BATCH_SIZE} (${NUM_GPUS} GPUs * ${PER_GPU_BATCH})"
echo "[TRAIN] HARD_NEG_K=${HARD_NEG_K} NEG_BANK_REFRESH_STEPS=${NEG_BANK_REFRESH_STEPS} (DDP-parallel refresh)"
echo "[TRAIN] TCM: lambda=${TCM_LOSS_WEIGHT} T_beta=${TCM_POS_THRESHOLD} T_alpha=${TCM_NEG_THRESHOLD} form=${TCM_LOSS_FORM} reduction=${TCM_REDUCTION}"
echo "[TRAIN] HCL: beta=${HCL_BETA} (disabled)"
echo "[TRAIN] MARGIN=${MARGIN}"

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
    --eval_top100_samples 3 \
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

echo "[TRAIN] Variant E completed at $(date)"
