#!/bin/bash
#SBATCH --job-name=q3_ablate
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=2:45:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ablate_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ablate_%x.err

# 2x2 factorial ablation: { vanilla InfoNCE, HCL reweighting } x
# { no TCM, TCM auxiliary loss }.
#
# Variants (select via env var VARIANT = A | B | C | D):
#   A: vanilla InfoNCE                     (reference baseline)
#   B: InfoNCE + HCL  (beta=1)             (hard-neg importance reweighting only)
#   C: InfoNCE + TCM  (lambda=0.1)         (absolute-threshold calibration only)
#   D: InfoNCE + HCL + TCM                 (interaction: HCL focuses TCM's neg push)
#
# Margin is set to 0 across all variants so TCM's pos-side effect can be
# isolated from CosFace margin's redundant pos-pushing.
#
# Budget knobs (env-overridable):
#   MAX_TRAIN_SECONDS defaults to 7200 (2h training wall-time per variant,
#                     chosen so every variant sees the same compute budget
#                     independent of step/epoch count).
#   EPOCHS            defaults to 1.  NOTE: the cosine LR schedule spans
#                     total_steps = steps_per_epoch * EPOCHS, so if EPOCHS is
#                     large while MAX_TRAIN_SECONDS is short the run will
#                     spend its entire budget inside LR warmup and the model
#                     will effectively not train.  Keep EPOCHS aligned with
#                     the wall-time budget (1 epoch ~ 2h at this batch size).
#   TRAIN_LIMIT       defaults to 0 (full dataset).
#
# Resume is disabled (training from scratch per 2x2 protocol).
#
# Submit:
#   VARIANT=A sbatch run_tcm_hcl_ablation_aries.sh
#   VARIANT=B sbatch run_tcm_hcl_ablation_aries.sh
#   VARIANT=C sbatch run_tcm_hcl_ablation_aries.sh
#   VARIANT=D sbatch run_tcm_hcl_ablation_aries.sh

set -euo pipefail

# ======Configuration=====
VARIANT="${VARIANT:-C}"

# --- Variant-specific loss flags (default: all off) ---
MARGIN="0.0"         # Zero margin for clean TCM-vs-InfoNCE single variable.
TCM_LOSS_WEIGHT="0.0"
HCL_BETA="0.0"
case "${VARIANT}" in
    A)  # InfoNCE baseline
        VARIANT_SUFFIX="A_infonce"
        ;;
    B)  # InfoNCE + HCL hard-negative reweighting
        HCL_BETA="1.0"
        VARIANT_SUFFIX="B_hcl_b1"
        ;;
    C)  # InfoNCE + TCM absolute-threshold calibration
        TCM_LOSS_WEIGHT="0.1"
        VARIANT_SUFFIX="C_tcm_l01"
        ;;
    D)  # InfoNCE + HCL + TCM (interaction)
        HCL_BETA="1.0"
        TCM_LOSS_WEIGHT="0.1"
        VARIANT_SUFFIX="D_hcl_b1_tcm_l01"
        ;;
    *)
        echo "[ERROR] Unknown VARIANT=${VARIANT}; expected one of {A, B, C, D}."
        exit 1
        ;;
esac

# --- Shared recipe (mirrors run_3variant_1m_aries_gc12k_maxsim_mfa.sh) ---
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
# Pre-flight below discovers clean GPUs dynamically and sets
# CUDA_VISIBLE_DEVICES; fails loud if fewer than NUM_GPUS are free (Aries
# occasionally has out-of-band workloads squatting on some GPUs).
NUM_GPUS=8
MASTER_ADDR="127.0.0.1"
# Pick a distinct port per variant so concurrent submissions don't collide.
case "${VARIANT}" in
    A) MASTER_PORT=29961 ;;
    B) MASTER_PORT=29962 ;;
    C) MASTER_PORT=29963 ;;
    D) MASTER_PORT=29964 ;;
esac

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
# /mnt/gemini/home: 4.2T free (vs /mnt/aries/data4 at 339G / 96% used);
# NFS not nvme, but ckpt I/O happens only on eval improvements so the slower
# backend is acceptable.
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
# Walltime-capped training: 2 hours of training wall-time per variant so all
# variants see the same compute budget regardless of step/epoch count.
# IMPORTANT: EPOCHS also controls the cosine LR schedule horizon via
# total_steps = steps_per_epoch * EPOCHS.  Setting EPOCHS >> wall-time budget
# leaves the run stuck in warmup; keep EPOCHS equal to the number of epochs
# the wall-time can actually cover (1 epoch ~ 2h at this batch size).
# The training script hard-errors if max_train_seconds > 0 and epochs > 2.
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-7200}"
EPOCHS="${EPOCHS:-1}"
NUM_WORKERS=4
LR="1.7e-4"
TEMPERATURE="0.07"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
WIKI_RANK=1000000
NOISY_RATIO=0.0
ONLINE_HARD_NEG_K=0

GLOSSARY_NEG_PATH=""
GLOSSARY_NEG_REFRESH_STEPS=0
HARD_NEG_K=0
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=0

# Relaxed eval/save for ablation speed: no periodic save (epoch save only),
# sparse sampled eval (~2 mid-epoch points + end-of-epoch full eval).
SAVE_STEPS=999999
EVAL_STEPS_SAMPLE=40
KEEP_CHECKPOINTS=2
EVAL_TOPK=10

ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
EVAL_GLOSSARY_SIZES="1000 10000"
BEST_METRIC="eval_acl6060/recall@10_gs1000"
BEST_METRIC_SECONDARY="eval_acl6060/recall@10_gs10000"

# TCM thresholds: shared across variants (only TCM_LOSS_WEIGHT toggles).
TCM_POS_THRESHOLD="0.7"
TCM_NEG_THRESHOLD="0.4"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"
# ======Configuration=====

mkdir -p "${SAVE_DIR}"

# Pre-flight: enumerate physical GPUs, pick NUM_GPUS clean ones, and export
# CUDA_VISIBLE_DEVICES accordingly.  Fail loud if fewer than NUM_GPUS clean
# GPUs exist after retries (Aries has non-SLURM vLLM workloads squatting on
# random GPUs).
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
VERSION="3var_clean_gc_wr$((WIKI_RANK / 1000))k_m${MARGIN}_maxsim_mfa_ablate_${VARIANT_SUFFIX}_2h"
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_t=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="ablate_${SAVE_NAME}"

echo "[TRAIN] VARIANT=${VARIANT} (${VARIANT_SUFFIX}) MAX_TRAIN_SECONDS=${MAX_TRAIN_SECONDS} (EPOCHS cap=${EPOCHS})"
echo "[TRAIN] Save: ${SAVE_PATH}"
echo "[TRAIN] Batch: ${BATCH_SIZE} (${NUM_GPUS} GPUs * ${PER_GPU_BATCH})"
echo "[TRAIN] TCM: lambda=${TCM_LOSS_WEIGHT} T_beta=${TCM_POS_THRESHOLD} T_alpha=${TCM_NEG_THRESHOLD} form=${TCM_LOSS_FORM} reduction=${TCM_REDUCTION}"
echo "[TRAIN] HCL: beta=${HCL_BETA}"
echo "[TRAIN] MARGIN=${MARGIN} (set to 0 for clean ablation)"

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

echo "[TRAIN] Variant ${VARIANT} completed at $(date)"
