#!/bin/bash
#SBATCH --job-name=q3_variantE_ps_k1024
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=28:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_variantE_ps_k1024_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_variantE_ps_k1024_%x.err

# Variant E (per-sample HN K=1024, cold-start 5 ep).  Rationale:
#   * Replaces the old shared-pool mining (--hard_neg_k) which dedup-collapsed
#     150k-300k unique HN columns into a single bank shared across the batch,
#     causing cross-row contamination (row A's HN ~= row B's GT) and
#     cold-start mode collapse when k >= 128.
#   * New strategy: each anchor mines its OWN top-K bank terms (no dedup,
#     no cross-row sharing).  Similarity matrix is extended by exactly K=1024
#     per-row private columns (see compute_masked_contrastive_loss in
#     qwen3_glossary_neg_train.py).
#   * Advisor guidance: since the new path is a per-row matmul, K can be
#     pushed well above the pool limit without OOM.  Start at K=1024 (same
#     nominal budget as the failed k=1024 pool run for apples-to-apples);
#     scale up to K=4096 in a follow-up if OOD still plateaus.
#
# Budget: 5 full epochs from scratch (matches k=64 baseline / variantE).
# ~530 steps/epoch * 5 = 2650 steps.  Expected wall time: ~15-20 s/step * 2650
# = 11-15h.  slurm --time=28h gives 2x safety margin.
#
# Submit: sbatch run_hardneg_per_sample_k1024_cold_aries.sh        (full 5-ep)
#         MAX_STEPS=50 sbatch run_hardneg_per_sample_k1024_cold_aries.sh  (smoke)

set -euo pipefail

# ======Configuration=====
# --- Loss (kept identical to variant E baseline) ---
MARGIN="0.0"
HCL_BETA="0.0"
TCM_LOSS_WEIGHT="1.0"
TCM_POS_THRESHOLD="0.85"
TCM_NEG_THRESHOLD="0.25"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"

# --- Data-side hard-neg: per-sample, NOT pool (mutually exclusive) ---
HARD_NEG_K=0
HARD_NEG_K_PER_SAMPLE=1024
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=50

# --- Shared recipe (mirrors run_hardneg_k1024_aries.sh baseline) ---
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

# Use /tmp (not /dev/shm) to avoid systemd/SLURM tmpfs cleanup race that
# killed earlier runs with FileNotFoundError on pymp-* dirs.
LOCAL_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-local}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
trap '[ -n "${LOCAL_TMP_DIR:-}" ] && rm -rf "${LOCAL_TMP_DIR}"' EXIT

export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO

NUM_GPUS=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29962

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

# HF cache on aries local partition (writable everywhere we might run).
export HF_HOME="/mnt/aries/home/jiaxuanluo/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="/mnt/aries/data4/jiaxuanluo/cache/torch"
export XDG_CACHE_HOME="/mnt/aries/data4/jiaxuanluo/cache"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

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
# Per-sample HN adds an [B_local, K, D] detached tensor to phase-2 loss.
# K=1024, D=1024, bf16 => 3 GB / rank.  The old shared-pool path at k=1024
# used chunk=256 comfortably (19 GB / 49 GB).  Per-sample is cheaper on the
# in-batch sim matmul side (N_global stays == B_global), so chunk=256 is safe.
GRAD_CACHE_CHUNK_SIZE=256
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-0}"
EPOCHS="${EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-0}"
NUM_WORKERS=4
LR="1.7e-4"
TEMPERATURE="0.07"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
WIKI_RANK=1000000
NOISY_RATIO=0.0
ONLINE_HARD_NEG_K=0

GLOSSARY_NEG_PATH=""
GLOSSARY_NEG_REFRESH_STEPS=0

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

# Pre-flight: pick NUM_GPUS clean GPUs (defensive against root-owned processes
# holding memory without SLURM accounting - see 43808 postmortem).
PREFLIGHT_OUT="$(python3 - "$NUM_GPUS" <<'PYEOF'
import subprocess, sys, time
needed = int(sys.argv[1])
threshold_mib = 500
max_retry = 6
sleep_s = 20
for attempt in range(1, max_retry + 1):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    free, busy = [], []
    for line in out.strip().splitlines():
        idx, used = [x.strip() for x in line.split(",")]
        if int(used) <= threshold_mib:
            free.append(idx)
        else:
            busy.append((idx, used))
    print(f"[PREFLIGHT] attempt={attempt} free={free} busy={busy}", file=sys.stderr)
    if len(free) >= needed:
        print(",".join(free[:needed]))
        sys.exit(0)
    time.sleep(sleep_s)
print(f"[PREFLIGHT] only {len(free)}/{needed} clean GPUs after {max_retry*sleep_s}s, aborting.", file=sys.stderr)
sys.exit(1)
PYEOF
)"
if [ -z "${PREFLIGHT_OUT}" ]; then
    echo "[PREFLIGHT] failed to pick ${NUM_GPUS} clean GPUs" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${PREFLIGHT_OUT}"
echo "[PREFLIGHT] selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Defensive re-mkdir right before torchrun (systemd tmpfs cleaner hit us before).
mkdir -p "${LOCAL_TMP_DIR}"
if [ ! -d "${LOCAL_TMP_DIR}" ]; then
    echo "[PREFLIGHT][FATAL] TMPDIR ${LOCAL_TMP_DIR} is not writable" >&2
    exit 1
fi
echo "[PREFLIGHT] TMPDIR=${LOCAL_TMP_DIR} exists=$(stat -c %F "${LOCAL_TMP_DIR}")"

BS_ABBR=$((BATCH_SIZE / 1024))k
if [ $((BATCH_SIZE % 1024)) -ne 0 ]; then
    BS_ABBR="${BATCH_SIZE}"
fi

TEXT_TAG="tr${TEXT_LORA_RANK}"
MODE_NAME="scale_lora-r${LORA_RANK}-${TEXT_TAG}"
SMOKE_TAG=""
if [ "${MAX_STEPS}" -gt 0 ]; then
    SMOKE_TAG="_smoke${MAX_STEPS}"
fi
VERSION="3var_clean_gc_wr$((WIKI_RANK / 1000))k_m${MARGIN}_maxsim_mfa_variantE_hardneg_per_sample_k${HARD_NEG_K_PER_SAMPLE}_tcm_ep${EPOCHS}_cold${SMOKE_TAG}"
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_t=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="variantE_ps_k${HARD_NEG_K_PER_SAMPLE}_${SAVE_NAME}"

echo "[TRAIN] VARIANT=E_per_sample_k${HARD_NEG_K_PER_SAMPLE} (cold start, 5 epoch)"
echo "[TRAIN] MAX_TRAIN_SECONDS=${MAX_TRAIN_SECONDS} EPOCHS=${EPOCHS} MAX_STEPS=${MAX_STEPS}"
echo "[TRAIN] Save: ${SAVE_PATH}"
echo "[TRAIN] Batch: ${BATCH_SIZE} (${NUM_GPUS} * ${PER_GPU_BATCH})  grad_cache_chunk=${GRAD_CACHE_CHUNK_SIZE}"
echo "[TRAIN] HARD_NEG_K_PER_SAMPLE=${HARD_NEG_K_PER_SAMPLE} NEG_BANK_REFRESH_STEPS=${NEG_BANK_REFRESH_STEPS}"
echo "[TRAIN] TCM: lambda=${TCM_LOSS_WEIGHT} T_beta=${TCM_POS_THRESHOLD} T_alpha=${TCM_NEG_THRESHOLD}"

OPTS=""
if [ "${USE_LORA}" = "true" ]; then OPTS="${OPTS} --use_lora"; fi
if [ "${USE_MAXSIM}" = "true" ]; then OPTS="${OPTS} --use_maxsim"; fi
if [ "${MFA_SUPERVISED}" = "true" ]; then OPTS="${OPTS} --mfa_supervised_maxsim"; fi
if [ "${WIKI_RANK}" -gt 0 ]; then OPTS="${OPTS} --wiki_rank ${WIKI_RANK}"; fi
if [ "${MAX_STEPS}" -gt 0 ]; then OPTS="${OPTS} --max_steps ${MAX_STEPS}"; fi

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
    --hard_neg_k_per_sample "${HARD_NEG_K_PER_SAMPLE}" \
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
    --eval_minimal_metrics \
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

echo "[TRAIN] Variant E per_sample k${HARD_NEG_K_PER_SAMPLE} completed at $(date)"
