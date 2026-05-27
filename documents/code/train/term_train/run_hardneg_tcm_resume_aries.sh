#!/bin/bash
#SBATCH --job-name=q3_variantE_resume
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=20:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_variantE_resume_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_variantE_resume_%x.err

# Variant E resume: continue training from the ep5 best-acl6060_gs10000 checkpoint
# for 5 MORE epochs to diagnose whether the in-domain plateau (DEV gs10k ~0.85 at
# tau=0.85) simply needs more gradient steps rather than a structural change.
#
# Resume points:
#   - model / text encoder weights
#   - optimizer (AdamW momentum/variance)
#   - best_metric_value  (so later best-ckpts only fire on improvement)
#   - epoch / global_step counters
#
# Scheduler: --reset_scheduler so the cosine curve is re-drawn over the NEW
#            total_steps = 530 * EPOCHS (EPOCHS=10 -> 5300 steps) and fast-
#            forwarded to the resume step.  This gives the remaining 2650
#            steps a fresh, non-zero LR decay tail.  Without --reset_scheduler
#            we would snap to LR=0 immediately and make no further progress.
#
# All other hyperparameters are frozen at variant E values so only "more
# steps" is the independent variable.
#
# Usage:
#   sbatch run_hardneg_tcm_resume_aries.sh

set -euo pipefail

# ======Configuration=====
# --- Resume source (fully qualified cross-node path). ---
RESUME_CKPT="/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_tcm_ep5_best_acl6060_gs10000.pt"

# --- Variant E loss/neg-bank recipe (identical to run_hardneg_tcm_aries.sh). ---
MARGIN="0.0"
HCL_BETA="0.0"
TCM_LOSS_WEIGHT="1.0"
TCM_POS_THRESHOLD="0.85"
TCM_NEG_THRESHOLD="0.25"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"

HARD_NEG_K=64
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=50

# --- Training budget: extend from ep5 -> ep10 (5 MORE epochs over 6.5M samples). ---
EPOCHS=10
MAX_TRAIN_SECONDS=0

# --- Env (fully qualified paths; mirrors run_hardneg_ablation_aries.sh after
#     the /tmp + HF_HOME fix). ---
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

# Per-job isolated tmp on node-local /tmp (NOT /dev/shm).  /dev/shm on aries
# has been observed to silently disappear mid-run under systemd/slurm tmpfs
# cleanup, causing DataLoader worker mkdtemp() to fail with ENOENT.
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
MASTER_PORT=29968

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

# HuggingFace cache at the path where Qwen3 Audio is already materialized on
# aries (node-local $HOME is not portable, and /mnt/aries/data4 gave
# write-permission errors on some nodes).
export HF_HOME="/mnt/aries/home/jiaxuanluo/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
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
MFA_WINDOW_SELECTION="hard_max"
MFA_LSE_TEMPERATURE=1.0

TEXT_LR="0"
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
TEXT_TARGET_MODULES="query key value dense"
TEXT_POOLING="cls"
SPARSE_WEIGHT="0.0"

PER_GPU_BATCH=1536
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
GRAD_CACHE_CHUNK_SIZE=256
NUM_WORKERS=4
LR="1.7e-4"
TEMPERATURE="0.07"
TRAIN_LIMIT=0
WIKI_RANK=1000000
NOISY_RATIO=0.0
ONLINE_HARD_NEG_K=0

GLOSSARY_NEG_PATH=""
GLOSSARY_NEG_REFRESH_STEPS=0

# Checkpoint / eval cadence: save best-only (step save disabled) + eval every
# 80 steps to keep the resume run's wandb curve comparable to the ablation
# launcher while not spamming logs.
SAVE_STEPS=999999
EVAL_STEPS_SAMPLE=80
KEEP_CHECKPOINTS=2
EVAL_TOPK=10

ACL_DEV_JSONL="/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"
EVAL_GLOSSARY_SIZES="1000 10000"
BEST_METRIC="eval_acl6060/recall@10_gs1000"
BEST_METRIC_SECONDARY="eval_acl6060/recall@10_gs10000"
# ======Configuration=====

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[RESUME] missing resume checkpoint: ${RESUME_CKPT}" >&2
    exit 1
fi
mkdir -p "${SAVE_DIR}"

# Preflight: verify every visible GPU is actually free.  SLURM does NOT
# isolate us from non-SLURM root processes (e.g. swift/HF exports running
# under root on the same node), and those can silently hold 20+ GiB per card
# which causes the resume run to OOM ~4 minutes into the first forward pass
# (see job 43808 postmortem).  Fail fast here if <NUM_GPUS clean cards so
# we don't burn a scheduling slot.
PREFLIGHT_GPU_THRESHOLD_MIB=500
PREFLIGHT_RETRIES=6
PREFLIGHT_SLEEP_S=20
preflight_out=""
for attempt in $(seq 1 ${PREFLIGHT_RETRIES}); do
    preflight_out="$(python3 - "$NUM_GPUS" "$PREFLIGHT_GPU_THRESHOLD_MIB" <<'PYEOF'
import subprocess, sys
needed = int(sys.argv[1])
threshold_mib = int(sys.argv[2])
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
free, busy = [], []
for line in out.strip().splitlines():
    idx, used = [x.strip() for x in line.split(",")]
    (free if int(used) <= threshold_mib else busy).append((idx, used))
print(f"free={[i for i,_ in free]} busy={busy}", file=sys.stderr)
if len(free) >= needed:
    print(",".join(i for i, _ in free[:needed]))
    sys.exit(0)
sys.exit(1)
PYEOF
    )" && break
    echo "[PREFLIGHT] attempt=${attempt}/${PREFLIGHT_RETRIES}: fewer than ${NUM_GPUS} clean GPUs, sleeping ${PREFLIGHT_SLEEP_S}s..." >&2
    sleep "${PREFLIGHT_SLEEP_S}"
done
if [ -z "${preflight_out}" ]; then
    echo "[PREFLIGHT][FATAL] could not find ${NUM_GPUS} clean GPUs after ${PREFLIGHT_RETRIES} tries" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${preflight_out}"
echo "[PREFLIGHT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

BS_ABBR=$((BATCH_SIZE / 1024))k
if [ $((BATCH_SIZE % 1024)) -ne 0 ]; then
    BS_ABBR="${BATCH_SIZE}"
fi

TEXT_TAG="tr${TEXT_LORA_RANK}"
MODE_NAME="scale_lora-r${LORA_RANK}-${TEXT_TAG}"
VERSION="3var_clean_gc_wr$((WIKI_RANK / 1000))k_m${MARGIN}_maxsim_mfa_variantE_hardneg_tcm_ep${EPOCHS}_resume_from_ep5best"
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_t=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="variantE_resume_${SAVE_NAME}"

echo "[TRAIN] VARIANT=E RESUME (continue 5 more epochs from ep5 best)"
echo "[TRAIN] RESUME_CKPT=${RESUME_CKPT}"
echo "[TRAIN] EPOCHS=${EPOCHS} (was 5, +5 more) -> total_steps ~= 530*${EPOCHS}"
echo "[TRAIN] Save: ${SAVE_PATH}"
echo "[TRAIN] Batch: ${BATCH_SIZE} (${NUM_GPUS} GPUs * ${PER_GPU_BATCH})"
echo "[TRAIN] HARD_NEG_K=${HARD_NEG_K} NEG_BANK_REFRESH_STEPS=${NEG_BANK_REFRESH_STEPS}"
echo "[TRAIN] TCM: lambda=${TCM_LOSS_WEIGHT} T_beta=${TCM_POS_THRESHOLD} T_alpha=${TCM_NEG_THRESHOLD}"
echo "[TRAIN] LR schedule: --reset_scheduler (cosine re-draw over ${EPOCHS} epochs)"

OPTS=""
if [ "${USE_LORA}" = "true" ]; then OPTS="${OPTS} --use_lora"; fi
if [ "${USE_MAXSIM}" = "true" ]; then OPTS="${OPTS} --use_maxsim"; fi
if [ "${MFA_SUPERVISED}" = "true" ]; then OPTS="${OPTS} --mfa_supervised_maxsim"; fi
if [ "${WIKI_RANK}" -gt 0 ]; then OPTS="${OPTS} --wiki_rank ${WIKI_RANK}"; fi

mkdir -p "${LOCAL_TMP_DIR}"
if [ ! -d "${LOCAL_TMP_DIR}" ]; then
    echo "[PREFLIGHT][FATAL] TMPDIR ${LOCAL_TMP_DIR} is not writable" >&2
    exit 1
fi
echo "[PREFLIGHT] TMPDIR=${LOCAL_TMP_DIR} exists=$(stat -c %F "${LOCAL_TMP_DIR}")"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_PATH}" \
    --resume "${RESUME_CKPT}" \
    --reset_scheduler \
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
    --mfa_window_selection "${MFA_WINDOW_SELECTION}" \
    --mfa_lse_temperature "${MFA_LSE_TEMPERATURE}" \
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
    --eval_top100_samples 0 \
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

echo "[TRAIN] Variant E resume completed at $(date)"
