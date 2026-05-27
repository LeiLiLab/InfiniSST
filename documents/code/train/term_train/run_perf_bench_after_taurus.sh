#!/bin/bash
#SBATCH --job-name=q3_perf_bench_after
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=0-02:30:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_perf_bench_after.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_perf_bench_after.err

# 200-step A/B bench: qwen3_glossary_neg_train AFTER nonblock+cache+sync-removal.
# Cfg cloned from 43849 (run_ablation_A_k1024_normalize_aggressive_taurus.sh)
# with max_steps=200, no eval, no save, NUM_GPUS=2, WandB enabled.
# Compares step_time_ms to 43849's 25.5s/it tqdm steady-state.

set -euo pipefail

export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

LOCAL_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-local}/pytorch_tmp"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
trap '[ -n "${LOCAL_TMP_DIR:-}" ] && rm -rf "${LOCAL_TMP_DIR}"' EXIT

export NCCL_TIMEOUT=1800
export TORCH_DISTRIBUTED_DEBUG=INFO

export HF_HOME="/mnt/aries/home/jiaxuanluo/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}"

TRAIN_JSONL="/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_perf_bench.md"
SAVE_DIR="/tmp/${USER}_${SLURM_JOB_ID:-local}/perf_bench_out"
SAVE_PATH="${SAVE_DIR}/perf_bench.pt"
mkdir -p "${SAVE_DIR}"

STAMP="$(date +%Y%m%d-%H%M)"
WANDB_EXP_NAME="retriever_perf__after_nonblock_cache_sync__${STAMP}__2gpu_200steps"

# Clone of 43849 config
HARD_NEG_K=0
HARD_NEG_K_PER_SAMPLE=1024
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=50
TERM_ID_NORMALIZE="aggressive"

USE_LORA="true"
LORA_RANK=128
LORA_ALPHA=256
TARGET_DIM=1024
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"

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

NUM_GPUS=2
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29972

PER_GPU_BATCH=2048
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
GRAD_CACHE_CHUNK_SIZE=512
MAX_STEPS=200
NUM_WORKERS=4
LR="1.7e-4"
TEMPERATURE="0.07"
WIKI_RANK=1000000
NOISY_RATIO=0.0
ONLINE_HARD_NEG_K=0

TCM_LOSS_WEIGHT="1.0"
TCM_POS_THRESHOLD="0.85"
TCM_NEG_THRESHOLD="0.25"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"
MARGIN="0.0"
HCL_BETA="0.0"

PREFLIGHT_OUT="$(python3 - "$NUM_GPUS" <<'PYEOF'
import subprocess, sys, time
needed = int(sys.argv[1])
threshold_mib = 500
max_retry = 8
sleep_s = 30
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
print(f"[PREFLIGHT] only {len(free)}/{needed} clean GPUs, aborting.", file=sys.stderr)
sys.exit(1)
PYEOF
)"
if [ -z "${PREFLIGHT_OUT}" ]; then
    echo "[PREFLIGHT] failed to pick ${NUM_GPUS} clean GPUs" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${PREFLIGHT_OUT}"
echo "[PREFLIGHT] selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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
    --epochs 1 \
    --max_steps "${MAX_STEPS}" \
    --num_workers "${NUM_WORKERS}" \
    --temperature "${TEMPERATURE}" \
    --target_dim "${TARGET_DIM}" \
    --pooling_type "transformer" \
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
    --neg_bank_size "${NEG_BANK_SIZE}" \
    --neg_bank_refresh_steps "${NEG_BANK_REFRESH_STEPS}" \
    --hard_neg_k "${HARD_NEG_K}" \
    --hard_neg_k_per_sample "${HARD_NEG_K_PER_SAMPLE}" \
    --noisy_ratio "${NOISY_RATIO}" \
    --margin "${MARGIN}" \
    --online_hard_neg_k "${ONLINE_HARD_NEG_K}" \
    --grad_cache_chunk_size "${GRAD_CACHE_CHUNK_SIZE}" \
    --save_steps 999999 \
    --eval_steps_sample 0 \
    --keep_checkpoints 1 \
    --tcm_loss_weight "${TCM_LOSS_WEIGHT}" \
    --tcm_pos_threshold "${TCM_POS_THRESHOLD}" \
    --tcm_neg_threshold "${TCM_NEG_THRESHOLD}" \
    --tcm_loss_form "${TCM_LOSS_FORM}" \
    --tcm_reduction "${TCM_REDUCTION}" \
    --hcl_beta "${HCL_BETA}" \
    --term_id_normalize "${TERM_ID_NORMALIZE}" \
    --enable_wandb \
    --wandb_project "qwen3_rag" \
    --wandb_exp_name "${WANDB_EXP_NAME}" \
    --wandb_log_interval 5 \
    --experiment_family "retriever_perf" \
    --data_tag "3variant_1m_mfa" \
    --task_tag "smoke" \
    --extra_wandb_tags "gpus:2" "steps:200" "variant:after_nonblock_cache_sync" \
    --baseline_run_ids "43849" \
    --notes_file "${NOTES_FILE}" \
    ${OPTS}

echo "[PERF BENCH AFTER] done at $(date)"
