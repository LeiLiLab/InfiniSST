#!/bin/bash
#SBATCH --job-name=q3_mfa_hnps_k512_6gpu
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:6
#SBATCH --time=28:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_mfa_hnps_k512_6gpu_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_mfa_hnps_k512_6gpu_%x.err

# Diffs vs run_mfa_smallest_dense_k1024_aries.sh (43848 / zv28ve3q):
#   * HARD_NEG_K=0, HARD_NEG_K_PER_SAMPLE=512 (was pool k=64).
#     43848 finalize showed delta_gs10000=-0.0171 vs per-sample k=1024
#     (r0xi5xkt best_sec=0.8915), so the pool HN is the OOD bottleneck.
#   * NUM_GPUS=6 (was 8), PER_GPU_BATCH=2048 (was 1536); BATCH_SIZE=12288
#     preserved. Leaves 2 aries GPUs idle by design per launch plan.
#     Memory: at B=2048 W=230 D=1024 bf16 the max-sim tensor ~0.93 GB/rank
#     on 48GB A6000, comfortable.
#   * Passes the mandatory experiment_tracking flags
#     (--experiment_family / --data_tag / --task_tag / --extra_wandb_tags /
#     --baseline_run_ids / --notes_file) that 43848's launcher silently
#     skipped.
#
# Submit:
#   sbatch run_mfa_smallest_dense_hnps_k512_6gpu_aries.sh
#   MAX_STEPS=50 sbatch run_mfa_smallest_dense_hnps_k512_6gpu_aries.sh  (smoke)

set -euo pipefail

# ======Configuration=====
# --- Loss (kept identical to 43848 variantE baseline) ---
MARGIN="0.0"
HCL_BETA="0.0"
TCM_LOSS_WEIGHT="1.0"
TCM_POS_THRESHOLD="0.85"
TCM_NEG_THRESHOLD="0.25"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"

# --- Data-side hard-neg (PER-SAMPLE k=512, the variable under test) ---
HARD_NEG_K=0
HARD_NEG_K_PER_SAMPLE=512
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=50

# --- term_id normalization (bug fix, same as 43848) ---
TERM_ID_NORMALIZE="aggressive"

# --- Same smallest+dense MFA recipe as 43848 ---
MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"
MAXSIM_STRIDE=2
MFA_WINDOW_SELECTION="smallest"

# --- Shared recipe ---
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

export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO

NUM_GPUS=6
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29964

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="qwen3_rag"

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
NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_hnps_k512.md"

USE_LORA="true"
LORA_RANK=128
LORA_ALPHA=256
TARGET_DIM=1024
TARGET_MODULES="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2"
POOLING_TYPE="transformer"

USE_MAXSIM="true"
MFA_SUPERVISED="true"

# TEXT_LR=0 silently defaults to args.lr inside the trainer
# (qwen3_glossary_neg_train.py:3641). We keep 0 to match 43848's behavior
# (text LoRA at args.lr=1.7e-4, not frozen).
TEXT_LR="0"
TEXT_LORA_RANK=128
TEXT_LORA_ALPHA=256
TEXT_TARGET_MODULES="query key value dense"
TEXT_POOLING="cls"
SPARSE_WEIGHT="0.0"

PER_GPU_BATCH=2048
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
GRAD_CACHE_CHUNK_SIZE=256
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-0}"
EPOCHS="${EPOCHS:-3}"
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

# Experiment tracking (mandatory per .cursor/rules/experiment_tracking.mdc)
EXPERIMENT_FAMILY="sst_ood_hardneg"
DATA_TAG="3variant_1m_mfa"
TASK_TAG="train"
EXTRA_WANDB_TAGS="variant:hnps_k512_smallest_dense_normAGGR_6gpu compute:aries-6gpu"
BASELINE_RUN_IDS="zv28ve3q r0xi5xkt"
# ======Configuration=====

mkdir -p "${SAVE_DIR}"

# Pre-flight: informational GPU check; never block.
python3 - "$NUM_GPUS" <<'PYEOF'
import subprocess, sys
needed = int(sys.argv[1])
threshold_mib = 500
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
free, busy = [], []
for line in out.strip().splitlines():
    idx, used = [x.strip() for x in line.split(",")]
    (free if int(used) <= threshold_mib else busy).append((idx, used))
print(f"[PREFLIGHT] free={free} busy={busy}", file=sys.stderr)
if len(free) < needed:
    print(
        f"[PREFLIGHT][WARN] only {len(free)}/{needed} GPUs under "
        f"{threshold_mib}MiB; proceeding with SLURM-allocated 0..{needed-1}.",
        file=sys.stderr,
    )
PYEOF
SLURM_DEV_LIST="$(seq -s, 0 $((NUM_GPUS - 1)))"
export CUDA_VISIBLE_DEVICES="${SLURM_DEV_LIST}"
echo "[PREFLIGHT] selected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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
VERSION="3var_clean_gc_wr$((WIKI_RANK / 1000))k_m${MARGIN}_maxsim_mfa_variantE_hardneg_ps_k${HARD_NEG_K_PER_SAMPLE}_tcm_ep${EPOCHS}_cold_smallest_dense_normAGGR_6gpu${SMOKE_TAG}"
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_t=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
WANDB_EXP_NAME="variantE_hnps_k${HARD_NEG_K_PER_SAMPLE}_smallest_dense_normAGGR_6gpu_${SAVE_NAME}"

echo "[TRAIN] VARIANT=E_hnps_k${HARD_NEG_K_PER_SAMPLE}_smallest_dense_normAGGR_6gpu"
echo "[TRAIN] MFA_WINDOW_SELECTION=${MFA_WINDOW_SELECTION}"
echo "[TRAIN] MAXSIM_WINDOWS=${MAXSIM_WINDOWS}"
echo "[TRAIN] MAX_TRAIN_SECONDS=${MAX_TRAIN_SECONDS} EPOCHS=${EPOCHS} MAX_STEPS=${MAX_STEPS}"
echo "[TRAIN] Save: ${SAVE_PATH}"
echo "[TRAIN] Batch: ${BATCH_SIZE} (${NUM_GPUS} * ${PER_GPU_BATCH})  grad_cache_chunk=${GRAD_CACHE_CHUNK_SIZE}"
echo "[TRAIN] HARD_NEG_K=${HARD_NEG_K} HARD_NEG_K_PER_SAMPLE=${HARD_NEG_K_PER_SAMPLE} NEG_BANK_REFRESH_STEPS=${NEG_BANK_REFRESH_STEPS}"
echo "[TRAIN] TERM_ID_NORMALIZE=${TERM_ID_NORMALIZE}"
echo "[TRAIN] TCM: lambda=${TCM_LOSS_WEIGHT} T_beta=${TCM_POS_THRESHOLD} T_alpha=${TCM_NEG_THRESHOLD}"
echo "[TRAIN] experiment_family=${EXPERIMENT_FAMILY} data_tag=${DATA_TAG} task_tag=${TASK_TAG}"
echo "[TRAIN] extra_wandb_tags=${EXTRA_WANDB_TAGS}"
echo "[TRAIN] baseline_run_ids=${BASELINE_RUN_IDS}"
echo "[TRAIN] notes_file=${NOTES_FILE}"

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
    --mfa_window_selection "${MFA_WINDOW_SELECTION}" \
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
    --term_id_normalize "${TERM_ID_NORMALIZE}" \
    --max_train_seconds "${MAX_TRAIN_SECONDS}" \
    --experiment_family "${EXPERIMENT_FAMILY}" \
    --data_tag "${DATA_TAG}" \
    --task_tag "${TASK_TAG}" \
    --extra_wandb_tags ${EXTRA_WANDB_TAGS} \
    --baseline_run_ids ${BASELINE_RUN_IDS} \
    --notes_file "${NOTES_FILE}" \
    ${OPTS}

echo "[TRAIN] Variant E HN per-sample k=${HARD_NEG_K_PER_SAMPLE} smallest+dense 6GPU completed at $(date)"
