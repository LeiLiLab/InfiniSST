#!/bin/bash

# Common launcher body for HN-depth scouts on Aries.
# Wrapper scripts provide SBATCH headers plus variant-specific env vars such as
# HN depth, notes file, naming, and time budget.

set -euo pipefail

# ====== Variant-specific env (must be set by wrapper) ======
: "${VARIANT_TAG:?VARIANT_TAG env var is required}"
: "${VERSION:?VERSION env var is required}"
: "${WANDB_EXP_NAME:?WANDB_EXP_NAME env var is required}"
: "${NOTES_FILE:?NOTES_FILE env var is required}"

# ======Configuration=====
MARGIN="${MARGIN:-0.0}"
HCL_BETA="${HCL_BETA:-0.0}"
RESUME="${RESUME:-}"
RESET_SCHEDULER="${RESET_SCHEDULER:-false}"
RESET_BEST_ON_RESUME="${RESET_BEST_ON_RESUME:-false}"
CONSTANT_LR="${CONSTANT_LR:-0.0}"
RESUME_COSINE_DECAY_TO_MAX_STEPS="${RESUME_COSINE_DECAY_TO_MAX_STEPS:-false}"
RUN_VERDICT="${RUN_VERDICT:-}"
EVAL_ONLY="${EVAL_ONLY:-false}"

# This sweep is explicitly non-TCM. Keep all TCM branches at zero so HN depth
# is the only semantic variable moving.
TCM_LOSS_WEIGHT="${TCM_LOSS_WEIGHT:-0.0}"
TCM_POS_LOSS_WEIGHT="${TCM_POS_LOSS_WEIGHT:-0.0}"
TCM_NEG_LOSS_WEIGHT="${TCM_NEG_LOSS_WEIGHT:-0.0}"
TCM_POS_THRESHOLD="${TCM_POS_THRESHOLD:-0.85}"
TCM_NEG_THRESHOLD="${TCM_NEG_THRESHOLD:-0.25}"
TCM_LOSS_FORM="${TCM_LOSS_FORM:-hinge}"
TCM_REDUCTION="${TCM_REDUCTION:-mean_viol}"
TCM_NEG_SCOPE="${TCM_NEG_SCOPE:-all}"
TCM_NEG_TOPK="${TCM_NEG_TOPK:-0}"
TCM_WARMUP_STEPS="${TCM_WARMUP_STEPS:-0}"
TCM_SWEEP_FBETA="${TCM_SWEEP_FBETA:-3.0}"

HARD_NEG_K="${HARD_NEG_K:-0}"
HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-1024}"
NEG_BANK_SIZE="${NEG_BANK_SIZE:-0}"
NEG_BANK_REFRESH_STEPS="${NEG_BANK_REFRESH_STEPS:-50}"

TERM_ID_NORMALIZE="${TERM_ID_NORMALIZE:-aggressive}"
MAXSIM_WINDOWS="${MAXSIM_WINDOWS:-2 3 4 5 6 7 8 10 12 16 20 24}"
MAXSIM_STRIDE="${MAXSIM_STRIDE:-2}"
MFA_WINDOW_SELECTION="${MFA_WINDOW_SELECTION:-smallest}"
MFA_POSITIVE_SCOPE="${MFA_POSITIVE_SCOPE:-auto}"

export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/tmp/${USER}_${SLURM_JOB_ID:-local}/pytorch_tmp}"
mkdir -p "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export TMP="${LOCAL_TMP_DIR}"
export TEMP="${LOCAL_TMP_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
trap '[ -n "${LOCAL_TMP_DIR:-}" ] && rm -rf "${LOCAL_TMP_DIR}"' EXIT

export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO

NUM_GPUS_ENV="${NUM_GPUS:-}"
NUM_GPUS="${NUM_GPUS:-8}"
REQUESTED_CUDA_DEVICES="${CUDA_DEVICE_LIST:-${GPU_LIST:-}}"
REQUESTED_CUDA_DEVICE_COUNT=0
if [ -n "${REQUESTED_CUDA_DEVICES}" ]; then
    REQUESTED_CUDA_DEVICES="$(python3 - "${REQUESTED_CUDA_DEVICES}" <<'PYEOF'
import re
import sys

raw = sys.argv[1]
parts = [p.strip() for p in re.split(r"[,\s]+", raw) if p.strip()]
bad = [p for p in parts if not p.isdigit()]
if not parts:
    print("[ERROR] CUDA_DEVICE_LIST/GPU_LIST did not contain any GPU ids", file=sys.stderr)
    sys.exit(2)
if bad:
    print(f"[ERROR] CUDA_DEVICE_LIST/GPU_LIST has non-integer ids: {bad}", file=sys.stderr)
    sys.exit(2)
if len(set(parts)) != len(parts):
    print(f"[ERROR] CUDA_DEVICE_LIST/GPU_LIST has duplicate ids: {parts}", file=sys.stderr)
    sys.exit(2)
print(",".join(parts))
PYEOF
)"
    REQUESTED_CUDA_DEVICE_COUNT="$(awk -F, '{print NF}' <<< "${REQUESTED_CUDA_DEVICES}")"
    if [ -n "${NUM_GPUS_ENV}" ] && [ "${NUM_GPUS_ENV}" -ne "${REQUESTED_CUDA_DEVICE_COUNT}" ]; then
        echo "[ERROR] NUM_GPUS=${NUM_GPUS_ENV} but CUDA_DEVICE_LIST/GPU_LIST has ${REQUESTED_CUDA_DEVICE_COUNT} ids: ${REQUESTED_CUDA_DEVICES}" >&2
        exit 2
    fi
    NUM_GPUS="${REQUESTED_CUDA_DEVICE_COUNT}"
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29971}"

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=online
WANDB_PROJECT="${WANDB_PROJECT:-qwen3_rag}"

export HF_HOME="${HF_HOME:-/mnt/aries/home/jiaxuanluo/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TORCH_HOME="${TORCH_HOME:-/mnt/aries/data4/jiaxuanluo/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/aries/data4/jiaxuanluo/cache}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m_mfa.jsonl}"
DEV_JSONL="${DEV_JSONL-/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl}"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
SAVE_DIR="/mnt/gemini/home/jiaxuanluo/train_outputs"

USE_LORA="${USE_LORA:-true}"
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
TARGET_DIM="${TARGET_DIM:-1024}"
TARGET_MODULES="${TARGET_MODULES:-q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2}"
POOLING_TYPE="${POOLING_TYPE:-transformer}"
AUDIO_ENCODER_PRESET="${AUDIO_ENCODER_PRESET:-qwen3-omni}"
AUDIO_ENCODER_TYPE="${AUDIO_ENCODER_TYPE:-qwen3_omni}"
AUDIO_MODEL_ID="${AUDIO_MODEL_ID:-Atotti/Qwen3-Omni-AudioTransformer}"
AUDIO_FEATURE_EXTRACTOR_ID="${AUDIO_FEATURE_EXTRACTOR_ID:-openai/whisper-large-v3}"
AUDIO_INPUT_DTYPE="${AUDIO_INPUT_DTYPE:-auto}"
AUDIO_HIDDEN_DIM="${AUDIO_HIDDEN_DIM:-0}"

USE_MAXSIM="${USE_MAXSIM:-true}"
MFA_SUPERVISED="${MFA_SUPERVISED:-true}"

TEXT_ENCODER_PRESET="${TEXT_ENCODER_PRESET:-bge-m3}"
TEXT_MODEL_ID="${TEXT_MODEL_ID:-BAAI/bge-m3}"
TEXT_INPUT_PREFIX="${TEXT_INPUT_PREFIX:-}"
TEXT_LR="${TEXT_LR:-0}"
TEXT_LORA_RANK="${TEXT_LORA_RANK:-128}"
TEXT_LORA_ALPHA="${TEXT_LORA_ALPHA:-256}"
TEXT_TARGET_MODULES="${TEXT_TARGET_MODULES:-query key value dense}"
TEXT_POOLING="${TEXT_POOLING:-cls}"
SPARSE_WEIGHT="${SPARSE_WEIGHT:-0.0}"

# Keep global batch overrideable for uneven GPU-count ablations.
PER_GPU_BATCH="${PER_GPU_BATCH:-1536}"
BATCH_SIZE="${BATCH_SIZE:-$((NUM_GPUS * PER_GPU_BATCH))}"
if [ "${BATCH_SIZE}" -lt "${NUM_GPUS}" ]; then
    echo "[ERROR] BATCH_SIZE=${BATCH_SIZE} is smaller than NUM_GPUS=${NUM_GPUS}" >&2
    exit 2
fi
GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-256}"
FIXED_AUDIO_SECONDS="${FIXED_AUDIO_SECONDS:-0}"
EVAL_FIXED_AUDIO_SECONDS="${EVAL_FIXED_AUDIO_SECONDS:-0}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-0}"
EPOCHS="${EPOCHS:-1}"
SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1.7e-4}"
TEMPERATURE="${TEMPERATURE:-0.07}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
WIKI_RANK="${WIKI_RANK:-1000000}"
NOISY_RATIO="${NOISY_RATIO:-0.0}"
ONLINE_HARD_NEG_K="${ONLINE_HARD_NEG_K:-0}"

GLOSSARY_NEG_PATH="${GLOSSARY_NEG_PATH:-}"
GLOSSARY_NEG_REFRESH_STEPS="${GLOSSARY_NEG_REFRESH_STEPS:-0}"
TRAIN_EXCLUDE_TERM_GLOSSARIES="${TRAIN_EXCLUDE_TERM_GLOSSARIES:-}"
STRICT_TRAIN_EVAL_TERM_FILTER="${STRICT_TRAIN_EVAL_TERM_FILTER:-false}"

SAVE_STEPS="${SAVE_STEPS:-999999}"
SAVE_LATEST_STEPS="${SAVE_LATEST_STEPS:-0}"
SAVE_LATEST_ON_EVAL="${SAVE_LATEST_ON_EVAL:-true}"
EVAL_STEPS_SAMPLE="${EVAL_STEPS_SAMPLE:-40}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"
EVAL_TOPK="${EVAL_TOPK:-10}"
EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-0}"
EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-0}"
ACL_EVAL_SAMPLE_LIMIT="${ACL_EVAL_SAMPLE_LIMIT:-0}"
TAGGED_ACL_EVAL_SAMPLE_LIMIT="${TAGGED_ACL_EVAL_SAMPLE_LIMIT:-0}"
MEDICINE_EVAL_SAMPLE_LIMIT="${MEDICINE_EVAL_SAMPLE_LIMIT:-0}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-17}"
EVAL_SCORE_DEVICE="${EVAL_SCORE_DEVICE:-cuda}"
EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-256}"
EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-1024}"
EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS:-2}"
EVAL_METRIC_DENOMINATOR="${EVAL_METRIC_DENOMINATOR:-fixed_raw}"
TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS-0.75}"

ACL_DEV_JSONL="${ACL_DEV_JSONL-/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl}"
TAGGED_ACL_DEV_JSONL="${TAGGED_ACL_DEV_JSONL:-}"
MEDICINE_DEV_JSONL="${MEDICINE_DEV_JSONL:-}"
EVAL_WIKI_GLOSSARY="${EVAL_WIKI_GLOSSARY-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json}"
EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-1000 10000}"
EVAL_METRICS_GLOSSARY="${EVAL_METRICS_GLOSSARY:-}"
ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-}"
ACL_EVAL_GLOSSARY_SIZES="${ACL_EVAL_GLOSSARY_SIZES:-}"
ACL_EVAL_METRICS_GLOSSARY="${ACL_EVAL_METRICS_GLOSSARY:-}"
TAGGED_ACL_EVAL_WIKI_GLOSSARY="${TAGGED_ACL_EVAL_WIKI_GLOSSARY:-}"
TAGGED_ACL_EVAL_GLOSSARY_SIZES="${TAGGED_ACL_EVAL_GLOSSARY_SIZES:-}"
TAGGED_ACL_EVAL_METRICS_GLOSSARY="${TAGGED_ACL_EVAL_METRICS_GLOSSARY:-}"
MEDICINE_EVAL_WIKI_GLOSSARY="${MEDICINE_EVAL_WIKI_GLOSSARY:-}"
MEDICINE_EVAL_GLOSSARY_SIZES="${MEDICINE_EVAL_GLOSSARY_SIZES:-}"
MEDICINE_EVAL_METRICS_GLOSSARY="${MEDICINE_EVAL_METRICS_GLOSSARY:-}"
FULL_EVAL_WIKI_GLOSSARY="${FULL_EVAL_WIKI_GLOSSARY:-}"
FULL_EVAL_GLOSSARY_SIZES="${FULL_EVAL_GLOSSARY_SIZES:-}"
FULL_EVAL_EVERY_N_EVALS="${FULL_EVAL_EVERY_N_EVALS:-0}"
FULL_EVAL_NAME="${FULL_EVAL_NAME:-dev_full}"
BEST_METRIC="${BEST_METRIC:-eval_acl6060/recall@10_gs1000}"
BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY-eval_acl6060/recall@10}"
EARLY_STOP_BEST_PATIENCE_EVALS="${EARLY_STOP_BEST_PATIENCE_EVALS:-0}"
DUMP_SIM_DISTRIBUTIONS="${DUMP_SIM_DISTRIBUTIONS:-}"
DUMP_EVAL_MISSES_DIR="${DUMP_EVAL_MISSES_DIR:-}"
DUMP_EVAL_MISSES_EVAL_NAMES="${DUMP_EVAL_MISSES_EVAL_NAMES:-}"
DUMP_EVAL_MISSES_BANKS="${DUMP_EVAL_MISSES_BANKS:-}"
DUMP_EVAL_MISSES_TOPN="${DUMP_EVAL_MISSES_TOPN:-80}"
AUTO_FULL_EVAL_ON_BEST="${AUTO_FULL_EVAL_ON_BEST:-false}"
AUTO_FULL_EVAL_LAUNCHER="${AUTO_FULL_EVAL_LAUNCHER:-}"
AUTO_FULL_EVAL_PARTITION="${AUTO_FULL_EVAL_PARTITION:-}"
AUTO_FULL_EVAL_MIN_STEP_DELTA="${AUTO_FULL_EVAL_MIN_STEP_DELTA:-0}"
AUTO_FULL_EVAL_EXTRA_ENV="${AUTO_FULL_EVAL_EXTRA_ENV:-}"

EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_hardneg}"
DATA_TAG="${DATA_TAG:-3variant_1m_mfa}"
TASK_TAG="${TASK_TAG:-train}"
EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-compute:aries-8gpu}"
BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-tys70s0y r0xi5xkt zv28ve3q}"
# ======Configuration=====

mkdir -p "${SAVE_DIR}"

SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-false}"
if [ -n "${REQUESTED_CUDA_DEVICES}" ]; then
    python3 - "${REQUESTED_CUDA_DEVICES}" "${SELECT_CLEAN_GPUS}" <<'PYEOF'
import os
import subprocess
import sys

requested = sys.argv[1].split(",")
select_clean = sys.argv[2].lower() == "true"
threshold_mib = 500
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
mem = {}
for line in out.strip().splitlines():
    idx, used = [x.strip() for x in line.split(",")]
    mem[idx] = int(used)
missing = [idx for idx in requested if idx not in mem]
if missing:
    print(f"[PREFLIGHT][FATAL] requested GPUs not present in nvidia-smi: {missing}", file=sys.stderr)
    sys.exit(1)
slurm_raw = os.environ.get("SLURM_JOB_GPUS") or ""
if slurm_raw:
    print(
        f"[PREFLIGHT] explicit CUDA_DEVICE_LIST={requested} wins over SLURM_JOB_GPUS={slurm_raw}",
        file=sys.stderr,
    )
status = [(idx, mem[idx]) for idx in requested]
print(f"[PREFLIGHT] requested={status}", file=sys.stderr)
if select_clean:
    busy = [(idx, used) for idx, used in status if used > threshold_mib]
    if busy:
        print(
            f"[PREFLIGHT][FATAL] requested GPUs over {threshold_mib}MiB: {busy}",
            file=sys.stderr,
        )
        sys.exit(1)
PYEOF
    CUDA_DEV_LIST="${REQUESTED_CUDA_DEVICES}"
elif [ "${SELECT_CLEAN_GPUS}" = "true" ]; then
    CUDA_DEV_LIST="$(python3 - "$NUM_GPUS" <<'PYEOF'
import os
import subprocess, sys
needed = int(sys.argv[1])
threshold_mib = 500
allowed_raw = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""
allowed = None
if allowed_raw:
    parsed = []
    for part in allowed_raw.replace(" ", ",").split(","):
        part = part.strip()
        if part.isdigit():
            parsed.append(part)
    if parsed:
        allowed = set(parsed)
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
free, busy = [], []
for line in out.strip().splitlines():
    idx, used = [x.strip() for x in line.split(",")]
    if allowed is not None and idx not in allowed:
        continue
    (free if int(used) <= threshold_mib else busy).append((idx, used))
scope = f"allowed={sorted(allowed)} " if allowed is not None else ""
print(f"[PREFLIGHT] {scope}free={free} busy={busy}", file=sys.stderr)
if len(free) < needed:
    print(
        f"[PREFLIGHT][FATAL] only {len(free)}/{needed} allocated GPUs under "
        f"{threshold_mib}MiB; refusing to launch on busy GPUs.",
        file=sys.stderr,
    )
    sys.exit(1)
print(",".join(idx for idx, _ in free[:needed]))
PYEOF
)"
else
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
    CUDA_DEV_LIST="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_DEV_LIST}"
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
PER_RANK_EFFECTIVE_BATCH=$((BATCH_SIZE / NUM_GPUS))
EFFECTIVE_GLOBAL_BATCH=$((PER_RANK_EFFECTIVE_BATCH * NUM_GPUS))
if [ "${EFFECTIVE_GLOBAL_BATCH}" -ne "${BATCH_SIZE}" ]; then
    echo "[PREFLIGHT][WARN] BATCH_SIZE=${BATCH_SIZE} is not divisible by NUM_GPUS=${NUM_GPUS}; train loader will use per_rank_bs=${PER_RANK_EFFECTIVE_BATCH}, effective_global_batch=${EFFECTIVE_GLOBAL_BATCH}" >&2
fi

TEXT_TAG="tr${TEXT_LORA_RANK}"
MODE_NAME="scale_lora-r${LORA_RANK}-${TEXT_TAG}"
SMOKE_TAG=""
if [ "${MAX_STEPS}" -gt 0 ]; then
    SMOKE_TAG="_smoke${MAX_STEPS}"
fi
SAVE_NAME="q3rag_${MODE_NAME}_bs${BS_ABBR}_t=${TEMPERATURE}_${VERSION}${SMOKE_TAG}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"

echo "[TRAIN] VARIANT=${VARIANT_TAG}"
echo "[TRAIN] VERSION=${VERSION}"
echo "[TRAIN] RESUME=${RESUME:-<none>} CONSTANT_LR=${CONSTANT_LR} RESET_SCHEDULER=${RESET_SCHEDULER} RESET_BEST_ON_RESUME=${RESET_BEST_ON_RESUME} RESUME_COSINE_DECAY_TO_MAX_STEPS=${RESUME_COSINE_DECAY_TO_MAX_STEPS}"
echo "[TRAIN] MAX_TRAIN_SECONDS=${MAX_TRAIN_SECONDS} EPOCHS=${EPOCHS} SCHEDULER_EPOCHS=${SCHEDULER_EPOCHS} MAX_STEPS=${MAX_STEPS} EVAL_ONLY=${EVAL_ONLY}"
echo "[TRAIN] Save: ${SAVE_PATH}"
echo "[TRAIN] save_latest_on_eval=${SAVE_LATEST_ON_EVAL} save_latest_steps=${SAVE_LATEST_STEPS} save_steps=${SAVE_STEPS}"
echo "[TRAIN] Batch: requested_global=${BATCH_SIZE} effective_global=${EFFECTIVE_GLOBAL_BATCH} (${NUM_GPUS} * per_rank=${PER_RANK_EFFECTIVE_BATCH}; PER_GPU_BATCH=${PER_GPU_BATCH}) grad_cache_chunk=${GRAD_CACHE_CHUNK_SIZE} fixed_audio_seconds=${FIXED_AUDIO_SECONDS} eval_fixed_audio_seconds=${EVAL_FIXED_AUDIO_SECONDS}"
echo "[TRAIN] Audio encoder: preset=${AUDIO_ENCODER_PRESET} type=${AUDIO_ENCODER_TYPE} model=${AUDIO_MODEL_ID} feature_extractor=${AUDIO_FEATURE_EXTRACTOR_ID} input_dtype=${AUDIO_INPUT_DTYPE} hidden_dim=${AUDIO_HIDDEN_DIM}"
echo "[TRAIN] Text encoder: preset=${TEXT_ENCODER_PRESET} model=${TEXT_MODEL_ID} input_prefix='${TEXT_INPUT_PREFIX}' pooling=${TEXT_POOLING}"
echo "[TRAIN] HARD_NEG_K=${HARD_NEG_K} HARD_NEG_K_PER_SAMPLE=${HARD_NEG_K_PER_SAMPLE} NEG_BANK_REFRESH_STEPS=${NEG_BANK_REFRESH_STEPS}"
echo "[TRAIN] TERM_ID_NORMALIZE=${TERM_ID_NORMALIZE}"
echo "[TRAIN] TCM: base=${TCM_LOSS_WEIGHT} pos_w=${TCM_POS_LOSS_WEIGHT} neg_w=${TCM_NEG_LOSS_WEIGHT} pos=${TCM_POS_THRESHOLD} neg=${TCM_NEG_THRESHOLD} warmup=${TCM_WARMUP_STEPS}"
echo "[TRAIN] experiment_family=${EXPERIMENT_FAMILY} data_tag=${DATA_TAG} task_tag=${TASK_TAG}"
echo "[TRAIN] extra_wandb_tags=${EXTRA_WANDB_TAGS}"
echo "[TRAIN] baseline_run_ids=${BASELINE_RUN_IDS}"
echo "[TRAIN] notes_file=${NOTES_FILE}"
echo "[TRAIN] wandb_exp_name=${WANDB_EXP_NAME}"
echo "[TRAIN] eval_wiki_glossary=${EVAL_WIKI_GLOSSARY} sizes=${EVAL_GLOSSARY_SIZES} eval_steps=${EVAL_STEPS_SAMPLE}"
echo "[TRAIN] eval_sample_limit dev=${EVAL_SAMPLE_LIMIT} acl=${ACL_EVAL_SAMPLE_LIMIT} tagged_acl=${TAGGED_ACL_EVAL_SAMPLE_LIMIT} medicine=${MEDICINE_EVAL_SAMPLE_LIMIT} seed=${EVAL_SAMPLE_SEED}"
echo "[TRAIN] eval_score_device=${EVAL_SCORE_DEVICE} query_chunk=${EVAL_SCORE_QUERY_CHUNK} text_chunk=${EVAL_SCORE_TEXT_CHUNK}"
echo "[TRAIN] eval_glossary_match_min_norm_chars=${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS}"
echo "[TRAIN] eval_metric_denominator=${EVAL_METRIC_DENOMINATOR}"
echo "[TRAIN] eval_metrics_glossary=${EVAL_METRICS_GLOSSARY:-<raw/base bank>}"
echo "[TRAIN] acl_eval_wiki_glossary=${ACL_EVAL_WIKI_GLOSSARY:-<same>} sizes=${ACL_EVAL_GLOSSARY_SIZES:-<same>}"
echo "[TRAIN] acl_eval_metrics_glossary=${ACL_EVAL_METRICS_GLOSSARY:-<raw/base bank>}"
echo "[TRAIN] tagged_acl_dev_jsonl=${TAGGED_ACL_DEV_JSONL:-<none>}"
echo "[TRAIN] tagged_acl_eval_wiki_glossary=${TAGGED_ACL_EVAL_WIKI_GLOSSARY:-<same>} sizes=${TAGGED_ACL_EVAL_GLOSSARY_SIZES:-<same>}"
echo "[TRAIN] tagged_acl_eval_metrics_glossary=${TAGGED_ACL_EVAL_METRICS_GLOSSARY:-<raw/base bank>}"
echo "[TRAIN] medicine_dev_jsonl=${MEDICINE_DEV_JSONL:-<none>}"
echo "[TRAIN] medicine_eval_wiki_glossary=${MEDICINE_EVAL_WIKI_GLOSSARY:-<same>} sizes=${MEDICINE_EVAL_GLOSSARY_SIZES:-<same>}"
echo "[TRAIN] medicine_eval_metrics_glossary=${MEDICINE_EVAL_METRICS_GLOSSARY:-<raw/base bank>}"
echo "[TRAIN] train_exclude_term_glossaries=${TRAIN_EXCLUDE_TERM_GLOSSARIES:-<none>}"
echo "[TRAIN] strict_train_eval_term_filter=${STRICT_TRAIN_EVAL_TERM_FILTER}"
echo "[TRAIN] full_eval_wiki_glossary=${FULL_EVAL_WIKI_GLOSSARY:-<none>} sizes=${FULL_EVAL_GLOSSARY_SIZES:-<none>} every_n_evals=${FULL_EVAL_EVERY_N_EVALS} name=${FULL_EVAL_NAME}"
echo "[TRAIN] early_stop_best_patience_evals=${EARLY_STOP_BEST_PATIENCE_EVALS}"
echo "[TRAIN] dump_sim_distributions=${DUMP_SIM_DISTRIBUTIONS:-<none>}"
echo "[TRAIN] dump_eval_misses_dir=${DUMP_EVAL_MISSES_DIR:-<none>} eval_names=${DUMP_EVAL_MISSES_EVAL_NAMES:-<all>} banks=${DUMP_EVAL_MISSES_BANKS:-<all>} topn=${DUMP_EVAL_MISSES_TOPN}"
echo "[TRAIN] auto_full_eval_on_best=${AUTO_FULL_EVAL_ON_BEST} launcher=${AUTO_FULL_EVAL_LAUNCHER:-<none>} partition=${AUTO_FULL_EVAL_PARTITION:-<default>} min_step_delta=${AUTO_FULL_EVAL_MIN_STEP_DELTA}"

OPTS=""
if [ "${USE_LORA}" = "true" ]; then OPTS="${OPTS} --use_lora"; fi
if [ "${USE_MAXSIM}" = "true" ]; then OPTS="${OPTS} --use_maxsim"; fi
if [ "${MFA_SUPERVISED}" = "true" ]; then OPTS="${OPTS} --mfa_supervised_maxsim"; fi
if [ "${WIKI_RANK}" -gt 0 ]; then OPTS="${OPTS} --wiki_rank ${WIKI_RANK}"; fi
if [ "${MAX_STEPS}" -gt 0 ]; then OPTS="${OPTS} --max_steps ${MAX_STEPS}"; fi
if [ -n "${RESUME}" ]; then OPTS="${OPTS} --resume ${RESUME}"; fi
if [ "${SCHEDULER_EPOCHS}" -gt 0 ]; then OPTS="${OPTS} --scheduler_epochs ${SCHEDULER_EPOCHS}"; fi
if [ "${EVAL_ONLY}" = "true" ]; then OPTS="${OPTS} --eval_only"; fi
if [ "${RESET_SCHEDULER}" = "true" ]; then OPTS="${OPTS} --reset_scheduler"; fi
if [ "${RESET_BEST_ON_RESUME}" = "true" ]; then OPTS="${OPTS} --reset_best_on_resume"; fi
if [ "${CONSTANT_LR}" != "0.0" ]; then OPTS="${OPTS} --constant_lr ${CONSTANT_LR}"; fi
if [ "${RESUME_COSINE_DECAY_TO_MAX_STEPS}" = "true" ]; then OPTS="${OPTS} --resume_cosine_decay_to_max_steps"; fi
if [ "${SAVE_LATEST_ON_EVAL}" = "true" ]; then OPTS="${OPTS} --save_latest_on_eval"; fi
if [ "${SAVE_LATEST_STEPS}" -gt 0 ]; then OPTS="${OPTS} --save_latest_steps ${SAVE_LATEST_STEPS}"; fi
if [ -n "${FULL_EVAL_WIKI_GLOSSARY}" ]; then OPTS="${OPTS} --full_eval_wiki_glossary ${FULL_EVAL_WIKI_GLOSSARY}"; fi
if [ -n "${FULL_EVAL_GLOSSARY_SIZES}" ]; then OPTS="${OPTS} --full_eval_glossary_sizes ${FULL_EVAL_GLOSSARY_SIZES}"; fi
OPTS="${OPTS} --eval_metric_denominator ${EVAL_METRIC_DENOMINATOR}"
if [ -n "${EVAL_METRICS_GLOSSARY}" ]; then OPTS="${OPTS} --eval_metrics_glossary ${EVAL_METRICS_GLOSSARY}"; fi
if [ -n "${ACL_EVAL_WIKI_GLOSSARY}" ]; then OPTS="${OPTS} --acl_eval_wiki_glossary ${ACL_EVAL_WIKI_GLOSSARY}"; fi
if [ -n "${ACL_EVAL_GLOSSARY_SIZES}" ]; then OPTS="${OPTS} --acl_eval_glossary_sizes ${ACL_EVAL_GLOSSARY_SIZES}"; fi
if [ -n "${ACL_EVAL_METRICS_GLOSSARY}" ]; then OPTS="${OPTS} --acl_eval_metrics_glossary ${ACL_EVAL_METRICS_GLOSSARY}"; fi
if [ -n "${TAGGED_ACL_EVAL_WIKI_GLOSSARY}" ]; then OPTS="${OPTS} --tagged_acl_eval_wiki_glossary ${TAGGED_ACL_EVAL_WIKI_GLOSSARY}"; fi
if [ -n "${TAGGED_ACL_EVAL_GLOSSARY_SIZES}" ]; then OPTS="${OPTS} --tagged_acl_eval_glossary_sizes ${TAGGED_ACL_EVAL_GLOSSARY_SIZES}"; fi
if [ -n "${TAGGED_ACL_EVAL_METRICS_GLOSSARY}" ]; then OPTS="${OPTS} --tagged_acl_eval_metrics_glossary ${TAGGED_ACL_EVAL_METRICS_GLOSSARY}"; fi
if [ -n "${MEDICINE_EVAL_WIKI_GLOSSARY}" ]; then OPTS="${OPTS} --medicine_eval_wiki_glossary ${MEDICINE_EVAL_WIKI_GLOSSARY}"; fi
if [ -n "${MEDICINE_EVAL_GLOSSARY_SIZES}" ]; then OPTS="${OPTS} --medicine_eval_glossary_sizes ${MEDICINE_EVAL_GLOSSARY_SIZES}"; fi
if [ -n "${MEDICINE_EVAL_METRICS_GLOSSARY}" ]; then OPTS="${OPTS} --medicine_eval_metrics_glossary ${MEDICINE_EVAL_METRICS_GLOSSARY}"; fi
if [ "${EVAL_SAMPLE_LIMIT}" -gt 0 ]; then OPTS="${OPTS} --eval_sample_limit ${EVAL_SAMPLE_LIMIT}"; fi
if [ "${ACL_EVAL_SAMPLE_LIMIT}" -gt 0 ]; then OPTS="${OPTS} --acl_eval_sample_limit ${ACL_EVAL_SAMPLE_LIMIT}"; fi
if [ "${TAGGED_ACL_EVAL_SAMPLE_LIMIT}" -gt 0 ]; then OPTS="${OPTS} --tagged_acl_eval_sample_limit ${TAGGED_ACL_EVAL_SAMPLE_LIMIT}"; fi
if [ "${MEDICINE_EVAL_SAMPLE_LIMIT}" -gt 0 ]; then OPTS="${OPTS} --medicine_eval_sample_limit ${MEDICINE_EVAL_SAMPLE_LIMIT}"; fi
OPTS="${OPTS} --eval_sample_seed ${EVAL_SAMPLE_SEED}"
OPTS="${OPTS} --eval_score_device ${EVAL_SCORE_DEVICE}"
OPTS="${OPTS} --eval_score_query_chunk ${EVAL_SCORE_QUERY_CHUNK}"
OPTS="${OPTS} --eval_score_text_chunk ${EVAL_SCORE_TEXT_CHUNK}"
if [ "${FULL_EVAL_EVERY_N_EVALS}" -gt 0 ]; then OPTS="${OPTS} --full_eval_every_n_evals ${FULL_EVAL_EVERY_N_EVALS}"; fi
if [ -n "${FULL_EVAL_NAME}" ]; then OPTS="${OPTS} --full_eval_name ${FULL_EVAL_NAME}"; fi
if [ "${EARLY_STOP_BEST_PATIENCE_EVALS}" -gt 0 ]; then OPTS="${OPTS} --early_stop_best_patience_evals ${EARLY_STOP_BEST_PATIENCE_EVALS}"; fi
if [ -n "${DUMP_SIM_DISTRIBUTIONS}" ]; then OPTS="${OPTS} --dump_sim_distributions ${DUMP_SIM_DISTRIBUTIONS}"; fi
if [ -n "${DUMP_EVAL_MISSES_DIR}" ]; then
    OPTS="${OPTS} --dump_eval_misses_dir ${DUMP_EVAL_MISSES_DIR}"
    if [ -n "${DUMP_EVAL_MISSES_EVAL_NAMES}" ]; then OPTS="${OPTS} --dump_eval_misses_eval_names ${DUMP_EVAL_MISSES_EVAL_NAMES}"; fi
    if [ -n "${DUMP_EVAL_MISSES_BANKS}" ]; then OPTS="${OPTS} --dump_eval_misses_banks ${DUMP_EVAL_MISSES_BANKS}"; fi
    OPTS="${OPTS} --dump_eval_misses_topn ${DUMP_EVAL_MISSES_TOPN}"
fi
if [ -n "${TRAIN_EXCLUDE_TERM_GLOSSARIES}" ]; then OPTS="${OPTS} --train_exclude_term_glossaries ${TRAIN_EXCLUDE_TERM_GLOSSARIES}"; fi
if [ "${STRICT_TRAIN_EVAL_TERM_FILTER}" = "true" ]; then OPTS="${OPTS} --strict_train_eval_term_filter"; fi
if [ "${AUTO_FULL_EVAL_ON_BEST}" = "true" ]; then OPTS="${OPTS} --auto_full_eval_on_best"; fi
if [ -n "${AUTO_FULL_EVAL_LAUNCHER}" ]; then OPTS="${OPTS} --auto_full_eval_launcher ${AUTO_FULL_EVAL_LAUNCHER}"; fi
if [ -n "${AUTO_FULL_EVAL_PARTITION}" ]; then OPTS="${OPTS} --auto_full_eval_partition ${AUTO_FULL_EVAL_PARTITION}"; fi
if [ "${AUTO_FULL_EVAL_MIN_STEP_DELTA}" -gt 0 ]; then OPTS="${OPTS} --auto_full_eval_min_step_delta ${AUTO_FULL_EVAL_MIN_STEP_DELTA}"; fi
if [ -n "${AUTO_FULL_EVAL_EXTRA_ENV}" ]; then OPTS="${OPTS} --auto_full_eval_extra_env ${AUTO_FULL_EVAL_EXTRA_ENV}"; fi

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
    --fixed_audio_seconds "${FIXED_AUDIO_SECONDS}" \
    --eval_fixed_audio_seconds "${EVAL_FIXED_AUDIO_SECONDS}" \
    --train_limit "${TRAIN_LIMIT}" \
    --num_workers "${NUM_WORKERS}" \
    --temperature "${TEMPERATURE}" \
    --target_dim "${TARGET_DIM}" \
    --audio_encoder_preset "${AUDIO_ENCODER_PRESET}" \
    --audio_encoder_type "${AUDIO_ENCODER_TYPE}" \
    --audio_model_id "${AUDIO_MODEL_ID}" \
    --audio_feature_extractor_id "${AUDIO_FEATURE_EXTRACTOR_ID}" \
    --audio_input_dtype "${AUDIO_INPUT_DTYPE}" \
    --audio_hidden_dim "${AUDIO_HIDDEN_DIM}" \
    --text_encoder_preset "${TEXT_ENCODER_PRESET}" \
    --text_model_id "${TEXT_MODEL_ID}" \
    --text_input_prefix "${TEXT_INPUT_PREFIX}" \
    --pooling_type "${POOLING_TYPE}" \
    --maxsim_windows ${MAXSIM_WINDOWS} \
    --maxsim_stride "${MAXSIM_STRIDE}" \
    --mfa_window_selection "${MFA_WINDOW_SELECTION}" \
    --mfa_positive_scope "${MFA_POSITIVE_SCOPE}" \
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
    --eval_glossary_match_min_norm_chars "${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS}" \
    --keep_checkpoints "${KEEP_CHECKPOINTS}" \
    --acl_dev_jsonl "${ACL_DEV_JSONL}" \
    --tagged_acl_dev_jsonl "${TAGGED_ACL_DEV_JSONL}" \
    --medicine_dev_jsonl "${MEDICINE_DEV_JSONL}" \
    --eval_wiki_glossary "${EVAL_WIKI_GLOSSARY}" \
    --eval_glossary_sizes ${EVAL_GLOSSARY_SIZES} \
    --best_metric "${BEST_METRIC}" \
    --best_metric_secondary "${BEST_METRIC_SECONDARY}" \
    --eval_top100_samples "${EVAL_TOP100_SAMPLES}" \
    --eval_minimal_metrics \
    --enable_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${WANDB_EXP_NAME}" \
    --tcm_loss_weight "${TCM_LOSS_WEIGHT}" \
    --tcm_pos_loss_weight "${TCM_POS_LOSS_WEIGHT}" \
    --tcm_neg_loss_weight "${TCM_NEG_LOSS_WEIGHT}" \
    --tcm_pos_threshold "${TCM_POS_THRESHOLD}" \
    --tcm_neg_threshold "${TCM_NEG_THRESHOLD}" \
    --tcm_loss_form "${TCM_LOSS_FORM}" \
    --tcm_reduction "${TCM_REDUCTION}" \
    --tcm_neg_scope "${TCM_NEG_SCOPE}" \
    --tcm_neg_topk "${TCM_NEG_TOPK}" \
    --tcm_sweep_thresholds ${TCM_SWEEP_THRESHOLDS} \
    --tcm_sweep_fbeta "${TCM_SWEEP_FBETA}" \
    --tcm_warmup_steps "${TCM_WARMUP_STEPS}" \
    --hcl_beta "${HCL_BETA}" \
    --term_id_normalize "${TERM_ID_NORMALIZE}" \
    --max_train_seconds "${MAX_TRAIN_SECONDS}" \
    --experiment_family "${EXPERIMENT_FAMILY}" \
    --data_tag "${DATA_TAG}" \
    --task_tag "${TASK_TAG}" \
    --extra_wandb_tags ${EXTRA_WANDB_TAGS} \
    --baseline_run_ids ${BASELINE_RUN_IDS} \
    --notes_file "${NOTES_FILE}" \
    --run_verdict "${RUN_VERDICT}" \
    ${OPTS}

echo "[TRAIN] ${VARIANT_TAG} completed at $(date)"
