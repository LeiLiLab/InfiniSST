#!/bin/bash
#SBATCH --job-name=q3_vpool_a1
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=200G
#SBATCH --gres=gpu:6
#SBATCH --time=28:00:00
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vpool_a1_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_vpool_a1_%x.err

# A1: Voice-Pool (GigaSpeech v2 one-per-opus, CLEAN-ONLY) retriever ablation.
# Clone of run_ablation_A_k1024_normalize_aggressive_taurus.sh (43849) with
# ONLY the upstream TTS audio set changed.  See notes_voicepool_a1.md for
# hypothesis and expected deltas.
#
# Key diffs vs 43849:
#   - TRAIN_JSONL -> new v2 clean-only MFA output.
#   - VERSION / SAVE_NAME / WANDB_EXP_NAME bumped to reflect A1.
#   - NOTES_FILE required by experiment_tracking schema.
#   - NOISY_RATIO=0.0 (v2 has no noisy branch; matches 43849's value already).
# Intentionally nothing else changes: same LR, same batch, same MaxSim windows,
# same TCM coef, same HN k=1024 + normAGGR.  Everything else identical keeps
# A1 a clean isolation of the audio-domain variable.

set -euo pipefail

# ======Configuration=====
MARGIN="0.0"
HCL_BETA="0.0"
TCM_LOSS_WEIGHT="1.0"
TCM_POS_THRESHOLD="0.85"
TCM_NEG_THRESHOLD="0.25"
TCM_LOSS_FORM="squared_hinge"
TCM_REDUCTION="mean_viol"

HARD_NEG_K=0
HARD_NEG_K_PER_SAMPLE=1024
NEG_BANK_SIZE=0
NEG_BANK_REFRESH_STEPS=50

TERM_ID_NORMALIZE="aggressive"

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
MASTER_PORT=29965

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

# NEW v2 TRAIN_JSONL -- built downstream by:
#   TTS (clean-only, v2 pool) -> merge -> MFA -> post-proc + Gigaspeech merge.
# Must exist before sbatch.  See documents/code/data_pre/wiki_synth/PIPELINE.md.
TRAIN_JSONL="/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_clean_mfa.jsonl"
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py"
SAVE_DIR="/mnt/gemini/home/jiaxuanluo/train_outputs"
NOTES_FILE="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes_voicepool_a1.md"

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

PER_GPU_BATCH=2048
BATCH_SIZE=$((NUM_GPUS * PER_GPU_BATCH))
GRAD_CACHE_CHUNK_SIZE=512
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

if [ ! -f "${TRAIN_JSONL}" ]; then
    echo "[ERROR] TRAIN_JSONL not found: ${TRAIN_JSONL}" >&2
    echo "  Upstream pipeline (TTS v2 clean -> merge -> MFA -> gigaspeech merge)" >&2
    echo "  must complete before submitting this launcher." >&2
    exit 1
fi

mkdir -p "${SAVE_DIR}"

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
VERSION="3var_gsv2_clean_gc_wr$((WIKI_RANK / 1000))k_m${MARGIN}_maxsim_mfa_hn_k${HARD_NEG_K_PER_SAMPLE}_tcm_ep${EPOCHS}_normAGGR${SMOKE_TAG}"
SAVE_NAME="q3rag_vpool_a1_${MODE_NAME}_bs${BS_ABBR}_t=${TEMPERATURE}_${VERSION}"
SAVE_PATH="${SAVE_DIR}/${SAVE_NAME}.pt"
TIMESTAMP="$(date +%Y%m%d-%H%M)"
WANDB_EXP_NAME="family_voice_pool__a1_gsv2_clean__${TIMESTAMP}__${SAVE_NAME}"

echo "[TRAIN] A1 VOICE POOL: TTS=GSv2 clean-only (9989 unique-opus prompts)"
echo "[TRAIN] Baseline: 43848 smallest+dense + pool_k=64 + normAGGR"
echo "[TRAIN] TRAIN_JSONL=${TRAIN_JSONL}"
echo "[TRAIN] NOTES=${NOTES_FILE}"
echo "[TRAIN] EPOCHS=${EPOCHS} MAX_STEPS=${MAX_STEPS} MAX_TRAIN_SECONDS=${MAX_TRAIN_SECONDS}"
echo "[TRAIN] Save: ${SAVE_PATH}"
echo "[TRAIN] Batch: ${BATCH_SIZE} (${NUM_GPUS} * ${PER_GPU_BATCH})  grad_cache_chunk=${GRAD_CACHE_CHUNK_SIZE}"

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
    --experiment_family "voice_pool" \
    --data_tag "3variant_gsv2_clean_mfa" \
    --task_tag "train" \
    --extra_wandb_tags "variant:a1_gsv2_clean" "gpus:6" "nonblock+cache+sync-fix" \
    --baseline_run_ids "43848" "43849" \
    --notes_file "${NOTES_FILE}" \
    --tcm_loss_weight "${TCM_LOSS_WEIGHT}" \
    --tcm_pos_threshold "${TCM_POS_THRESHOLD}" \
    --tcm_neg_threshold "${TCM_NEG_THRESHOLD}" \
    --tcm_loss_form "${TCM_LOSS_FORM}" \
    --tcm_reduction "${TCM_REDUCTION}" \
    --hcl_beta "${HCL_BETA}" \
    --max_train_seconds "${MAX_TRAIN_SECONDS}" \
    --term_id_normalize "${TERM_ID_NORMALIZE}" \
    ${OPTS}

echo "[TRAIN] A1 voice-pool run completed at $(date)"
