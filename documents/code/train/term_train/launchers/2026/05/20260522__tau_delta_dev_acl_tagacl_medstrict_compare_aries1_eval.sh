#!/bin/bash
#SBATCH --job-name=q3_tau_cmp_eval
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=260G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_tau_cmp_eval_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_tau_cmp_eval_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"

: "${MODEL_TAG:?MODEL_TAG is required, e.g. nohn_best_secondary or lh1b88kw_main}"
: "${RESUME:?RESUME checkpoint path is required}"
: "${NOTES_FILE:?NOTES_FILE is required}"

if [ ! -f "${RESUME}" ]; then
    echo "[FATAL] checkpoint not found: ${RESUME}" >&2
    exit 1
fi

export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-6}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export WAIT_FOR_CLEAN_GPUS="${WAIT_FOR_CLEAN_GPUS:-true}"
export GPU_CLEAN_THRESHOLD_MIB="${GPU_CLEAN_THRESHOLD_MIB:-500}"
export GPU_WAIT_INTERVAL_SEC="${GPU_WAIT_INTERVAL_SEC:-60}"
export GPU_WAIT_TIMEOUT_SEC="${GPU_WAIT_TIMEOUT_SEC:-172800}"
COMPUTE_TAG="${SLURM_JOB_PARTITION:-aries}-${NUM_GPUS}gpu"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

wait_for_clean_gpus() {
    local gpu_list="$1"
    local threshold_mib="$2"
    local interval_sec="$3"
    local timeout_sec="$4"
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        if python3 - "${gpu_list}" "${threshold_mib}" <<'PYEOF'
import subprocess
import sys

requested = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
threshold = int(sys.argv[2])
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
mem = {}
for line in out.strip().splitlines():
    idx, used = [part.strip() for part in line.split(",")]
    mem[idx] = int(used)
missing = [idx for idx in requested if idx not in mem]
busy = [(idx, mem[idx]) for idx in requested if idx in mem and mem[idx] > threshold]
print(f"[GPU_WAIT] requested={[(idx, mem.get(idx)) for idx in requested]} threshold_mib={threshold}", flush=True)
if missing or busy:
    if missing:
        print(f"[GPU_WAIT] missing GPUs: {missing}", file=sys.stderr, flush=True)
    if busy:
        print(f"[GPU_WAIT] busy GPUs: {busy}", file=sys.stderr, flush=True)
    sys.exit(1)
PYEOF
        then
            break
        fi
        if [ $(( $(date +%s) - start_ts )) -ge "${timeout_sec}" ]; then
            echo "[GPU_WAIT][FATAL] timed out waiting for GPUs ${gpu_list} under ${threshold_mib}MiB" >&2
            exit 3
        fi
        sleep "${interval_sec}"
    done
}

if [ "${WAIT_FOR_CLEAN_GPUS}" = "true" ]; then
    wait_for_clean_gpus "${CUDA_DEVICE_LIST}" "${GPU_CLEAN_THRESHOLD_MIB}" "${GPU_WAIT_INTERVAL_SEC}" "${GPU_WAIT_TIMEOUT_SEC}"
fi

export VARIANT_TAG="${VARIANT_TAG:-${MODEL_TAG}_tau_delta_dev_acl_tagacl_medstrict}"
export VERSION="${VERSION:-3var_gsdedup_vctx576_${MODEL_TAG}_tau_delta_dev_acl_tagacl_medstrict_${COMPUTE_TAG}_${RUN_STAMP}}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-${MODEL_TAG}_tau_delta_dev_acl_tagacl_medstrict_${COMPUTE_TAG}_${RUN_STAMP}}"
export TASK_TAG="eval"
export DATA_TAG="${DATA_TAG:-vctx576_tau_delta_dev_acl_tagacl_medstrict}"
export EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-sst_ood_hardneg_eval_compare}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:${MODEL_TAG} compute:${COMPUTE_TAG} comparison:nohn_vs_lh1b88kw calibration:dev_delta readout:acl_tagged_medstrict protocol:fixed-raw-denominator}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw 40fgbr2y 4g108a3w nrxiasfm qjy4m1x9}"

export TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl}"
export DEV_JSONL="${DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl}"
export ACL_DEV_JSONL="${ACL_DEV_JSONL-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl}"
export TAGGED_ACL_DEV_JSONL="${TAGGED_ACL_DEV_JSONL-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_tagged_glossary_varctx2p88_3p84_4p80_5p76/acl6060_tagged_dev_dataset.jsonl}"
export MEDICINE_DEV_JSONL="${MEDICINE_DEV_JSONL-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl}"

# Match the lh1b88kw / no-HN architecture family.
export FIXED_AUDIO_SECONDS=5.76
export EVAL_FIXED_AUDIO_SECONDS=5.76
export AUDIO_ENCODER_PRESET="qwen3-omni"
export AUDIO_ENCODER_TYPE="qwen3_omni"
export AUDIO_MODEL_ID="Atotti/Qwen3-Omni-AudioTransformer"
export AUDIO_FEATURE_EXTRACTOR_ID="openai/whisper-large-v3"
export AUDIO_INPUT_DTYPE="auto"
export TEXT_ENCODER_PRESET="bge-m3"
export TEXT_MODEL_ID="BAAI/bge-m3"
export TEXT_POOLING="cls"
export TEXT_LORA_RANK=128
export TEXT_LORA_ALPHA=256
export LORA_RANK=128
export LORA_ALPHA=256
export TARGET_DIM=1024
export POOLING_TYPE="transformer"
export USE_MAXSIM=true
export MFA_SUPERVISED=true
export MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"
export MAXSIM_STRIDE=2
export MFA_WINDOW_SELECTION="smallest"
export MFA_POSITIVE_SCOPE="auto"
export TERM_ID_NORMALIZE="aggressive"

# Eval-only: disable all train-time losses/negative banks.
export EVAL_ONLY=true
export HARD_NEG_K=0
export HARD_NEG_K_PER_SAMPLE=0
export NEG_BANK_SIZE=0
export GLOSSARY_NEG_PATH=""
export GLOSSARY_NEG_REFRESH_STEPS=0
export TCM_LOSS_WEIGHT=0.0
export TCM_POS_LOSS_WEIGHT=0.0
export TCM_NEG_LOSS_WEIGHT=0.0
export TCM_POS_THRESHOLD=0.85
export TCM_NEG_THRESHOLD=0.60
export TCM_LOSS_FORM="hinge"
export TCM_REDUCTION="mean_viol"
export TCM_NEG_SCOPE="all"
export TCM_NEG_TOPK=0
export TCM_WARMUP_STEPS=0

export GRAD_CACHE_CHUNK_SIZE=256
export NUM_WORKERS=0
export EPOCHS=1
export SCHEDULER_EPOCHS=1
export MAX_STEPS=0
export MAX_TRAIN_SECONDS=0
export RESET_SCHEDULER=false
export RESET_BEST_ON_RESUME=false
export RESUME_COSINE_DECAY_TO_MAX_STEPS=false
export MASTER_PORT="${MASTER_PORT:-30420}"

# Dev selects the no-HN inference tau. ACL, tagged ACL, and medicine are readout only.
export EVAL_METRIC_DENOMINATOR="${EVAL_METRIC_DENOMINATOR:-fixed_raw}"
export EVAL_WIKI_GLOSSARY="${EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-10000 100000}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json}"
export ACL_EVAL_GLOSSARY_SIZES="${ACL_EVAL_GLOSSARY_SIZES-1000 10000}"
export TAGGED_ACL_EVAL_WIKI_GLOSSARY="${TAGGED_ACL_EVAL_WIKI_GLOSSARY-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
export TAGGED_ACL_EVAL_GLOSSARY_SIZES="${TAGGED_ACL_EVAL_GLOSSARY_SIZES-1000 10000}"
export MEDICINE_EVAL_WIKI_GLOSSARY="${MEDICINE_EVAL_WIKI_GLOSSARY-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json}"
export MEDICINE_EVAL_GLOSSARY_SIZES="${MEDICINE_EVAL_GLOSSARY_SIZES-1000 10000}"
export FULL_EVAL_WIKI_GLOSSARY="${FULL_EVAL_WIKI_GLOSSARY-}"
export FULL_EVAL_GLOSSARY_SIZES="${FULL_EVAL_GLOSSARY_SIZES-}"
export FULL_EVAL_EVERY_N_EVALS=0
export FULL_EVAL_NAME="dev_full"
export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs100000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY-}"
export EVAL_STEPS_SAMPLE="${EVAL_STEPS_SAMPLE:-0}"
export EVAL_TOP100_SAMPLES=0
export EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-0}"
export ACL_EVAL_SAMPLE_LIMIT="${ACL_EVAL_SAMPLE_LIMIT:-0}"
export TAGGED_ACL_EVAL_SAMPLE_LIMIT="${TAGGED_ACL_EVAL_SAMPLE_LIMIT:-0}"
export MEDICINE_EVAL_SAMPLE_LIMIT="${MEDICINE_EVAL_SAMPLE_LIMIT:-0}"
export EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-17}"
export EVAL_SCORE_DEVICE="${EVAL_SCORE_DEVICE:-cuda}"
export EVAL_SCORE_QUERY_CHUNK="${EVAL_SCORE_QUERY_CHUNK:-256}"
export EVAL_SCORE_TEXT_CHUNK="${EVAL_SCORE_TEXT_CHUNK:-1024}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS=2
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.65 0.66 0.67 0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.90}"
export TCM_SWEEP_FBETA=3.0
export AUTO_FULL_EVAL_ON_BEST=false
export RUN_VERDICT="${RUN_VERDICT:-Eval-only no-HN vs lh1b88kw comparison. Metrics denominator is fixed to the strict raw/base glossary; retriever candidate bank changes for gs1k/gs10k/gs100k. Tau selection uses dev only before held-out readout.}"

echo "[TAU_COMPARE_EVAL] model=${MODEL_TAG}"
echo "[TAU_COMPARE_EVAL] ckpt=${RESUME}"
echo "[TAU_COMPARE_EVAL] eval_metric_denominator=${EVAL_METRIC_DENOMINATOR}"
echo "[TAU_COMPARE_EVAL] dev=${DEV_JSONL} glossary=${EVAL_WIKI_GLOSSARY} sizes=${EVAL_GLOSSARY_SIZES}"
echo "[TAU_COMPARE_EVAL] acl=${ACL_DEV_JSONL} glossary=${ACL_EVAL_WIKI_GLOSSARY} sizes=${ACL_EVAL_GLOSSARY_SIZES}"
echo "[TAU_COMPARE_EVAL] tagged_acl=${TAGGED_ACL_DEV_JSONL} glossary=${TAGGED_ACL_EVAL_WIKI_GLOSSARY} sizes=${TAGGED_ACL_EVAL_GLOSSARY_SIZES}"
echo "[TAU_COMPARE_EVAL] medicine=${MEDICINE_DEV_JSONL} glossary=${MEDICINE_EVAL_WIKI_GLOSSARY} sizes=${MEDICINE_EVAL_GLOSSARY_SIZES}"
echo "[TAU_COMPARE_EVAL] tau_grid=${TCM_SWEEP_THRESHOLDS}"
echo "[TAU_COMPARE_EVAL] cuda_device_list=${CUDA_DEVICE_LIST} compute=${COMPUTE_TAG}"

source "${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
