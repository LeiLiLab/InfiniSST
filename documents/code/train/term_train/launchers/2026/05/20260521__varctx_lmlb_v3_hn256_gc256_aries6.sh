#!/bin/bash
#SBATCH --job-name=q3_hn256_a6
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:6
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn256_a6_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_hn256_a6_%x.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_LAUNCHER="${REPO_ROOT}/documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh"

export HARD_NEG_K="${HARD_NEG_K:-0}"
export HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-256}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-256}"
export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,4,5}"
CUDA_DEVICE_COUNT="$(python3 - "${CUDA_DEVICE_LIST}" <<'PYEOF'
import re
import sys

parts = [p.strip() for p in re.split(r"[,\s]+", sys.argv[1]) if p.strip()]
if not parts:
    raise SystemExit("CUDA_DEVICE_LIST is empty")
print(len(parts))
PYEOF
)"
CUDA_DEVICE_TAG="$(tr -cd '0-9' <<< "${CUDA_DEVICE_LIST}")"
export NUM_GPUS="${NUM_GPUS:-${CUDA_DEVICE_COUNT}}"
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-8192}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-$((TARGET_GLOBAL_BATCH / NUM_GPUS))}"
export BATCH_SIZE="${BATCH_SIZE:-$((PER_GPU_BATCH * NUM_GPUS))}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_STEPS="${MAX_STEPS:-0}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export MASTER_PORT="${MASTER_PORT:-29998}"

export SAVE_LATEST_ON_EVAL="${SAVE_LATEST_ON_EVAL:-true}"
export EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-0}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.75}"
export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs10000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-eval_acl6060/recall@10}"
export EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS="${EVAL_GLOSSARY_MATCH_MIN_NORM_CHARS:-2}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_gt_union_gs10000_min_norm2_backfill.json}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/dev/shm/q3_hn256_a6_${USER}_${RUN_STAMP}}"
export WANDB_DIR="${WANDB_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/aries/data4/jiaxuanluo/cache/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/aries/data4/jiaxuanluo/cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/mnt/aries/data4/jiaxuanluo/config}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
mkdir -p "${LOCAL_TMP_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}"

wait_for_clean_gpus() {
    local gpu_list="$1"
    local threshold_mib="${GPU_CLEAN_THRESHOLD_MIB:-500}"
    local interval_sec="${GPU_WAIT_INTERVAL_SEC:-60}"
    local timeout_sec="${GPU_WAIT_TIMEOUT_SEC:-172800}"
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
print(f"[GPU_WAIT] requested={[(idx, mem.get(idx)) for idx in requested]} threshold_mib={threshold}")
if missing or busy:
    if missing:
        print(f"[GPU_WAIT] missing GPUs: {missing}", file=sys.stderr)
    if busy:
        print(f"[GPU_WAIT] busy GPUs: {busy}", file=sys.stderr)
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

if [ "${WAIT_FOR_CLEAN_GPUS:-true}" = "true" ]; then
    wait_for_clean_gpus "${CUDA_DEVICE_LIST}"
fi

export VARIANT_TAG="${VARIANT_TAG:-hn256_varctx576_v3_gc${GRAD_CACHE_CHUNK_SIZE}_gpu${CUDA_DEVICE_TAG}_tcmoff_ep6}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_bs${BATCH_SIZE}_gc${GRAD_CACHE_CHUNK_SIZE}_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep${EPOCHS}_v3_smallest_dense_normAGGR_gpu${CUDA_DEVICE_TAG}_aries}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn256_gsv2full_gsdedup_varctx576_v3_bs${BATCH_SIZE}_gc${GRAD_CACHE_CHUNK_SIZE}_tcmoff_ep${EPOCHS}_gpu${CUDA_DEVICE_TAG}_aries}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260521__varctx_lmlb_v3_hn256_gc256_aries6.md}"

export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn256_varctx576_v3 compute:aries-6gpu ablation:hard_neg256 source:lh1b88kw}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw bgz7akb6 40fgbr2y ah9u1bao dxwrgbln}"
export RUN_VERDICT="${RUN_VERDICT:-HN256 ablation from lh1b88kw recipe: hard_neg_k_per_sample=256, grad_cache_chunk_size=${GRAD_CACHE_CHUNK_SIZE}, target global batch 8192 with effective equal-rank batch ${BATCH_SIZE}; primary best metric is eval_dev/recall@10_gs10000 and secondary best metric is eval_acl6060/recall@10.}"

source "${BASE_LAUNCHER}"
