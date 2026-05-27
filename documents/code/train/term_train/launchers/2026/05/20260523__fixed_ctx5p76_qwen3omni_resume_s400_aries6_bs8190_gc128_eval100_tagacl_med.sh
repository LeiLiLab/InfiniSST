#!/bin/bash
#SBATCH --job-name=q3_ctx576_r_a6
#SBATCH --partition=aries
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:6
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/InfiniSST
#SBATCH --output=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ctx576_r_a6_%x.out
#SBATCH --error=/mnt/gemini/data1/jiaxuanluo/logs/%j_q3_ctx576_r_a6_%x.err

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${REPO_ROOT}/documents/code/train/term_train/launchers/2026/05/20260519__fixed_ctx5p76_qwen3omni_taurus8_bs8k_gc128_eval100_tagacl_med.sh}"

export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,6,7}"
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

# The training path uses equal per-rank DDP batches. With six GPUs, exact 8192
# would floor to 8190 internally, so make the effective value explicit.
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-8192}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-$((TARGET_GLOBAL_BATCH / NUM_GPUS))}"
export BATCH_SIZE="${BATCH_SIZE:-$((PER_GPU_BATCH * NUM_GPUS))}"

export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-128}"
export MASTER_PORT="${MASTER_PORT:-20042}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"

export RESUME="${RESUME:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_ctx5p76_bs8k_gc128_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_q3o_taurus8_best.pt}"
export RESET_BEST_ON_RESUME="${RESET_BEST_ON_RESUME:-false}"
export RESET_SCHEDULER="${RESET_SCHEDULER:-false}"
export RESUME_COSINE_DECAY_TO_MAX_STEPS="${RESUME_COSINE_DECAY_TO_MAX_STEPS:-false}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/dev/shm/q3_ctx576_resume_a6_${USER}_${RUN_STAMP}}"
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

export VARIANT_TAG="${VARIANT_TAG:-hn1024_ctx5p76_q3o_a6_b${BATCH_SIZE}_g${GRAD_CACHE_CHUNK_SIZE}_resume_s400}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_ctx5p76_bs${BATCH_SIZE}_gc${GRAD_CACHE_CHUNK_SIZE}_eval100_tagacl_med_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep${EPOCHS:-6}_v3_q3o_resume_s400_gpu${CUDA_DEVICE_TAG}_aries6}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn1024_gsv2full_gsdedup_ctx5p76_q3o_aries6_bs${BATCH_SIZE}_gc${GRAD_CACHE_CHUNK_SIZE}_eval100_tagacl_med_resume_s400}"
export NOTES_FILE="${NOTES_FILE:-${REPO_ROOT}/documents/code/train/term_train/notes/2026/05/20260523__fixed_ctx5p76_qwen3omni_resume_s400_aries6_bs8190_gc128_eval100_tagacl_med.md}"

export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn1024_ctx5p76_q3o_a6_b8190_g128 compute:aries-6gpu source:zseptpl0 resume:step400}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-zseptpl0 lh1b88kw ah9u1bao dxwrgbln}"
export RUN_VERDICT="${RUN_VERDICT:-Resume fixed 5.76s Qwen3-Omni retriever control from zseptpl0 step-400 primary checkpoint on Aries GPUs 0,1,2,3,6,7; target batch 8192, effective equal-rank batch 8190, eval every 100 steps, dev-primary checkpoint selection with tagged ACL and medicine readouts.}"

source "${BASE_LAUNCHER}"
