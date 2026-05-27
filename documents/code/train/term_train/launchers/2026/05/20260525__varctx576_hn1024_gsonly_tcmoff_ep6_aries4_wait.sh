#!/bin/bash
set -euo pipefail

# Direct Aries 4-GPU launcher for the GigaSpeech-only data ablation.  It waits
# for the requested physical GPUs to be idle, then delegates to the shared
# HN1024 varctx training body.

export RUN_STAMP="${RUN_STAMP:-20260525T1248_gsonly_hn1024_varctx576_aries4}"
export CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export SELECT_CLEAN_GPUS="${SELECT_CLEAN_GPUS:-true}"
export WAIT_FOR_CUDA_DEVICES="${WAIT_FOR_CUDA_DEVICES:-true}"
export WAIT_FOR_CUDA_TIMEOUT_SEC="${WAIT_FOR_CUDA_TIMEOUT_SEC:-43200}"
export WAIT_FOR_CUDA_POLL_SEC="${WAIT_FOR_CUDA_POLL_SEC:-60}"
export WAIT_FOR_CUDA_MAX_USED_MIB="${WAIT_FOR_CUDA_MAX_USED_MIB:-500}"

export TRAIN_JSONL="${TRAIN_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2full_gsdedup_varctx576_gsonly.jsonl}"
export DEV_JSONL="${DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/term_dev_dataset_varctx2p88_3p84_4p80_5p76_new_version.jsonl}"
export ACL_DEV_JSONL="${ACL_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary_varctx2p88_3p84_4p80_5p76/acl6060_dev_dataset.jsonl}"
export MEDICINE_DEV_JSONL="${MEDICINE_DEV_JSONL:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_dev_dataset.jsonl}"

export EVAL_WIKI_GLOSSARY="${EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample10000.json}"
export EVAL_GLOSSARY_SIZES="${EVAL_GLOSSARY_SIZES:-10000}"
export ACL_EVAL_WIKI_GLOSSARY="${ACL_EVAL_WIKI_GLOSSARY:-/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json}"
export ACL_EVAL_GLOSSARY_SIZES="${ACL_EVAL_GLOSSARY_SIZES:-10000}"
export MEDICINE_EVAL_WIKI_GLOSSARY="${MEDICINE_EVAL_WIKI_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/medicine_eval_varctx2p88_3p84_4p80_5p76_clean_mfa_exact_only/medicine_glossary_gt_plus_medicine_wiki_gs10000.json}"
export MEDICINE_EVAL_GLOSSARY_SIZES="${MEDICINE_EVAL_GLOSSARY_SIZES:-10000}"

export VARIANT_TAG="${VARIANT_TAG:-hn1024_varctx576_gsonly_v3_tcmoff_ep6}"
export DATA_TAG="${DATA_TAG:-gsv2full_gsdedup_varctx576_gsonly}"
export VERSION="${VERSION:-3var_gsv2full_gsdedup_varctx576_gsonly_bs8192_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_4gpu_aries}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-variantE_hn1024_gsonly_varctx576_v3_bs8192_gc128_tcmoff_ep6_aries4_${RUN_STAMP}}"
export NOTES_FILE="${NOTES_FILE:-/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/notes/2026/05/20260525__varctx576_hn1024_gsonly_tcmoff_ep6_aries4.md}"
export EXTRA_WANDB_TAGS="${EXTRA_WANDB_TAGS:-variant:hn1024_gsonly_v3 compute:aries-4gpu ablation:data-gsonly}"
export BASELINE_RUN_IDS="${BASELINE_RUN_IDS:-lh1b88kw ah9u1bao dxwrgbln}"

export NUM_WORKERS="${NUM_WORKERS:-4}"
export PER_GPU_BATCH="${PER_GPU_BATCH:-2048}"
export BATCH_SIZE="${BATCH_SIZE:-8192}"
export GRAD_CACHE_CHUNK_SIZE="${GRAD_CACHE_CHUNK_SIZE:-128}"
export EPOCHS="${EPOCHS:-6}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-6}"
export MAX_STEPS="${MAX_STEPS:-0}"
export SAVE_STEPS="${SAVE_STEPS:-999999}"
export SAVE_LATEST_ON_EVAL="${SAVE_LATEST_ON_EVAL:-true}"
export SAVE_LATEST_STEPS="${SAVE_LATEST_STEPS:-0}"
export EVAL_STEPS_SAMPLE="${EVAL_STEPS_SAMPLE:-80}"
export EVAL_TOP100_SAMPLES="${EVAL_TOP100_SAMPLES:-3}"
export KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"

export HARD_NEG_K="${HARD_NEG_K:-0}"
export HARD_NEG_K_PER_SAMPLE="${HARD_NEG_K_PER_SAMPLE:-1024}"
export TCM_LOSS_WEIGHT="${TCM_LOSS_WEIGHT:-0.0}"
export TCM_POS_LOSS_WEIGHT="${TCM_POS_LOSS_WEIGHT:-0.0}"
export TCM_NEG_LOSS_WEIGHT="${TCM_NEG_LOSS_WEIGHT:-0.0}"
export TCM_POS_THRESHOLD="${TCM_POS_THRESHOLD:-0.80}"
export TCM_NEG_THRESHOLD="${TCM_NEG_THRESHOLD:-0.60}"
export TCM_SWEEP_THRESHOLDS="${TCM_SWEEP_THRESHOLDS:-0.85 0.80 0.75 0.70}"

export FIXED_AUDIO_SECONDS="${FIXED_AUDIO_SECONDS:-5.76}"
export EVAL_FIXED_AUDIO_SECONDS="${EVAL_FIXED_AUDIO_SECONDS:-5.76}"
export BEST_METRIC="${BEST_METRIC:-eval_dev/recall@10_gs10000}"
export BEST_METRIC_SECONDARY="${BEST_METRIC_SECONDARY:-eval_acl6060/recall@10}"
export STRICT_TRAIN_EVAL_TERM_FILTER="${STRICT_TRAIN_EVAL_TERM_FILTER:-false}"
export TRAIN_EXCLUDE_TERM_GLOSSARIES="${TRAIN_EXCLUDE_TERM_GLOSSARIES:-}"
export LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/tmp/jx_gsonly_${RUN_STAMP}}"
export MASTER_PORT="${MASTER_PORT:-20157}"

REQUIRED_PATHS=(
  "${TRAIN_JSONL}"
  "${DEV_JSONL}"
  "${ACL_DEV_JSONL}"
  "${MEDICINE_DEV_JSONL}"
  "${EVAL_WIKI_GLOSSARY}"
  "${ACL_EVAL_WIKI_GLOSSARY}"
  "${MEDICINE_EVAL_WIKI_GLOSSARY}"
  "${NOTES_FILE}"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
  if [ ! -f "${required_path}" ]; then
    echo "[ERROR] required file missing: ${required_path}" >&2
    exit 2
  fi
done

for tag in \
  "family:${EXPERIMENT_FAMILY:-sst_ood_hardneg}" \
  "task:${TASK_TAG:-train}" \
  "data:${DATA_TAG}" \
  "status:running" \
  ${EXTRA_WANDB_TAGS}; do
  if [ "${#tag}" -lt 1 ] || [ "${#tag}" -gt 64 ]; then
    echo "[ERROR] invalid WandB tag length (${#tag}): ${tag}" >&2
    exit 2
  fi
done

if [ "${WAIT_FOR_CUDA_DEVICES}" = "true" ]; then
  python3 - "${CUDA_DEVICE_LIST}" "${WAIT_FOR_CUDA_TIMEOUT_SEC}" "${WAIT_FOR_CUDA_POLL_SEC}" "${WAIT_FOR_CUDA_MAX_USED_MIB}" <<'PYEOF'
import subprocess
import sys
import time

requested = [x.strip() for x in sys.argv[1].replace(" ", ",").split(",") if x.strip()]
timeout = int(sys.argv[2])
poll = int(sys.argv[3])
threshold = int(sys.argv[4])
deadline = time.time() + timeout
while True:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    mem = {}
    for line in out.strip().splitlines():
        idx, used = [part.strip() for part in line.split(",")]
        mem[idx] = int(used)
    status = [(idx, mem.get(idx, -1)) for idx in requested]
    missing = [idx for idx, used in status if used < 0]
    busy = [(idx, used) for idx, used in status if used > threshold]
    print(f"[WAIT_GPU] requested={status} busy={busy}", flush=True)
    if missing:
        raise SystemExit(f"missing requested GPUs: {missing}")
    if not busy:
        break
    if time.time() >= deadline:
        raise SystemExit(f"timeout waiting for GPUs {requested} under {threshold} MiB")
    time.sleep(poll)
PYEOF
fi

source "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_mfa_smallest_dense_hn_depth_common_8gpu_aries.sh"
