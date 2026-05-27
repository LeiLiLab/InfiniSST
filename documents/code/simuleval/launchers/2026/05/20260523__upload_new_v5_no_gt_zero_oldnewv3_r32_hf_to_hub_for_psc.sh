#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/mnt/aries/data6/jiaxuanluo/slm/speech_llm_new_v5_no_gt_zero_llm_variant_aug_oldnewv3_zh_r32a64_tp2_aries2/keep1.0_r32/v0-20260523-050346-hf}"
HF_REPO_ID="${HF_REPO_ID:-gavinlaw/infinisst-new-v5-no-gt-zero-oldnewv3-r32a64-keep1p0-r32-zh}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STAGE_DIR="${STAGE_DIR:-/mnt/aries/data6/jiaxuanluo/hf_upload_staging/new_v5_no_gt_zero_oldnewv3_r32_hf}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[ERROR] Missing MODEL_DIR: ${MODEL_DIR}" >&2
  exit 3
fi
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "[ERROR] Missing HF config.json under MODEL_DIR: ${MODEL_DIR}" >&2
  exit 3
fi

shard_count="$(find "${MODEL_DIR}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
if [[ "${shard_count}" != "15" ]]; then
  echo "[ERROR] Expected 15 safetensor shards, found ${shard_count}: ${MODEL_DIR}" >&2
  exit 3
fi

echo "[INFO] Uploading HF model directory"
echo "[INFO] model_dir=${MODEL_DIR}"
echo "[INFO] hf_repo_id=${HF_REPO_ID}"
echo "[INFO] num_workers=${NUM_WORKERS}"
echo "[INFO] stage_dir=${STAGE_DIR}"
du -sh "${MODEL_DIR}"
hf auth whoami

# The exported HF directory is root-owned on aries, so `hf upload-large-folder`
# cannot write its resumable metadata there.  Use a writable symlink mirror:
# file contents are read from MODEL_DIR, while `.cache/huggingface` lives here.
mkdir -p "${STAGE_DIR}"
find "${MODEL_DIR}" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' f; do
  ln -sfn "${f}" "${STAGE_DIR}/$(basename "${f}")"
done

hf upload-large-folder \
  --repo-type model \
  --private \
  --num-workers "${NUM_WORKERS}" \
  --no-bars \
  "${HF_REPO_ID}" \
  "${STAGE_DIR}"

echo "[INFO] Upload finished: https://huggingface.co/${HF_REPO_ID}"
