#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)}"
GPU_PAIR="${GPU_PAIR:-6,7}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/v9_v10_dense_top10_train_eval_${RUN_STAMP}}"

BUILD_LAUNCHER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260521__build_v9_v10_dense_top10_notau_backfill_zh.sh"
TRAIN_V9="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260521__speech_llm_v9_dense_top10_notau_backfill_zh_aries2_r8a32.sh"
TRAIN_V10="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260521__speech_llm_v10_marker_dense_top10_notau_backfill_zh_aries2_r8a32.sh"
EVAL_LAUNCHER="${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260521__tagged_acl_v9_v10_quick_zh_lm2_raw_aries67.sh"

for p in "${BUILD_LAUNCHER}" "${TRAIN_V9}" "${TRAIN_V10}" "${EVAL_LAUNCHER}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] Missing launcher: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${LOG_ROOT}"
cd "${ROOT_DIR}"

echo "[PIPELINE] run_stamp=${RUN_STAMP}"
echo "[PIPELINE] gpu_pair=${GPU_PAIR}"
echo "[PIPELINE] log_root=${LOG_ROOT}"

echo "[PIPELINE] Step 1/4: build V9/V10 data"
GPU_DEVICES_CSV_OVERRIDE="${GPU_PAIR}" \
NUM_SHARDS_OVERRIDE=2 \
bash "${BUILD_LAUNCHER}" \
  > "${LOG_ROOT}/01_build_data.out" \
  2> "${LOG_ROOT}/01_build_data.err"

echo "[PIPELINE] Step 2/5: train V9"
HOST_GPU_DEVICES_OVERRIDE="${GPU_PAIR}" \
MASTER_PORT=30591 \
bash "${TRAIN_V9}" \
  > "${LOG_ROOT}/02_train_v9.out" \
  2> "${LOG_ROOT}/02_train_v9.err"

echo "[PIPELINE] Step 3/5: quick eval V9"
RUN_STAMP="${RUN_STAMP}" \
GPU_PAIR="${GPU_PAIR}" \
EVAL_VARIANTS="v9_dense" \
bash "${EVAL_LAUNCHER}" \
  > "${LOG_ROOT}/03_eval_v9.out" \
  2> "${LOG_ROOT}/03_eval_v9.err"

echo "[PIPELINE] Step 4/5: train V10"
HOST_GPU_DEVICES_OVERRIDE="${GPU_PAIR}" \
MASTER_PORT=30592 \
bash "${TRAIN_V10}" \
  > "${LOG_ROOT}/04_train_v10.out" \
  2> "${LOG_ROOT}/04_train_v10.err"

echo "[PIPELINE] Step 5/5: quick eval V10"
RUN_STAMP="${RUN_STAMP}" \
GPU_PAIR="${GPU_PAIR}" \
EVAL_VARIANTS="v10_marker" \
bash "${EVAL_LAUNCHER}" \
  > "${LOG_ROOT}/05_eval_v10.out" \
  2> "${LOG_ROOT}/05_eval_v10.err"

echo "[PIPELINE] ALL DONE"
