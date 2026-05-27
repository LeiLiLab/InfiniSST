#!/usr/bin/env bash
set -euo pipefail

HOST_LABEL="${HOST_LABEL:?set HOST_LABEL}"
DEST_BASE="${DEST_BASE:?set DEST_BASE}"
LOG_ROOT="${LOG_ROOT:?set LOG_ROOT}"
ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
BW_LIMIT_KB="${BW_LIMIT_KB:-80000}"

mkdir -p "${LOG_ROOT}"
cd "${ROOT_DIR}"

setsid bash -lc \
  "HOST_LABEL='${HOST_LABEL}' DEST_BASE='${DEST_BASE}' BW_LIMIT_KB='${BW_LIMIT_KB}' bash documents/code/simuleval/launchers/2026/05/20260525__stage_de_slm_local_cache.sh" \
  > "${LOG_ROOT}/stage.out" \
  2> "${LOG_ROOT}/stage.err" \
  < /dev/null &

echo "$!" > "${LOG_ROOT}/stage.pid"
echo "[INFO] launched stage cache pid=$(cat "${LOG_ROOT}/stage.pid") log_root=${LOG_ROOT}"
