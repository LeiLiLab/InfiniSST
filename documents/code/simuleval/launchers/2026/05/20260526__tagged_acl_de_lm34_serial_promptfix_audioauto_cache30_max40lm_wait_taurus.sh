#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260526T015250_tagged_acl_de_lm34_serial_promptfix_audioauto_cache30_max40lm_wait_taurus}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260526__tagged_acl_de_lm3_serial_promptfix_cache30_audioauto_max40lm_taurus.sh}"
LOG_ROOT="${LOG_ROOT:-/mnt/data1/jiaxuanluo/logs/tagged_acl_de_lm34_serial_promptfix_audioauto_cache30_max40lm_wait_${RUN_STAMP}}"
INDEX_PATH="${INDEX_PATH:-/mnt/data1/jiaxuanluo/maxsim_index_cache/tagged_acl_de_lm3_serial_promptfix_cache30_audioauto_max40lm/maxsim_acl6060_tagged_gt_raw_min_norm2_ebc26806ed693f1a_tr128_ta256.pt}"
POLL_SECS="${POLL_SECS:-30}"
MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-2048}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-25}"

mkdir -p "${LOG_ROOT}"
echo "$$" > "${LOG_ROOT}/watcher.pid"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_file() {
  [[ -s "$1" ]] || fail "Missing/empty required file: $1"
}

gpu_is_idle() {
  local gpu="$1" line mem util
  line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, -v g="${gpu}" '$1 + 0 == g {print $0}')"
  [[ -n "${line}" ]] || return 1
  mem="$(awk -F, '{gsub(/[[:space:]]/, "", $2); print $2}' <<< "${line}")"
  util="$(awk -F, '{gsub(/[[:space:]]/, "", $3); print $3}' <<< "${line}")"
  (( mem <= MAX_IDLE_GPU_MEM_MB && util <= MAX_IDLE_GPU_UTIL ))
}

wait_pair_idle() {
  local pair="$1" g0 g1
  IFS=',' read -r g0 g1 <<< "${pair}"
  while true; do
    if gpu_is_idle "${g0}" && gpu_is_idle "${g1}"; then
      return 0
    fi
    echo "[WAIT] $(date -u +%Y-%m-%dT%H:%M:%SZ) pair=${pair} busy; retry in ${POLL_SECS}s" | tee -a "${LOG_ROOT}/watcher.log"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits | tee -a "${LOG_ROOT}/watcher.log" >/dev/null || true
    sleep "${POLL_SECS}"
  done
}

launch_one() {
  local lm="$1" pair="$2"
  local max_new=$((40 * lm))
  local stamp="${RUN_STAMP}_lm${lm}"
  local child_log="/mnt/data1/jiaxuanluo/logs/tagged_acl_de_lm${lm}_serial_promptfix_audioauto_cache30_max40lm_${stamp}"
  mkdir -p "${child_log}"
  echo "[LAUNCH] $(date -u +%Y-%m-%dT%H:%M:%SZ) lm=${lm} pair=${pair}" | tee -a "${LOG_ROOT}/watcher.log"
  RUN_STAMP="${stamp}" \
  LM="${lm}" \
  MAX_NEW_TOKENS="${max_new}" \
  GPU_PAIR="${pair}" \
  VLLM_LIMIT_AUDIO="auto" \
  VLLM_LIMIT_AUDIO_OVERRIDE="auto" \
  EVAL_TMPDIR="/tmp/jxde${lm}auto" \
  INDEX_PATH_OVERRIDE="${INDEX_PATH}" \
  setsid bash "${BASE_LAUNCHER}" \
    > "${child_log}/launcher.out" \
    2> "${child_log}/launcher.err" \
    < /dev/null &
  local pid=$!
  echo "${pid}" > "${child_log}/setsid.pid"
  echo "[PID] lm=${lm} pid=${pid} child_log=${child_log}" | tee -a "${LOG_ROOT}/watcher.log"
}

main() {
  [[ "$(hostname -s)" == taurus* ]] || fail "This launcher is Taurus-only; current host=$(hostname -s)"
  require_file "${BASE_LAUNCHER}"
  require_file "${INDEX_PATH}"
  echo "run_stamp=${RUN_STAMP}" | tee "${LOG_ROOT}/run_meta.txt"
  echo "index_path=${INDEX_PATH}" | tee -a "${LOG_ROOT}/run_meta.txt"
  echo "policy=VLLM_LIMIT_AUDIO=auto cache=30/30 max_new_tokens=40*lm" | tee -a "${LOG_ROOT}/run_meta.txt"

  wait_pair_idle "0,1"
  launch_one 3 "0,1"

  wait_pair_idle "2,3"
  launch_one 4 "2,3"

  date -u +%Y-%m-%dT%H:%M:%SZ > "${LOG_ROOT}/submitted.txt"
}

main "$@"
