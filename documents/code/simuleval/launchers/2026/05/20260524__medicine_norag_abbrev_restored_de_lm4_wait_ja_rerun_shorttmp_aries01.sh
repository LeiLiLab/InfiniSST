#!/usr/bin/env bash
set -euo pipefail

JA_WRAPPER_PID="${JA_WRAPPER_PID_OVERRIDE:-3352336}"
ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
MEDICINE_LAUNCHER="${MEDICINE_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__medicine_abbrev_restored_norag_streaming_batched_aries67.sh}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm4_aries01}"
CACHE_BASE="${CACHE_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/cache/medicine_norag_de_rerun_shorttmp_20260524T0352/lm4_aries01}"
TMPDIR_SHORT="${TMPDIR_SHORT_OVERRIDE:-/dev/shm/jxde4}"
RUN_STAMP="${RUN_STAMP_OVERRIDE:-20260524T0352}"
WAIT_SECONDS="${WAIT_SECONDS_OVERRIDE:-60}"
MAX_POLLS="${MAX_POLLS_OVERRIDE:-720}"

OUT_LOG="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_de_rerun_shorttmp_lm4_aries01.out"
ERR_LOG="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_de_rerun_shorttmp_lm4_aries01.err"
PID_FILE="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_de_rerun_shorttmp_lm4_aries01.wrapper.pid"

mkdir -p "${LOG_ROOT}" "${TMPDIR_SHORT}" "${CACHE_BASE}/xdg" "${CACHE_BASE}/triton" "${CACHE_BASE}/torchinductor" "${CACHE_BASE}/cuda"

for _ in $(seq 1 "${MAX_POLLS}"); do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if ps -p "${JA_WRAPPER_PID}" >/dev/null 2>&1; then
    echo "[${ts}] waiting for ja wrapper pid=${JA_WRAPPER_PID} before starting de lm4"
    sleep "${WAIT_SECONDS}"
    continue
  fi
  echo "[${ts}] ja wrapper exited; starting de lm4 on GPUs 0,1"
  break
done

if ps -p "${JA_WRAPPER_PID}" >/dev/null 2>&1; then
  echo "[ERROR] timed out waiting for ja wrapper pid=${JA_WRAPPER_PID}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_BASE}" ]]; then
  echo "[ERROR] output base already exists: ${OUTPUT_BASE}" >&2
  exit 3
fi

ssh aries "setsid bash -lc 'cd ${ROOT_DIR} && export TMPDIR=${TMPDIR_SHORT} XDG_CACHE_HOME=${CACHE_BASE}/xdg TRITON_CACHE_DIR=${CACHE_BASE}/triton TORCHINDUCTOR_CACHE_DIR=${CACHE_BASE}/torchinductor CUDA_CACHE_PATH=${CACHE_BASE}/cuda && MASTER_PORT=20444 LANGS_OVERRIDE=de TARGET_LMS_OVERRIDE=4 CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV=0:1 OUTPUT_BASE_OVERRIDE=${OUTPUT_BASE} FORCE_RERUN_OVERRIDE=1 bash ${MEDICINE_LAUNCHER}' > ${OUT_LOG} 2> ${ERR_LOG} < /dev/null & echo \$! > ${PID_FILE}; cat ${PID_FILE}"
echo "[STARTED] de lm4 wrapper pid=$(cat "${PID_FILE}" 2>/dev/null || true)"
