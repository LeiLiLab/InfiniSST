#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
MEDICINE_LAUNCHER="${MEDICINE_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__medicine_abbrev_restored_norag_streaming_batched_aries67.sh}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm4_taurus45}"
CACHE_BASE="${CACHE_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/cache/medicine_norag_de_rerun_shorttmp_20260524T0403/lm4_taurus45}"
TMPDIR_SHORT="${TMPDIR_SHORT_OVERRIDE:-/dev/shm/jxde4t}"
RUN_STAMP="${RUN_STAMP_OVERRIDE:-20260524T0403}"

OUT_LOG="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_de_rerun_shorttmp_lm4_taurus45.out"
ERR_LOG="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_de_rerun_shorttmp_lm4_taurus45.err"
PID_FILE="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_de_rerun_shorttmp_lm4_taurus45.inner.pid"

if [[ -e "${OUTPUT_BASE}" ]]; then
  echo "[ERROR] output base already exists: ${OUTPUT_BASE}" >&2
  exit 3
fi

mkdir -p "${LOG_ROOT}" "${TMPDIR_SHORT}" "${CACHE_BASE}/xdg" "${CACHE_BASE}/triton" "${CACHE_BASE}/torchinductor" "${CACHE_BASE}/cuda"

(
  cd "${ROOT_DIR}"
  export TMPDIR="${TMPDIR_SHORT}"
  export XDG_CACHE_HOME="${CACHE_BASE}/xdg"
  export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/torchinductor"
  export CUDA_CACHE_PATH="${CACHE_BASE}/cuda"
  MASTER_PORT="${MASTER_PORT_OVERRIDE:-20544}" \
  LANGS_OVERRIDE="de" \
  TARGET_LMS_OVERRIDE="4" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-4:5}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  FORCE_RERUN_OVERRIDE="${FORCE_RERUN_OVERRIDE:-1}" \
  bash "${MEDICINE_LAUNCHER}"
) >"${OUT_LOG}" 2>"${ERR_LOG}" &

echo "$!" > "${PID_FILE}"
printf '[SUBMITTED] lang=de lm=4 host=taurus gpu_csv=%s pid=%s output_base=%s tmp_dir=%s cache_base=%s out_log=%s err_log=%s\n' \
  "${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-4:5}" "$(cat "${PID_FILE}")" "${OUTPUT_BASE}" "${TMPDIR_SHORT}" "${CACHE_BASE}" "${OUT_LOG}" "${ERR_LOG}"

wait "$(cat "${PID_FILE}")"
