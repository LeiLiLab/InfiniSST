#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
MEDICINE_LAUNCHER="${MEDICINE_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260522__medicine_abbrev_restored_norag_streaming_batched_aries67.sh}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs}"
OUT_ROOT_PREFIX="${OUT_ROOT_PREFIX_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_de}"
RUN_STAMP="${RUN_STAMP_OVERRIDE:-20260524T0305}"

mkdir -p "${LOG_ROOT}"

run_one() {
  local lm="$1"
  local gpu_csv="$2"
  local port="$3"
  local tag="$4"
  local output_base="${OUT_ROOT_PREFIX}_lm${lm}_${tag}"
  local out_log="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_de_lm${lm}_${tag}.out"
  local err_log="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_de_lm${lm}_${tag}.err"
  local pid_file="${LOG_ROOT}/${RUN_STAMP}_medicine_norag_abbrev_restored_de_lm${lm}_${tag}.inner.pid"

  (
    cd "${ROOT_DIR}"
    MASTER_PORT="${port}" \
    LANGS_OVERRIDE="de" \
    TARGET_LMS_OVERRIDE="${lm}" \
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="${gpu_csv}" \
    OUTPUT_BASE_OVERRIDE="${output_base}" \
    FORCE_RERUN_OVERRIDE="${FORCE_RERUN_OVERRIDE:-1}" \
    bash "${MEDICINE_LAUNCHER}"
  ) >"${out_log}" 2>"${err_log}" &
  echo "$!" > "${pid_file}"
  printf '[SUBMITTED] lang=de lm=%s gpu_csv=%s pid=%s output_base=%s out_log=%s err_log=%s\n' \
    "${lm}" "${gpu_csv}" "$(cat "${pid_file}")" "${output_base}" "${out_log}" "${err_log}"
}

run_one 1 "0:1" 20141 "aries01"
run_one 2 "2:3" 20142 "aries23"
run_one 3 "4:5" 20143 "aries45"
run_one 4 "6:7" 20144 "aries67"

wait
