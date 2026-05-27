#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260526T035925_medicine_de_lm1234_serial_promptfix_cache30_audioauto_max40lm_taurus}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260526__medicine_de_serial_promptfix_cache30_audioauto_max40lm_taurus.sh}"
OUT_ROOT="${OUT_ROOT:-/mnt/data1/jiaxuanluo/medicine_de_lm1234_serial_promptfix_cache30_audioauto_max40lm_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/data1/jiaxuanluo/logs/medicine_de_lm1234_serial_promptfix_cache30_audioauto_max40lm_${RUN_STAMP}}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/top.pid}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
echo "$$" > "${PID_FILE}"

run_lm() {
  local lm="$1"
  local gpu_pair="$2"
  local child_stamp="${RUN_STAMP}_lm${lm}"
  local child_out="${OUT_ROOT}/lm${lm}"
  local child_log="${LOG_ROOT}/lm${lm}"
  local max_new=$((lm * 40))
  mkdir -p "${child_out}" "${child_log}"
  {
    echo "[INFO] start medicine de lm=${lm} gpu_pair=${gpu_pair} max_new_tokens=${max_new}"
    RUN_STAMP="${child_stamp}" \
    OUT_ROOT="${child_out}" \
    LOG_ROOT="${child_log}" \
    GPU_PAIR="${gpu_pair}" \
    LM="${lm}" \
    MAX_NEW_TOKENS="${max_new}" \
    MAX_CACHE_CHUNKS=30 \
    KEEP_CACHE_CHUNKS=30 \
    VLLM_LIMIT_AUDIO=auto \
    VLLM_LIMIT_AUDIO_OVERRIDE=auto \
    EVAL_TMPDIR="/tmp/jxmdde${lm}c30" \
    DENSITY_TAG="medhard_de_lm${lm}_serial_promptfix_cache30_audioauto_max40lm" \
    bash "${BASE_LAUNCHER}"
  } > "${child_log}/wrapper.stdout" 2> "${child_log}/wrapper.stderr"
}

echo "run_stamp=${RUN_STAMP}" | tee "${OUT_ROOT}/run_meta.txt"
echo "host=$(hostname -s)" | tee -a "${OUT_ROOT}/run_meta.txt"
echo "out_root=${OUT_ROOT}" | tee -a "${OUT_ROOT}/run_meta.txt"
echo "log_root=${LOG_ROOT}" | tee -a "${OUT_ROOT}/run_meta.txt"
echo "launcher=${BASE_LAUNCHER}" | tee -a "${OUT_ROOT}/run_meta.txt"
echo "policy=serial_promptfix cache_chunks=30/30 vllm_limit_audio=auto max_new_tokens=40*lm" | tee -a "${OUT_ROOT}/run_meta.txt"
df -h /mnt/data1 /mnt/gemini/data1 | tee "${LOG_ROOT}/df_prelaunch.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_prelaunch.csv"

run_lm 1 "0,1" &
pid_lm1=$!
run_lm 2 "4,5" &
pid_lm2=$!
run_lm 3 "6,7" &
pid_lm3=$!

echo "lm1_pid=${pid_lm1}" | tee "${LOG_ROOT}/child_pids.txt"
echo "lm2_pid=${pid_lm2}" | tee -a "${LOG_ROOT}/child_pids.txt"
echo "lm3_pid=${pid_lm3}" | tee -a "${LOG_ROOT}/child_pids.txt"

set +e
wait "${pid_lm2}"
status_lm2=$?
set -e
echo "lm2_status=${status_lm2}" | tee "${LOG_ROOT}/lm2.status"

run_lm 4 "4,5" &
pid_lm4=$!
echo "lm4_pid=${pid_lm4}" | tee -a "${LOG_ROOT}/child_pids.txt"

set +e
wait "${pid_lm1}"
status_lm1=$?
wait "${pid_lm3}"
status_lm3=$?
wait "${pid_lm4}"
status_lm4=$?
set -e

{
  echo "lm1_status=${status_lm1}"
  echo "lm2_status=${status_lm2}"
  echo "lm3_status=${status_lm3}"
  echo "lm4_status=${status_lm4}"
} | tee "${LOG_ROOT}/child_status.tsv"

python3 - "${OUT_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for lm in (1, 2, 3, 4):
    paths = sorted((root / f"lm{lm}").glob(f"__summary__/summary_medicine_hardraw_de_lm{lm}_serial_promptfix.tsv"))
    if paths:
        with paths[0].open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f, delimiter="\t"))
summary = root / "__summary__"
summary.mkdir(exist_ok=True)
out = summary / "summary_medicine_hardraw_de_lm1_lm2_lm3_lm4_serial_promptfix.tsv"
if rows:
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] wrote combined summary: {out}")
else:
    print("[WARN] no per-lm summaries found")
PY

date -u +%Y-%m-%dT%H:%M:%SZ > "${LOG_ROOT}/done.txt"
if [[ "${status_lm1}" != "0" || "${status_lm2}" != "0" || "${status_lm3}" != "0" || "${status_lm4}" != "0" ]]; then
  exit 1
fi
