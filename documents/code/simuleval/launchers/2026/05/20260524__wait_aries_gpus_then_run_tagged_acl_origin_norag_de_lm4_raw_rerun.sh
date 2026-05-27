#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
HOLD_JOB_ID="${HOLD_JOB_ID:-45290}"
RUN_STAMP="${RUN_STAMP:-20260524T160830_tagacl_origin_norag_de_lm4_raw_rerun}"
GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-0,1;2,3;4,5;6,7}"
FREE_MEM_MAX_MIB="${FREE_MEM_MAX_MIB:-1000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"

MAIN_LAUNCHER="${MAIN_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__tagged_acl_origin_norag_de_lm4_raw_rerun.sh}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_origin_norag_de_lm4_raw_rerun_${RUN_STAMP}}"

mkdir -p "${LOG_ROOT}"

ts() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

choose_pair() {
  local probe
  probe="$(
    srun --jobid="${HOLD_JOB_ID}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 \
      --chdir="${ROOT_DIR}" \
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
  )"
  PROBE="${probe}" python - "${GPU_PAIRS_CSV}" "${FREE_MEM_MAX_MIB}" <<'PY'
import os
import sys

pairs_s, limit_s = sys.argv[1:3]
limit = int(limit_s)
mem = {}
for line in os.environ.get("PROBE", "").splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        continue
    try:
        mem[int(parts[0])] = int(float(parts[1]))
    except ValueError:
        continue
for pair in [p.strip() for p in pairs_s.split(";") if p.strip()]:
    ids = [int(x) for x in pair.split(",")]
    if all(mem.get(i, 10**9) <= limit for i in ids):
        print(pair)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

main() {
  echo "[$(ts)] waiting for an Aries GPU pair in job ${HOLD_JOB_ID}; pairs=${GPU_PAIRS_CSV}; free_mem_max=${FREE_MEM_MAX_MIB} MiB"
  while true; do
    if pair="$(choose_pair)"; then
      echo "[$(ts)] selected Aries GPU pair: ${pair}"
      exec srun --jobid="${HOLD_JOB_ID}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 \
        --chdir="${ROOT_DIR}" \
        env RUN_STAMP="${RUN_STAMP}" GPU_PAIR="${pair}" ROOT_DIR="${ROOT_DIR}" \
          bash "${MAIN_LAUNCHER}"
    fi
    echo "[$(ts)] no free pair yet; sleeping ${CHECK_INTERVAL_SEC}s"
    sleep "${CHECK_INTERVAL_SEC}"
  done
}

main "$@"
