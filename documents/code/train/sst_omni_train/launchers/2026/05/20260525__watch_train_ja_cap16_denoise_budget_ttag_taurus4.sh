#!/usr/bin/env bash
# Wait for four Taurus GPUs to be idle, then run Japanese cap16 denoise-budget SFT.
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi
TRAIN_LAUNCHER="${TRAIN_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260525__speech_llm_ja_cap16_denoise_budget_ttag_taurus4_r32a32_ep1.sh}"
POLL_SEC="${POLL_SEC_OVERRIDE:-60}"
MAX_GPU_MEM_MIB="${MAX_GPU_MEM_MIB_OVERRIDE:-1000}"
IDLE_GPU_COUNT="${IDLE_GPU_COUNT_OVERRIDE:-4}"
GPU_CSV="${HOST_GPU_DEVICES_OVERRIDE:-}"

for p in "${ROOT_DIR}" "${TRAIN_LAUNCHER}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

if [[ -n "${GPU_CSV}" ]]; then
  IFS=',' read -r -a EXPECTED_GPUS <<< "${GPU_CSV}"
  if (( ${#EXPECTED_GPUS[@]} != IDLE_GPU_COUNT )); then
    echo "[ERROR] Expected ${IDLE_GPU_COUNT} GPU ids, got ${GPU_CSV}" >&2
    exit 2
  fi
fi

select_idle_gpus() {
  python3 - "${GPU_CSV}" "${IDLE_GPU_COUNT}" "${MAX_GPU_MEM_MIB}" <<'PY'
import subprocess
import sys

requested = sys.argv[1].strip()
need = int(sys.argv[2])
limit = int(sys.argv[3])
requested_set = None
if requested:
    requested_set = {int(x) for x in requested.split(",") if x.strip()}
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
idle = []
busy = []
seen = set()
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        continue
    idx, mem, util = map(int, parts)
    if requested_set is not None and idx not in requested_set:
        continue
    seen.add(idx)
    if mem <= limit:
        idle.append(idx)
    else:
        busy.append((idx, mem, util))
if requested_set is not None:
    missing = sorted(requested_set - seen)
    if missing:
        print(f"missing GPUs: {missing}", file=sys.stderr)
        sys.exit(2)
if len(idle) >= need:
    print(",".join(str(x) for x in sorted(idle)[:need]))
    sys.exit(0)
print("idle GPUs: " + (",".join(str(x) for x in sorted(idle)) or "<none>"), file=sys.stderr)
print("busy GPUs: " + ", ".join(f"{i}:mem={m}MiB util={u}%" for i, m, u in busy), file=sys.stderr)
sys.exit(1)
PY
}

echo "[WATCH] started at $(date -u --iso-8601=seconds)"
echo "[WATCH] ROOT_DIR=${ROOT_DIR}"
echo "[WATCH] TRAIN_LAUNCHER=${TRAIN_LAUNCHER}"
echo "[WATCH] requested GPU_CSV=${GPU_CSV:-<any>} IDLE_GPU_COUNT=${IDLE_GPU_COUNT} MAX_GPU_MEM_MIB=${MAX_GPU_MEM_MIB}"

SELECTED_GPUS=""
while true; do
  if SELECTED_GPUS="$(select_idle_gpus 2> >(sed 's/^/[WATCH] /' >&2))"; then
    break
  fi
  echo "[WATCH] waiting $(date -u --iso-8601=seconds)"
  sleep "${POLL_SEC}"
done

echo "[WATCH] selected GPUs=${SELECTED_GPUS}"
echo "[WATCH] launching training at $(date -u --iso-8601=seconds)"
cd "${ROOT_DIR}"
ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
HOST_GPU_DEVICES_OVERRIDE="${SELECTED_GPUS}" \
MASTER_PORT_OVERRIDE="${MASTER_PORT_OVERRIDE:-29724}" \
bash "${TRAIN_LAUNCHER}"
