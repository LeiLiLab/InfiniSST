#!/usr/bin/env bash
# Wait for all 8 Taurus GPUs to be idle, then run German retriever cap16 exact-boundary SFT.
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi
TRAIN_LAUNCHER="${TRAIN_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260525__speech_llm_de_retriever_cap16_exactboundary_taurus8_r32a32_ep4.sh}"
POLL_SEC="${POLL_SEC_OVERRIDE:-60}"
MAX_GPU_MEM_MIB="${MAX_GPU_MEM_MIB_OVERRIDE:-1000}"
GPU_CSV="${HOST_GPU_DEVICES_OVERRIDE:-0,1,2,3,4,5,6,7}"

for p in "${ROOT_DIR}" "${TRAIN_LAUNCHER}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

IFS=',' read -r -a EXPECTED_GPUS <<< "${GPU_CSV}"
if (( ${#EXPECTED_GPUS[@]} != 8 )); then
  echo "[ERROR] Expected 8 GPU ids, got ${GPU_CSV}" >&2
  exit 2
fi

gpu_idle() {
  python3 - "${GPU_CSV}" "${MAX_GPU_MEM_MIB}" <<'PY'
import subprocess
import sys

expected = {int(x) for x in sys.argv[1].split(",") if x.strip()}
limit = int(sys.argv[2])
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
seen = set()
busy = []
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        continue
    idx, mem, util = map(int, parts)
    if idx not in expected:
        continue
    seen.add(idx)
    if mem > limit:
        busy.append((idx, mem, util))
missing = sorted(expected - seen)
if missing:
    print(f"missing GPUs: {missing}")
    sys.exit(2)
if busy:
    print("busy GPUs: " + ", ".join(f"{i}:mem={m}MiB util={u}%" for i, m, u in busy))
    sys.exit(1)
print("all expected GPUs idle")
PY
}

echo "[WATCH] started at $(date -u --iso-8601=seconds)"
echo "[WATCH] ROOT_DIR=${ROOT_DIR}"
echo "[WATCH] TRAIN_LAUNCHER=${TRAIN_LAUNCHER}"
echo "[WATCH] GPU_CSV=${GPU_CSV} MAX_GPU_MEM_MIB=${MAX_GPU_MEM_MIB}"

while true; do
  if gpu_idle; then
    break
  fi
  echo "[WATCH] waiting $(date -u --iso-8601=seconds)"
  sleep "${POLL_SEC}"
done

echo "[WATCH] launching training at $(date -u --iso-8601=seconds)"
cd "${ROOT_DIR}"
ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
HOST_GPU_DEVICES_OVERRIDE="${GPU_CSV}" \
MASTER_PORT_OVERRIDE="${MASTER_PORT_OVERRIDE:-29704}" \
bash "${TRAIN_LAUNCHER}"
