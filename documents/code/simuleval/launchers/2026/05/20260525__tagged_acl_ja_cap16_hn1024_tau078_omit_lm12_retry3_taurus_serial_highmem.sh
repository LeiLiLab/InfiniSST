#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP_OVERRIDE:-20260525T1344_jacap16_hn1024_tau078_omit_lm12_retry3_taurus_serial_highmem}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_ja_cap16_hn1024_tau078_omit_lm12_retry3_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_ja_cap16_hn1024_tau078_omit_lm12_retry3_${RUN_STAMP}}"
HOST_LAUNCHER="${HOST_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260525__tagged_acl_ja_cap16_hn1024_tau078_omit_batch_host.sh}"
OUTPUT_BASE="${OUT_ROOT}/ja_retr_cap16_exact_r32a32_ep4_hn1024_tau078_omit_batch_max80"
SUMMARY_DIR="${OUTPUT_BASE}/__summary__"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${SUMMARY_DIR}"

run_lm() {
  local lm="$1"
  echo "[RUN_LM] lm=${lm} pair=0,7 gpu_memory_utilization=0.72"
  env \
    RUN_STAMP_OVERRIDE="${RUN_STAMP}" \
    HOST_LABEL_OVERRIDE=taurus \
    LMS_OVERRIDE="${lm}" \
    AUTO_PAIR_OVERRIDE=0 \
    GPU_PAIRS_CSV_OVERRIDE="0,7" \
    OUT_ROOT_OVERRIDE="${OUT_ROOT}" \
    LOG_ROOT_OVERRIDE="${LOG_ROOT}" \
    EVAL_TMPDIR_ROOT_OVERRIDE=/tmp/jx_jc12r3 \
    WANDB_LOG_OVERRIDE=1 \
    POLL_SECS_OVERRIDE=20 \
    GPU_MEMORY_UTILIZATION_OVERRIDE=0.72 \
    bash "${HOST_LAUNCHER}"
}

run_lm 1
run_lm 2

python - "${SUMMARY_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

summary_dir = Path(sys.argv[1])
paths = [summary_dir / "summary_ja_lm1.tsv", summary_dir / "summary_ja_lm2.tsv"]
rows = []
for path in paths:
    if not path.is_file():
        raise SystemExit(f"missing summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(data)}")
    rows.extend(data)
out = summary_dir / "summary_ja_lm12.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: int(row["lm"])))
print(f"[SUMMARY] {out}", flush=True)
PY

echo "[ALL DONE] output_base=${OUTPUT_BASE}"
