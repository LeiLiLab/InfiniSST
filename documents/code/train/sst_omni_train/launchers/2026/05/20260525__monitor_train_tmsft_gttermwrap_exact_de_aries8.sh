#!/usr/bin/env bash
# Monitor the Aries TM-SFT exact GT term-wrap training and register completion.
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
EVENT_ID="20260525T0055__speech_llm_train__tmsft_gttermwrap_exact_de_r32a32_ep4_aries8"
MANIFEST="${ROOT_DIR}/documents/code/train/sst_omni_train/manifests/2026/05/${EVENT_ID}.json"
REMOTE_HOST="${REMOTE_HOST_OVERRIDE:-aries}"
REMOTE_PID="${REMOTE_PID_OVERRIDE:-3508143}"
WANDB_RUN_ID="${WANDB_RUN_ID_OVERRIDE:-sf6nw09x}"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_tmsft_gttermwrap_exact_de_r32a32_ep4_aries8}"
TRAIN_LOG="${TRAIN_LOG_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/speech_llm_tmsft_gttermwrap_exact_de_r32a32_ep4_aries8/train_keep1.0_r32_20260525_083648.log}"
DETACHED_OUT="${DETACHED_OUT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/train_tmsft_gttermwrap_exact_de_aries8_20260525T0055.out}"
DETACHED_ERR="${DETACHED_ERR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/train_tmsft_gttermwrap_exact_de_aries8_20260525T0055.err}"
POLL_SEC="${POLL_SEC_OVERRIDE:-60}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[ERROR] Missing manifest: ${MANIFEST}" >&2
  exit 3
fi

cd "${ROOT_DIR}"

echo "[MONITOR] started at $(date -u --iso-8601=seconds)"
echo "[MONITOR] remote=${REMOTE_HOST} pid=${REMOTE_PID}"
echo "[MONITOR] manifest=${MANIFEST}"
echo "[MONITOR] run_root=${RUN_ROOT}"

while ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "ps -p ${REMOTE_PID} >/dev/null 2>&1"; do
  echo "[MONITOR] still running $(date -u --iso-8601=seconds)"
  sleep "${POLL_SEC}"
done

echo "[MONITOR] remote pid exited at $(date -u --iso-8601=seconds)"

mapfile -t HF_DIRS < <(find "${RUN_ROOT}" -maxdepth 5 -type d -name '*-hf' -print 2>/dev/null | sort)
FAIL_HITS=""
if [[ -f "${TRAIN_LOG}" ]]; then
  FAIL_HITS="$(grep -En 'Traceback|ERROR|RuntimeError|CUDA out of memory|Training failed' "${TRAIN_LOG}" || true)"
fi
if [[ -f "${DETACHED_ERR}" ]]; then
  err_hits="$(grep -En 'Traceback|ERROR|RuntimeError|CUDA out of memory|Training failed' "${DETACHED_ERR}" || true)"
  if [[ -n "${err_hits}" ]]; then
    FAIL_HITS="${FAIL_HITS}"$'\n'"${err_hits}"
  fi
fi

if (( ${#HF_DIRS[@]} > 0 )) && [[ -z "${FAIL_HITS//[[:space:]]/}" ]]; then
  FINAL_STATUS="success_hf_exported"
else
  FINAL_STATUS="failed_or_incomplete_no_clean_hf_export"
fi

HF_JSON="$(python3 - "${HF_DIRS[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:], ensure_ascii=True))
PY
)"

python3 - "${MANIFEST}" "${FINAL_STATUS}" "${HF_JSON}" "${TRAIN_LOG}" "${DETACHED_OUT}" "${DETACHED_ERR}" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
status = sys.argv[2]
hf_dirs = json.loads(sys.argv[3])
train_log = sys.argv[4]
detached_out = sys.argv[5]
detached_err = sys.argv[6]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = status
metadata = manifest.setdefault("metadata", {})
metadata["completion_checked_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
metadata["completion_monitor"] = {
    "script": "documents/code/train/sst_omni_train/launchers/2026/05/20260525__monitor_train_tmsft_gttermwrap_exact_de_aries8.sh",
    "hf_export_dirs": hf_dirs,
    "train_log": train_log,
    "detached_stdout": detached_out,
    "detached_stderr": detached_err,
}
artifacts = manifest.setdefault("artifacts", [])
existing = {(a.get("role"), a.get("path")) for a in artifacts}
for hf_dir in hf_dirs:
    key = ("hf_export", hf_dir)
    if key not in existing:
        artifacts.append({
            "role": "hf_export",
            "type": "hf_model_dir",
            "direction": "output",
            "path": hf_dir,
            "metadata": {},
        })
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
subprocess.run(["python", "documents/code/general/experiment_event.py", "register", str(manifest_path)], check=True)
PY

python "${ROOT_DIR}/documents/code/general/wandb_tool.py" --project sst_omni db-sync --runs "${WANDB_RUN_ID}" || true

if command -v "${HOME}/bin/codex-notify" >/dev/null 2>&1; then
  "${HOME}/bin/codex-notify" --delay 8 --detach --workspace "${ROOT_DIR}" \
    "TM-SFT exact term-wrap training finished: ${FINAL_STATUS}, run ${WANDB_RUN_ID}"
fi

echo "[MONITOR] final_status=${FINAL_STATUS}"
printf '[MONITOR] hf_dirs=%s\n' "${HF_DIRS[*]:-}"
