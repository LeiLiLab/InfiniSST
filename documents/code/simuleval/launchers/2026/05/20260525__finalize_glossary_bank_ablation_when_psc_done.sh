#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
PSC_HOST="${PSC_HOST:-jluo7@bridges2.psc.edu}"
SSH_OPTS="${SSH_OPTS:--S /home/jiaxuanluo/.ssh/sockets/jluo7@bridges2.psc.edu:22 -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=2}"
PSC_JOB_IDS="${PSC_JOB_IDS:-40974420}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/documents/code/simuleval/manifests/2026/05/20260525T0122__analysis__glossary_bank_ablation_zh_fixedraw.json}"
REPOSTEVAL="${REPOSTEVAL:-${ROOT_DIR}/documents/code/simuleval/src/reposteval_psc_medicine_gs_fixedraw_20260525.py}"
BUILD="${BUILD:-${ROOT_DIR}/documents/code/simuleval/src/build_glossary_bank_ablation_20260525.py}"

notify() {
  local msg="$1"
  if [[ -x "${HOME}/bin/codex-notify" ]]; then
    "${HOME}/bin/codex-notify" --delay 0 --detach --workspace "${ROOT_DIR}" "${msg}" || true
  fi
}

ssh_remote() {
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${PSC_HOST}" "$@"
}

job_snapshot() {
  ssh_remote "date -u; squeue -j ${PSC_JOB_IDS} -o '%i|%j|%T|%M|%l|%R' --noheader || true; sacct -j ${PSC_JOB_IDS} --format=JobIDRaw,JobName,State,Elapsed,Timelimit,ExitCode,NodeList -P -n || true"
}

terminal_failed() {
  local snapshot="$1"
  grep -E '(^|[|])(FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY)([|]|$)' <<<"${snapshot}" >/dev/null
}

mark_manifest_status() {
  local status="$1"
  local snapshot_file="$2"
  python - "${MANIFEST}" "${status}" "${snapshot_file}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
status = sys.argv[2]
snapshot_file = sys.argv[3]
data = json.loads(path.read_text(encoding="utf-8"))
data["status"] = status
meta = data.setdefault("metadata", {})
meta["finalizer_last_update_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
meta["finalizer_last_snapshot"] = snapshot_file
if status == "complete":
    meta["rows_collected"] = 24
    meta["missing_rows"] = []
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  python "${ROOT_DIR}/documents/code/general/experiment_event.py" register "${MANIFEST}" || true
}

main() {
  cd "${ROOT_DIR}"
  local start now elapsed snapshot snapshot_dir snapshot_file
  start="$(date +%s)"
  snapshot_dir="/mnt/gemini/data1/jiaxuanluo/logs/glossary_bank_ablation_finalizer_20260525T0122"
  mkdir -p "${snapshot_dir}"
  notify "Glossary-bank finalizer started: waiting for PSC job(s) ${PSC_JOB_IDS}"

  while true; do
    now="$(date +%s)"
    elapsed=$((now - start))
    snapshot_file="${snapshot_dir}/snapshot_$(date -u +%Y%m%dT%H%M%SZ).txt"
    snapshot="$(job_snapshot || true)"
    printf '%s\n' "${snapshot}" | tee "${snapshot_file}"

    if python "${REPOSTEVAL}" && python "${BUILD}"; then
      mark_manifest_status "complete" "${snapshot_file}"
      notify "Glossary-bank ablation complete: 24/24 rows, figure rebuilt at latex/figures/glossary_bank_ablation_zh_fixedraw.pdf"
      exit 0
    fi

    if terminal_failed "${snapshot}"; then
      mark_manifest_status "failed_pending_row" "${snapshot_file}"
      notify "Glossary-bank ablation finalizer stopped: PSC job failed; snapshot=${snapshot_file}"
      exit 1
    fi

    if (( elapsed >= MAX_WAIT_SECONDS )); then
      mark_manifest_status "finalizer_timeout" "${snapshot_file}"
      notify "Glossary-bank ablation finalizer timed out after ${elapsed}s; snapshot=${snapshot_file}"
      exit 2
    fi

    sleep "${INTERVAL_SECONDS}"
  done
}

main "$@"
