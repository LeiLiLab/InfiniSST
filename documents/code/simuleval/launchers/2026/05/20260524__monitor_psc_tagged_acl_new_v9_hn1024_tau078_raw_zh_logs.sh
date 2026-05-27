#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/documents/code/simuleval/manifests/2026/05/20260524T0400__simuleval__psc_tagged_acl_new_v9_hn1024_tau078_raw_zh.json}"
PSC_HOST="${PSC_HOST:-jluo7@bridges2.psc.edu}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ControlPath=/home/jiaxuanluo/.ssh/sockets/jluo7@bridges2.psc.edu:22}"
PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
RUN_STAMP_BASE="${RUN_STAMP_BASE:-20260524T0400_psc_tagacl_newv9_hn1024_tau078_raw_zh}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-1800}"
STATE_DIR="${STATE_DIR:-/mnt/gemini/data1/jiaxuanluo/logs/psc_tagacl_new_v9_monitor_20260524T0400_state}"
NOTIFY_BIN="${NOTIFY_BIN:-${HOME}/bin/codex-notify}"
SEND_START_NOTIFICATION="${SEND_START_NOTIFICATION:-1}"

REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${PSC_BASE}/logs/tagged_acl_new_v9_hn1024_tau078_raw/${RUN_STAMP_BASE}}"
REMOTE_OUT_ROOT="${REMOTE_OUT_ROOT:-${PSC_BASE}/outputs/tagged_acl_new_v9_hn1024_tau078_raw/${RUN_STAMP_BASE}}"

mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/monitor.lock"
if ! flock -n 9; then
  echo "[INFO] another monitor already holds ${STATE_DIR}/monitor.lock"
  exit 0
fi

notify() {
  local msg="$1"
  if [[ -x "${NOTIFY_BIN}" ]]; then
    "${NOTIFY_BIN}" --workspace "${ROOT_DIR}" "${msg}" || true
  else
    echo "[WARN] notify helper not executable: ${NOTIFY_BIN}" >&2
  fi
}

manifest_fields() {
  python - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
metadata = data.get("metadata", {})
full_jobs = metadata.get("psc_full_jobs") or []
full_ids = [str(item.get("job_id", "")) for item in full_jobs if item.get("job_id")]
print("|".join([
    data.get("status", "unknown"),
    str(metadata.get("psc_model_pull_job_id", "")),
    str(metadata.get("psc_smoke_job_id", "")),
    ",".join(full_ids),
    str(len(full_ids)),
]))
PY
}

collect_snapshot() {
  local job_ids="$1"
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${PSC_HOST}" "bash -s" -- "${job_ids}" "${REMOTE_LOG_ROOT}" "${REMOTE_OUT_ROOT}" <<'REMOTE'
set -euo pipefail
job_ids="$1"
log_root="$2"
out_root="$3"
echo "__REMOTE_DATE__"
date
echo "__JOBS__"
if [[ -n "${job_ids}" ]]; then
  squeue -h -j "${job_ids}" -o "%i|%j|%T|%R" 2>/dev/null || true
  sacct -n -P -j "${job_ids}" -X --format=JobID,JobName,State,NodeList,ExitCode 2>/dev/null || true
else
  echo "no_job_ids"
fi
echo "__FILES__"
roots=()
if [[ -d "${log_root}" ]]; then
  roots+=("${log_root}")
fi
if [[ -d "${out_root}" ]]; then
  roots+=("${out_root}")
fi
if (( ${#roots[@]} > 0 )); then
  find "${roots[@]}" \
    \( -path "${log_root}/model_pull" -o -path "${log_root}/model_pull/*" \) -prune -o \
    -type f -size +0c -printf '%T@|%s|%p\n' 2>/dev/null | sort
fi
REMOTE
}

section_digest() {
  local section="$1"
  local snapshot="$2"
  awk -v target="${section}" '
    $0 == target {flag=1; next}
    /^__/ && flag {flag=0}
    flag {print}
  ' "${snapshot}" | sha256sum | awk '{print $1}'
}

section_count() {
  local section="$1"
  local snapshot="$2"
  awk -v target="${section}" '
    $0 == target {flag=1; next}
    /^__/ && flag {flag=0}
    flag && NF {count++}
    END {print count + 0}
  ' "${snapshot}"
}

section_oneline() {
  local section="$1"
  local snapshot="$2"
  awk -v target="${section}" '
    $0 == target {flag=1; next}
    /^__/ && flag {flag=0}
    flag && NF {print}
  ' "${snapshot}" | tail -8 | tr '\n' '; ' | cut -c1-500
}

latest_files_oneline() {
  local snapshot="$1"
  awk '
    $0 == "__FILES__" {flag=1; next}
    /^__/ && flag {flag=0}
    flag && NF {
      split($0, parts, "|")
      path=parts[3]
      n=split(path, segs, "/")
      print segs[n] "(" parts[2] "B)"
    }
  ' "${snapshot}" | tail -8 | tr '\n' ' ' | cut -c1-500
}

if [[ "${SEND_START_NOTIFICATION}" == "1" && ! -f "${STATE_DIR}/started" ]]; then
  notify "PSC tagACL new_v9 log monitor started: interval=${MONITOR_INTERVAL_SECONDS}s manifest=${MANIFEST}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${STATE_DIR}/started"
fi

while true; do
  IFS='|' read -r status pull_job smoke_job full_csv full_count < <(manifest_fields)
  job_ids=""
  for id in "${pull_job}" "${smoke_job}"; do
    if [[ -n "${id}" ]]; then
      job_ids="${job_ids:+${job_ids},}${id}"
    fi
  done
  if [[ -n "${full_csv}" ]]; then
    job_ids="${job_ids:+${job_ids},}${full_csv}"
  fi

  snapshot="${STATE_DIR}/snapshot_$(date -u +%Y%m%dT%H%M%SZ).txt"
  if ! collect_snapshot "${job_ids}" > "${snapshot}.tmp" 2> "${snapshot}.ssh_err"; then
    {
      echo "__REMOTE_DATE__"
      date -u +%Y-%m-%dT%H:%M:%SZ
      echo "__JOBS__"
      echo "ssh_failed"
      echo "__FILES__"
    } > "${snapshot}.tmp"
    cat "${snapshot}.ssh_err" >> "${snapshot}.tmp"
  fi
  mv "${snapshot}.tmp" "${snapshot}"
  ln -sfn "${snapshot}" "${STATE_DIR}/latest_snapshot.txt"

  jobs_digest="$(section_digest "__JOBS__" "${snapshot}")"
  files_digest="$(section_digest "__FILES__" "${snapshot}")"
  jobs_count="$(section_count "__JOBS__" "${snapshot}")"
  files_count="$(section_count "__FILES__" "${snapshot}")"

  last_jobs_digest="$(cat "${STATE_DIR}/last_jobs.digest" 2>/dev/null || true)"
  last_files_digest="$(cat "${STATE_DIR}/last_files.digest" 2>/dev/null || true)"

  if [[ "${jobs_digest}" != "${last_jobs_digest}" || "${files_digest}" != "${last_files_digest}" ]]; then
    jobs_line="$(section_oneline "__JOBS__" "${snapshot}")"
    files_line="$(latest_files_oneline "${snapshot}")"
    notify "PSC tagACL new_v9 monitor: status=${status} pull=${pull_job:-none} smoke=${smoke_job:-none} full_jobs=${full_count}; jobs=${jobs_count}; log_files=${files_count}; jobs: ${jobs_line:-none}; files: ${files_line:-none}"
    printf '%s\n' "${jobs_digest}" > "${STATE_DIR}/last_jobs.digest"
    printf '%s\n' "${files_digest}" > "${STATE_DIR}/last_files.digest"
  fi

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status} pull=${pull_job:-none} smoke=${smoke_job:-none} full_jobs=${full_count} jobs=${jobs_count} log_files=${files_count}"
  sleep "${MONITOR_INTERVAL_SECONDS}"
done
