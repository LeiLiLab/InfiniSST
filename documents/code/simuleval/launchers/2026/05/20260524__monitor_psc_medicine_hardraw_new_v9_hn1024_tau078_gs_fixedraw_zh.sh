#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
MANIFEST="${MANIFEST:-${ROOT_DIR}/documents/code/simuleval/manifests/2026/05/20260524T1226__simuleval__psc_medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw_zh.json}"
PSC_HOST="${PSC_HOST:-jluo7@bridges2.psc.edu}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ControlPath=/home/jiaxuanluo/.ssh/sockets/jluo7@bridges2.psc.edu:22}"
PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
RUN_STAMP_BASE="${RUN_STAMP_BASE:-20260524T1226_psc_med_newv9_hn1024_tau078_gs1k_gs10k_fixedraw_zh}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-1800}"
STATE_DIR="${STATE_DIR:-/mnt/gemini/data1/jiaxuanluo/logs/psc_medicine_new_v9_gs_fixedraw_monitor_20260524T1226_state}"
NOTIFY_BIN="${NOTIFY_BIN:-${HOME}/bin/codex-notify}"
SEND_START_NOTIFICATION="${SEND_START_NOTIFICATION:-1}"

REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${PSC_BASE}/logs/medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP_BASE}}"
REMOTE_OUT_ROOT="${REMOTE_OUT_ROOT:-${PSC_BASE}/outputs/medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP_BASE}}"

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
jobs = metadata.get("psc_full_jobs") or []
ids = [str(item.get("job_id", "")) for item in jobs if item.get("job_id")]
print("|".join([
    data.get("status", "unknown"),
    ",".join(ids),
    str(len(ids)),
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
date -u +%Y-%m-%dT%H:%M:%SZ

squeue_lines=""
sacct_lines=""
if [[ -n "${job_ids}" ]]; then
  squeue_lines="$(squeue -h -j "${job_ids}" -o "%i|%j|%T|%R" 2>/dev/null || true)"
  sacct_lines="$(sacct -n -P -j "${job_ids}" -X --format=JobID,JobName,State,NodeList,ExitCode 2>/dev/null || true)"
fi

echo "__JOBS__"
if [[ -n "${squeue_lines}" ]]; then
  printf '%s\n' "${squeue_lines}"
fi
if [[ -n "${sacct_lines}" ]]; then
  printf '%s\n' "${sacct_lines}"
fi
if [[ -z "${squeue_lines}${sacct_lines}" ]]; then
  echo "no_job_state"
fi

echo "__SUMMARY__"
eval_count=0
instances_count=0
strip_count=0
if [[ -d "${out_root}" ]]; then
  eval_count="$(find "${out_root}" -type f -name 'eval_results.tsv' -size +0c 2>/dev/null | wc -l | tr -d ' ')"
  instances_count="$(find "${out_root}" -type f -name 'instances.log' -size +0c 2>/dev/null | wc -l | tr -d ' ')"
  strip_count="$(find "${out_root}" -type f -name 'instances.strip_term.log' -size +0c 2>/dev/null | wc -l | tr -d ' ')"
fi
active_count=0
if [[ -n "${squeue_lines}" ]]; then
  active_count="$(printf '%s\n' "${squeue_lines}" | awk 'NF {c++} END {print c+0}')"
fi
terminal_count=0
failed_count=0
if [[ -n "${sacct_lines}" ]]; then
  terminal_count="$(printf '%s\n' "${sacct_lines}" | awk -F'|' '$3 ~ /^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)/ {c++} END {print c+0}')"
  failed_count="$(printf '%s\n' "${sacct_lines}" | awk -F'|' '$3 ~ /^(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)/ {c++} END {print c+0}')"
fi
printf 'eval_results=%s\ninstances=%s\nstrip_instances=%s\nactive_jobs=%s\nterminal_jobs=%s\nfailed_jobs=%s\n' \
  "${eval_count}" "${instances_count}" "${strip_count}" "${active_count}" "${terminal_count}" "${failed_count}"

echo "__FILES__"
roots=()
if [[ -d "${log_root}" ]]; then roots+=("${log_root}"); fi
if [[ -d "${out_root}" ]]; then roots+=("${out_root}"); fi
if (( ${#roots[@]} > 0 )); then
  find "${roots[@]}" -type f -size +0c -printf '%T@|%s|%p\n' 2>/dev/null | sort | tail -80
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

section_oneline() {
  local section="$1"
  local snapshot="$2"
  awk -v target="${section}" '
    $0 == target {flag=1; next}
    /^__/ && flag {flag=0}
    flag && NF {print}
  ' "${snapshot}" | tail -10 | tr '\n' '; ' | cut -c1-700
}

summary_value() {
  local key="$1"
  local snapshot="$2"
  awk -F= -v key="${key}" '
    $0 == "__SUMMARY__" {flag=1; next}
    /^__/ && flag {flag=0}
    flag && $1 == key {print $2; exit}
  ' "${snapshot}"
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
  ' "${snapshot}" | tail -10 | tr '\n' ' ' | cut -c1-700
}

if [[ "${SEND_START_NOTIFICATION}" == "1" && ! -f "${STATE_DIR}/started" ]]; then
  notify "PSC medicine new_v9 gs fixedraw monitor started: interval=${MONITOR_INTERVAL_SECONDS}s manifest=${MANIFEST}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "${STATE_DIR}/started"
fi

while true; do
  IFS='|' read -r status full_csv expected_jobs < <(manifest_fields)
  snapshot="${STATE_DIR}/snapshot_$(date -u +%Y%m%dT%H%M%SZ).txt"
  if ! collect_snapshot "${full_csv}" > "${snapshot}.tmp" 2> "${snapshot}.ssh_err"; then
    {
      echo "__REMOTE_DATE__"
      date -u +%Y-%m-%dT%H:%M:%SZ
      echo "__JOBS__"
      echo "ssh_failed"
      echo "__SUMMARY__"
      echo "eval_results=0"
      echo "instances=0"
      echo "strip_instances=0"
      echo "active_jobs=0"
      echo "terminal_jobs=0"
      echo "failed_jobs=0"
      echo "__FILES__"
      cat "${snapshot}.ssh_err"
    } > "${snapshot}.tmp"
  fi
  mv "${snapshot}.tmp" "${snapshot}"
  ln -sfn "${snapshot}" "${STATE_DIR}/latest_snapshot.txt"

  jobs_digest="$(section_digest "__JOBS__" "${snapshot}")"
  files_digest="$(section_digest "__FILES__" "${snapshot}")"
  last_jobs_digest="$(cat "${STATE_DIR}/last_jobs.digest" 2>/dev/null || true)"
  last_files_digest="$(cat "${STATE_DIR}/last_files.digest" 2>/dev/null || true)"
  changed=0
  if [[ "${jobs_digest}" != "${last_jobs_digest}" || "${files_digest}" != "${last_files_digest}" ]]; then
    changed=1
  fi
  printf '%s\n' "${jobs_digest}" > "${STATE_DIR}/last_jobs.digest"
  printf '%s\n' "${files_digest}" > "${STATE_DIR}/last_files.digest"

  eval_count="$(summary_value eval_results "${snapshot}")"
  instances_count="$(summary_value instances "${snapshot}")"
  strip_count="$(summary_value strip_instances "${snapshot}")"
  active_count="$(summary_value active_jobs "${snapshot}")"
  terminal_count="$(summary_value terminal_jobs "${snapshot}")"
  failed_count="$(summary_value failed_jobs "${snapshot}")"
  jobs_line="$(section_oneline "__JOBS__" "${snapshot}")"
  files_line="$(latest_files_oneline "${snapshot}")"

  notify "PSC medicine new_v9 heartbeat: status=${status}, eval=${eval_count:-0}/${expected_jobs}, instances=${instances_count:-0}, active=${active_count:-0}, terminal=${terminal_count:-0}, failed=${failed_count:-0}, changed=${changed}; jobs: ${jobs_line:-none}; files: ${files_line:-none}"

  if [[ "${failed_count:-0}" != "0" && ! -f "${STATE_DIR}/failure_notified" ]]; then
    notify "PSC medicine new_v9 failure detected: failed_jobs=${failed_count}; snapshot=${snapshot}"
    date -u +%Y-%m-%dT%H:%M:%SZ > "${STATE_DIR}/failure_notified"
  fi

  if [[ "${eval_count:-0}" -ge "${expected_jobs:-999}" && "${expected_jobs:-0}" -gt 0 ]]; then
    notify "PSC medicine new_v9 all evals complete: eval=${eval_count}/${expected_jobs}; output=${REMOTE_OUT_ROOT}; snapshot=${snapshot}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) complete eval=${eval_count}/${expected_jobs}"
    exit 0
  fi

  if [[ "${terminal_count:-0}" -ge "${expected_jobs:-999}" && "${expected_jobs:-0}" -gt 0 ]]; then
    notify "PSC medicine new_v9 jobs terminal but outputs incomplete: eval=${eval_count:-0}/${expected_jobs}, failed=${failed_count:-0}; output=${REMOTE_OUT_ROOT}; snapshot=${snapshot}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) terminal_incomplete eval=${eval_count:-0}/${expected_jobs}"
    exit 2
  fi

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) status=${status} eval=${eval_count:-0}/${expected_jobs} active=${active_count:-0} terminal=${terminal_count:-0} failed=${failed_count:-0}"
  sleep "${MONITOR_INTERVAL_SECONDS}"
done
