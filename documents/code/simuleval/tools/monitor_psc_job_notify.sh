#!/usr/bin/env bash
set -euo pipefail

# Poll one PSC Slurm job and send codex-notify updates on meaningful events.
# This is intended to run detached from a local workstation/login node.

JOB_ID="${JOB_ID:?set JOB_ID}"
PSC_HOST="${PSC_HOST:-jluo7@bridges2.psc.edu}"
CONTROL_PATH="${CONTROL_PATH:-${HOME}/.ssh/sockets/jluo7@bridges2.psc.edu:22}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:?set REMOTE_LOG_DIR}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
MAX_SECONDS="${MAX_SECONDS:-43200}"
STATE_DIR="${STATE_DIR:-/mnt/gemini/data1/jiaxuanluo/logs/psc_job_monitors/${JOB_ID}}"
WORKSPACE="${WORKSPACE:-/home/jiaxuanluo/InfiniSST}"
NOTIFY_BIN="${NOTIFY_BIN:-${HOME}/bin/codex-notify}"

mkdir -p "${STATE_DIR}"

notify() {
  local message="$1"
  if [[ -x "${NOTIFY_BIN}" ]]; then
    "${NOTIFY_BIN}" --workspace "${WORKSPACE}" "${message}" || true
  fi
}

remote_snapshot() {
  ssh \
    -o BatchMode=yes \
    -o ConnectTimeout=20 \
    -o ControlPath="${CONTROL_PATH}" \
    "${PSC_HOST}" \
    JOB_ID="${JOB_ID}" REMOTE_LOG_DIR="${REMOTE_LOG_DIR}" 'bash -s' <<'REMOTE'
set -euo pipefail

echo "UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SQUEUE_BEGIN"
squeue -h -j "${JOB_ID}" -o "%i|%T|%M|%l|%S|%D|%R" || true
echo "SQUEUE_END"
echo "SACCT_BEGIN"
sacct -n -P -j "${JOB_ID}" --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End 2>/dev/null || true
echo "SACCT_END"
echo "LOGS_BEGIN"
if [[ -d "${REMOTE_LOG_DIR}" ]]; then
  find "${REMOTE_LOG_DIR}" -maxdepth 1 -type f -name "*${JOB_ID}*" \
    -printf "%f|%s|%TY-%Tm-%TdT%TH:%TM:%TS\n" | sort || true
fi
echo "LOGS_END"
echo "TAIL_BEGIN"
if [[ -d "${REMOTE_LOG_DIR}" ]]; then
  for f in "${REMOTE_LOG_DIR}"/*"${JOB_ID}"*; do
    [[ -f "${f}" ]] || continue
    echo "--- ${f} ---"
    tail -n 40 "${f}" || true
  done
fi
echo "TAIL_END"
REMOTE
}

extract_main_state() {
  awk -F'|' -v job="${JOB_ID}" '
    /^SACCT_BEGIN$/ {in_sacct=1; next}
    /^SACCT_END$/ {in_sacct=0}
    in_sacct && $1 == job {sacct_state=$3}
    /^SQUEUE_BEGIN$/ {in_squeue=1; next}
    /^SQUEUE_END$/ {in_squeue=0}
    in_squeue && $1 == job {squeue_state=$2}
    END {
      if (sacct_state != "") print sacct_state;
      else if (squeue_state != "") print squeue_state;
      else print "UNKNOWN";
    }
  ' "$1"
}

has_nonempty_logs() {
  awk -F'|' '
    in_logs && NF >= 2 && $2 + 0 > 0 {found=1}
    /^LOGS_BEGIN$/ {in_logs=1; next}
    /^LOGS_END$/ {in_logs=0}
    END {exit found ? 0 : 1}
  ' "$1"
}

tail_summary() {
  awk '
    /^TAIL_BEGIN$/ {in_tail=1; next}
    /^TAIL_END$/ {in_tail=0}
    in_tail {print}
  ' "$1" | tail -n 20
}

start_epoch="$(date +%s)"
last_state=""
notified_logs=0
notified_done=0
notified_running=0
notify "PSC monitor started: job ${JOB_ID}; polling every ${INTERVAL_SECONDS}s; logs ${REMOTE_LOG_DIR}"

while true; do
  now_epoch="$(date +%s)"
  if (( now_epoch - start_epoch > MAX_SECONDS )); then
    notify "PSC monitor timeout: job ${JOB_ID}; no terminal state after ${MAX_SECONDS}s"
    exit 0
  fi

  snapshot="${STATE_DIR}/snapshot_$(date -u +%Y%m%dT%H%M%SZ).txt"
  if ! remote_snapshot >"${snapshot}" 2>"${snapshot}.ssh_err"; then
    notify "PSC monitor SSH check failed: job ${JOB_ID}; see ${snapshot}.ssh_err"
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  state="$(extract_main_state "${snapshot}")"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) state=${state}" >>"${STATE_DIR}/monitor.log"

  if [[ "${state}" != "${last_state}" ]]; then
    notify "PSC job ${JOB_ID} state: ${last_state:-START} -> ${state}"
    last_state="${state}"
  fi

  if [[ "${state}" == "RUNNING" && "${notified_running}" == "0" ]]; then
    notify "PSC job ${JOB_ID} is running. Logs: ${REMOTE_LOG_DIR}"
    notified_running=1
  fi

  if has_nonempty_logs "${snapshot}" && [[ "${notified_logs}" == "0" ]]; then
    summary="$(tail_summary "${snapshot}")"
    notify "PSC job ${JOB_ID} has non-empty logs. Remote log dir: ${REMOTE_LOG_DIR}"$'\n'"${summary}"
    notified_logs=1
  fi

  case "${state}" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
      if [[ "${notified_done}" == "0" ]]; then
        summary="$(tail_summary "${snapshot}")"
        notify "PSC job ${JOB_ID} terminal state: ${state}. Logs: ${REMOTE_LOG_DIR}"$'\n'"${summary}"
        notified_done=1
      fi
      exit 0
      ;;
  esac

  sleep "${INTERVAL_SECONDS}"
done
