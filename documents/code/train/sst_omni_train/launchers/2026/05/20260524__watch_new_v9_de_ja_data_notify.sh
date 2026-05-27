#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
REMOTE_PORT="${REMOTE_PORT_OVERRIDE:-20042}"
REMOTE_HOST="${REMOTE_HOST_OVERRIDE:-localhost}"
REMOTE_USER="${REMOTE_USER_OVERRIDE:-jiaxuanluo}"
REMOTE_PID="${REMOTE_PID_OVERRIDE:-473400}"
POLL_SEC="${POLL_SEC_OVERRIDE:-300}"
MAX_POLLS="${MAX_POLLS_OVERRIDE:-288}"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o UserKnownHostsFile=/dev/null
  -o StrictHostKeyChecking=no
  -p "${REMOTE_PORT}"
)

OUT_BASE="/mnt/gemini/data1/jiaxuanluo"
LOG_OUT="${OUT_BASE}/logs/20260524_new_v9_de_ja_continue_variant_no_gt_zero_termtag.out"
LOG_ERR="${OUT_BASE}/logs/20260524_new_v9_de_ja_continue_variant_no_gt_zero_termtag.err"

DE_DIR="${OUT_BASE}/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_de_20260524"
JA_DIR="${OUT_BASE}/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_ja_20260524"
DE_FINAL="${DE_DIR}/train_s_de_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3.jsonl"
JA_FINAL="${JA_DIR}/train_s_ja_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3.jsonl"
DE_SUMMARY="${DE_DIR}/new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_de_summary.json"
JA_SUMMARY="${JA_DIR}/new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_ja_summary.json"

notify() {
  local message="$1"
  if [[ -x "${HOME}/bin/codex-notify" ]]; then
    "${HOME}/bin/codex-notify" --workspace "${ROOT_DIR}" "${message}" || true
  else
    echo "[WARN] codex-notify not available: ${message}" >&2
  fi
}

remote_check() {
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "set -euo pipefail
     alive=0
     if ps -p '${REMOTE_PID}' >/dev/null 2>&1; then alive=1; fi
     ok=0
     if [[ -s '${DE_FINAL}' && -s '${JA_FINAL}' && -s '${DE_SUMMARY}' && -s '${JA_SUMMARY}' ]]; then ok=1; fi
     echo alive=\${alive}
     echo ok=\${ok}
     ls -lh '${DE_FINAL}' '${JA_FINAL}' '${DE_SUMMARY}' '${JA_SUMMARY}' 2>/dev/null || true
     echo '--- last stdout ---'
     tail -8 '${LOG_OUT}' 2>/dev/null || true
     echo '--- last stderr ---'
     tail -8 '${LOG_ERR}' 2>/dev/null || true"
}

echo "[WATCH] remote=${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT} pid=${REMOTE_PID} poll_sec=${POLL_SEC} max_polls=${MAX_POLLS}"
for poll in $(seq 1 "${MAX_POLLS}"); do
  echo "[WATCH] poll=${poll} time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status="$(remote_check || true)"
  printf '%s\n' "${status}"
  alive="$(printf '%s\n' "${status}" | awk -F= '$1=="alive"{print $2; exit}')"
  ok="$(printf '%s\n' "${status}" | awk -F= '$1=="ok"{print $2; exit}')"
  if [[ "${ok}" == "1" ]]; then
    notify "Codex finished: de/ja New V9 data completed. Outputs: ${DE_FINAL} ; ${JA_FINAL}"
    exit 0
  fi
  if [[ "${alive}" == "0" ]]; then
    notify "Codex alert: de/ja New V9 data job exited before final outputs were complete. Check ${LOG_OUT} and ${LOG_ERR}"
    exit 1
  fi
  sleep "${POLL_SEC}"
done

notify "Codex alert: de/ja New V9 data watcher timed out after ${MAX_POLLS} polls. Check ${LOG_OUT}"
exit 2
