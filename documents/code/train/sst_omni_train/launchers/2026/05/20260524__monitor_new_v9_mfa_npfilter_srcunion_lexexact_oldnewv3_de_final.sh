#!/usr/bin/env bash
# Detached monitor for de New V9 SFT data final JSONL.
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_20260524}"
CONTINUE_PID="${CONTINUE_PID_OVERRIDE:-3677627}"
SLEEP_SECONDS="${SLEEP_SECONDS_OVERRIDE:-60}"

FINAL_TRAIN="${OUT_DIR}/train_s_de_new_v9_mfa_openai_rewrite_oldnewv3.jsonl"
FINAL_DEV="${OUT_DIR}/dev_s_de_new_v9_mfa_openai_rewrite_oldnewv3_first355.jsonl"
SUMMARY_JSON="${OUT_DIR}/new_v9_mfa_openai_rewrite_oldnewv3_de_summary.json"
TAG_STATS="${OUT_DIR}/stage4_assistant_termtag_stats.json"
TAG_SAMPLES="${OUT_DIR}/stage4_assistant_termtag_samples.json"
MANIFEST="${ROOT_DIR}/documents/code/train/sst_omni_train/manifests/2026/05/20260524T1530__data_prepare__new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de.json"
NOTES="${ROOT_DIR}/documents/code/train/sst_omni_train/notes/2026/05/20260524__new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_data.md"
REGISTER="${ROOT_DIR}/documents/code/general/experiment_event.py"
NOTIFY="${HOME}/bin/codex-notify"

cd "${ROOT_DIR}"

notify() {
  local msg="$1"
  if [[ -x "${NOTIFY}" ]]; then
    "${NOTIFY}" --delay 8 --detach --workspace "${ROOT_DIR}" "${msg}" || true
  else
    echo "[WARN] notifier not executable: ${NOTIFY}" >&2
  fi
}

update_manifest() {
  local status="$1"
  local reason="$2"
  python3 - "${MANIFEST}" "${status}" "${reason}" "${FINAL_TRAIN}" "${FINAL_DEV}" "${SUMMARY_JSON}" "${TAG_STATS}" "${TAG_SAMPLES}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest, status, reason, final_train, final_dev, summary_json, tag_stats, tag_samples = sys.argv[1:]
p = Path(manifest)
obj = json.loads(p.read_text(encoding="utf-8"))
obj["status"] = status
meta = obj.setdefault("metadata", {})
meta["monitor_status"] = status
meta["monitor_reason"] = reason
meta["monitor_updated_at_utc"] = datetime.now(timezone.utc).isoformat()

def add_artifact(role, typ, direction, path):
    artifacts = obj.setdefault("artifacts", [])
    if any(a.get("path") == path for a in artifacts):
        return
    artifacts.append({"role": role, "type": typ, "direction": direction, "path": path})

if status == "success":
    add_artifact("final_train_jsonl", "dataset", "output", final_train)
    add_artifact("final_dev_jsonl", "dataset", "output", final_dev)
    add_artifact("summary_json", "report", "output", summary_json)
    add_artifact("tag_stats", "report", "output", tag_stats)
    add_artifact("tag_samples", "report", "output", tag_samples)

p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
PY
  python3 "${REGISTER}" register "${MANIFEST}" || true
}

while true; do
  if [[ -s "${FINAL_TRAIN}" && -s "${FINAL_DEV}" && -s "${SUMMARY_JSON}" && -s "${TAG_STATS}" ]]; then
    rows="$(wc -l < "${FINAL_TRAIN}")"
    dev_rows="$(wc -l < "${FINAL_DEV}")"
    if (( rows > 0 && dev_rows > 0 )); then
      msg="Codex finished: de New V9 final JSONL ready (${rows} train rows). ${FINAL_TRAIN}"
      echo "[OK] ${msg}"
      update_manifest "success" "final JSONL, dev JSONL, summary, and tag stats exist"
      {
        echo
        echo "## Monitor completion"
        echo
        echo "- Final train JSONL: \`${FINAL_TRAIN}\`"
        echo "- Train rows: \`${rows}\`"
        echo "- Final dev JSONL: \`${FINAL_DEV}\`"
        echo "- Dev rows: \`${dev_rows}\`"
        echo "- Summary: \`${SUMMARY_JSON}\`"
      } >> "${NOTES}"
      notify "${msg}"
      exit 0
    fi
  fi

  if [[ -n "${CONTINUE_PID}" ]] && ! kill -0 "${CONTINUE_PID}" 2>/dev/null; then
    msg="Codex alert: de New V9 data-prep process exited but final JSONL is missing. Check ${OUT_DIR}"
    echo "[ERROR] ${msg}" >&2
    update_manifest "failed" "continuation process exited before final JSONL appeared"
    notify "${msg}"
    exit 2
  fi

  sleep "${SLEEP_SECONDS}"
done
