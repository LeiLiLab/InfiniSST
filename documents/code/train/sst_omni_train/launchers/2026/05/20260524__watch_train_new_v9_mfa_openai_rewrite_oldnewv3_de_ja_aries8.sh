#!/usr/bin/env bash
# Wait for clean de/ja New V9 data, then start both 4-GPU SFT jobs on aries.
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs}"
TRAIN_LAUNCHER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260524__speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_de_ja_taurus4_r32a64_tp2.sh"
POLL_SEC="${POLL_SEC_OVERRIDE:-60}"
mkdir -p "${LOG_ROOT}"

validate_ready() {
  local lang="$1"
  local data_dir="${DATA_ROOT}/speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_${lang}_20260524"
  local train="${data_dir}/train_s_${lang}_new_v9_mfa_openai_rewrite_oldnewv3.jsonl"
  local dev="${data_dir}/dev_s_${lang}_new_v9_mfa_openai_rewrite_oldnewv3_first355.jsonl"
  local summary="${data_dir}/new_v9_mfa_openai_rewrite_oldnewv3_${lang}_summary.json"
  [[ -s "${train}" && -s "${dev}" && -s "${summary}" ]] || return 1
  python3 - "${summary}" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
s = json.loads(p.read_text(encoding="utf-8"))
v = s.get("final_validation") or {}
errors = []
if int(v.get("rows", 0)) <= 0:
    errors.append("rows<=0")
if int(v.get("gt_terms", 0)) <= 0:
    errors.append("gt_terms<=0")
if float(v.get("gt_in_term_map_rate", 0.0)) < 0.999:
    errors.append(f"gt_in_term_map_rate={v.get('gt_in_term_map_rate')}")
if float(v.get("no_gt_zero_rate", 0.0)) < 0.999:
    errors.append(f"no_gt_zero_rate={v.get('no_gt_zero_rate')}")
if int(v.get("malformed_tag_assistant_messages", 1)) != 0:
    errors.append(f"malformed_tag_assistant_messages={v.get('malformed_tag_assistant_messages')}")
if int(v.get("latin_word_cut_tag_messages", 1)) != 0:
    errors.append(f"latin_word_cut_tag_messages={v.get('latin_word_cut_tag_messages')}")
if errors:
    raise SystemExit("not_ready: " + ", ".join(errors))
print("ready", json.dumps(v, ensure_ascii=False, sort_keys=True))
PY
}

start_train() {
  local lang="$1"
  local gpus="$2"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%S)"
  local run_tag="speech_llm_new_v9_mfa_openai_${lang}_aries4_${stamp}"
  local out="${LOG_ROOT}/${run_tag}.out"
  local err="${LOG_ROOT}/${run_tag}.err"
  local pidfile="${LOG_ROOT}/${run_tag}.pid"
  local marker="${LOG_ROOT}/${run_tag}.started"

  if ls "${LOG_ROOT}"/speech_llm_new_v9_mfa_openai_"${lang}"_aries4_*.pid >/dev/null 2>&1; then
    for p in "${LOG_ROOT}"/speech_llm_new_v9_mfa_openai_"${lang}"_aries4_*.pid; do
      if [[ -s "${p}" ]] && kill -0 "$(cat "${p}")" 2>/dev/null; then
        echo "[INFO] ${lang}: existing train is live via ${p}; skip"
        return 0
      fi
    done
  fi

  setsid bash -lc "
    set -euo pipefail
    cd '${ROOT_DIR}'
    LANG_CODE_OVERRIDE='${lang}' \
    ROOT_DIR_OVERRIDE='${ROOT_DIR}' \
    HOST_GPU_DEVICES_OVERRIDE='${gpus}' \
    SAVE_BASE_OVERRIDE='${DATA_ROOT}/slm/speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_${lang}_r32a64_tp2_aries4' \
    TRAIN_LOG_DIR_OVERRIDE='${LOG_ROOT}/speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_${lang}_r32a64_tp2_aries4' \
    WANDB_EXP_PREFIX_OVERRIDE='speech-llm-new_v9-mfa-openai-oldnewv3-${lang}-r32a64-tp2-aries4' \
    COMPUTE_TAG_OVERRIDE='aries4' \
    bash '${TRAIN_LAUNCHER}'
  " > "${out}" 2> "${err}" < /dev/null &
  echo "$!" > "${pidfile}"
  echo "${stamp}" > "${marker}"
  echo "[STARTED] ${lang} gpus=${gpus} pid=$(cat "${pidfile}") out=${out} err=${err}"
}

echo "[INFO] watcher started at $(date -u --iso-8601=seconds)"
echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] data_root=${DATA_ROOT}"
echo "[INFO] train_launcher=${TRAIN_LAUNCHER}"

de_started=0
ja_started=0
while (( de_started == 0 || ja_started == 0 )); do
  if (( de_started == 0 )); then
    if validate_ready de; then
      start_train de "0,1,2,3"
      de_started=1
    else
      echo "[WAIT] de not ready at $(date -u --iso-8601=seconds)"
    fi
  fi
  if (( ja_started == 0 )); then
    if validate_ready ja; then
      start_train ja "4,5,6,7"
      ja_started=1
    else
      echo "[WAIT] ja not ready at $(date -u --iso-8601=seconds)"
    fi
  fi
  if (( de_started == 0 || ja_started == 0 )); then
    sleep "${POLL_SEC}"
  fi
done

echo "[OK] watcher launched all jobs at $(date -u --iso-8601=seconds)"
