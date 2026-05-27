#!/usr/bin/env bash
set -euo pipefail

# Polls the ja Speech LLM training export every 15 minutes.  Once the HF
# checkpoint is complete, launches tagged ACL raw ja lm=1..4 evals, one lm per
# available two-GPU slot on taurus or aries.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP_OVERRIDE:-20260524T1843_tagacl_newv9_mfa_npfilter_ja_aftertrain}"
MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_r32a64_tp2_aries4/keep1.0_r32/v0-20260525-020815-hf}"
TRAIN_LOG="${TRAIN_LOG_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/train_speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_aries4_20260524T1806.out}"
TRAIN_PIDFILE="${TRAIN_PIDFILE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/train_speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_aries4_20260524T1806.pid}"
TRAIN_HOST="${TRAIN_HOST_OVERRIDE:-aries}"
TRAIN_WANDB_RUN="${TRAIN_WANDB_RUN_OVERRIDE:-332v0v6n}"

EVAL_LAUNCHER="${EVAL_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_ja_same_lm_batch.sh}"
MANIFEST="${MANIFEST_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/manifests/2026/05/20260524T1843__simuleval__tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_ja_after_train.json}"
OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_ja_lm1to4_${RUN_STAMP}}"
MODEL_LABEL="${MODEL_LABEL:-new_v9_mfa_npfilter_lexexact_oldnewv3_ja_r32a64_hn1024_tau078_same_lm_batch_v1}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_ja_lm1to4_${RUN_STAMP}}"
STATUS_DIR="${STATUS_DIR:-${LOG_ROOT}/status}"
SUMMARY_DIR="${SUMMARY_DIR:-${OUTPUT_BASE}/__summary__}"

POLL_SECS="${POLL_SECS_OVERRIDE:-900}"
HOSTS="${HOSTS_OVERRIDE:-taurus aries}"
LMS="${LMS_OVERRIDE:-1 2 3 4}"
MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-2048}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-25}"

mkdir -p "${LOG_ROOT}" "${STATUS_DIR}" "${SUMMARY_DIR}" "$(dirname "${MANIFEST}")"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_ROOT}/watcher.log"
}

notify() {
  local msg="$1"
  if command -v "${HOME}/bin/codex-notify" >/dev/null 2>&1 || [[ -x "${HOME}/bin/codex-notify" ]]; then
    "${HOME}/bin/codex-notify" --delay 8 --detach --workspace "${ROOT_DIR}" "${msg}" || true
  else
    log "notify helper missing: ${msg}"
  fi
}

run_host() {
  local host="$1"
  shift
  if [[ "${host}" == "taurus" || "${host}" == "$(hostname -s)" || "${host}" == "$(hostname)" ]]; then
    bash -lc "$*"
  else
    ssh "${host}" "$*"
  fi
}

hf_ready() {
  [[ -s "${MODEL_NAME}/config.json" ]] || return 1
  [[ -s "${MODEL_NAME}/generation_config.json" ]] || return 1
  [[ -s "${MODEL_NAME}/model.safetensors.index.json" ]] || return 1
  [[ -s "${MODEL_NAME}/tokenizer_config.json" ]] || return 1
  local shard_count
  shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || return 1
}

train_alive_or_done() {
  hf_ready && return 0
  local pid
  pid="$(cat "${TRAIN_PIDFILE}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 1
  if [[ "${TRAIN_HOST}" == "taurus" ]]; then
    kill -0 "${pid}" 2>/dev/null
  else
    ssh "${TRAIN_HOST}" "kill -0 ${pid} 2>/dev/null"
  fi
}

idle_pair_for_host() {
  local host="$1" csv pair
  csv="$(run_host "${host}" "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null || true)"
  [[ -n "${csv}" ]] || return 1
  pair="$(awk -F, -v max_mem="${MAX_IDLE_GPU_MEM_MB}" -v max_util="${MAX_IDLE_GPU_UTIL}" '
    {
      gsub(/[[:space:]]/, "", $1);
      gsub(/[[:space:]]/, "", $2);
      gsub(/[[:space:]]/, "", $3);
      if (($2 + 0) <= max_mem && ($3 + 0) <= max_util) print $1;
    }
  ' <<< "${csv}" | head -n 2 | paste -sd, -)"
  [[ "${pair}" == *,* ]] || return 1
  echo "${pair}"
}

remote_root_for_host() {
  local host="$1"
  if [[ "${host}" == "aries" ]]; then
    echo "/mnt/taurus/home/jiaxuanluo/InfiniSST"
  else
    echo "${ROOT_DIR}"
  fi
}

process_running() {
  local host="$1" pid="$2"
  [[ -n "${pid}" ]] || return 1
  run_host "${host}" "kill -0 ${pid} 2>/dev/null"
}

lm_is_done() {
  local lm="$1"
  [[ -s "${STATUS_DIR}/lm${lm}.done" ]]
}

lm_is_failed() {
  local lm="$1"
  [[ -s "${STATUS_DIR}/lm${lm}.failed" ]]
}

lm_is_submitted() {
  local lm="$1"
  [[ -s "${STATUS_DIR}/lm${lm}.submitted" ]]
}

launch_lm() {
  local lm="$1" host="$2" pair="$3"
  local remote_root remote_eval_launcher lm_log pid_file done_file failed_file submitted_file
  remote_root="$(remote_root_for_host "${host}")"
  remote_eval_launcher="${remote_root}/documents/code/simuleval/launchers/2026/05/20260524__tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_ja_same_lm_batch.sh"
  lm_log="${LOG_ROOT}/lm${lm}_${host}_gpu${pair//,/}"
  pid_file="${STATUS_DIR}/lm${lm}.pid"
  done_file="${STATUS_DIR}/lm${lm}.done"
  failed_file="${STATUS_DIR}/lm${lm}.failed"
  submitted_file="${STATUS_DIR}/lm${lm}.submitted"
  mkdir -p "${lm_log}"
  log "launch lm=${lm} host=${host} pair=${pair}"
  run_host "${host}" "mkdir -p '${lm_log}' '${STATUS_DIR}' && cd '${remote_root}' && setsid bash -lc 'ROOT_DIR_OVERRIDE=\"${remote_root}\" RUN_STAMP_OVERRIDE=\"${RUN_STAMP}\" LM_OVERRIDE=\"${lm}\" CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE=\"${pair}\" MODEL_NAME_OVERRIDE=\"${MODEL_NAME}\" OUT_ROOT=\"${OUT_ROOT}\" OUTPUT_BASE=\"${OUTPUT_BASE}\" LOG_ROOT=\"${lm_log}\" WANDB_COMPUTE_TAG_OVERRIDE=\"compute:${host}_gpu${pair//,/}\" bash \"${remote_eval_launcher}\" && echo ok > \"${done_file}\" || { rc=\$?; echo failed:\$rc > \"${failed_file}\"; exit \$rc; }' > '${lm_log}/watch_submit.out' 2> '${lm_log}/watch_submit.err' < /dev/null & echo \$! > '${pid_file}'"
  {
    echo "host=${host}"
    echo "gpu_pair=${pair}"
    echo "pid=$(cat "${pid_file}" 2>/dev/null || true)"
    echo "log=${lm_log}"
  } > "${submitted_file}"
}

refresh_submitted_states() {
  local lm host pid
  for lm in ${LMS}; do
    lm_is_done "${lm}" && continue
    lm_is_failed "${lm}" && continue
    lm_is_submitted "${lm}" || continue
    host="$(awk -F= '$1=="host"{print $2}' "${STATUS_DIR}/lm${lm}.submitted" 2>/dev/null || true)"
    pid="$(cat "${STATUS_DIR}/lm${lm}.pid" 2>/dev/null || true)"
    if [[ -n "${host}" && -n "${pid}" ]] && process_running "${host}" "${pid}"; then
      continue
    fi
    if ! lm_is_done "${lm}" && ! lm_is_failed "${lm}"; then
      echo "process exited without done marker; host=${host} pid=${pid}" > "${STATUS_DIR}/lm${lm}.failed"
      notify "Codex eval failed: ja tagged ACL lm${lm} process exited without done marker; see ${LOG_ROOT}"
    fi
  done
}

all_eval_done() {
  local lm
  for lm in ${LMS}; do
    lm_is_done "${lm}" || return 1
  done
}

any_eval_failed() {
  local lm
  for lm in ${LMS}; do
    lm_is_failed "${lm}" && return 0
  done
  return 1
}

write_summary() {
  python3 - "${OUTPUT_BASE}" "${SUMMARY_DIR}" ${LMS} <<'PY'
import csv
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
summary_dir = Path(sys.argv[2])
lms = sys.argv[3:]
summary_dir.mkdir(parents=True, exist_ok=True)
rows = []
for lm in lms:
    paths = sorted(output_base.glob(f"ja/**_lm{lm}_*/eval_results.tsv"))
    if len(paths) != 1:
        raise SystemExit(f"expected one eval_results.tsv for lm={lm}, found {len(paths)}")
    with paths[0].open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {paths[0]}, got {len(data)}")
    row = data[0]
    rows.append({
        "lang": "ja",
        "lm": lm,
        "glossary": "raw",
        "BLEU": row.get("BLEU", ""),
        "TERM_ACC": row.get("TERM_ACC", ""),
        "REAL_TERM_ADOPT": row.get("REAL_TERM_ADOPT", ""),
        "TERM_FCR": row.get("TERM_FCR", ""),
        "StreamLAAL": row.get("StreamLAAL", ""),
        "eval_results": str(paths[0]),
    })
fields = ["lang", "lm", "glossary", "BLEU", "TERM_ACC", "REAL_TERM_ADOPT", "TERM_FCR", "StreamLAAL", "eval_results"]
tsv = summary_dir / "summary_ja_raw_lm1to4_same_lm_batch_v1.tsv"
with tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

def pct(x):
    try:
        return f"{float(x) * 100:.2f}"
    except Exception:
        return str(x)

lines = [
    "# Tagged ACL ja raw: clean New V9 MFA/npfilter + HN1024 tau=0.78",
    "",
    "| lang | lm | glossary | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | StreamLAAL |",
    "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    lines.append(
        f"| {r['lang']} | {r['lm']} | {r['glossary']} | "
        f"{float(r['BLEU']):.2f} | {pct(r['TERM_ACC'])} | {pct(r['REAL_TERM_ADOPT'])} | "
        f"{pct(r['TERM_FCR'])} | {float(r['StreamLAAL']):.0f} |"
    )
md = summary_dir / "summary_ja_raw_lm1to4_same_lm_batch_v1.md"
lines += ["", f"Summary TSV: `{tsv}`", f"Output base: `{output_base}`"]
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md)
PY
}

update_manifest_status() {
  local status="$1"
  python3 - "${MANIFEST}" "${status}" "${OUTPUT_BASE}" "${SUMMARY_DIR}" <<'PY' || true
import json
import sys
from pathlib import Path
manifest = Path(sys.argv[1])
status = sys.argv[2]
output_base = sys.argv[3]
summary_dir = sys.argv[4]
data = json.loads(manifest.read_text(encoding="utf-8"))
data["status"] = status
meta = data.setdefault("metadata", {})
meta["watcher_status"] = status
meta["output_base"] = output_base
meta["summary_dir"] = summary_dir
manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  python3 "${ROOT_DIR}/documents/code/general/experiment_event.py" register "${MANIFEST}" >> "${LOG_ROOT}/watcher.log" 2>&1 || true
}

log "watcher started; model=${MODEL_NAME}; poll_secs=${POLL_SECS}"
while true; do
  if ! hf_ready; then
    shard_count="0"
    if [[ -d "${MODEL_NAME}" ]]; then
      shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
    fi
    log "training/export not ready yet; hf_shards=${shard_count}/15; next check in ${POLL_SECS}s"
    if ! train_alive_or_done; then
      log "training process is not alive and HF export is incomplete"
      update_manifest_status "failed_train_incomplete"
      notify "Codex failed: ja New V9 training process ended before complete HF export; see ${TRAIN_LOG}"
      exit 4
    fi
    sleep "${POLL_SECS}"
    continue
  fi

  if [[ ! -s "${STATUS_DIR}/train_done_notified" ]]; then
    log "HF export ready; launching evals when GPUs are available"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_DIR}/train_done_notified"
    update_manifest_status "eval_running"
    notify "Codex update: ja New V9 training/export finished; starting tagged ACL lm1-4 eval dispatch."
  fi

  refresh_submitted_states
  if any_eval_failed; then
    update_manifest_status "failed_eval"
    log "at least one lm failed; status dir=${STATUS_DIR}"
    exit 5
  fi
  if all_eval_done; then
    summary_path="$(write_summary | tail -n 1)"
    update_manifest_status "success"
    notify "Codex finished: ja tagged ACL lm1-4 eval complete. Summary: ${summary_path}"
    log "all evals complete; summary=${summary_path}"
    exit 0
  fi

  for lm in ${LMS}; do
    lm_is_done "${lm}" && continue
    lm_is_submitted "${lm}" && continue
    for host in ${HOSTS}; do
      pair="$(idle_pair_for_host "${host}" | tail -n 1 || true)"
      if [[ -n "${pair}" ]]; then
        launch_lm "${lm}" "${host}" "${pair}"
        sleep 20
        break
      fi
    done
  done

  refresh_submitted_states
  if all_eval_done; then
    continue
  fi
  log "eval dispatch loop sleeping ${POLL_SECS}s"
  sleep "${POLL_SECS}"
done
