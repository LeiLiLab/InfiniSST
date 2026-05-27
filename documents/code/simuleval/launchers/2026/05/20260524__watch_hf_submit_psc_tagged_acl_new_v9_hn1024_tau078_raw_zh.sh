#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
PSC_HOST="${PSC_HOST:-jluo7@bridges2.psc.edu}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ControlPath=/home/jiaxuanluo/.ssh/sockets/jluo7@bridges2.psc.edu:22}"
PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
PSC_ROOT="${PSC_ROOT:-${PSC_BASE}/src/InfiniSST}"
PSC_ENV="${PSC_ENV:-${PSC_BASE}/envs/spaCyEnv_20260518}"
PSC_LAUNCHER="${PSC_LAUNCHER:-${PSC_ROOT}/documents/code/simuleval/launchers/2026/05/20260524__psc_tagged_acl_new_v9_hn1024_tau078_raw_zh.sh}"
APPTAINER_SIF="${APPTAINER_SIF:-${PSC_BASE}/containers/ubuntu_22_04_gcc.sif}"

HF_REPO_ID="${HF_REPO_ID:-gavinlaw/infinisst-new-v9-termtag-delay-oldnewv3-r32a64-keep1p0-r32-zh}"
PSC_MODEL_ROOT="${PSC_MODEL_ROOT:-${PSC_BASE}/models/new_v9_termtag_delay_oldnewv3_r32a64/keep1.0_r32}"
PSC_MODEL_DIR="${PSC_MODEL_DIR:-${PSC_MODEL_ROOT}/v0-20260524-062743-hf}"
RUN_STAMP_BASE="${RUN_STAMP_BASE:-20260524T0400_psc_tagacl_newv9_hn1024_tau078_raw_zh}"
EVENT_MANIFEST="${EVENT_MANIFEST:-${ROOT_DIR}/documents/code/simuleval/manifests/2026/05/20260524T0400__simuleval__psc_tagged_acl_new_v9_hn1024_tau078_raw_zh.json}"
WATCH_SLEEP_SECONDS="${WATCH_SLEEP_SECONDS:-120}"
HF_READY_TIMEOUT_SECONDS="${HF_READY_TIMEOUT_SECONDS:-43200}"

REMOTE_OUTPUT_ROOT="${PSC_BASE}/outputs/tagged_acl_new_v9_hn1024_tau078_raw/${RUN_STAMP_BASE}"
REMOTE_LOG_ROOT="${PSC_BASE}/logs/tagged_acl_new_v9_hn1024_tau078_raw/${RUN_STAMP_BASE}"

ssh_remote() {
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${PSC_HOST}" "$@"
}

put_remote_file() {
  local path="$1"
  local dir
  dir="$(dirname "${path}")"
  # Do not let the mkdir SSH call consume stdin when this function is used in a pipeline.
  ssh ${SSH_OPTS} "${PSC_HOST}" "mkdir -p $(printf '%q' "${dir}")" < /dev/null
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${PSC_HOST}" "cat > $(printf '%q' "${path}")"
}

update_manifest() {
  local status="$1"
  local update_json="$2"
  MANIFEST_UPDATE_JSON="${update_json}" python - "${EVENT_MANIFEST}" "${status}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
status = sys.argv[2]
update = json.loads(os.environ["MANIFEST_UPDATE_JSON"])
data = json.loads(path.read_text(encoding="utf-8"))
data["status"] = status
metadata = data.setdefault("metadata", {})
metadata.update(update)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  python "${ROOT_DIR}/documents/code/general/experiment_event.py" register "${EVENT_MANIFEST}"
}

repo_ready() {
  python - "${HF_REPO_ID}" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
try:
    files = HfApi().list_repo_files(repo_id, repo_type="model")
except Exception:
    raise SystemExit(1)
shards = [f for f in files if f.startswith("model-") and f.endswith(".safetensors")]
if "config.json" not in files or len(shards) != 15:
    raise SystemExit(1)
print(f"ready files={len(files)} shards={len(shards)}")
PY
}

wait_for_repo_ready() {
  local start now
  start="$(date +%s)"
  until repo_ready; do
    now="$(date +%s)"
    if (( now - start >= HF_READY_TIMEOUT_SECONDS )); then
      echo "[ERROR] HF repo not ready after ${HF_READY_TIMEOUT_SECONDS}s: ${HF_REPO_ID}" >&2
      update_manifest "failed_hf_timeout" "{\"hf_repo_id\":\"${HF_REPO_ID}\",\"hf_ready_timeout_seconds\":${HF_READY_TIMEOUT_SECONDS}}"
      return 4
    fi
    echo "[WAIT] HF repo not ready yet: ${HF_REPO_ID}"
    sleep "${WATCH_SLEEP_SECONDS}"
  done
}

submit_pull_job() {
  local script="${REMOTE_LOG_ROOT}/model_pull/pull_new_v9_model.sbatch"
  cat <<REMOTE | put_remote_file "${script}"
#!/usr/bin/env bash
#SBATCH -A cis260009p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 01:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH -J tagacl_newv9_pull
#SBATCH -o ${REMOTE_LOG_ROOT}/model_pull/slurm_%j.out
#SBATCH -e ${REMOTE_LOG_ROOT}/model_pull/slurm_%j.err
set -euo pipefail
ENV_DIR=${PSC_ENV}
MODEL_DIR=${PSC_MODEL_DIR}
HF_REPO_ID=${HF_REPO_ID}
PSC_BASE=${PSC_BASE}
export PATH="\${ENV_DIR}/bin:\${PATH}"
export HF_HOME="\${HF_HOME:-\${PSC_BASE}/cache/hf}"
export TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE:-\${PSC_BASE}/cache/hf/transformers}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-\${PSC_BASE}/cache/hf/datasets}"
mkdir -p "\$(dirname "\${MODEL_DIR}")" "\${HF_HOME}" "\${TRANSFORMERS_CACHE}" "\${HF_DATASETS_CACHE}"
validate_model_dir() {
  [[ -f "\$1/config.json" ]] || return 1
  local shards
  shards="\$(find "\$1" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "\${shards}" == "15" ]]
}
if validate_model_dir "\${MODEL_DIR}"; then
  echo "[SKIP] model already valid: \${MODEL_DIR}"
  exit 0
fi
rm -rf "\${MODEL_DIR}.tmp"
mkdir -p "\${MODEL_DIR}.tmp"
hf download "\${HF_REPO_ID}" --repo-type model --local-dir "\${MODEL_DIR}.tmp"
validate_model_dir "\${MODEL_DIR}.tmp"
rm -rf "\${MODEL_DIR}"
mv "\${MODEL_DIR}.tmp" "\${MODEL_DIR}"
echo "[DONE] model ready: \${MODEL_DIR}"
REMOTE
  ssh_remote "sbatch --parsable $(printf '%q' "${script}")"
}

submit_eval_job() {
  local name="$1" mode="$2" lms="$3" time_limit="$4" dependency="${5:-}"
  local suffix="$6"
  local out_root="${REMOTE_OUTPUT_ROOT}/${suffix}"
  local log_root="${REMOTE_LOG_ROOT}/${suffix}"
  local script="${log_root}/${name}.sbatch"
  cat <<REMOTE | put_remote_file "${script}"
#!/usr/bin/env bash
#SBATCH -A cis260009p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:4
#SBATCH -t ${time_limit}
#SBATCH --cpus-per-gpu=4
#SBATCH -J ${name}
#SBATCH -o ${log_root}/slurm_%j.out
#SBATCH -e ${log_root}/slurm_%j.err
set -euo pipefail
cd ${PSC_ROOT}
env \\
  PSC_BASE=${PSC_BASE} \\
  ROOT_DIR=${PSC_ROOT} \\
  ENV_DIR=${PSC_ENV} \\
  USE_APPTAINER=1 \\
  APPTAINER_SIF=${APPTAINER_SIF} \\
  RUN_STAMP=${RUN_STAMP_BASE}_${suffix} \\
  MODE=${mode} \\
  LMS="${lms}" \\
  SMOKE_LM_OVERRIDE="${lms}" \\
  SMOKE_GLOSSARY_KIND_OVERRIDE=raw \\
  GPU_PAIR=0,1,2,3 \\
  MODEL_ROOT=${PSC_MODEL_ROOT} \\
  MODEL_DIR=${PSC_MODEL_DIR} \\
  OUT_ROOT=${out_root} \\
  LOG_ROOT=${log_root} \\
  VLLM_TP_SIZE_OVERRIDE=4 \\
  VLLM_MAX_MODEL_LEN_OVERRIDE=8192 \\
  GPU_MEMORY_UTILIZATION_OVERRIDE=0.70 \\
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \\
  VLLM_MOE_USE_DEEP_GEMM=0 \\
  VLLM_USE_FUSED_MOE_GROUPED_TOPK=0 \\
  RAG_GPU_OVERRIDE=cuda:3 \\
  bash ${PSC_LAUNCHER}
REMOTE
  if [[ -n "${dependency}" ]]; then
    ssh_remote "sbatch --parsable --dependency=$(printf '%q' "${dependency}") $(printf '%q' "${script}")"
  else
    ssh_remote "sbatch --parsable $(printf '%q' "${script}")"
  fi
}

wait_for_terminal() {
  local job_id="$1"
  local state=""
  while true; do
    state="$(ssh_remote "sacct -n -P -j $(printf '%q' "${job_id}") -X --format=State | head -1 | awk -F'|' '{print \$1}'" | tr -d ' ')"
    echo "[POLL] job=${job_id} state=${state:-unknown}"
    case "${state}" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
        printf '%s\n' "${state}"
        return 0
        ;;
    esac
    sleep "${WATCH_SLEEP_SECONDS}"
  done
}

check_smoke_outputs() {
  local out_root="${REMOTE_OUTPUT_ROOT}/smoke"
  ssh_remote "set -euo pipefail; evals=\$(find $(printf '%q' "${out_root}") -name eval_results.tsv -size +0 2>/dev/null | wc -l | tr -d ' '); inst=\$(find $(printf '%q' "${out_root}") -name instances.log -size +0 2>/dev/null | wc -l | tr -d ' '); strip=\$(find $(printf '%q' "${out_root}") -name instances.strip_term.log -size +0 2>/dev/null | wc -l | tr -d ' '); echo evals=\$evals inst=\$inst strip=\$strip; test \$evals -ge 1; test \$inst -ge 1; test \$strip -ge 1"
}

main() {
  echo "[INFO] watcher started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  update_manifest "waiting_hf_repo" "{\"hf_repo_id\":\"${HF_REPO_ID}\",\"psc_model_dir\":\"${PSC_MODEL_DIR}\",\"watcher_pid\":$$}"
  wait_for_repo_ready
  update_manifest "hf_ready_submitting_psc_pull" "{\"hf_repo_id\":\"${HF_REPO_ID}\",\"hf_ready_at_utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"

  local pull_job smoke_job smoke_state
  pull_job="$(submit_pull_job)"
  echo "[SUBMIT] pull_job=${pull_job}"
  update_manifest "psc_model_pull_submitted" "{\"psc_model_pull_job_id\":\"${pull_job}\",\"psc_model_dir\":\"${PSC_MODEL_DIR}\"}"

  smoke_job="$(submit_eval_job tagacl_newv9_smoke smoke "2" "01:30:00" "afterok:${pull_job}" smoke)"
  echo "[SUBMIT] smoke_job=${smoke_job}"
  update_manifest "psc_smoke_submitted" "{\"psc_model_pull_job_id\":\"${pull_job}\",\"psc_smoke_job_id\":\"${smoke_job}\",\"psc_smoke_mode\":\"smoke\",\"psc_smoke_lms\":[\"2\"]}"

  smoke_state="$(wait_for_terminal "${smoke_job}" | tail -1)"
  echo "[DONE] smoke_job=${smoke_job} state=${smoke_state}"
  if [[ "${smoke_state}" != "COMPLETED" ]]; then
    update_manifest "failed_psc_smoke" "{\"psc_smoke_job_id\":\"${smoke_job}\",\"psc_smoke_state\":\"${smoke_state}\"}"
    exit 5
  fi
  check_smoke_outputs
  update_manifest "psc_smoke_passed_submitting_full" "{\"psc_smoke_job_id\":\"${smoke_job}\",\"psc_smoke_state\":\"${smoke_state}\",\"psc_smoke_output_root\":\"${REMOTE_OUTPUT_ROOT}/smoke\"}"

  local jobs_json="[]"
  local lm job
  for lm in 1 2 3 4; do
    job="$(submit_eval_job "tagacl_newv9_lm${lm}" full "${lm}" "02:00:00" "" "lm${lm}")"
    echo "[SUBMIT] lm=${lm} job=${job}"
    jobs_json="$(python - "${jobs_json}" "${lm}" "${job}" "${REMOTE_OUTPUT_ROOT}/lm${lm}" <<'PY'
import json
import sys
items = json.loads(sys.argv[1])
items.append({"lm": sys.argv[2], "job_id": sys.argv[3], "output_root": sys.argv[4]})
print(json.dumps(items))
PY
)"
  done
  update_manifest "psc_full_jobs_submitted" "{\"psc_full_jobs\":${jobs_json},\"psc_full_output_root\":\"${REMOTE_OUTPUT_ROOT}\"}"
  echo "[ALL DONE] submitted full jobs: ${jobs_json}"
}

main "$@"
