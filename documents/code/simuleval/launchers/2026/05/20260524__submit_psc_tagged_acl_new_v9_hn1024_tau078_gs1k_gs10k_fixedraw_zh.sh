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
PSC_MODEL_ROOT="${PSC_MODEL_ROOT:-${PSC_BASE}/models/new_v9_termtag_delay_oldnewv3_r32a64/keep1.0_r32}"
PSC_MODEL_DIR="${PSC_MODEL_DIR:-${PSC_MODEL_ROOT}/v0-20260524-062743-hf}"
RAW_GLOSSARY="${RAW_GLOSSARY:-${PSC_BASE}/glossaries/acl6060_tagged_gt_raw_min_norm2.json}"

RUN_STAMP_BASE="${RUN_STAMP_BASE:-20260524T0520_psc_tagacl_newv9_hn1024_tau078_gs1k_gs10k_fixedraw_zh}"
EVENT_MANIFEST="${EVENT_MANIFEST:-${ROOT_DIR}/documents/code/simuleval/manifests/2026/05/20260524T0520__simuleval__psc_tagged_acl_new_v9_hn1024_tau078_gs1k_gs10k_fixedraw_zh.json}"
LMS="${LMS:-1 2 3 4}"
GLOSSARY_KINDS="${GLOSSARY_KINDS:-gs1k gs10k}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
ACCOUNT="${ACCOUNT:-cis260009p}"
PARTITION="${PARTITION:-GPU-shared}"
GRES="${GRES:-gpu:v100-32:4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-4}"

REMOTE_OUTPUT_ROOT="${PSC_BASE}/outputs/tagged_acl_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP_BASE}"
REMOTE_LOG_ROOT="${PSC_BASE}/logs/tagged_acl_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP_BASE}"

ssh_remote() {
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${PSC_HOST}" "$@"
}

put_remote_file() {
  local path="$1"
  local dir
  dir="$(dirname "${path}")"
  # Do not consume pipeline stdin with the mkdir SSH call.
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${PSC_HOST}" "mkdir -p $(printf '%q' "${dir}")" < /dev/null
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${PSC_HOST}" "cat > $(printf '%q' "${path}")"
}

validate_remote_preflight() {
  ssh_remote "set -euo pipefail
for p in \
  $(printf '%q' "${PSC_ROOT}") \
  $(printf '%q' "${PSC_ENV}/bin/python") \
  $(printf '%q' "${PSC_LAUNCHER}") \
  $(printf '%q' "${APPTAINER_SIF}") \
  $(printf '%q' "${PSC_MODEL_DIR}/config.json") \
  $(printf '%q' "${RAW_GLOSSARY}"); do
  test -e \"\$p\"
done
shards=\$(find $(printf '%q' "${PSC_MODEL_DIR}") -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')
test \"\$shards\" = 15
bash -n $(printf '%q' "${PSC_LAUNCHER}")
"
}

submit_setting_job() {
  local kind="$1"
  local lm="$2"
  local suffix="${kind}/lm${lm}"
  local safe_kind="${kind//[^A-Za-z0-9]/_}"
  local job_name="tagacl_n9_${safe_kind}_lm${lm}"
  local out_root="${REMOTE_OUTPUT_ROOT}/${suffix}"
  local log_root="${REMOTE_LOG_ROOT}/${suffix}"
  local script="${log_root}/${job_name}.sbatch"

  cat <<REMOTE | put_remote_file "${script}"
#!/usr/bin/env bash
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH -t ${TIME_LIMIT}
#SBATCH --cpus-per-gpu=${CPUS_PER_GPU}
#SBATCH -J ${job_name}
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
  RUN_STAMP=${RUN_STAMP_BASE}_${safe_kind}_lm${lm} \\
  MODE=full \\
  LMS="${lm}" \\
  GLOSSARY_KINDS="${kind}" \\
  GPU_PAIR=0,1,2,3 \\
  MODEL_ROOT=${PSC_MODEL_ROOT} \\
  MODEL_DIR=${PSC_MODEL_DIR} \\
  OUT_ROOT=${out_root} \\
  LOG_ROOT=${log_root} \\
  EVAL_GLOSSARY_PATH_GLOBAL=${RAW_GLOSSARY} \\
  EVAL_GLOSSARY_FOLLOWS_KIND=0 \\
  DENSITY_TAG_OVERRIDE=tagacl_new_v9_hn1024_tau078_gs_fixedraw \\
  WANDB_RUN_PREFIX_OVERRIDE=new_v9_hn1024_tau078_gs_fixedraw \\
  WANDB_EXPERIMENT_FAMILY_OVERRIDE=tagged_acl_new_v9_hn1024_tau078 \\
  WANDB_VARIANT_PREFIX_OVERRIDE=new_v9_hn1024_tau078_gs_fixedraw \\
  VLLM_TP_SIZE_OVERRIDE=4 \\
  VLLM_MAX_MODEL_LEN_OVERRIDE=8192 \\
  GPU_MEMORY_UTILIZATION_OVERRIDE=0.70 \\
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \\
  VLLM_MOE_USE_DEEP_GEMM=0 \\
  VLLM_USE_FUSED_MOE_GROUPED_TOPK=0 \\
  RAG_GPU_OVERRIDE=cuda:3 \\
  bash ${PSC_LAUNCHER}
REMOTE

  ssh_remote "sbatch --parsable $(printf '%q' "${script}")"
}

update_manifest_submitted() {
  local jobs_json="$1"
  MANIFEST_UPDATE_JSON="${jobs_json}" \
  REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT}" \
  REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT}" \
  RUN_STAMP_BASE="${RUN_STAMP_BASE}" \
  python - "${EVENT_MANIFEST}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
jobs = json.loads(os.environ["MANIFEST_UPDATE_JSON"])
data = json.loads(path.read_text(encoding="utf-8"))
data["status"] = "psc_full_jobs_submitted_direct"
metadata = data.setdefault("metadata", {})
metadata.update({
    "psc_full_jobs": jobs,
    "psc_full_jobs_submitted_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat().replace("+00:00", "Z"),
    "psc_full_output_root": os.environ["REMOTE_OUTPUT_ROOT"],
    "psc_full_log_root": os.environ["REMOTE_LOG_ROOT"],
    "run_stamp_base": os.environ["RUN_STAMP_BASE"],
})
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  python "${ROOT_DIR}/documents/code/general/experiment_event.py" register "${EVENT_MANIFEST}"
}

main() {
  validate_remote_preflight

  local jobs_json="[]"
  local kind lm job
  for kind in ${GLOSSARY_KINDS}; do
    for lm in ${LMS}; do
      job="$(submit_setting_job "${kind}" "${lm}")"
      echo "[SUBMIT] kind=${kind} lm=${lm} job=${job}"
      jobs_json="$(python - "${jobs_json}" "${kind}" "${lm}" "${job}" "${REMOTE_OUTPUT_ROOT}/${kind}/lm${lm}" "${REMOTE_LOG_ROOT}/${kind}/lm${lm}" <<'PY'
import json
import sys

items = json.loads(sys.argv[1])
items.append({
    "glossary_kind": sys.argv[2],
    "lm": sys.argv[3],
    "job_id": sys.argv[4],
    "output_root": sys.argv[5],
    "log_root": sys.argv[6],
})
print(json.dumps(items))
PY
)"
    done
  done
  update_manifest_submitted "${jobs_json}"
  echo "[ALL DONE] submitted gs fixed-raw jobs: ${jobs_json}"
}

main "$@"
