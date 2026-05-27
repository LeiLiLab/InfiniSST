#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
PSC_HOST="${PSC_HOST:-jluo7@bridges2.psc.edu}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ControlPath=/home/jiaxuanluo/.ssh/sockets/jluo7@bridges2.psc.edu:22}"
PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
PSC_ROOT="${PSC_ROOT:-${PSC_BASE}/src/InfiniSST}"
PSC_ENV="${PSC_ENV:-${PSC_BASE}/envs/spaCyEnv_20260518}"
PSC_LAUNCHER="${PSC_LAUNCHER:-${PSC_ROOT}/documents/code/simuleval/launchers/2026/05/20260524__psc_medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw_zh.sh}"
BATCH_LAUNCHER="${BATCH_LAUNCHER:-${PSC_ROOT}/documents/code/simuleval/launchers/2026/05/20260524__medicine_hardraw_lm1to4_5samples_hn1024_tau078_new_v9_batch.sh}"
APPTAINER_SIF="${APPTAINER_SIF:-${PSC_BASE}/containers/ubuntu_22_04_gcc.sif}"
PSC_MODEL_DIR="${PSC_MODEL_DIR:-${PSC_BASE}/models/new_v9_termtag_delay_oldnewv3_r32a64/keep1.0_r32/v0-20260524-062743-hf}"
HN1024_CKPT="${HN1024_CKPT:-${PSC_BASE}/checkpoints/lh1b88kw_best_eval_acl6060_recallat10.pt}"
FIXED_RAW_GLOSSARY="${FIXED_RAW_GLOSSARY:-${PSC_BASE}/glossaries/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json}"
GS1K_GLOSSARY="${GS1K_GLOSSARY:-${PSC_BASE}/glossaries/medicine_hardraw_plus_gtwiki_gs1000_translated_fixedraw.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY:-${PSC_BASE}/glossaries/medicine_hardraw_plus_gtwiki_gs10000_translated_fixedraw.json}"
ESO_TEST_ROOT="${ESO_TEST_ROOT:-${PSC_BASE}/data/eso_medicine_abbrev_restored/test}"
NOTES_FILE="${NOTES_FILE:-${PSC_ROOT}/documents/code/simuleval/notes/2026/05/20260524__psc_medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw_zh.md}"

RUN_STAMP_BASE="${RUN_STAMP_BASE:-20260524T1226_psc_med_newv9_hn1024_tau078_gs1k_gs10k_fixedraw_zh}"
EVENT_MANIFEST="${EVENT_MANIFEST:-${ROOT_DIR}/documents/code/simuleval/manifests/2026/05/20260524T1226__simuleval__psc_medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw_zh.json}"
LMS="${LMS:-1 2 3 4}"
GLOSSARY_KINDS="${GLOSSARY_KINDS:-gs1k gs10k}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
ACCOUNT="${ACCOUNT:-cis260009p}"
PARTITION="${PARTITION:-GPU-shared}"
GRES="${GRES:-gpu:v100-32:4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-4}"

REMOTE_OUTPUT_ROOT="${PSC_BASE}/outputs/medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP_BASE}"
REMOTE_LOG_ROOT="${PSC_BASE}/logs/medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP_BASE}"

ssh_remote() {
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${PSC_HOST}" "$@"
}

put_remote_file() {
  local path="$1"
  local dir
  dir="$(dirname "${path}")"
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
  $(printf '%q' "${BATCH_LAUNCHER}") \
  $(printf '%q' "${APPTAINER_SIF}") \
  $(printf '%q' "${PSC_MODEL_DIR}/config.json") \
  $(printf '%q' "${HN1024_CKPT}") \
  $(printf '%q' "${FIXED_RAW_GLOSSARY}") \
  $(printf '%q' "${GS1K_GLOSSARY}") \
  $(printf '%q' "${GS10K_GLOSSARY}") \
  $(printf '%q' "${NOTES_FILE}"); do
  test -e \"\$p\"
done
for s in 404 545006 596001 605000 606; do
  test -s $(printf '%q' "${ESO_TEST_ROOT}")/sample_\${s}_v2/full_sample_v2.json
  test -s $(printf '%q' "${ESO_TEST_ROOT}")/sample_\${s}_v2/\${s}_v2.wav
  test -s $(printf '%q' "${ESO_TEST_ROOT}")/sample_\${s}_v2/sentences_v2.json
  test -s $(printf '%q' "${ESO_TEST_ROOT}")/sample_\${s}_v2/metadata_v2.json
done
shards=\$(find $(printf '%q' "${PSC_MODEL_DIR}") -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')
test \"\$shards\" = 15
python - <<'PY'
import json
from pathlib import Path
checks = [
  ('hardraw', Path('${FIXED_RAW_GLOSSARY}'), 212),
  ('gs1k', Path('${GS1K_GLOSSARY}'), 1000),
  ('gs10k', Path('${GS10K_GLOSSARY}'), 10000),
]
def norm(x): return ' '.join(str(x or '').casefold().split())
for name, path, expected in checks:
    data = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(data, list), (name, type(data))
    assert len(data) == expected, (name, len(data), expected)
    keys = [norm((e.get('term') or e.get('source')) if isinstance(e, dict) else '') for e in data]
    assert len(set(keys)) == expected, (name, len(set(keys)), expected)
PY
bash -n $(printf '%q' "${PSC_LAUNCHER}")
bash -n $(printf '%q' "${BATCH_LAUNCHER}")
"
}

submit_setting_job() {
  local kind="$1"
  local lm="$2"
  local safe_kind="${kind//[^A-Za-z0-9]/_}"
  local job_name="med_n9_${safe_kind}_lm${lm}"
  local out_root="${REMOTE_OUTPUT_ROOT}/${kind}/lm${lm}"
  local log_root="${REMOTE_LOG_ROOT}/${kind}/lm${lm}"
  local index_root="${PSC_BASE}/cache/maxsim_index_cache/medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/${RUN_STAMP_BASE}/${kind}/lm${lm}"
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
  GLOSSARY_KIND=${kind} \\
  TARGET_LMS=${lm} \\
  GPU_PAIR=0,1,2,3 \\
  MODEL_DIR=${PSC_MODEL_DIR} \\
  HN1024_CKPT=${HN1024_CKPT} \\
  FIXED_RAW_GLOSSARY=${FIXED_RAW_GLOSSARY} \\
  GS1K_GLOSSARY=${GS1K_GLOSSARY} \\
  GS10K_GLOSSARY=${GS10K_GLOSSARY} \\
  ESO_TEST_ROOT=${ESO_TEST_ROOT} \\
  OUT_ROOT=${out_root} \\
  LOG_ROOT=${log_root} \\
  INDEX_CACHE_DIR=${index_root} \\
  NOTES_FILE=${NOTES_FILE} \\
  VLLM_TP_SIZE_OVERRIDE=4 \\
  VLLM_MAX_MODEL_LEN_OVERRIDE=8192 \\
  VLLM_LIMIT_AUDIO_OVERRIDE=8 \\
  GPU_MEMORY_UTILIZATION_OVERRIDE=0.70 \\
  MAX_CACHE_SECONDS_OVERRIDE=80.0 \\
  KEEP_CACHE_SECONDS_OVERRIDE=60.0 \\
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
from datetime import datetime, timezone

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
jobs = json.loads(os.environ["MANIFEST_UPDATE_JSON"])
data["status"] = "psc_jobs_submitted"
metadata = data.setdefault("metadata", {})
metadata.update({
    "psc_full_jobs": jobs,
    "psc_full_jobs_submitted_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
    "lm": int(sys.argv[3]),
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
  echo "[ALL DONE] submitted PSC medicine gs fixed-raw jobs: ${jobs_json}"
}

main "$@"
