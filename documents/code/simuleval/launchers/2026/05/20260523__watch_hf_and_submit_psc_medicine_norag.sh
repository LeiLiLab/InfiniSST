#!/usr/bin/env bash
set -euo pipefail

# Local Taurus watcher.  It waits for each private HF origin model repo to
# contain a complete HF export, then submits the corresponding PSC 2xV100 job.

PSC_HOST="${PSC_HOST:-psc}"
PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
LAUNCHER="${LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260523__psc_medicine_norag_baseline_abbrev_restored.sh}"
APPTAINER_SIF="${APPTAINER_SIF:-${PSC_BASE}/containers/ubuntu_22_04_gcc.sif}"
RUN_STAMP_BASE="${RUN_STAMP_BASE:-20260523T2030_psc_medicine_norag}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_norag_psc_submit_20260523T2030}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-86400}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

mkdir -p "${LOCAL_STATE_DIR}"

repo_for_lang() {
  case "$1" in
    zh) echo "gavinlaw/infinisst-no-tmsft-origin-bsz4-zh" ;;
    de) echo "gavinlaw/infinisst-no-tmsft-origin-bsz4-de" ;;
    ja) echo "gavinlaw/infinisst-no-tmsft-origin-bsz4-ja" ;;
    *) return 2 ;;
  esac
}

lms_for_lang() {
  case "$1" in
    zh) echo "1 3 4" ;;
    de) echo "1 2 3 4" ;;
    ja) echo "1 2 3 4" ;;
    *) return 2 ;;
  esac
}

repo_ready() {
  local repo_id="$1"
python - "$repo_id" <<'PY'
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
PY
}

psc_data_ready() {
  ssh "${PSC_HOST}" 'bash -s' <<'REMOTE'
set -euo pipefail
root="/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/data/eso_medicine_abbrev_restored/test"
for sample in 404 545006 596001 605000 606; do
  test -s "${root}/sample_${sample}_v2/full_sample_v2.json"
  test -s "${root}/sample_${sample}_v2/${sample}_v2.wav"
done
REMOTE
}

submit_lang() {
  local lang="$1"
  local lms="$2"
  local run_stamp="${RUN_STAMP_BASE}_${lang}"
  local remote_cmd
  remote_cmd="$(
    printf 'lang=%q lms=%q run_stamp=%q time_limit=%q psc_base=%q root_dir=%q launcher=%q apptainer_sif=%q bash -s' \
      "${lang}" "${lms}" "${run_stamp}" "${TIME_LIMIT}" "${PSC_BASE}" "${ROOT_DIR}" "${LAUNCHER}" "${APPTAINER_SIF}"
  )"
  ssh "${PSC_HOST}" "${remote_cmd}" <<'REMOTE'
set -euo pipefail
log_root="${psc_base}/logs/medicine_norag_baseline_abbrev_restored/${run_stamp}"
out_root="${psc_base}/outputs/medicine_norag_baseline_abbrev_restored/${run_stamp}"
mkdir -p "${log_root}" "${out_root}"
wrap_cmd="$(
  printf 'cd %q && PSC_BASE=%q ROOT_DIR=%q RUN_STAMP=%q MED_LANG=%q TARGET_LMS=%q GPU_PAIR=0,1 OUTPUT_BASE=%q LOG_ROOT=%q USE_APPTAINER=1 APPTAINER_SIF=%q bash %q' \
    "${root_dir}" "${psc_base}" "${root_dir}" "${run_stamp}" "${lang}" "${lms}" "${out_root}" "${log_root}" "${apptainer_sif}" "${launcher}"
)"
sbatch --parsable \
  -A cis260009p \
  -p GPU-shared \
  --gres=gpu:v100-32:2 \
  --cpus-per-gpu=4 \
  -t "${time_limit}" \
  -J "med_norag_${lang}" \
  -o "${log_root}/${lang}_%j.out" \
  -e "${log_root}/${lang}_%j.err" \
  --wrap "${wrap_cmd}"
REMOTE
}

start_ts="$(date +%s)"
langs=(zh de ja)

while :; do
  submitted=0
  for lang in "${langs[@]}"; do
    if [[ -s "${LOCAL_STATE_DIR}/${lang}.job_id" ]]; then
      submitted="$((submitted + 1))"
      continue
    fi

    repo_id="$(repo_for_lang "${lang}")"
    if ! repo_ready "${repo_id}"; then
      echo "[WAIT] ${lang}: HF repo not ready: ${repo_id}"
      continue
    fi
    if ! psc_data_ready; then
      echo "[WAIT] ${lang}: PSC medicine data not ready"
      continue
    fi

    lms="$(lms_for_lang "${lang}")"
    job_id="$(submit_lang "${lang}" "${lms}")"
    printf '%s\n' "${job_id}" > "${LOCAL_STATE_DIR}/${lang}.job_id"
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${lang}" "${lms}" "${job_id}" \
      >> "${LOCAL_STATE_DIR}/submitted.tsv"
    echo "[SUBMITTED] ${lang} lms=${lms} job_id=${job_id}"
    submitted="$((submitted + 1))"
  done

  if [[ "${submitted}" == "${#langs[@]}" ]]; then
    echo "[ALL SUBMITTED] state_dir=${LOCAL_STATE_DIR}"
    exit 0
  fi

  now="$(date +%s)"
  if (( now - start_ts >= MAX_WAIT_SECONDS )); then
    echo "[ERROR] Timed out waiting for HF repos / PSC data." >&2
    exit 4
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done
