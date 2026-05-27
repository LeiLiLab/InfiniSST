#!/usr/bin/env bash
set -euo pipefail

# Fallback path used when private HF storage is insufficient for the origin
# model exports.  Transfers each model to PSC with file-size checks, then
# submits the matching medicine no-RAG 2xV100 job.

PSC_HOST="${PSC_HOST:-psc}"
PSC_BASE="${PSC_BASE:-/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval}"
ROOT_DIR="${ROOT_DIR:-${PSC_BASE}/src/InfiniSST}"
LAUNCHER="${LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260523__psc_medicine_norag_baseline_abbrev_restored.sh}"
APPTAINER_SIF="${APPTAINER_SIF:-${PSC_BASE}/containers/ubuntu_22_04_gcc.sif}"
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-/mnt/gemini/data/jiaxuanluo/owaski}"
REMOTE_MODEL_ROOT="${REMOTE_MODEL_ROOT:-${PSC_BASE}/models/owaski}"
RUN_STAMP_BASE="${RUN_STAMP_BASE:-20260523T2030_psc_medicine_norag}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_norag_psc_submit_20260523T2030}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

mkdir -p "${LOCAL_STATE_DIR}"

model_dir_for_lang() {
  case "$1" in
    zh) echo "gigaspeech-zh-s_origin-bsz4" ;;
    de) echo "gigaspeech-de-s_origin-bsz4" ;;
    ja) echo "gigaspeech-ja-s_origin-bsz4" ;;
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

local_validate_model() {
  local path="$1"
  test -f "${path}/config.json"
  local shards
  shards="$(find "${path}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shards}" == "15" ]]
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

remote_file_size() {
  local path="$1"
  ssh -n "${PSC_HOST}" "stat -c%s $(printf '%q' "${path}") 2>/dev/null || true"
}

remote_validate_model() {
  local remote_dir="$1"
  ssh "${PSC_HOST}" 'bash -s' -- "${remote_dir}" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
test -f "${remote_dir}/config.json"
shards="$(find "${remote_dir}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
test "${shards}" = "15"
REMOTE
}

tar_put_file() {
  local local_file="$1"
  local remote_dir="$2"
  local local_dir
  local base
  local_dir="$(dirname "${local_file}")"
  base="$(basename "${local_file}")"
  tar -C "${local_dir}" -cf - "${base}" | \
    ssh "${PSC_HOST}" "mkdir -p $(printf '%q' "${remote_dir}") && cd $(printf '%q' "${remote_dir}") && tar -xf -"
}

transfer_model_if_needed() {
  local lang="$1"
  local dirname local_dir remote_dir local_file base local_size remote_size tmp_remote
  dirname="$(model_dir_for_lang "${lang}")"
  local_dir="${LOCAL_MODEL_ROOT}/${dirname}"
  remote_dir="${REMOTE_MODEL_ROOT}/${dirname}"

  if ! local_validate_model "${local_dir}"; then
    echo "[ERROR] Local model validation failed: ${local_dir}" >&2
    exit 3
  fi
  if remote_validate_model "${remote_dir}"; then
    echo "[INFO] Remote model already validates: ${remote_dir}"
    return 0
  fi

  ssh "${PSC_HOST}" "mkdir -p $(printf '%q' "${remote_dir}")"
  while IFS= read -r -d '' local_file; do
    base="$(basename "${local_file}")"
    local_size="$(stat -c%s "${local_file}")"
    remote_size="$(remote_file_size "${remote_dir}/${base}")"
    if [[ "${remote_size}" == "${local_size}" ]]; then
      echo "[SKIP] ${lang}/${base}: size=${local_size}"
      continue
    fi
    echo "[COPY] ${lang}/${base}: ${local_size} bytes"
    tar_put_file "${local_file}" "${remote_dir}"
  done < <(find "${local_dir}" -maxdepth 1 -type f -print0 | sort -z)

  if ! remote_validate_model "${remote_dir}"; then
    echo "[ERROR] Remote model validation failed after transfer: ${remote_dir}" >&2
    exit 4
  fi
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
  printf 'cd %q && PSC_BASE=%q ROOT_DIR=%q RUN_STAMP=%q MED_LANG=%q TARGET_LMS=%q GPU_PAIR=0,1 OUTPUT_BASE=%q LOG_ROOT=%q USE_APPTAINER=1 APPTAINER_SIF=%q WAIT_FOR_HF_READY=0 bash %q' \
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

langs=(zh de ja)
for lang in "${langs[@]}"; do
  if [[ -s "${LOCAL_STATE_DIR}/${lang}.job_id" ]]; then
    echo "[SKIP] ${lang}: already submitted as $(cat "${LOCAL_STATE_DIR}/${lang}.job_id")"
    continue
  fi
  if ! psc_data_ready; then
    echo "[ERROR] PSC medicine data is not complete yet; rerun after data transfer finishes." >&2
    exit 4
  fi
  transfer_model_if_needed "${lang}"
  lms="$(lms_for_lang "${lang}")"
  job_id="$(submit_lang "${lang}" "${lms}")"
  printf '%s\n' "${job_id}" > "${LOCAL_STATE_DIR}/${lang}.job_id"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${lang}" "${lms}" "${job_id}" \
    >> "${LOCAL_STATE_DIR}/submitted.tsv"
  echo "[SUBMITTED] ${lang} lms=${lms} job_id=${job_id}"
done

echo "[ALL DONE] transfer-and-submit state_dir=${LOCAL_STATE_DIR}"
