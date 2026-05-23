#!/usr/bin/env bash
set -euo pipefail

# Streaming no-RAG baseline over restored ESO medicine samples.
# Runs one setting at a time on aries GPU 6,7 through the existing baseline
# SimulEval launcher. The sample-specific glossary is used for naming and later
# exact-match review only; RAG and oracle term-map injection stay disabled.

ROOT_DIR="/mnt/taurus/home/jiaxuanluo/InfiniSST"
BASE_SCRIPT="${ROOT_DIR}/documents/code/simuleval/rank16/baseline/bypass_simuleval_rank32_iter_0000452_hf_baseline_no_rag_sweep.sh"
PREP_SCRIPT="${ROOT_DIR}/documents/code/simuleval/prepare_medicine_one_talk_inputs.py"
CONDA_ENV_NAME="${CONDA_ENV_NAME_OVERRIDE:-spaCyEnv}"
DEFAULT_CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/${CONDA_ENV_NAME}"
if [[ -n "${CONDA_PREFIX_OVERRIDE:-}" ]]; then
  CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE}"
elif [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV_NAME}" ]]; then
  CONDA_PREFIX="${CONDA_PREFIX}"
else
  CONDA_PREFIX="${DEFAULT_CONDA_PREFIX}"
fi
export CONDA_PREFIX
if [[ -d "${CONDA_PREFIX}" ]]; then
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
PREP_PYTHON="${PREP_PYTHON_OVERRIDE:-${CONDA_PREFIX}/bin/python}"

ESO_TEST_ROOT="${ESO_TEST_ROOT_OVERRIDE:-/mnt/taurus/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2_abbrev_exact_match_abbrev_restored/test}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_20260522}"
STRICT_GLOSSARY="${STRICT_MEDICINE_GLOSSARY_OVERRIDE:-${OUTPUT_BASE}/strict_fixed_medicine_glossary.from_outputs_v2_terms.json}"

LANGS_TEXT="${LANGS_OVERRIDE:-zh de ja}"
SAMPLES_TEXT="${TARGET_SAMPLES_OVERRIDE:-404 545006 596001 605000 606}"
LMS_TEXT="${TARGET_LMS_OVERRIDE:-1 2 3 4}"
GPU_CSV="${CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV:-6:7}"
GPU_COMMA="${GPU_CSV//:/,}"
GPU_BUSY_THRESHOLD_MIB="${GPU_BUSY_THRESHOLD_MIB_OVERRIDE:-1000}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU_OVERRIDE:-0}"
FORCE_RERUN="${FORCE_RERUN_OVERRIDE:-0}"
RAG_K2_VALUE="${RAG_K2_VALUE_OVERRIDE:-10}"
TERM_SOURCE="${TERM_SOURCE_OVERRIDE:-glossary_match}"
GLOSSARY_SOURCE_FILTER="${GLOSSARY_SOURCE_FILTER_OVERRIDE:-strict_fixed_medicine_glossary}"

TIMING_TSV="${OUTPUT_BASE}/timing.tsv"
HYP_TSV="${OUTPUT_BASE}/hypotheses.tsv"
RUN_SUMMARY="${OUTPUT_BASE}/run_summary.txt"

read -r -a LANGS <<< "${LANGS_TEXT}"
read -r -a TARGET_SAMPLES <<< "${SAMPLES_TEXT}"
read -r -a TARGET_LMS <<< "${LMS_TEXT}"

model_for_lang() {
  case "$1" in
    zh) echo "/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4" ;;
    de) echo "/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4" ;;
    ja) echo "/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-ja-s_origin-bsz4" ;;
    *) echo "[ERROR] Unsupported language: $1" >&2; return 2 ;;
  esac
}

segment_for_lm() {
  python3 - "$1" <<'PY'
import sys
print(f"{0.96 * float(sys.argv[1]):.2f}")
PY
}

output_dir_for() {
  local lang="$1"
  local model="$2"
  local glossary_tag="$3"
  local lm="$4"
  local model_short
  local segment
  model_short="$(basename "${model}")"
  segment="$(segment_for_lm "${lm}")"
  printf '%s/%s/%s_g%s_cs%s_hs0.48_lm%s_k2%s_k110_th0p0' \
    "${OUTPUT_BASE}" "${lang}" "${model_short}" "${glossary_tag}" \
    "${segment}" "${lm}" "${RAG_K2_VALUE}"
}

clean_shm() {
  local me
  me="$(id -un)"
  find /dev/shm -maxdepth 1 -user "${me}" -mmin +1 \
    \( -name 'psm_*' -o -name 'loky-*' -o -name 'torch_*' -o -name 'vllm_*' \) \
    -delete 2>/dev/null || true
}

check_gpus() {
  local gpu
  local used
  IFS=',' read -r -a gpu_ids <<< "${GPU_COMMA}"
  for gpu in "${gpu_ids[@]}"; do
    used="$(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, -v target="${gpu}" '$1 ~ "^[[:space:]]*" target "[[:space:]]*$" {gsub(/ /, "", $2); print $2}'
    )"
    if [[ -z "${used}" ]]; then
      echo "[ERROR] Could not read memory for GPU ${gpu}" >&2
      return 3
    fi
    if [[ "${ALLOW_BUSY_GPU}" != "1" && "${used}" -gt "${GPU_BUSY_THRESHOLD_MIB}" ]]; then
      echo "[ERROR] GPU ${gpu} is busy (${used} MiB > ${GPU_BUSY_THRESHOLD_MIB} MiB)" >&2
      return 4
    fi
  done
}

prepare_inputs() {
  local lang="$1"
  local sample="$2"
  local input_dir="${OUTPUT_BASE}/${lang}/__medicine_inputs__/lists"
  local prefix="medicine_${sample}"
  local glossary_tag="strict_fixed_medicine_glossary_abbrev_restored__${prefix}"
  local term_map_tag="strict_fixed_medicine_glossary.match__${prefix}"

  mkdir -p "${input_dir}"
  "${PREP_PYTHON}" "${PREP_SCRIPT}" \
    --sample-id "${sample}" \
    --lang-code "${lang}" \
    --eso-test-root "${ESO_TEST_ROOT}" \
    --strict-glossary "${STRICT_GLOSSARY}" \
    --term-source "${TERM_SOURCE}" \
    --oracle-glossary "${STRICT_GLOSSARY}" \
    --eval-glossary "${STRICT_GLOSSARY}" \
    --glossary-source-filter "${GLOSSARY_SOURCE_FILTER}" \
    --glossary-tag "${glossary_tag}" \
    --oracle-term-map-tag "${term_map_tag}" \
    --output-dir "${input_dir}" >/dev/null

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${input_dir}" \
    "${input_dir}/medicine.source__${prefix}.txt" \
    "${input_dir}/medicine.target.${lang}__${prefix}.txt" \
    "${input_dir}/${glossary_tag}.json" \
    "${input_dir}/${term_map_tag}.json" \
    "${glossary_tag}"
}

append_hypothesis() {
  local lang="$1"
  local lm="$2"
  local sample="$3"
  local output_dir="$4"
  local instances_log="${output_dir}/instances.log"

  if [[ ! -s "${instances_log}" ]]; then
    echo "[ERROR] Missing non-empty instances.log: ${instances_log}" >&2
    return 3
  fi

  "${PREP_PYTHON}" - "$HYP_TSV" "$lang" "$lm" "$sample" "$output_dir" "$instances_log" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
lang, lm, sample, output_dir, instances_log = sys.argv[2:7]
out.parent.mkdir(parents=True, exist_ok=True)
exists = out.exists() and out.stat().st_size > 0
fields = ["lang", "lm", "sample_id", "instance_index", "output_dir", "instances_log", "hypothesis"]
with out.open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
    if not exists:
        writer.writeheader()
    with Path(instances_log).open("r", encoding="utf-8", errors="replace") as inst:
        for line in inst:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            writer.writerow({
                "lang": lang,
                "lm": lm,
                "sample_id": sample,
                "instance_index": obj.get("index", ""),
                "output_dir": output_dir,
                "instances_log": instances_log,
                "hypothesis": obj.get("prediction", ""),
            })
PY
}

record_timing() {
  local lang="$1"
  local lm="$2"
  local sample="$3"
  local status="$4"
  local seconds="$5"
  local output_dir="$6"
  local rc="$7"
  if [[ ! -f "${TIMING_TSV}" ]]; then
    printf 'lang\tlm\tsample_id\tstatus\tseconds\tminutes\trc\toutput_dir\n' > "${TIMING_TSV}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%.3f\t%s\t%s\n' \
    "${lang}" "${lm}" "${sample}" "${status}" "${seconds}" \
    "$(python3 - "$seconds" <<'PY'
import sys
print(float(sys.argv[1]) / 60.0)
PY
)" \
    "${rc}" "${output_dir}" >> "${TIMING_TSV}"
}

run_one() {
  local lang="$1"
  local lm="$2"
  local sample="$3"
  local model="$4"
  local prep_row="$5"
  local input_dir src_list tgt_list glossary_path term_map_path glossary_tag
  local output_dir start end elapsed rc resume_mode clean_output

  IFS=$'\t' read -r input_dir src_list tgt_list glossary_path term_map_path glossary_tag <<< "${prep_row}"
  output_dir="$(output_dir_for "${lang}" "${model}" "${glossary_tag}" "${lm}")"
  resume_mode="1"
  clean_output="0"
  if [[ "${FORCE_RERUN}" == "1" ]]; then
    resume_mode="0"
    clean_output="1"
  fi

  echo "[SETTING START] lang=${lang} lm=${lm} sample=${sample} output=${output_dir}"
  clean_shm
  check_gpus
  start="$(date +%s)"
  set +e
  CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX}" \
  GLOSSARY_PATHS_OVERRIDE="${glossary_path}" \
  SRC_LIST_OVERRIDE="${src_list}" \
  TGT_LIST_OVERRIDE="${tgt_list}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  MODEL_NAME_OVERRIDE="${model}" \
  LANG_CODE_OVERRIDE="${lang}" \
  LATENCY_MULTIPLIERS_OVERRIDE="${lm}" \
  RAG_K2_VALUES_OVERRIDE="${RAG_K2_VALUE}" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_COMMA}" \
  RESUME_MODE="${resume_mode}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${clean_output}" \
  BACKUP_PARTIAL_RUNS="1" \
  bash "${BASE_SCRIPT}"
  rc="$?"
  set -e
  end="$(date +%s)"
  elapsed="$((end - start))"

  if [[ "${rc}" -ne 0 ]]; then
    record_timing "${lang}" "${lm}" "${sample}" "failed" "${elapsed}" "${output_dir}" "${rc}"
    echo "[SETTING FAILED] lang=${lang} lm=${lm} sample=${sample} rc=${rc} seconds=${elapsed}" >&2
    return "${rc}"
  fi

  append_hypothesis "${lang}" "${lm}" "${sample}" "${output_dir}"
  record_timing "${lang}" "${lm}" "${sample}" "success" "${elapsed}" "${output_dir}" "0"
  echo "[SETTING DONE] lang=${lang} lm=${lm} sample=${sample} seconds=${elapsed} output=${output_dir}"
  clean_shm
}

main() {
  mkdir -p "${OUTPUT_BASE}"
  {
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "output_base=${OUTPUT_BASE}"
    echo "eso_test_root=${ESO_TEST_ROOT}"
    echo "langs=${LANGS_TEXT}"
    echo "samples=${SAMPLES_TEXT}"
    echo "lms=${LMS_TEXT}"
    echo "gpus=${GPU_COMMA}"
  } > "${RUN_SUMMARY}"

  for p in "${BASE_SCRIPT}" "${PREP_SCRIPT}" "${PREP_PYTHON}" "${STRICT_GLOSSARY}" "${ESO_TEST_ROOT}"; do
    if [[ ! -e "${p}" ]]; then
      echo "[ERROR] Missing required path: ${p}" >&2
      exit 3
    fi
  done

  echo "[INFO] LANGS=${LANGS[*]}"
  echo "[INFO] SAMPLES=${TARGET_SAMPLES[*]}"
  echo "[INFO] LMS=${TARGET_LMS[*]}"
  echo "[INFO] GPU_COMMA=${GPU_COMMA}"
  echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"

  local lang sample lm model prep_row
  for lang in "${LANGS[@]}"; do
    model="$(model_for_lang "${lang}")"
    if [[ ! -d "${model}" ]]; then
      echo "[ERROR] Missing model for ${lang}: ${model}" >&2
      exit 3
    fi
    for lm in "${TARGET_LMS[@]}"; do
      for sample in "${TARGET_SAMPLES[@]}"; do
        prep_row="$(prepare_inputs "${lang}" "${sample}")"
        run_one "${lang}" "${lm}" "${sample}" "${model}" "${prep_row}"
      done
    done
  done

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RUN_SUMMARY}"
  echo "[ALL DONE] output_base=${OUTPUT_BASE}"
}

main "$@"
