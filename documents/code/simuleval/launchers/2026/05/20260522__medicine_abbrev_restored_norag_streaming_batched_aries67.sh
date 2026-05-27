#!/usr/bin/env bash
set -euo pipefail

# Batched streaming no-RAG baseline over restored ESO medicine samples.
# Each setting is one (language, latency multiplier) run containing all target
# medicine samples. This amortizes Qwen3-Omni/vLLM model loading while still
# producing one hypothesis row per sample in instances.log.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
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
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260522}"
STRICT_GLOSSARY="${STRICT_MEDICINE_GLOSSARY_OVERRIDE:-${OUTPUT_BASE}/strict_fixed_medicine_glossary.from_outputs_v2_terms.json}"

LANGS_TEXT="${LANGS_OVERRIDE:-zh de ja}"
SAMPLES_TEXT="${TARGET_SAMPLES_OVERRIDE:-404 545006 596001 605000 606}"
LMS_TEXT="${TARGET_LMS_OVERRIDE:-2}"
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
    zh) echo "${MODEL_ZH_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4}" ;;
    de) echo "${MODEL_DE_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4}" ;;
    ja) echo "${MODEL_JA_OVERRIDE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-ja-s_origin-bsz4}" ;;
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

build_strict_glossary_from_eso_terms() {
  if [[ -n "${STRICT_MEDICINE_GLOSSARY_OVERRIDE:-}" ]]; then
    return 0
  fi
  if [[ -s "${STRICT_GLOSSARY}" && "${STRICT_MEDICINE_GLOSSARY_FORCE_REBUILD_OVERRIDE:-0}" != "1" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "${STRICT_GLOSSARY}")"
  "${PREP_PYTHON}" - "${ESO_TEST_ROOT}" "${STRICT_GLOSSARY}" "${TARGET_SAMPLES[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
sample_ids = sys.argv[3:]


def norm_space(text):
    return " ".join(str(text or "").split())


def sample_dir(sample_id):
    for candidate in (root / f"sample_{sample_id}_v2", root / f"sample_{sample_id}"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No sample dir for {sample_id} under {root}")


entries = {}
stats = {
    "sample_count": len(sample_ids),
    "sentence_count": 0,
    "term_mentions": 0,
    "skipped_empty_term": 0,
    "skipped_empty_translations": 0,
}

for sample_id in sample_ids:
    path = sample_dir(sample_id) / "full_sample_v2.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    sentences = data.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError(f"Expected list at sentences in {path}")
    for sent in sentences:
        stats["sentence_count"] += 1
        sentence_id = str(sent.get("sentence_id") or "")
        for term_entry in sent.get("terms") or []:
            if not isinstance(term_entry, dict):
                continue
            stats["term_mentions"] += 1
            term = norm_space(term_entry.get("term"))
            if not term:
                stats["skipped_empty_term"] += 1
                continue
            target_translations = term_entry.get("target_translations") or {}
            translations = {
                str(lang): norm_space(value)
                for lang, value in target_translations.items()
                if isinstance(value, str) and norm_space(value)
            }
            if not translations:
                stats["skipped_empty_translations"] += 1
                continue
            translation_key = json.dumps(translations, ensure_ascii=False, sort_keys=True)
            key = (term.casefold(), translation_key)
            row = entries.setdefault(
                key,
                {
                    "term": term,
                    "source": "strict_fixed_medicine_glossary",
                    "target_translations": translations,
                    "sample_ids": set(),
                    "sentence_ids": set(),
                },
            )
            row["sample_ids"].add(str(sample_id))
            if sentence_id:
                row["sentence_ids"].add(sentence_id)

if not entries:
    raise ValueError(f"No terms extracted from {root} for samples={sample_ids}")

rows = []
for row in entries.values():
    row = dict(row)
    row["sample_ids"] = sorted(row["sample_ids"])
    row["sentence_ids"] = sorted(row["sentence_ids"])
    rows.append(row)
rows.sort(
    key=lambda x: (
        x["term"].casefold(),
        json.dumps(x["target_translations"], ensure_ascii=False, sort_keys=True),
    )
)

out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
stats["output_path"] = str(out_path)
stats["entries"] = len(rows)
stats["unique_term_strings"] = len({row["term"].casefold() for row in rows})
print(json.dumps(stats, ensure_ascii=False, indent=2))
PY
}

prepare_sample_inputs() {
  local lang="$1"
  local sample="$2"
  local input_dir="${OUTPUT_BASE}/${lang}/__medicine_inputs__/per_sample"
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

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${sample}" \
    "${input_dir}/medicine.source__${prefix}.txt" \
    "${input_dir}/medicine.target.${lang}__${prefix}.txt" \
    "${input_dir}/${glossary_tag}.json" \
    "${input_dir}/medicine_inputs_manifest__${prefix}.json"
}

prepare_combined_inputs() {
  local lang="$1"
  local combined_dir="${OUTPUT_BASE}/${lang}/__medicine_inputs__/combined"
  local src_list="${combined_dir}/medicine5.source.txt"
  local tgt_list="${combined_dir}/medicine5.target.${lang}.txt"
  local sent_audio_yaml="${combined_dir}/medicine5.audio.yaml"
  local sent_ref="${combined_dir}/medicine5.ref.${lang}.sentences.txt"
  local sent_source="${combined_dir}/medicine5.source_text.en.sentences.txt"
  local sample_map="${combined_dir}/medicine5.sample_map.tsv"
  local glossary_tag="strict_fixed_medicine_glossary_abbrev_restored__medicine5"
  local glossary_path="${combined_dir}/${glossary_tag}.json"
  local prep_rows="${combined_dir}/medicine5.prep_rows.tsv"
  local idx=0
  local sample row src tgt gloss manifest

  mkdir -p "${combined_dir}"
  : > "${src_list}"
  : > "${tgt_list}"
  : > "${sent_audio_yaml}"
  : > "${sent_ref}"
  : > "${sent_source}"
  printf 'instance_index\tsample_id\tsource_list\ttarget_list\tglossary_path\tmanifest_path\n' > "${sample_map}"
  : > "${prep_rows}"

  for sample in "${TARGET_SAMPLES[@]}"; do
    row="$(prepare_sample_inputs "${lang}" "${sample}")"
    IFS=$'\t' read -r sample src tgt gloss manifest <<< "${row}"
    cat "${src}" >> "${src_list}"
    cat "${tgt}" >> "${tgt_list}"
    cat "${OUTPUT_BASE}/${lang}/__medicine_inputs__/per_sample/medicine.audio__medicine_${sample}.yaml" >> "${sent_audio_yaml}"
    cat "${OUTPUT_BASE}/${lang}/__medicine_inputs__/per_sample/medicine.ref.${lang}__medicine_${sample}.txt" >> "${sent_ref}"
    cat "${OUTPUT_BASE}/${lang}/__medicine_inputs__/per_sample/medicine.source_text.en__medicine_${sample}.txt" >> "${sent_source}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${idx}" "${sample}" "${src}" "${tgt}" "${gloss}" "${manifest}" >> "${sample_map}"
    printf '%s\n' "${row}" >> "${prep_rows}"
    idx="$((idx + 1))"
  done

  "${PREP_PYTHON}" - "${prep_rows}" "${glossary_path}" <<'PY'
import json
import sys
from pathlib import Path

rows = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
out = Path(sys.argv[2])
merged = {}
for row in rows:
    if not row.strip():
        continue
    sample, _src, _tgt, gloss, _manifest = row.split("\t")
    data = json.loads(Path(gloss).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.items()
    else:
        items = enumerate(data)
    for key, value in items:
        if isinstance(value, dict):
            value = dict(value)
            value.setdefault("sample_id", sample)
        merged[f"{sample}::{key}"] = value
out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

  printf '%s\t%s\t%s\t%s\n' "${src_list}" "${tgt_list}" "${glossary_path}" "${glossary_tag}"
}

validate_source_paths() {
  local combined_row="$1"
  local src_list tgt_list glossary_path glossary_tag missing=0
  IFS=$'\t' read -r src_list tgt_list glossary_path glossary_tag <<< "${combined_row}"
  if [[ ! -s "${src_list}" ]]; then
    echo "[ERROR] Missing source list: ${src_list}" >&2
    return 3
  fi
  while IFS= read -r wav_path; do
    [[ -n "${wav_path}" ]] || continue
    if [[ ! -r "${wav_path}" ]]; then
      echo "[ERROR] Source wav is not readable on this host: ${wav_path}" >&2
      missing="$((missing + 1))"
    fi
  done < "${src_list}"
  if (( missing > 0 )); then
    echo "[ERROR] ${missing} source wav path(s) missing from ${src_list}" >&2
    return 3
  fi
}

append_hypotheses() {
  local lang="$1"
  local lm="$2"
  local sample_map="$3"
  local output_dir="$4"
  local instances_log="${output_dir}/instances.log"

  if [[ ! -s "${instances_log}" ]]; then
    echo "[ERROR] Missing non-empty instances.log: ${instances_log}" >&2
    return 3
  fi

  "${PREP_PYTHON}" - "$HYP_TSV" "$lang" "$lm" "$sample_map" "$output_dir" "$instances_log" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
lang, lm, sample_map_path, output_dir, instances_log = sys.argv[2:7]
out.parent.mkdir(parents=True, exist_ok=True)
exists = out.exists() and out.stat().st_size > 0
index_to_sample = {}
with Path(sample_map_path).open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        index_to_sample[int(row["instance_index"])] = row["sample_id"]
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
            idx = int(obj.get("index", 0))
            writer.writerow({
                "lang": lang,
                "lm": lm,
                "sample_id": index_to_sample.get(idx, ""),
                "instance_index": idx,
                "output_dir": output_dir,
                "instances_log": instances_log,
                "hypothesis": obj.get("prediction", ""),
            })
PY
}

record_timing() {
  local lang="$1"
  local lm="$2"
  local status="$3"
  local seconds="$4"
  local output_dir="$5"
  local rc="$6"
  if [[ ! -f "${TIMING_TSV}" ]]; then
    printf 'lang\tlm\tsample_group\tsample_count\tstatus\tseconds\tminutes\trc\toutput_dir\n' > "${TIMING_TSV}"
  fi
  printf '%s\t%s\tmedicine5\t%s\t%s\t%s\t%.3f\t%s\t%s\n' \
    "${lang}" "${lm}" "${#TARGET_SAMPLES[@]}" "${status}" "${seconds}" \
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
  local model="$3"
  local combined_row="$4"
  local sample_map="$5"
  local src_list tgt_list glossary_path glossary_tag output_dir start end elapsed rc resume_mode clean_output

  IFS=$'\t' read -r src_list tgt_list glossary_path glossary_tag <<< "${combined_row}"
  output_dir="$(output_dir_for "${lang}" "${model}" "${glossary_tag}" "${lm}")"
  resume_mode="1"
  clean_output="0"
  if [[ "${FORCE_RERUN}" == "1" ]]; then
    resume_mode="0"
    clean_output="1"
  fi

  echo "[SETTING START] lang=${lang} lm=${lm} samples=${SAMPLES_TEXT} output=${output_dir}"
  clean_shm
  check_gpus
  start="$(date +%s)"
  set +e
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  DATA_ROOT_OVERRIDE="$(dirname "${src_list}")" \
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
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.8}" \
  MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_OVERRIDE:-80.0}" \
  KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_OVERRIDE:-60.0}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-2}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-32768}" \
  VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE:-}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0}" \
  RESUME_MODE="${resume_mode}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${clean_output}" \
  BACKUP_PARTIAL_RUNS="1" \
  bash "${BASE_SCRIPT}"
  rc="$?"
  set -e
  end="$(date +%s)"
  elapsed="$((end - start))"

  if [[ "${rc}" -ne 0 ]]; then
    record_timing "${lang}" "${lm}" "failed" "${elapsed}" "${output_dir}" "${rc}"
    echo "[SETTING FAILED] lang=${lang} lm=${lm} rc=${rc} seconds=${elapsed}" >&2
    return "${rc}"
  fi

  append_hypotheses "${lang}" "${lm}" "${sample_map}" "${output_dir}"
  record_timing "${lang}" "${lm}" "success" "${elapsed}" "${output_dir}" "0"
  echo "[SETTING DONE] lang=${lang} lm=${lm} seconds=${elapsed} output=${output_dir}"
  clean_shm
}

main() {
  mkdir -p "${OUTPUT_BASE}"
  {
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "output_base=${OUTPUT_BASE}"
    echo "eso_test_root=${ESO_TEST_ROOT}"
    echo "strict_glossary=${STRICT_GLOSSARY}"
    echo "glossary_source_filter=${GLOSSARY_SOURCE_FILTER}"
    echo "langs=${LANGS_TEXT}"
    echo "samples=${SAMPLES_TEXT}"
    echo "lms=${LMS_TEXT}"
    echo "gpus=${GPU_COMMA}"
    echo "batching=one_language_lm_run_contains_all_samples"
  } > "${RUN_SUMMARY}"

  for p in "${BASE_SCRIPT}" "${PREP_SCRIPT}" "${PREP_PYTHON}" "${ESO_TEST_ROOT}"; do
    if [[ ! -e "${p}" ]]; then
      echo "[ERROR] Missing required path: ${p}" >&2
      exit 3
    fi
  done

  build_strict_glossary_from_eso_terms

  if [[ ! -s "${STRICT_GLOSSARY}" ]]; then
    echo "[ERROR] Missing generated strict glossary: ${STRICT_GLOSSARY}" >&2
    exit 3
  fi

  echo "[INFO] LANGS=${LANGS[*]}"
  echo "[INFO] SAMPLES=${TARGET_SAMPLES[*]}"
  echo "[INFO] LMS=${TARGET_LMS[*]}"
  echo "[INFO] GPU_COMMA=${GPU_COMMA}"
  echo "[INFO] OUTPUT_BASE=${OUTPUT_BASE}"
  echo "[INFO] STRICT_GLOSSARY=${STRICT_GLOSSARY}"

  local lang lm model combined_row sample_map
  for lang in "${LANGS[@]}"; do
    model="$(model_for_lang "${lang}")"
    if [[ ! -d "${model}" ]]; then
      echo "[ERROR] Missing model for ${lang}: ${model}" >&2
      exit 3
    fi
    combined_row="$(prepare_combined_inputs "${lang}")"
    validate_source_paths "${combined_row}"
    sample_map="${OUTPUT_BASE}/${lang}/__medicine_inputs__/combined/medicine5.sample_map.tsv"
    for lm in "${TARGET_LMS[@]}"; do
      run_one "${lang}" "${lm}" "${model}" "${combined_row}" "${sample_map}"
    done
  done

  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RUN_SUMMARY}"
  echo "[ALL DONE] output_base=${OUTPUT_BASE}"
}

main "$@"
