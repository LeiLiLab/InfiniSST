#!/usr/bin/env bash
set -euo pipefail

# En-De tagged ACL raw lm=4 InfiniSST/no-RAG one-sample-at-a-time rerun.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260524T2300_tagacl_origin_norag_de_lm4_raw_onebyone_aries01}"
GPU_PAIR="${GPU_PAIR:-0,1}"

BASELINE_SCRIPT="${BASELINE_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/rank16/baseline/bypass_simuleval_rank32_iter_0000452_hf_baseline_no_rag_sweep.sh}"
OFFLINE_EVAL_SCRIPT="${OFFLINE_EVAL_SCRIPT:-${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py}"
DATA_ROOT="${DATA_ROOT:-/mnt/taurus/data/siqiouyang/datasets/acl6060}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
MODEL_DE="${MODEL_DE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
STREAM_LAAL_TOOL_REL="${STREAM_LAAL_TOOL_REL:-examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm4_raw_onebyone_aries01_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_origin_norag_de_lm4_raw_onebyone_aries01_${RUN_STAMP}}"
CACHE_BASE="${CACHE_BASE:-/mnt/gemini/data1/jiaxuanluo/cache/tagged_acl_origin_norag_de_lm4_raw_onebyone_aries01_${RUN_STAMP}}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_tacl_d4o}"
INPUT_DIR="${OUT_ROOT}/__inputs__"
SAMPLE_DIR="${OUT_ROOT}/__samples__"
COMBINED_DIR="${OUT_ROOT}/combined/de/gigaspeech-de-s_origin-bsz4_gacl6060_tagged_gt_raw_min_norm2_cs3.84_hs0.48_lm4_k210_k110_th0p0_onebyone"
SUMMARY_DIR="${OUT_ROOT}/__summary__"

CONDA_BASE="${CONDA_BASE:-/mnt/taurus/home/jiaxuanluo/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-spaCyEnv}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}/bin/python}"

FULL_SOURCE="${DATA_ROOT}/dev.source"
FULL_TARGET="${DATA_ROOT}/dev.target.de"
SRC_LIST_PORTABLE="${INPUT_DIR}/dev.source.portable"
FULL_REF="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.de.txt"
FULL_SRC_TEXT="${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt"
AUDIO_YAML="${DATA_ROOT}/dev.yaml"
GLOSSARY_TAG="acl6060_tagged_gt_raw_min_norm2"
SETTING_REL="de/gigaspeech-de-s_origin-bsz4_gacl6060_tagged_gt_raw_min_norm2_cs3.84_hs0.48_lm4_k210_k110_th0p0"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_file() {
  local path="$1"
  [[ -s "${path}" ]] || fail "Missing/empty required file: ${path}"
}

gpu_is_idle() {
  local gpu="$1" csv line mem util
  csv="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  line="$(awk -F, -v g="${gpu}" '$1 + 0 == g {print $0}' <<< "${csv}")"
  [[ -n "${line}" ]] || return 1
  mem="$(awk -F, '{gsub(/[[:space:]]/, "", $2); print $2}' <<< "${line}")"
  util="$(awk -F, '{gsub(/[[:space:]]/, "", $3); print $3}' <<< "${line}")"
  (( mem <= 2048 && util <= 25 ))
}

if [[ -e "${OUT_ROOT}/.started" && "${FORCE_RERUN_OVERRIDE:-0}" != "1" ]]; then
  fail "output root already started: ${OUT_ROOT}"
fi

for path in \
  "${BASELINE_SCRIPT}" \
  "${OFFLINE_EVAL_SCRIPT}" \
  "${MODEL_DE}/config.json" \
  "${RAW_GLOSSARY}" \
  "${FULL_SOURCE}" \
  "${FULL_TARGET}" \
  "${FULL_REF}" \
  "${FULL_SRC_TEXT}" \
  "${AUDIO_YAML}"; do
  require_file "${path}"
done

IFS=',' read -r gpu0 gpu1 <<< "${GPU_PAIR}"
gpu_is_idle "${gpu0}" || fail "GPU ${gpu0} is not idle"
gpu_is_idle "${gpu1}" || fail "GPU ${gpu1} is not idle"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${CACHE_BASE}" "${EVAL_TMPDIR}" "${INPUT_DIR}" "${SAMPLE_DIR}" "${COMBINED_DIR}" "${SUMMARY_DIR}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.started"

export TMPDIR="${EVAL_TMPDIR}"
export TMP="${EVAL_TMPDIR}"
export TEMP="${EVAL_TMPDIR}"
export XDG_CACHE_HOME="${CACHE_BASE}/xdg"
export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/torchinductor"
export CUDA_CACHE_PATH="${CACHE_BASE}/cuda"
export HF_HOME="${CACHE_BASE}/hf"
export HF_HUB_CACHE="${CACHE_BASE}/hf/hub"
export TRANSFORMERS_CACHE="${CACHE_BASE}/hf/transformers"
export VLLM_CACHE_ROOT="${CACHE_BASE}/vllm"
export NUMBA_CACHE_DIR="${CACHE_BASE}/numba"
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" "${NUMBA_CACHE_DIR}"

sed \
  -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
  -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
  "${FULL_SOURCE}" > "${SRC_LIST_PORTABLE}"
if grep -q '^/mnt/data/' "${SRC_LIST_PORTABLE}"; then
  fail "portable source rewrite left node-local /mnt/data paths in ${SRC_LIST_PORTABLE}"
fi
while IFS= read -r wav_path; do
  [[ -n "${wav_path}" ]] || continue
  require_file "${wav_path}"
done < "${SRC_LIST_PORTABLE}"

"${PYTHON_BIN}" - "${SRC_LIST_PORTABLE}" "${FULL_TARGET}" "${SAMPLE_DIR}" <<'PY'
import sys
from pathlib import Path

source = [x.rstrip("\n") for x in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()]
target = [x.rstrip("\n") for x in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()]
out = Path(sys.argv[3])
if len(source) != len(target):
    raise SystemExit(f"source/target mismatch: {len(source)} vs {len(target)}")
if len(source) != 5:
    raise SystemExit(f"expected 5 ACL dev talks, got {len(source)}")
for i, (src, tgt) in enumerate(zip(source, target)):
    (out / f"sample_{i}.source").write_text(src + "\n", encoding="utf-8")
    (out / f"sample_{i}.target.de").write_text(tgt + "\n", encoding="utf-8")
print(f"[SPLIT] wrote {len(source)} one-sample source/target pairs under {out}")
PY

df -h /mnt/gemini/data1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_prelaunch.csv"

{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "gpu_pair=${GPU_PAIR}"
  echo "model=${MODEL_DE}"
  echo "data_root=${DATA_ROOT}"
  echo "source_list_portable=${SRC_LIST_PORTABLE}"
  echo "raw_glossary=${RAW_GLOSSARY}"
  echo "output_root=${OUT_ROOT}"
  echo "lang=de"
  echo "lm=4"
  echo "mode=one_sample_per_simuleval_invocation"
  echo "runtime_rag=0"
  echo "max_new_tokens=40"
} | tee "${OUT_ROOT}/run_meta.txt"

for idx in 0 1 2 3 4; do
  sample_output_base="${OUT_ROOT}/sample_${idx}/origin_norag"
  sample_setting_dir="${sample_output_base}/${SETTING_REL}"
  sample_log="${LOG_ROOT}/sample_${idx}.generation.log"
  echo "[RUN] sample=${idx} output=${sample_setting_dir}" | tee -a "${LOG_ROOT}/onebyone.progress.log"
  mkdir -p "${EVAL_TMPDIR}/s${idx}"
  TMPDIR="${EVAL_TMPDIR}/s${idx}" \
  TMP="${EVAL_TMPDIR}/s${idx}" \
  TEMP="${EVAL_TMPDIR}/s${idx}" \
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  DATA_ROOT_OVERRIDE="${DATA_ROOT}" \
  CONDA_BASE_OVERRIDE="${CONDA_BASE}" \
  CONDA_ENV_NAME_OVERRIDE="${CONDA_ENV_NAME}" \
  MODEL_NAME_OVERRIDE="${MODEL_DE}" \
  OUTPUT_BASE_OVERRIDE="${sample_output_base}" \
  LANG_CODE_OVERRIDE="de" \
  TARGET_LANG_OVERRIDE="German" \
  LATENCY_MULTIPLIERS_OVERRIDE="4" \
  GLOSSARY_PATHS_OVERRIDE="${RAW_GLOSSARY}" \
  SRC_LIST_OVERRIDE="${SAMPLE_DIR}/sample_${idx}.source" \
  TGT_LIST_OVERRIDE="${SAMPLE_DIR}/sample_${idx}.target.de" \
  RAG_K2_VALUES_OVERRIDE="10" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="1" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.78}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-2}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-16384}" \
  MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-40}" \
  MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_OVERRIDE:-80.0}" \
  KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_OVERRIDE:-60.0}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}" \
  bash "${BASELINE_SCRIPT}" 2>&1 | tee "${sample_log}"

  require_file "${sample_setting_dir}/instances.log"
  line_count="$(wc -l < "${sample_setting_dir}/instances.log")"
  [[ "${line_count}" == "1" ]] || fail "sample ${idx} expected one instance row, got ${line_count}: ${sample_setting_dir}/instances.log"
done

: > "${COMBINED_DIR}/instances.log"
for idx in 0 1 2 3 4; do
  cat "${OUT_ROOT}/sample_${idx}/origin_norag/${SETTING_REL}/instances.log" >> "${COMBINED_DIR}/instances.log"
done
combined_rows="$(wc -l < "${COMBINED_DIR}/instances.log")"
[[ "${combined_rows}" == "5" ]] || fail "combined instances expected 5 rows, got ${combined_rows}"

echo "[RUN] fixed-raw tagged StreamLAAL/TERM_ACC post-eval for onebyone combined instances"
"${PYTHON_BIN}" "${OFFLINE_EVAL_SCRIPT}" \
  --mode acl6060 \
  --instances-log "${COMBINED_DIR}/instances.log" \
  --lang-code de \
  --ref-file "${FULL_REF}" \
  --source-file "${FULL_SRC_TEXT}" \
  --audio-yaml "${AUDIO_YAML}" \
  --glossary-acl6060 "${RAW_GLOSSARY}" \
  --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
  --stream-laal-tool-rel "${STREAM_LAAL_TOOL_REL}" \
  --strip-output-tags none \
  --term-fcr-policy source_ref_negative_sentence \
  --output-tsv "${COMBINED_DIR}/eval_results.tsv" \
  --output-log "${COMBINED_DIR}/eval_results.log" \
  2>&1 | tee "${LOG_ROOT}/posteval.log"

"${PYTHON_BIN}" - "${COMBINED_DIR}/eval_results.tsv" "${SUMMARY_DIR}/summary_de_lm4_onebyone.tsv" "${SUMMARY_DIR}/summary_de_lm4_onebyone.md" "${COMBINED_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

eval_tsv = Path(sys.argv[1])
out_tsv = Path(sys.argv[2])
out_md = Path(sys.argv[3])
combined_dir = Path(sys.argv[4])
rows = list(csv.DictReader(eval_tsv.open("r", encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one eval row, got {len(rows)}")
row = rows[0]
fields = ["method_key", "lang", "lm", "mode", "BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL", "eval_results", "instances_log"]
record = {
    "method_key": "origin_norag",
    "lang": "de",
    "lm": "4",
    "mode": "onebyone",
    "BLEU": row.get("BLEU", ""),
    "StreamLAAL": row.get("StreamLAAL", ""),
    "StreamLAAL_CA": row.get("StreamLAAL_CA", ""),
    "TERM_ACC": row.get("TERM_ACC", ""),
    "TERM_CORRECT": row.get("TERM_CORRECT", ""),
    "TERM_TOTAL": row.get("TERM_TOTAL", ""),
    "eval_results": str(eval_tsv),
    "instances_log": str(combined_dir / "instances.log"),
}
with out_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
out_md.write_text(
    "\n".join([
        "# Tagged ACL origin no-RAG de lm4 one-by-one rerun",
        "",
        "| method | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC | correct/total |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| origin_norag onebyone | {float(record['BLEU']):.4f} | {float(record['StreamLAAL']):.4f} | {float(record['StreamLAAL_CA']):.4f} | {float(record['TERM_ACC']):.4f} | {record['TERM_CORRECT']}/{record['TERM_TOTAL']} |",
        "",
        f"- eval_results: `{eval_tsv}`",
        f"- instances: `{combined_dir / 'instances.log'}`",
    ]) + "\n",
    encoding="utf-8",
)
print(f"[SUMMARY] wrote {out_tsv}")
print(f"[SUMMARY] wrote {out_md}")
PY

date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.success"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_postrun.csv"
echo "[ALL DONE] ${SUMMARY_DIR}/summary_de_lm4_onebyone.md"
