#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
RUN_STAMP="${RUN_STAMP:-20260525T131354_tagacl_origin_norag_de_lm1_serial_taurus07}"
GPU_PAIR="${GPU_PAIR:-0,7}"

BASELINE_SCRIPT="${BASELINE_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/rank16/baseline/bypass_simuleval_rank32_iter_0000452_hf_baseline_no_rag_sweep.sh}"
OFFLINE_EVAL_SCRIPT="${OFFLINE_EVAL_SCRIPT:-${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_origin_norag_de_lm1_serial_taurus07.md}"

DATA_ROOT="${DATA_ROOT:-/mnt/taurus/data/siqiouyang/datasets/acl6060}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
MODEL_DE="${MODEL_DE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
STREAM_LAAL_TOOL_REL="${STREAM_LAAL_TOOL_REL:-examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm1_serial_taurus07_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_origin_norag_de_lm1_serial_taurus07_${RUN_STAMP}}"
BASELINE_OUTPUT_BASE="${OUT_ROOT}/origin_norag"
SUMMARY_DIR="${OUT_ROOT}/__summary__"
INPUT_DIR="${OUT_ROOT}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_DIR}/dev.source.portable"
TGT_LIST_DE="${DATA_ROOT}/dev.target.de"

EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_tacl_de1s}"
CONDA_BASE="${CONDA_BASE:-/mnt/taurus/home/jiaxuanluo/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-spaCyEnv}"
CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-${CONDA_BASE}/envs/${CONDA_ENV_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"

GLOSSARY_TAG="acl6060_tagged_gt_raw_min_norm2"
SETTING_DIR="${BASELINE_OUTPUT_BASE}/de/gigaspeech-de-s_origin-bsz4_g${GLOSSARY_TAG}_cs0.96_hs0.48_lm1_k210_k110_th0p0"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_path() {
  [[ -e "$1" ]] || fail "missing required path: $1"
}

setup_env() {
  mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${SUMMARY_DIR}" "${INPUT_DIR}" "${EVAL_TMPDIR}" \
    "${EVAL_TMPDIR}/torchinductor" "${EVAL_TMPDIR}/triton"
  export TMPDIR="${EVAL_TMPDIR}"
  export TMP="${EVAL_TMPDIR}"
  export TEMP="${EVAL_TMPDIR}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${EVAL_TMPDIR}/torchinductor}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${EVAL_TMPDIR}/triton}"
  export VLLM_WORKER_MULTIPROC_METHOD="spawn"
  export VLLM_NO_USAGE_STATS="1"
  export VLLM_PORT="${VLLM_PORT:-28651}"
  export MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
  export PATH="${CONDA_PREFIX}/bin:${MWERSEGMENTER_ROOT}:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
}

prepare_inputs() {
  require_path "${DATA_ROOT}/dev.source"
  require_path "${TGT_LIST_DE}"
  sed \
    -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
    -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
    "${DATA_ROOT}/dev.source" > "${SRC_LIST_PORTABLE}"

  if grep -q '^/mnt/data/' "${SRC_LIST_PORTABLE}"; then
    fail "portable source rewrite left node-local /mnt/data paths in ${SRC_LIST_PORTABLE}"
  fi
  while IFS= read -r wav_path; do
    [[ -n "${wav_path}" ]] || continue
    require_path "${wav_path}"
  done < "${SRC_LIST_PORTABLE}"
}

run_generation() {
  echo "[RUN] serial no-RAG baseline generation: de lm1 -> ${SETTING_DIR}"
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  DATA_ROOT_OVERRIDE="${DATA_ROOT}" \
  CONDA_BASE_OVERRIDE="${CONDA_BASE}" \
  CONDA_ENV_NAME_OVERRIDE="${CONDA_ENV_NAME}" \
  MODEL_NAME_OVERRIDE="${MODEL_DE}" \
  OUTPUT_BASE_OVERRIDE="${BASELINE_OUTPUT_BASE}" \
  LANG_CODE_OVERRIDE="de" \
  TARGET_LANG_OVERRIDE="German" \
  LATENCY_MULTIPLIERS_OVERRIDE="1" \
  GLOSSARY_PATHS_OVERRIDE="${RAW_GLOSSARY}" \
  SRC_LIST_OVERRIDE="${SRC_LIST_PORTABLE}" \
  TGT_LIST_OVERRIDE="${TGT_LIST_DE}" \
  RAG_K2_VALUES_OVERRIDE="10" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-1}" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.78}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE_OVERRIDE:-2}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-16384}" \
  MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-40}" \
  MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_OVERRIDE:-80.0}" \
  KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_OVERRIDE:-60.0}" \
  VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE:-128}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}" \
  bash "${BASELINE_SCRIPT}" 2>&1 | tee "${LOG_ROOT}/${RUN_STAMP}.generation.log"
}

run_posteval() {
  require_path "${SETTING_DIR}/instances.log"
  echo "[RUN] fixed-raw tagged StreamLAAL/TERM_ACC post-eval"
  "${PYTHON_BIN}" "${OFFLINE_EVAL_SCRIPT}" \
    --mode acl6060 \
    --instances-log "${SETTING_DIR}/instances.log" \
    --lang-code de \
    --ref-file "${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.de.txt" \
    --source-file "${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt" \
    --audio-yaml "${DATA_ROOT}/dev.yaml" \
    --glossary-acl6060 "${RAW_GLOSSARY}" \
    --fbk-fairseq-root "${FBK_FAIRSEQ_ROOT}" \
    --stream-laal-tool-rel "${STREAM_LAAL_TOOL_REL}" \
    --strip-output-tags none \
    --term-fcr-policy source_ref_negative_sentence \
    --output-tsv "${SETTING_DIR}/eval_results.tsv" \
    --output-log "${SETTING_DIR}/eval_results.log" \
    2>&1 | tee "${LOG_ROOT}/${RUN_STAMP}.posteval.log"
}

write_summary() {
  "${PYTHON_BIN}" - "${SETTING_DIR}/eval_results.tsv" "${SUMMARY_DIR}/tagged_acl_origin_norag_de_lm1_serial_taurus07.tsv" "${SETTING_DIR}" "${OUT_ROOT}/runtime_seconds.txt" <<'PY'
import csv
import sys
from pathlib import Path

eval_tsv = Path(sys.argv[1])
out_tsv = Path(sys.argv[2])
setting_dir = Path(sys.argv[3])
runtime_file = Path(sys.argv[4])
rows = list(csv.DictReader(eval_tsv.open("r", encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one eval row in {eval_tsv}, got {len(rows)}")
inst = setting_dir / "instances.log"
inst_rows = sum(1 for line in inst.open("r", encoding="utf-8", errors="replace") if line.strip())
row = rows[0]
fields = [
    "method_key", "lang", "lm", "max_new_tokens", "protocol", "BLEU",
    "StreamLAAL", "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT",
    "TERM_TOTAL", "instances_log_rows", "runtime_seconds",
    "eval_results", "instances_log",
]
record = {
    "method_key": "origin_norag_serial_infinisst",
    "lang": row.get("lang_code", "de"),
    "lm": "1",
    "max_new_tokens": "40",
    "protocol": "serial_simuleval_agent_v4",
    "BLEU": row.get("BLEU", ""),
    "StreamLAAL": row.get("StreamLAAL", ""),
    "StreamLAAL_CA": row.get("StreamLAAL_CA", ""),
    "TERM_ACC": row.get("TERM_ACC", ""),
    "TERM_CORRECT": row.get("TERM_CORRECT", ""),
    "TERM_TOTAL": row.get("TERM_TOTAL", ""),
    "instances_log_rows": str(inst_rows),
    "runtime_seconds": runtime_file.read_text(encoding="utf-8").strip() if runtime_file.exists() else "",
    "eval_results": str(eval_tsv),
    "instances_log": str(inst),
}
out_tsv.parent.mkdir(parents=True, exist_ok=True)
with out_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
print(out_tsv)
PY
}

main() {
  local start end
  start="$(date +%s)"
  setup_env
  for p in "${BASELINE_SCRIPT}" "${OFFLINE_EVAL_SCRIPT}" "${NOTES_FILE}" "${RAW_GLOSSARY}" \
    "${MODEL_DE}" "${DATA_ROOT}/dev.source" "${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.de.txt" \
    "${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt"; do
    require_path "${p}"
  done
  prepare_inputs
  run_generation
  run_posteval
  end="$(date +%s)"
  echo "$((end - start))" > "${OUT_ROOT}/runtime_seconds.txt"
  write_summary
  date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.success"
  echo "[ALL DONE] ${SUMMARY_DIR}/tagged_acl_origin_norag_de_lm1_serial_taurus07.tsv"
}

main "$@"
