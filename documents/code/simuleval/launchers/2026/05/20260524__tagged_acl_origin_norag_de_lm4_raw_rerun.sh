#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
RUN_STAMP="${RUN_STAMP:-20260524T160830_tagacl_origin_norag_de_lm4_raw_rerun}"
GPU_PAIR="${GPU_PAIR:-4,5}"

BASELINE_SCRIPT="${BASELINE_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/rank16/baseline/bypass_simuleval_rank32_iter_0000452_hf_baseline_no_rag_sweep.sh}"
OFFLINE_EVAL_SCRIPT="${OFFLINE_EVAL_SCRIPT:-${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py}"
WANDB_LOGGER="${WANDB_LOGGER:-${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__tagged_acl_origin_norag_de_lm4_raw_rerun.md}"

DATA_ROOT="${DATA_ROOT:-/mnt/taurus/data/siqiouyang/datasets/acl6060}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
MODEL_DE="${MODEL_DE:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4}"
FBK_FAIRSEQ_ROOT="${FBK_FAIRSEQ_ROOT:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
STREAM_LAAL_TOOL_REL="${STREAM_LAAL_TOOL_REL:-examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm4_raw_rerun_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_origin_norag_de_lm4_raw_rerun_${RUN_STAMP}}"
BASELINE_OUTPUT_BASE="${OUT_ROOT}/origin_norag"
SUMMARY_DIR="${OUT_ROOT}/__summary__"
SCAN_BASE="${OUT_ROOT}/wandb_scan"
INPUT_DIR="${OUT_ROOT}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_DIR}/dev.source.portable"
TGT_LIST_DE="${DATA_ROOT}/dev.target.de"

EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_tacl_de4}"
CONDA_BASE="${CONDA_BASE:-/mnt/taurus/home/jiaxuanluo/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-spaCyEnv}"
DEFAULT_CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
if [[ -n "${CONDA_PREFIX_OVERRIDE:-}" ]]; then
  CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE}"
elif [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV_NAME}" ]]; then
  CONDA_PREFIX="${CONDA_PREFIX}"
else
  CONDA_PREFIX="${DEFAULT_CONDA_PREFIX}"
fi
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"

GLOSSARY_TAG="acl6060_tagged_gt_raw_min_norm2"
SETTING_DIR="${BASELINE_OUTPUT_BASE}/de/gigaspeech-de-s_origin-bsz4_g${GLOSSARY_TAG}_cs3.84_hs0.48_lm4_k210_k110_th0p0"
SCAN_DIR="${SCAN_BASE}/de/dtagacl_origin_norag_lm4_k0_th0_g${GLOSSARY_TAG}"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_path() {
  [[ -e "$1" ]] || fail "missing required path: $1"
}

setup_env() {
  mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${SUMMARY_DIR}" "${SCAN_DIR}" "${INPUT_DIR}" "${EVAL_TMPDIR}" "${EVAL_TMPDIR}/torchinductor" "${EVAL_TMPDIR}/triton"
  export TMPDIR="${EVAL_TMPDIR}"
  export TMP="${EVAL_TMPDIR}"
  export TEMP="${EVAL_TMPDIR}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${EVAL_TMPDIR}/torchinductor}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${EVAL_TMPDIR}/triton}"
  export VLLM_WORKER_MULTIPROC_METHOD="spawn"
  export VLLM_NO_USAGE_STATS="1"
  export VLLM_PORT="${VLLM_PORT:-28654}"
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
  echo "[RUN] no-RAG baseline generation: de lm4 raw -> ${SETTING_DIR}"
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  DATA_ROOT_OVERRIDE="${DATA_ROOT}" \
  CONDA_BASE_OVERRIDE="${CONDA_BASE}" \
  CONDA_ENV_NAME_OVERRIDE="${CONDA_ENV_NAME}" \
  MODEL_NAME_OVERRIDE="${MODEL_DE}" \
  OUTPUT_BASE_OVERRIDE="${BASELINE_OUTPUT_BASE}" \
  LANG_CODE_OVERRIDE="de" \
  TARGET_LANG_OVERRIDE="German" \
  LATENCY_MULTIPLIERS_OVERRIDE="4" \
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

  cp "${SETTING_DIR}/eval_results.tsv" "${SCAN_DIR}/eval_results.tsv"
  cp "${SETTING_DIR}/instances.log" "${SCAN_DIR}/instances.log"
}

write_summary() {
  "${PYTHON_BIN}" - "${SETTING_DIR}/eval_results.tsv" "${SUMMARY_DIR}/tagged_acl_origin_norag_de_lm4_raw_rerun.tsv" "${SUMMARY_DIR}/tagged_acl_origin_norag_de_lm4_raw_rerun.md" "${SETTING_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

eval_tsv = Path(sys.argv[1])
out_tsv = Path(sys.argv[2])
out_md = Path(sys.argv[3])
setting_dir = Path(sys.argv[4])
rows = list(csv.DictReader(eval_tsv.open("r", encoding="utf-8"), delimiter="\t"))
if not rows:
    raise SystemExit(f"empty eval TSV: {eval_tsv}")
row = rows[-1]
fields = ["lang", "lm", "glossary", "BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL", "REAL_TERM_ADOPT", "TERM_FCR", "eval_results", "instances_log"]
record = {
    "lang": "de",
    "lm": "4",
    "glossary": "tagged_raw",
    "BLEU": row.get("BLEU", ""),
    "StreamLAAL": row.get("StreamLAAL", ""),
    "StreamLAAL_CA": row.get("StreamLAAL_CA", ""),
    "TERM_ACC": row.get("TERM_ACC", ""),
    "TERM_CORRECT": row.get("TERM_CORRECT", ""),
    "TERM_TOTAL": row.get("TERM_TOTAL", ""),
    "REAL_TERM_ADOPT": row.get("REAL_TERM_ADOPT", ""),
    "TERM_FCR": row.get("TERM_FCR", ""),
    "eval_results": str(eval_tsv),
    "instances_log": str(setting_dir / "instances.log"),
}
with out_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(record)

def fmt_pct(x):
    if x in {"", "NA", None}:
        return ""
    return f"{float(x) * 100:.2f}"

md = [
    "# Tagged ACL origin no-RAG de lm4 raw rerun",
    "",
    "| lang | lm | glossary | BLEU | TERM_ACC | StreamLAAL | StreamLAAL_CA | correct/total |",
    "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    f"| de | 4 | tagged_raw | {float(record['BLEU']):.4f} | {fmt_pct(record['TERM_ACC'])} | {float(record['StreamLAAL']):.4f} | {float(record['StreamLAAL_CA']):.4f} | {record['TERM_CORRECT']}/{record['TERM_TOTAL']} |",
    "",
    f"- eval_results: `{eval_tsv}`",
    f"- instances: `{setting_dir / 'instances.log'}`",
]
out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"[SUMMARY] wrote {out_tsv}")
print(f"[SUMMARY] wrote {out_md}")
PY
}

log_wandb() {
  echo "[RUN] logging rerun metrics to W&B"
  HOME="${HOME:-/home/jiaxuanluo}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${HOME:-/home/jiaxuanluo}/.config/wandb}" \
  "${PYTHON_BIN}" "${WANDB_LOGGER}" \
    --project simuleval_eval \
    --run-name "origin_norag__tagged_acl__de__lm4__raw__rerun" \
    --experiment-family "tagged_acl_origin_norag" \
    --data-tag "tagged_acl_raw_de" \
    --task-tag eval \
    --notes-file "${NOTES_FILE}" \
    --extra-tags "variant:origin_norag_de_raw_lm4" "method:norag_origin" "glossary:raw" "lang:de" "compute:aries45290" \
    --density "tagacl_origin_norag" \
    --rag-top-k "0" \
    --rag-score-threshold "0" \
    --output-base "${SCAN_BASE}" \
    --lang-code de \
    --latency-multipliers 4 \
    --glossary-tag "${GLOSSARY_TAG}" \
    --model-name "${MODEL_DE}" \
    --verdict "Tagged ACL raw En-De lm4 InfiniSST/no-TM-SFT no-RAG baseline rerun with fixed raw denominator." \
    2>&1 | tee "${LOG_ROOT}/${RUN_STAMP}.wandb.log"
  sed -n 's/.*simuleval_eval\/\\([A-Za-z0-9_-]*\\).*/\\1/p' "${LOG_ROOT}/${RUN_STAMP}.wandb.log" | tail -n 1 > "${SUMMARY_DIR}/wandb_run_id.txt" || true
}

main() {
  setup_env
  for p in "${BASELINE_SCRIPT}" "${OFFLINE_EVAL_SCRIPT}" "${WANDB_LOGGER}" "${NOTES_FILE}" "${RAW_GLOSSARY}" "${MODEL_DE}" "${DATA_ROOT}/dev.source" "${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.de.txt" "${DATA_ROOT}/dev/text/txt/ACL.6060.dev.en-xx.en.txt"; do
    require_path "${p}"
  done
  prepare_inputs
  run_generation
  run_posteval
  write_summary
  log_wandb
  echo "[ALL DONE] ${SUMMARY_DIR}/tagged_acl_origin_norag_de_lm4_raw_rerun.md"
}

main "$@"
