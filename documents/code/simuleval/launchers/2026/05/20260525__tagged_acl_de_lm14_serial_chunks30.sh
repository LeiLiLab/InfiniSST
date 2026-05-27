#!/usr/bin/env bash
set -euo pipefail

# Generic En-De tagged ACL raw serial SimulEval readout for lm=1,4.
# Runs the requested LMs sequentially on one 2-GPU pair.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)_tagacl_de_lm14_serial_chunks30}"
LMS="${LMS:-1 4}"
GPU_PAIR="${GPU_PAIR:-0,1}"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
DENSITY_TAG="${DENSITY_TAG:?DENSITY_TAG is required}"
TRAIN_EVENT_ID="${TRAIN_EVENT_ID:-unknown}"
EMPTY_TERM_MAP_POLICY="${EMPTY_TERM_MAP_POLICY:-omit}"
SYSTEM_PROMPT_STYLE="${SYSTEM_PROMPT_STYLE:-given_chunks}"
STRIP_OUTPUT_TAGS="${STRIP_OUTPUT_TAGS:-term}"
MAX_CACHE_CHUNKS="${MAX_CACHE_CHUNKS:-30}"
KEEP_CACHE_CHUNKS="${KEEP_CACHE_CHUNKS:-30}"

INPUT_DIR="${INPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_de_cap16_vs_denoise_lm14_chunks30.md}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/serial_simuleval}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_de_lm14_serial_chunks30_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_de_lm14_serial_chunks30_${RUN_STAMP}}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_dlm14s}"
INPUT_WORK_DIR="${OUT_ROOT}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_WORK_DIR}/source.portable.list"
SUMMARY_DIR="${OUTPUT_BASE}/__summary__"
MAX_NEW_TOKENS_PER_LM="${MAX_NEW_TOKENS_PER_LM:-40}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_path() {
  [[ -e "$1" ]] || fail "Missing required path: $1"
}

require_file() {
  [[ -s "$1" ]] || fail "Missing/empty required file: $1"
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

wait_pair_idle() {
  local g0 g1
  IFS=',' read -r g0 g1 <<< "${GPU_PAIR}"
  while true; do
    if gpu_is_idle "${g0}" && gpu_is_idle "${g1}"; then
      return 0
    fi
    echo "[WAIT] gpu_pair=${GPU_PAIR} not idle; retry in 30s" >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits >&2 || true
    sleep 30
  done
}

validate_model_dir() {
  require_file "${MODEL_NAME}/config.json"
  require_file "${MODEL_NAME}/generation_config.json"
  require_file "${MODEL_NAME}/model.safetensors.index.json"
  require_file "${MODEL_NAME}/tokenizer_config.json"
  local shard_count
  shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || fail "Expected 15 HF safetensor shards in ${MODEL_NAME}, found ${shard_count}"
}

write_lm_summary() {
  local lm="$1" max_new="$2"
  python - "${OUTPUT_BASE}" "${MODEL_LABEL}" "${lm}" "${max_new}" <<'PY'
import csv
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
model_label = sys.argv[2]
lm = sys.argv[3]
max_new = sys.argv[4]
paths = sorted(output_base.glob(f"de/*_lm{lm}_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one eval_results.tsv for lm={lm}, found {len(paths)}: {paths}")
eval_path = paths[0]
with eval_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {eval_path}, got {len(rows)}")
out_dir = eval_path.parent
inst = out_dir / "instances.log"
strip = out_dir / "instances.strip_term.log"
term_adopt = out_dir / "term_adoption.json"
for p in (inst, strip, term_adopt):
    if not p.is_file() or p.stat().st_size <= 0:
        raise SystemExit(f"missing/empty artifact: {p}")
inst_rows = sum(1 for _ in inst.open("r", encoding="utf-8"))
strip_rows = sum(1 for _ in strip.open("r", encoding="utf-8"))
if inst_rows != 5 or strip_rows != 5:
    raise SystemExit(f"expected 5 rows for lm={lm}, got raw={inst_rows} strip={strip_rows}")
row = rows[0]
summary_dir = output_base / "__summary__"
summary_dir.mkdir(parents=True, exist_ok=True)
record = {
    "method_key": model_label,
    "mode": "serial_simuleval",
    "lang": row.get("lang_code", "de"),
    "lm": lm,
    "max_new_tokens": max_new,
    "BLEU": row.get("BLEU", ""),
    "StreamLAAL": row.get("StreamLAAL", ""),
    "StreamLAAL_CA": row.get("StreamLAAL_CA", ""),
    "TERM_ACC": row.get("TERM_ACC", ""),
    "TERM_CORRECT": row.get("TERM_CORRECT", ""),
    "TERM_TOTAL": row.get("TERM_TOTAL", ""),
    "REAL_TERM_ADOPT": row.get("REAL_TERM_ADOPT", ""),
    "TERM_FCR": row.get("TERM_FCR", ""),
    "instances_log_rows": inst_rows,
    "instances_strip_term_log_rows": strip_rows,
    "eval_results": str(eval_path),
    "instances_log": str(inst),
    "instances_strip_term_log": str(strip),
    "term_adoption": str(term_adopt),
}
out = summary_dir / f"summary_de_lm{lm}.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(record), delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
print(
    "RESULT"
    f"\tmode=serial\tlm={lm}"
    f"\tBLEU={float(record['BLEU']):.4f}"
    f"\tStreamLAAL={float(record['StreamLAAL']):.4f}"
    f"\tStreamLAAL_CA={float(record['StreamLAAL_CA']):.4f}"
    f"\tTERM_ACC={float(record['TERM_ACC']):.4f}"
    f"\tTERM={record['TERM_CORRECT']}/{record['TERM_TOTAL']}"
    f"\teval={eval_path}",
    flush=True,
)
PY
}

merge_summaries() {
  python - "${SUMMARY_DIR}" ${LMS} <<'PY'
import csv
import sys
from pathlib import Path

summary_dir = Path(sys.argv[1])
lms = sys.argv[2:]
rows = []
for lm in lms:
    path = summary_dir / f"summary_de_lm{lm}.tsv"
    if not path.is_file():
        raise SystemExit(f"missing lm summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(data)}")
    rows.extend(data)
out = summary_dir / "summary_de_lm1_lm4.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: int(r["lm"])))
print(f"[SUMMARY] {out}", flush=True)
PY
}

run_one_lm() {
  local lm="$1" max_new lm_log compact_pair
  max_new="$((MAX_NEW_TOKENS_PER_LM * lm))"
  compact_pair="${GPU_PAIR//,/}"
  lm_log="${LOG_ROOT}/lm${lm}"
  mkdir -p "${lm_log}" "${EVAL_TMPDIR}/lm${lm}"
  wait_pair_idle
  echo "[RUN] serial lm=${lm} gpu_pair=${GPU_PAIR} max_new_tokens=${max_new}"
  ROOT_DIR="${ROOT_DIR}" \
  MODE="full" \
  RUN_GRANULARITY="full_corpus" \
  HOLD_JOB_ID=0 \
  INSIDE_HOLD_STEP=1 \
  MAX_PARALLEL_OVERRIDE=1 \
  LANGS_OVERRIDE="de" \
  LMS_OVERRIDE="${lm}" \
  GLOSSARY_KINDS_OVERRIDE="raw" \
  PAPERS_OVERRIDE="2022.acl-long.268 2022.acl-long.367 2022.acl-long.590 2022.acl-long.110 2022.acl-long.117" \
  GPU_PAIRS_CSV_OVERRIDE="${GPU_PAIR}" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  TERM_MAP_FORMAT_OVERRIDE="plain" \
  EMPTY_TERM_MAP_POLICY_OVERRIDE="${EMPTY_TERM_MAP_POLICY}" \
  SYSTEM_PROMPT_STYLE_OVERRIDE="${SYSTEM_PROMPT_STYLE}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  INPUT_ROOT_OVERRIDE="${OUT_ROOT}/__inputs__/${MODEL_LABEL}" \
  LOG_DIR_OVERRIDE="${LOG_ROOT}/${MODEL_LABEL}/lm${lm}" \
  SUMMARY_DIR_OVERRIDE="${SUMMARY_DIR}" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  RAW_GLOSSARY_OVERRIDE="${RAW_GLOSSARY}" \
  GS10K_GLOSSARY_OVERRIDE="${GS10K_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_GLOBAL_OVERRIDE="${RAW_GLOSSARY}" \
  EVAL_GLOSSARY_FOLLOWS_KIND_OVERRIDE=0 \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}/lm${lm}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-1}" \
  DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
  WANDB_LOG_OVERRIDE=0 \
  TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="0.78" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="1.92" \
  RAG_TOP_K_OVERRIDE=10 \
  STRIP_OUTPUT_TAGS_OVERRIDE="${STRIP_OUTPUT_TAGS}" \
  RAG_GPU_OVERRIDE="cuda:1" \
  VLLM_TP_SIZE_OVERRIDE=2 \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-12288}" \
  VLLM_LIMIT_AUDIO_OVERRIDE=128 \
  MAX_CACHE_SECONDS_OVERRIDE=0 \
  KEEP_CACHE_SECONDS_OVERRIDE=0 \
  MAX_CACHE_CHUNKS_OVERRIDE="${MAX_CACHE_CHUNKS}" \
  KEEP_CACHE_CHUNKS_OVERRIDE="${KEEP_CACHE_CHUNKS}" \
  MAX_NEW_TOKENS_OVERRIDE="${max_new}" \
  SRC_LIST_OVERRIDE="${SRC_LIST_PORTABLE}" \
  TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
  SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
  REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
  AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
  bash "${BASE_LAUNCHER}" \
    > "${lm_log}/serial_launcher.out" \
    2> "${lm_log}/serial_launcher.err"
  write_lm_summary "${lm}" "${max_new}" | tee "${lm_log}/result.txt"
}

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
require_path "${BASE_LAUNCHER}"
require_file "${RAW_GLOSSARY}"
require_file "${GS10K_GLOSSARY}"
require_file "${HN1024_CKPT}"
require_file "${NOTES_FILE}"
validate_model_dir
for f in source.list target.list source_text.txt ref.txt audio.yaml; do
  require_file "${INPUT_DIR}/${f}"
done

mkdir -p "${OUT_ROOT}" "${OUTPUT_BASE}" "${SUMMARY_DIR}" "${LOG_ROOT}" "${EVAL_TMPDIR}" "${INPUT_WORK_DIR}"
sed \
  -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
  -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
  "${INPUT_DIR}/source.list" > "${SRC_LIST_PORTABLE}"
if grep -q '^/mnt/data/' "${SRC_LIST_PORTABLE}"; then
  fail "portable source rewrite left node-local /mnt/data paths in ${SRC_LIST_PORTABLE}"
fi
while IFS= read -r wav_path; do
  [[ -n "${wav_path}" ]] || continue
  require_file "${wav_path}"
done < "${SRC_LIST_PORTABLE}"

df -h /mnt/gemini/data1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_prelaunch.csv"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "gpu_pair=${GPU_PAIR}"
  echo "model=${MODEL_NAME}"
  echo "model_label=${MODEL_LABEL}"
  echo "train_event_id=${TRAIN_EVENT_ID}"
  echo "density_tag=${DENSITY_TAG}"
  echo "empty_term_map_policy=${EMPTY_TERM_MAP_POLICY}"
  echo "system_prompt_style=${SYSTEM_PROMPT_STYLE}"
  echo "strip_output_tags=${STRIP_OUTPUT_TAGS}"
  echo "max_cache_chunks=${MAX_CACHE_CHUNKS}"
  echo "keep_cache_chunks=${KEEP_CACHE_CHUNKS}"
  echo "input_dir=${INPUT_DIR}"
  echo "source_list_portable=${SRC_LIST_PORTABLE}"
  echo "raw_glossary=${RAW_GLOSSARY}"
  echo "retriever=${HN1024_CKPT}"
  echo "output_base=${OUTPUT_BASE}"
  echo "lms=${LMS}"
  echo "eval_mode=serial_simuleval"
  echo "talks_per_lm=5"
  echo "tau=0.78"
  echo "lookback_sec=1.92"
  echo "max_new_tokens_policy=${MAX_NEW_TOKENS_PER_LM}*lm"
  echo "vllm_limit_audio=128"
  echo "vllm_max_model_len=12288"
} | tee "${OUT_ROOT}/run_meta.txt"

if [[ "${DRY_RUN_OVERRIDE:-0}" == "1" ]]; then
  echo "[DRY_RUN] validated inputs and wrote run_meta; not launching vLLM"
  exit 0
fi

for lm in ${LMS}; do
  run_one_lm "${lm}"
done

merge_summaries
date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.success"
echo "[ALL DONE] ${SUMMARY_DIR}/summary_de_lm1_lm4.tsv"
