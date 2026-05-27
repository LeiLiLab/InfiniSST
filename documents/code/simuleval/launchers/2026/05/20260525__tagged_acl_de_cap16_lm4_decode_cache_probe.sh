#!/usr/bin/env bash
set -euo pipefail

# One-config En-De tagged-ACL RASST probe for the de retriever-cap16 SLM.
# It waits for the requested GPU pair, runs one batched vLLM eval, then writes a
# compact summary with metric and length-ratio diagnostics.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
BATCH_LAUNCHER="${BATCH_LAUNCHER_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
RUN_STAMP="${RUN_STAMP_OVERRIDE:-$(date -u +%Y%m%dT%H%M%S)_de_lm4_decode_cache_probe}"
CONFIG_ID="${CONFIG_ID_OVERRIDE:-mt80_c80k60_ch16k8}"
GPU_PAIR="${GPU_PAIR_OVERRIDE:-0,1}"
LM_VALUE="${LM_VALUE_OVERRIDE:-4}"

MODEL_NAME="${MODEL_NAME_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_retriever_cap16_exactboundary_r32a32_ep4_taurus8/keep1.0_r32/v1-20260525-141908-hf}"
INPUT_DIR="${INPUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all}"
SOURCE_LIST_IN="${SOURCE_LIST_OVERRIDE:-${INPUT_DIR}/source.list}"
TARGET_LIST_IN="${TARGET_LIST_OVERRIDE:-${INPUT_DIR}/target.list}"
RAW_GLOSSARY="${RAW_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
HN1024_CKPT="${HN1024_CKPT_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
NOTES_FILE="${NOTES_FILE_OVERRIDE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_de_cap16_lm4_decode_cache_probe.md}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/batched_vllm}"

MAX_NEW_TOKENS_VALUE="${MAX_NEW_TOKENS_VALUE_OVERRIDE:-80}"
MAX_CACHE_SECONDS_VALUE="${MAX_CACHE_SECONDS_VALUE_OVERRIDE:-80}"
KEEP_CACHE_SECONDS_VALUE="${KEEP_CACHE_SECONDS_VALUE_OVERRIDE:-60}"
MAX_CACHE_CHUNKS_VALUE="${MAX_CACHE_CHUNKS_VALUE_OVERRIDE:-16}"
KEEP_CACHE_CHUNKS_VALUE="${KEEP_CACHE_CHUNKS_VALUE_OVERRIDE:-8}"
TEMPERATURE_VALUE="${TEMPERATURE_VALUE_OVERRIDE:-0.6}"
TOP_P_VALUE="${TOP_P_VALUE_OVERRIDE:-0.95}"
TOP_K_DECODE_VALUE="${TOP_K_DECODE_VALUE_OVERRIDE:-20}"
RAG_TOP_K_VALUE="${RAG_TOP_K_VALUE_OVERRIDE:-10}"
RAG_SCORE_THRESHOLD_VALUE="${RAG_SCORE_THRESHOLD_VALUE_OVERRIDE:-0.78}"
EMPTY_TERM_MAP_POLICY_VALUE="${EMPTY_TERM_MAP_POLICY_VALUE_OVERRIDE:-omit}"
EXPECTED_INSTANCE_ROWS="${EXPECTED_INSTANCE_ROWS_OVERRIDE:-5}"

MODEL_LABEL="${MODEL_LABEL_OVERRIDE:-de_retriever_cap16_lm4_${CONFIG_ID}}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_de_cap16_lm4_decode_cache_probe_${RUN_STAMP}_${CONFIG_ID}}"
OUTPUT_BASE="${OUTPUT_BASE_OVERRIDE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_de_cap16_lm4_decode_cache_probe_${RUN_STAMP}_${CONFIG_ID}}"
TMP_SLUG="$(printf '%s' "${CONFIG_ID}" | sha1sum | cut -c1-8)"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT_OVERRIDE:-/tmp/jxd4_${TMP_SLUG}}"
INPUT_WORK_DIR="${OUT_ROOT}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_WORK_DIR}/source.portable.list"

MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB_OVERRIDE:-2048}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL_OVERRIDE:-25}"
POLL_SECS="${POLL_SECS_OVERRIDE:-30}"

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
  (( mem <= MAX_IDLE_GPU_MEM_MB && util <= MAX_IDLE_GPU_UTIL ))
}

wait_pair_idle() {
  local g0 g1
  IFS=',' read -r g0 g1 <<< "${GPU_PAIR}"
  while true; do
    if gpu_is_idle "${g0}" && gpu_is_idle "${g1}"; then
      return 0
    fi
    echo "[WAIT] config=${CONFIG_ID} gpu_pair=${GPU_PAIR} not idle; retry in ${POLL_SECS}s" >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits >&2 || true
    sleep "${POLL_SECS}"
  done
}

write_summary() {
  "${PYTHON_BIN}" - "${OUTPUT_BASE}" "${CONFIG_ID}" "${LM_VALUE}" "${MAX_NEW_TOKENS_VALUE}" "${MAX_CACHE_SECONDS_VALUE}" "${KEEP_CACHE_SECONDS_VALUE}" "${MAX_CACHE_CHUNKS_VALUE}" "${KEEP_CACHE_CHUNKS_VALUE}" "${TEMPERATURE_VALUE}" "${TOP_P_VALUE}" "${TOP_K_DECODE_VALUE}" "${EXPECTED_INSTANCE_ROWS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
config_id, lm, max_new, max_cache_s, keep_cache_s, max_cache_c, keep_cache_c, temp, top_p, top_k, expected_rows_s = sys.argv[2:]
expected_rows = int(expected_rows_s)
paths = sorted(output_base.glob(f"de/*_lm{lm}_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one lm{lm} eval_results.tsv, found {len(paths)}: {paths}")
eval_path = paths[0]
with eval_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one eval row, got {len(rows)} in {eval_path}")
out_dir = eval_path.parent
inst = out_dir / "instances.log"
strip = out_dir / "instances.strip_term.log"
if not inst.is_file() or not strip.is_file():
    raise SystemExit("missing instances.log or instances.strip_term.log")
inst_rows = sum(1 for _ in inst.open("r", encoding="utf-8"))
strip_rows = []
for line in strip.open("r", encoding="utf-8"):
    row = json.loads(line)
    hyp = row.get("prediction", "")
    ref = row.get("reference", "")
    strip_rows.append(
        {
            "index": row.get("index"),
            "hyp_words": len(hyp.split()),
            "ref_words": len(ref.split()),
            "hyp_chars": len(hyp),
            "ref_chars": len(ref),
            "word_ratio": len(hyp.split()) / max(1, len(ref.split())),
            "char_ratio": len(hyp) / max(1, len(ref)),
        }
    )
if inst_rows != expected_rows or len(strip_rows) != expected_rows:
    raise SystemExit(
        f"expected {expected_rows} rows, got raw={inst_rows} strip={len(strip_rows)}"
    )
metric = rows[0]
summary_dir = output_base / "__summary__"
summary_dir.mkdir(parents=True, exist_ok=True)
summary_tsv = summary_dir / f"summary_de_lm{lm}.tsv"
fields = [
    "config_id", "lm", "max_new_tokens", "max_cache_seconds", "keep_cache_seconds",
    "max_cache_chunks", "keep_cache_chunks", "temperature", "top_p", "top_k_decode",
    "BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL",
    "REAL_TERM_ADOPT", "TERM_FCR", "sum_word_ratio", "sum_char_ratio",
    "max_word_ratio", "max_char_ratio", "instances_log_rows",
    "instances_strip_term_log_rows", "eval_results", "instances_log",
    "instances_strip_term_log",
]
total_hw = sum(x["hyp_words"] for x in strip_rows)
total_rw = sum(x["ref_words"] for x in strip_rows)
total_hc = sum(x["hyp_chars"] for x in strip_rows)
total_rc = sum(x["ref_chars"] for x in strip_rows)
record = {
    "config_id": config_id,
    "lm": lm,
    "max_new_tokens": max_new,
    "max_cache_seconds": max_cache_s,
    "keep_cache_seconds": keep_cache_s,
    "max_cache_chunks": max_cache_c,
    "keep_cache_chunks": keep_cache_c,
    "temperature": temp,
    "top_p": top_p,
    "top_k_decode": top_k,
    "BLEU": metric.get("BLEU", ""),
    "StreamLAAL": metric.get("StreamLAAL", ""),
    "StreamLAAL_CA": metric.get("StreamLAAL_CA", ""),
    "TERM_ACC": metric.get("TERM_ACC", ""),
    "TERM_CORRECT": metric.get("TERM_CORRECT", ""),
    "TERM_TOTAL": metric.get("TERM_TOTAL", ""),
    "REAL_TERM_ADOPT": metric.get("REAL_TERM_ADOPT", ""),
    "TERM_FCR": metric.get("TERM_FCR", ""),
    "sum_word_ratio": f"{total_hw / max(1, total_rw):.6f}",
    "sum_char_ratio": f"{total_hc / max(1, total_rc):.6f}",
    "max_word_ratio": f"{max(x['word_ratio'] for x in strip_rows):.6f}",
    "max_char_ratio": f"{max(x['char_ratio'] for x in strip_rows):.6f}",
    "instances_log_rows": str(inst_rows),
    "instances_strip_term_log_rows": str(len(strip_rows)),
    "eval_results": str(eval_path),
    "instances_log": str(inst),
    "instances_strip_term_log": str(strip),
}
with summary_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
(summary_dir / f"length_ratios_de_lm{lm}.json").write_text(
    json.dumps(strip_rows, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(
    "RESULT"
    f"\tconfig={config_id}"
    f"\tBLEU={float(record['BLEU']):.4f}"
    f"\tTERM_ACC={float(record['TERM_ACC']):.4f}"
    f"\tword_ratio={record['sum_word_ratio']}"
    f"\tchar_ratio={record['sum_char_ratio']}"
    f"\tsummary={summary_tsv}",
    flush=True,
)
PY
}

for p in \
  "${PYTHON_BIN}" \
  "${BATCH_LAUNCHER}" \
  "${MODEL_NAME}/config.json" \
  "${MODEL_NAME}/generation_config.json" \
  "${MODEL_NAME}/model.safetensors.index.json" \
  "${RAW_GLOSSARY}" \
  "${HN1024_CKPT}" \
  "${NOTES_FILE}" \
  "${INPUT_DIR}/source.list" \
  "${INPUT_DIR}/target.list" \
  "${INPUT_DIR}/source_text.txt" \
  "${INPUT_DIR}/ref.txt" \
  "${INPUT_DIR}/audio.yaml"; do
  require_file "${p}"
done
require_file "${SOURCE_LIST_IN}"
require_file "${TARGET_LIST_IN}"

mkdir -p "${OUT_ROOT}" "${OUTPUT_BASE}/__summary__" "${LOG_ROOT}" "${EVAL_TMPDIR_ROOT}" "${INPUT_WORK_DIR}"
sed \
  -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
  -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
  "${SOURCE_LIST_IN}" > "${SRC_LIST_PORTABLE}"
while IFS= read -r wav_path; do
  [[ -n "${wav_path}" ]] || continue
  require_file "${wav_path}"
done < "${SRC_LIST_PORTABLE}"

{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "config_id=${CONFIG_ID}"
  echo "gpu_pair=${GPU_PAIR}"
  echo "model=${MODEL_NAME}"
  echo "input_dir=${INPUT_DIR}"
  echo "source_list_in=${SOURCE_LIST_IN}"
  echo "target_list_in=${TARGET_LIST_IN}"
  echo "output_base=${OUTPUT_BASE}"
  echo "lm=${LM_VALUE}"
  echo "expected_instance_rows=${EXPECTED_INSTANCE_ROWS}"
  echo "max_new_tokens=${MAX_NEW_TOKENS_VALUE}"
  echo "max_cache_seconds=${MAX_CACHE_SECONDS_VALUE}"
  echo "keep_cache_seconds=${KEEP_CACHE_SECONDS_VALUE}"
  echo "max_cache_chunks=${MAX_CACHE_CHUNKS_VALUE}"
  echo "keep_cache_chunks=${KEEP_CACHE_CHUNKS_VALUE}"
  echo "temperature=${TEMPERATURE_VALUE}"
  echo "top_p=${TOP_P_VALUE}"
  echo "top_k_decode=${TOP_K_DECODE_VALUE}"
  echo "rag_top_k=${RAG_TOP_K_VALUE}"
  echo "tau=${RAG_SCORE_THRESHOLD_VALUE}"
  echo "empty_term_map_policy=${EMPTY_TERM_MAP_POLICY_VALUE}"
} | tee "${OUT_ROOT}/run_meta.txt"

wait_pair_idle

ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
LANG_CODE_OVERRIDE="de" \
LMS_OVERRIDE="${LM_VALUE}" \
RUN_TAG_OVERRIDE="${RUN_STAMP}_${CONFIG_ID}_de_lm${LM_VALUE}" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
VLLM_TP_SIZE_OVERRIDE=2 \
MAX_NUM_SEQS_OVERRIDE=5 \
SCHEDULER_BATCH_SIZE_OVERRIDE=5 \
SCHEDULE_MODE_OVERRIDE=round_robin \
VLLM_ENFORCE_EAGER_OVERRIDE=1 \
VLLM_ENABLE_PREFIX_CACHING=1 \
VLLM_LIMIT_AUDIO_OVERRIDE=128 \
SAFETENSORS_LOAD_STRATEGY_OVERRIDE=lazy \
MAX_MODEL_LEN_OVERRIDE=12288 \
VLLM_MAX_MODEL_LEN_OVERRIDE=12288 \
MAX_CACHE_SECONDS_OVERRIDE="${MAX_CACHE_SECONDS_VALUE}" \
KEEP_CACHE_SECONDS_OVERRIDE="${KEEP_CACHE_SECONDS_VALUE}" \
MAX_CACHE_CHUNKS_OVERRIDE="${MAX_CACHE_CHUNKS_VALUE}" \
KEEP_CACHE_CHUNKS_OVERRIDE="${KEEP_CACHE_CHUNKS_VALUE}" \
MIN_CACHE_CHUNKS_OVERRIDE=1 \
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_VALUE}" \
MAX_NEW_TOKENS_POLICY_OVERRIDE=fixed \
TEMPERATURE_OVERRIDE="${TEMPERATURE_VALUE}" \
TOP_P_OVERRIDE="${TOP_P_VALUE}" \
TOP_K_DECODE_OVERRIDE="${TOP_K_DECODE_VALUE}" \
GPU_MEMORY_UTILIZATION_OVERRIDE=0.72 \
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
SRC_LIST_OVERRIDE="${SRC_LIST_PORTABLE}" \
TGT_LIST_OVERRIDE="${TARGET_LIST_IN}" \
SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
DENSITY_TAG_OVERRIDE="tagacl_bv1_decap16_${CONFIG_ID}" \
GLOSSARY_TAG_OVERRIDE="acl6060_tagged_gt_raw_min_norm2" \
RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD_VALUE}" \
RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE=1.92 \
RAG_TOP_K_OVERRIDE="${RAG_TOP_K_VALUE}" \
RAG_DEVICE_OVERRIDE="cuda:0" \
RAG_BATCH_RETRIEVAL_OVERRIDE=1 \
INDEX_BUILD_DEVICE_OVERRIDE="cuda:0" \
INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
TERM_MAP_FORMAT_OVERRIDE=plain \
EMPTY_TERM_MAP_POLICY_OVERRIDE="${EMPTY_TERM_MAP_POLICY_VALUE}" \
TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
STRIP_OUTPUT_TAGS_OVERRIDE=term \
WANDB_LOG_OVERRIDE=0 \
NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
LOG_ROOT_OVERRIDE="${LOG_ROOT}/batch" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}" \
bash "${BATCH_LAUNCHER}" \
  > "${LOG_ROOT}/launcher.out" \
  2> "${LOG_ROOT}/launcher.err"

write_summary | tee "${OUT_ROOT}/summary_path.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_postrun.csv"
echo "[ALL DONE] output_base=${OUTPUT_BASE}"
