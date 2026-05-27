#!/usr/bin/env bash
set -euo pipefail

# Aries all-lm same-lm batch readout for Japanese tagged ACL raw RASST.
# Each latency multiplier runs the five ACL talks in one vLLM process on a
# dedicated 2-GPU NVLink pair.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)_tagacl_tmsftgtwrap_ja_hn1024_tau078_lm1to4_aries}"
LMS="${LMS:-1 2 3 4}"
GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-0,1;2,3;4,5;6,7}"

MODEL_LABEL="${MODEL_LABEL:-tmsft_gttermwrap_exact_ja_r32a32_ep4_hn1024_tau078_batch_max80}"
MODEL_NAME="${MODEL_NAME:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_tmsft_gttermwrap_exact_ja_r32a32_ep4_taurus8/keep1.0_r32/v0-20260525-104902-hf}"
INPUT_DIR="${INPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_20260524T0933_tagacl_newv9_hn1024_tau078_raw_ja_taurus8/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/ja/all}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_tmsft_gttermwrap_exact_ja_hn1024_tau078_batch_aries_lm1to4.md}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/batched_vllm}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_tmsft_gttermwrap_exact_ja_hn1024_tau078_batch_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_tmsft_gttermwrap_exact_ja_hn1024_tau078_batch_${RUN_STAMP}}"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT:-/tmp/jx_jatw_a}"
INPUT_WORK_DIR="${OUT_ROOT}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_WORK_DIR}/source.portable.list"
SUMMARY_DIR="${OUTPUT_BASE}/__summary__"

MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-2048}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-25}"
POLL_SECS="${POLL_SECS:-30}"

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
  local pair="$1" g0 g1
  IFS=',' read -r g0 g1 <<< "${pair}"
  while true; do
    if gpu_is_idle "${g0}" && gpu_is_idle "${g1}"; then
      return 0
    fi
    echo "[WAIT] gpu_pair=${pair} not idle; retry in ${POLL_SECS}s" >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits >&2 || true
    sleep "${POLL_SECS}"
  done
}

clean_shm() {
  local me
  me="$(id -un)"
  find /dev/shm -maxdepth 1 -user "${me}" -mmin +1 \
    \( -name 'psm_*' -o -name 'loky-*' -o -name 'torch_*' -o -name 'vllm_*' \) \
    -delete 2>/dev/null || true
}

validate_wandb_tags() {
  local tag lm pair
  local tags=(
    "family:tagged_acl_tmsft_gtwrap_ja_hn1024_tau078"
    "task:eval"
    "data:tagacl_raw_ja"
  )
  IFS=';' read -r -a pairs <<< "${GPU_PAIRS_CSV}"
  for pair in "${pairs[@]}"; do
    tags+=("compute:aries_gpu${pair//,/}")
  done
  for lm in ${LMS}; do
    tags+=("variant:tmsftgtwrap_t078_ja_raw_lm${lm}")
  done
  for tag in "${tags[@]}"; do
    (( ${#tag} >= 1 && ${#tag} <= 64 )) || fail "W&B tag length invalid (${#tag}): ${tag}"
  done
}

write_lm_summary() {
  local lm="$1"
  python - "${OUTPUT_BASE}" "${MODEL_LABEL}" "${lm}" <<'PY'
import csv
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
model_label = sys.argv[2]
lm = sys.argv[3]
paths = sorted(output_base.glob(f"ja/**_lm{lm}_*/eval_results.tsv"))
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
if not inst.is_file() or inst.stat().st_size <= 0:
    raise SystemExit(f"missing/empty instances.log for lm={lm}: {inst}")
if not strip.is_file() or strip.stat().st_size <= 0:
    raise SystemExit(f"missing/empty instances.strip_term.log for lm={lm}: {strip}")
inst_rows = sum(1 for _ in inst.open("r", encoding="utf-8"))
strip_rows = sum(1 for _ in strip.open("r", encoding="utf-8"))
if inst_rows != 5 or strip_rows != 5:
    raise SystemExit(f"expected 5 instance rows for lm={lm}, got raw={inst_rows} strip={strip_rows}")
row = rows[0]
summary_dir = output_base / "__summary__"
summary_dir.mkdir(parents=True, exist_ok=True)
out = summary_dir / f"summary_ja_lm{lm}.tsv"
record = {
    "method_key": model_label,
    "lang": row.get("lang_code", "ja"),
    "lm": lm,
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
}
fields = list(record)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
print(
    "RESULT"
    f"\tlm={lm}"
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
    path = summary_dir / f"summary_ja_lm{lm}.tsv"
    if not path.is_file():
        raise SystemExit(f"missing lm summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(data)}")
    rows.extend(data)
fields = list(rows[0])
out = summary_dir / "summary_ja_lm1to4.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: int(r["lm"])))
print(f"[SUMMARY] {out}", flush=True)
PY
}

run_one_lm() {
  local lm="$1" pair="$2" compact_pair lm_log
  compact_pair="${pair//,/}"
  lm_log="${LOG_ROOT}/lm${lm}"
  mkdir -p "${lm_log}" "${EVAL_TMPDIR_ROOT}/lm${lm}"
  wait_pair_idle "${pair}"
  clean_shm
  echo "[RUN] lm=${lm} gpu_pair=${pair}"
  RUN_TAG_OVERRIDE="${RUN_STAMP}_ja_lm${lm}" \
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  LANG_CODE_OVERRIDE="ja" \
  LMS_OVERRIDE="${lm}" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${pair}" \
  VLLM_TP_SIZE_OVERRIDE=2 \
  MAX_NUM_SEQS_OVERRIDE=5 \
  SCHEDULER_BATCH_SIZE_OVERRIDE=5 \
  SCHEDULE_MODE_OVERRIDE=round_robin \
  VLLM_ENFORCE_EAGER_OVERRIDE=1 \
  VLLM_ENABLE_PREFIX_CACHING=1 \
  VLLM_LIMIT_AUDIO_OVERRIDE=128 \
  MAX_MODEL_LEN_OVERRIDE=12288 \
  VLLM_MAX_MODEL_LEN_OVERRIDE=12288 \
  MAX_CACHE_SECONDS_OVERRIDE=80 \
  KEEP_CACHE_SECONDS_OVERRIDE=60 \
  MIN_CACHE_CHUNKS_OVERRIDE=1 \
  MAX_NEW_TOKENS_OVERRIDE=80 \
  MAX_NEW_TOKENS_POLICY_OVERRIDE=fixed \
  TEMPERATURE_OVERRIDE=0.6 \
  TOP_P_OVERRIDE=0.95 \
  TOP_K_DECODE_OVERRIDE=20 \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
  VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  SRC_LIST_OVERRIDE="${SRC_LIST_PORTABLE}" \
  TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
  SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
  REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
  AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
  GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  DENSITY_TAG_OVERRIDE="tagacl_bv1_tmsft_gtwrap_hn1024_tau078" \
  GLOSSARY_TAG_OVERRIDE="acl6060_tagged_gt_raw_min_norm2" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_SCORE_THRESHOLD_OVERRIDE=0.78 \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE=1.92 \
  RAG_TOP_K_OVERRIDE=10 \
  RAG_DEVICE_OVERRIDE="cuda:0" \
  RAG_BATCH_RETRIEVAL_OVERRIDE=1 \
  INDEX_BUILD_DEVICE_OVERRIDE="cuda:0" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
  TERM_MAP_FORMAT_OVERRIDE=plain \
  TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
  STRIP_OUTPUT_TAGS_OVERRIDE=term \
  WANDB_LOG_OVERRIDE="${WANDB_LOG_OVERRIDE:-1}" \
  WANDB_RUN_PREFIX_OVERRIDE="tmsftgtwrap_hn1024_t078" \
  WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_tmsft_gtwrap_ja_hn1024_tau078" \
  WANDB_VARIANT_PREFIX_OVERRIDE="tmsftgtwrap_t078" \
  WANDB_COMPUTE_TAG_OVERRIDE="compute:aries_gpu${compact_pair}" \
  WANDB_RUNTIME_GLOSSARY_LABEL_OVERRIDE="raw" \
  WANDB_DATA_TAG_OVERRIDE="tagacl_raw_ja" \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  LOG_ROOT_OVERRIDE="${lm_log}/batch" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}/lm${lm}" \
  bash "${BATCH_LAUNCHER}" \
    > "${lm_log}/launcher.out" \
    2> "${lm_log}/launcher.err"
  write_lm_summary "${lm}" | tee "${lm_log}/result.txt"
}

require_file "${BATCH_LAUNCHER}"
require_file "${MODEL_NAME}/config.json"
require_file "${MODEL_NAME}/generation_config.json"
require_file "${MODEL_NAME}/model.safetensors.index.json"
require_file "${RAW_GLOSSARY}"
require_file "${HN1024_CKPT}"
require_file "${NOTES_FILE}"
for f in source.list target.list source_text.txt ref.txt audio.yaml; do
  require_file "${INPUT_DIR}/${f}"
done
shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
[[ "${shard_count}" == "15" ]] || fail "Expected 15 HF safetensor shards in ${MODEL_NAME}, found ${shard_count}"
validate_wandb_tags

IFS=';' read -r -a gpu_pairs <<< "${GPU_PAIRS_CSV}"
IFS=' ' read -r -a lm_values <<< "${LMS}"
(( ${#gpu_pairs[@]} >= ${#lm_values[@]} )) || fail "Need ${#lm_values[@]} GPU pairs, got ${#gpu_pairs[@]}"

mkdir -p "${OUT_ROOT}" "${SUMMARY_DIR}" "${LOG_ROOT}" "${EVAL_TMPDIR_ROOT}" "${INPUT_WORK_DIR}"
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
  echo "gpu_pairs=${GPU_PAIRS_CSV}"
  echo "model=${MODEL_NAME}"
  echo "model_label=${MODEL_LABEL}"
  echo "input_dir=${INPUT_DIR}"
  echo "source_list_portable=${SRC_LIST_PORTABLE}"
  echo "raw_glossary=${RAW_GLOSSARY}"
  echo "retriever=${HN1024_CKPT}"
  echo "output_base=${OUTPUT_BASE}"
  echo "lms=${LMS}"
  echo "eval_mode=same_lm_batch_v1_parallel"
  echo "talks_per_lm=5"
  echo "tau=0.78"
  echo "lookback_sec=1.92"
  echo "max_new_tokens=80"
  echo "vllm_limit_audio=128"
  echo "vllm_max_model_len=12288"
  echo "strip_output_tags=term"
} | tee "${OUT_ROOT}/run_meta.txt"

if [[ "${DRY_RUN_OVERRIDE:-0}" == "1" ]]; then
  echo "[DRY_RUN] validated inputs and wrote run_meta; not launching vLLM"
  exit 0
fi

pids=()
for idx in "${!lm_values[@]}"; do
  lm="${lm_values[$idx]}"
  pair="${gpu_pairs[$idx]}"
  ( run_one_lm "${lm}" "${pair}" ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
(( status == 0 )) || fail "At least one lm batch eval failed; inspect ${LOG_ROOT}"
merge_summaries | tee "${SUMMARY_DIR}/summary_path.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_postrun.csv"
echo "[ALL DONE] output_base=${OUTPUT_BASE}"
