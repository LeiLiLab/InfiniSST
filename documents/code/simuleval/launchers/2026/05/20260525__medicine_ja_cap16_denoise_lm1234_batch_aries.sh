#!/usr/bin/env bash
set -euo pipefail

# Aries same-LM batch-vLLM medicine hardraw readout for JA cap16-denoise SLM.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

LANG_CODE="${LANG_CODE:-ja}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)_medicine_${LANG_CODE}_cap16_denoise_lm1234_batch_aries}"
MODEL_NAME="${MODEL_NAME:-/mnt/data3/jiaxuanluo/slm_local_cache/ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf}"
MODEL_LABEL="${MODEL_LABEL:-${LANG_CODE}_cap16_denoise_ttag_hn1024_tau078_omit_batch_chunks30}"
TRAIN_EVENT_ID="${TRAIN_EVENT_ID:-20260525T1550__speech_llm_train__ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_retry1}"

BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
MEDICINE_PREP_LAUNCHER="${MEDICINE_PREP_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__medicine_hardraw_lm1to4_5samples_hn1024_tau078_new_v9_batch.sh}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__medicine_ja_cap16_denoise_lm1234_batch_aries.md}"

HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
HARD_RAW_GLOSSARY="${HARD_RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/medicine_eval_hard_terms_llm_judge_manual_20260524/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json}"

OUTPUT_BASE="${OUTPUT_BASE:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_${LANG_CODE}_cap16_denoise_lm1234_batch_aries_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_${LANG_CODE}_cap16_denoise_lm1234_batch_aries_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_hardraw_${LANG_CODE}_cap16_denoise_lm1234_batch_aries_${RUN_STAMP%%_*}}"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT:-/tmp/jx_mjden_${RUN_STAMP%%_*}}"

GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-4,5;6,7}"
LMS="${LMS:-1 2 3 4}"
TARGET_SAMPLES="${TARGET_SAMPLES:-404 545006 596001 605000 606}"
MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-2048}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-25}"
POLL_SECS="${POLL_SECS:-30}"

RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD:-0.78}"
RAG_TOP_K="${RAG_TOP_K:-10}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC:-1.92}"
EMPTY_TERM_MAP_POLICY="${EMPTY_TERM_MAP_POLICY:-omit}"
RAG_PROMPT_POLICY="${RAG_PROMPT_POLICY:-given_chunks}"
STRIP_OUTPUT_TAGS="${STRIP_OUTPUT_TAGS:-term_t}"
MAX_CACHE_CHUNKS="${MAX_CACHE_CHUNKS:-30}"
KEEP_CACHE_CHUNKS="${KEEP_CACHE_CHUNKS:-30}"
MAX_NEW_TOKENS_PER_LM="${MAX_NEW_TOKENS_PER_LM:-20}"
VLLM_LIMIT_AUDIO="${VLLM_LIMIT_AUDIO:-128}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_GLOBAL_MERGE="${SKIP_GLOBAL_MERGE:-0}"
SKIP_SUCCESS_MARKER="${SKIP_SUCCESS_MARKER:-0}"

MED_INPUTS="${OUTPUT_BASE}/${LANG_CODE}/__medicine_inputs__/lists"
SUMMARY_DIR="${OUTPUT_BASE}/__summary__"
DENSITY_TAG="${DENSITY_TAG:-medhard5_${LANG_CODE}_cap16denoise_ttag_hn1024_tau0p78_omit_chunks30}"
MAX_NEW_TOKENS_PER_LM="${MAX_NEW_TOKENS_PER_LM:-40}"

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

validate_hf_model() {
  require_file "${MODEL_NAME}/config.json"
  require_file "${MODEL_NAME}/generation_config.json"
  require_file "${MODEL_NAME}/model.safetensors.index.json"
  local shard_count
  shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || fail "Expected 15 HF safetensor shards in ${MODEL_NAME}, found ${shard_count}"
}

prepare_inputs() {
  echo "[PREP] ${LANG_CODE} medicine hardraw five-sample inputs"
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  RUN_STAMP="${RUN_STAMP}_medicine_prep" \
  LANG_CODE_OVERRIDE="${LANG_CODE}" \
  TARGET_SAMPLES_OVERRIDE="${TARGET_SAMPLES}" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  MODEL_LABEL="${MODEL_LABEL}" \
  HARD_RAW_GLOSSARY_OVERRIDE="${HARD_RAW_GLOSSARY}" \
  RUNTIME_GLOSSARY_OVERRIDE="${HARD_RAW_GLOSSARY}" \
  FIXED_RAW_GLOSSARY_OVERRIDE="${HARD_RAW_GLOSSARY}" \
  HN1024_CKPT_OVERRIDE="${HN1024_CKPT}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  LOG_ROOT_OVERRIDE="${LOG_ROOT}/medicine_prepare" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}/medicine_prepare" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}/medicine_prepare" \
  PREP_ONLY_OVERRIDE=1 \
  FORCE_PREPARE_OVERRIDE="${FORCE_PREPARE_OVERRIDE:-0}" \
  bash "${MEDICINE_PREP_LAUNCHER}" \
    > "${LOG_ROOT}/medicine_prepare.out" \
    2> "${LOG_ROOT}/medicine_prepare.err"
}

write_lm_summary() {
  local lm="$1" max_new="$2"
  python3 - "${OUTPUT_BASE}" "${MODEL_LABEL}" "${LANG_CODE}" "${lm}" "${max_new}" <<'PY'
import csv
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
model_label, lang, lm, max_new = sys.argv[2:]
paths = sorted(output_base.glob(f"{lang}/*_lm{lm}_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one eval_results.tsv for lang={lang} lm={lm}, found {len(paths)}: {paths}")
eval_path = paths[0]
with eval_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {eval_path}, got {len(rows)}")
out_dir = eval_path.parent
inst = out_dir / "instances.log"
strip = out_dir / "instances.strip_term.log"
for p in (inst, strip):
    if not p.is_file() or p.stat().st_size <= 0:
        raise SystemExit(f"missing/empty log: {p}")
inst_rows = sum(1 for _ in inst.open("r", encoding="utf-8"))
strip_rows = sum(1 for _ in strip.open("r", encoding="utf-8"))
if inst_rows != 5 or strip_rows != 5:
    raise SystemExit(f"expected 5 rows for lm={lm}, got raw={inst_rows} strip={strip_rows}")
row = rows[0]
summary_dir = output_base / "__summary__"
summary_dir.mkdir(parents=True, exist_ok=True)
record = {
    "dataset": "medicine_hardraw",
    "method_key": model_label,
    "mode": "same_lm_batch",
    "lang": row.get("lang_code", lang),
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
}
out = summary_dir / f"summary_medicine_hardraw_{lang}_lm{lm}.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(record), delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
print(
    "RESULT"
    f"\tdataset=medicine_hardraw\tlang={lang}\tlm={lm}"
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
  python3 - "${OUTPUT_BASE}" "${LANG_CODE}" ${LMS} <<'PY'
import csv
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
lang = sys.argv[2]
lms = sys.argv[3:]
summary_dir = output_base / "__summary__"
rows = []
for lm in lms:
    path = summary_dir / f"summary_medicine_hardraw_{lang}_lm{lm}.tsv"
    if not path.is_file():
        raise SystemExit(f"missing summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(data)}")
    rows.extend(data)
suffix = "_".join(f"lm{lm}" for lm in lms)
out = summary_dir / f"summary_medicine_hardraw_{lang}_{suffix}.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: int(r["lm"])))
print(f"[SUMMARY] {out}", flush=True)
PY
}

run_batch_lm() {
  local lm="$1" pair="$2"
  local max_new="$((MAX_NEW_TOKENS_PER_LM * lm))"
  local lm_log="${LOG_ROOT}/medicine_hardraw_${LANG_CODE}_lm${lm}"
  local eval_glossary="${MED_INPUTS}/hard_medicine_raw__medicine5.json"
  local src_list="${MED_INPUTS}/medicine.source__medicine5_hardraw.txt"
  local tgt_list="${MED_INPUTS}/medicine.target.${LANG_CODE}__medicine5_hardraw.txt"
  local source_text="${MED_INPUTS}/medicine.source_text.en__medicine5_hardraw.txt"
  local ref_file="${MED_INPUTS}/medicine.ref.${LANG_CODE}__medicine5_hardraw.txt"
  local audio_yaml="${MED_INPUTS}/medicine.audio__medicine5_hardraw.yaml"
  for p in "${eval_glossary}" "${src_list}" "${tgt_list}" "${source_text}" "${ref_file}" "${audio_yaml}"; do
    require_file "${p}"
  done

  mkdir -p "${lm_log}" "${EVAL_TMPDIR_ROOT}/lm${lm}"
  wait_pair_idle "${pair}"
  clean_shm
  echo "[RUN] lang=${LANG_CODE} medicine_hardraw lm=${lm} gpu_pair=${pair} max_new_tokens=${max_new}"
  RUN_TAG_OVERRIDE="${RUN_STAMP}_medicine_hardraw_${LANG_CODE}_lm${lm}" \
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  LANG_CODE_OVERRIDE="${LANG_CODE}" \
  LMS_OVERRIDE="${lm}" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${pair}" \
  VLLM_TP_SIZE_OVERRIDE=2 \
  MAX_NUM_SEQS_OVERRIDE=5 \
  SCHEDULER_BATCH_SIZE_OVERRIDE=5 \
  SCHEDULE_MODE_OVERRIDE=round_robin \
  VLLM_ENFORCE_EAGER_OVERRIDE=1 \
  VLLM_ENABLE_PREFIX_CACHING=1 \
  VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO}" \
  SAFETENSORS_LOAD_STRATEGY_OVERRIDE=lazy \
  MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN}" \
  MAX_CACHE_SECONDS_OVERRIDE=0 \
  KEEP_CACHE_SECONDS_OVERRIDE=0 \
  MAX_CACHE_CHUNKS_OVERRIDE="${MAX_CACHE_CHUNKS}" \
  KEEP_CACHE_CHUNKS_OVERRIDE="${KEEP_CACHE_CHUNKS}" \
  MIN_CACHE_CHUNKS_OVERRIDE=1 \
  MAX_NEW_TOKENS_OVERRIDE="${max_new}" \
  MAX_NEW_TOKENS_POLICY_OVERRIDE=fixed \
  TEMPERATURE_OVERRIDE=0.6 \
  TOP_P_OVERRIDE=0.95 \
  TOP_K_DECODE_OVERRIDE=20 \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${VLLM_GPU_MEMORY_UTILIZATION}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
  VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  SRC_LIST_OVERRIDE="${src_list}" \
  TGT_LIST_OVERRIDE="${tgt_list}" \
  SOURCE_TEXT_FILE_OVERRIDE="${source_text}" \
  REF_FILE_OVERRIDE="${ref_file}" \
  AUDIO_YAML_OVERRIDE="${audio_yaml}" \
  GLOSSARY_PATH_OVERRIDE="${HARD_RAW_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${eval_glossary}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
  GLOSSARY_TAG_OVERRIDE="hard_medicine_raw__medicine5" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_DEVICE_OVERRIDE="cuda:0" \
  RAG_BATCH_RETRIEVAL_OVERRIDE=1 \
  INDEX_BUILD_DEVICE_OVERRIDE="cuda:0" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}/medicine_hardraw" \
  TERM_MAP_FORMAT_OVERRIDE=plain \
  EMPTY_TERM_MAP_POLICY_OVERRIDE="${EMPTY_TERM_MAP_POLICY}" \
  RAG_PROMPT_POLICY_OVERRIDE="${RAG_PROMPT_POLICY}" \
  TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
  STRIP_OUTPUT_TAGS_OVERRIDE="${STRIP_OUTPUT_TAGS}" \
  EVAL_MODE_OVERRIDE=acl6060 \
  WANDB_LOG_OVERRIDE=0 \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  LOG_ROOT_OVERRIDE="${lm_log}/batch" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}/lm${lm}" \
  bash "${BATCH_LAUNCHER}" \
    > "${lm_log}/launcher.out" \
    2> "${lm_log}/launcher.err"
  write_lm_summary "${lm}" "${max_new}" | tee "${lm_log}/result.txt"
}

run_lms_in_waves() {
  IFS=';' read -r -a gpu_pairs <<< "${GPU_PAIRS_CSV}"
  IFS=' ' read -r -a lm_values <<< "${LMS}"
  (( ${#gpu_pairs[@]} >= 1 )) || fail "Need at least one GPU pair"
  local idx=0
  while (( idx < ${#lm_values[@]} )); do
    local pids=()
    local pair_idx=0
    while (( pair_idx < ${#gpu_pairs[@]} && idx < ${#lm_values[@]} )); do
      ( run_batch_lm "${lm_values[$idx]}" "${gpu_pairs[$pair_idx]}" ) &
      pids+=("$!")
      idx=$((idx + 1))
      pair_idx=$((pair_idx + 1))
    done
    local status=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        status=1
      fi
    done
    [[ "${status}" == "0" ]] || fail "A medicine batch wave failed; see ${LOG_ROOT}"
  done
}

require_file "${BATCH_LAUNCHER}"
require_file "${MEDICINE_PREP_LAUNCHER}"
require_file "${NOTES_FILE}"
require_file "${HN1024_CKPT}"
require_file "${HARD_RAW_GLOSSARY}"
validate_hf_model

mkdir -p "${OUTPUT_BASE}" "${SUMMARY_DIR}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${EVAL_TMPDIR_ROOT}"
df -h /mnt/gemini/data1 /mnt/data3 /dev/shm || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_prelaunch.csv"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "lang=${LANG_CODE}"
  echo "gpu_pairs=${GPU_PAIRS_CSV}"
  echo "lms=${LMS}"
  echo "model=${MODEL_NAME}"
  echo "model_label=${MODEL_LABEL}"
  echo "train_event_id=${TRAIN_EVENT_ID}"
  echo "retriever=${HN1024_CKPT}"
  echo "medicine_glossary=${HARD_RAW_GLOSSARY}"
  echo "output_base=${OUTPUT_BASE}"
  echo "density_tag=${DENSITY_TAG}"
  echo "max_new_tokens_policy=${MAX_NEW_TOKENS_PER_LM}*lm"
  echo "empty_term_map_policy=${EMPTY_TERM_MAP_POLICY}"
  echo "rag_prompt_policy=${RAG_PROMPT_POLICY}"
  echo "strip_output_tags=${STRIP_OUTPUT_TAGS}"
  echo "max_cache_chunks=${MAX_CACHE_CHUNKS}"
  echo "keep_cache_chunks=${KEEP_CACHE_CHUNKS}"
  echo "vllm_limit_audio=${VLLM_LIMIT_AUDIO}"
  echo "vllm_max_model_len=${VLLM_MAX_MODEL_LEN}"
  echo "skip_prepare=${SKIP_PREPARE}"
  echo "skip_global_merge=${SKIP_GLOBAL_MERGE}"
  echo "skip_success_marker=${SKIP_SUCCESS_MARKER}"
} | tee "${OUTPUT_BASE}/run_meta.txt"

if [[ "${SKIP_PREPARE}" == "1" ]]; then
  echo "[PREP] skip existing medicine inputs"
else
  prepare_inputs
fi
run_lms_in_waves
if [[ "${SKIP_GLOBAL_MERGE}" == "1" ]]; then
  echo "[SUMMARY] skip global merge for partial LMS=${LMS}"
else
  merge_summaries
fi

if [[ "${SKIP_SUCCESS_MARKER}" == "1" ]]; then
  echo "[PARTIAL DONE] LMS=${LMS}; global success marker skipped"
else
  date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_BASE}/.success"
  echo "[ALL DONE] ${SUMMARY_DIR}/summary_medicine_hardraw_${LANG_CODE}_$(printf '%s\n' ${LMS} | awk '{printf "%slm%s", (NR==1?"":"_"), $1}').tsv"
fi
