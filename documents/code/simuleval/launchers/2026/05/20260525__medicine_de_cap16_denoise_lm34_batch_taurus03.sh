#!/usr/bin/env bash
set -euo pipefail

# Taurus-only medicine hardraw En-De lm=3,4 same-lm batch-vLLM readout.
# Uses GPU pairs 0,1 and 2,3 and writes to an independent output root.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260525T170456_medicine_de_cap16_denoise_lm34_batch_taurus03}"
MODEL_NAME="${MODEL_NAME:-/mnt/data1/jiaxuanluo/slm_local_cache/de_tagged_acl_20260525/cap16_denoise_ttag/v0-20260525-203735-hf}"
MODEL_LABEL="${MODEL_LABEL:-de_cap16_denoise_ttag_hn1024_tau078_omit_batch_chunks30_lm34_taurus03}"
BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__medicine_de_cap16_denoise_lm34_batch_taurus03.md}"

HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
HARD_RAW_GLOSSARY="${HARD_RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/medicine_eval_hard_terms_llm_judge_manual_20260524/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json}"
MED_INPUTS="${MED_INPUTS:-/mnt/gemini/data1/jiaxuanluo/de_cap16_denoise_medicine_acl_batch_taurus_20260525T165125_de_cap16_denoise_med_acl_batch_taurus/medicine_hardraw_de_cap16_denoise_ttag_hn1024_tau078_batch_chunks30/de/__medicine_inputs__/lists}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/de_cap16_denoise_medicine_acl_batch_taurus_20260525T165125/medicine_hardraw}"

OUTPUT_BASE="${OUTPUT_BASE:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_de_cap16_denoise_lm34_batch_taurus03_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_de_cap16_denoise_lm34_batch_taurus03_${RUN_STAMP}}"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT:-/tmp/jx_mdl34_${RUN_STAMP%%_*}}"

GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-0,1;2,3}"
DENSITY_TAG="${DENSITY_TAG:-medhard5_cap16denoise_ttag_hn1024_tau0p78_omit_chunks30_lm34_taurus03}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD:-0.78}"
RAG_TOP_K="${RAG_TOP_K:-10}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC:-1.92}"
EMPTY_TERM_MAP_POLICY="${EMPTY_TERM_MAP_POLICY:-omit}"
RAG_PROMPT_POLICY="${RAG_PROMPT_POLICY:-given_chunks}"
STRIP_OUTPUT_TAGS="${STRIP_OUTPUT_TAGS:-term_t}"
MAX_CACHE_CHUNKS="${MAX_CACHE_CHUNKS:-30}"
KEEP_CACHE_CHUNKS="${KEEP_CACHE_CHUNKS:-30}"
VLLM_LIMIT_AUDIO="${VLLM_LIMIT_AUDIO:-128}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.72}"
MAX_NEW_TOKENS_PER_LM="${MAX_NEW_TOKENS_PER_LM:-40}"

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
}
out = summary_dir / f"summary_medicine_hardraw_de_lm{lm}.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(record), delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
print(
    "RESULT"
    f"\tdataset=medicine_hardraw\tlm={lm}"
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
  python - "${OUTPUT_BASE}" <<'PY'
import csv
import sys
from pathlib import Path

summary_dir = Path(sys.argv[1]) / "__summary__"
rows = []
for lm in ("3", "4"):
    path = summary_dir / f"summary_medicine_hardraw_de_lm{lm}.tsv"
    if not path.is_file():
        raise SystemExit(f"missing summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(data)}")
    rows.extend(data)
out = summary_dir / "summary_medicine_hardraw_de_lm3_lm4.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: int(r["lm"])))
print(f"[SUMMARY] {out}", flush=True)
PY
}

run_lm() {
  local lm="$1" pair="$2" max_new lm_log
  max_new="$((MAX_NEW_TOKENS_PER_LM * lm))"
  lm_log="${LOG_ROOT}/lm${lm}"
  mkdir -p "${lm_log}" "${EVAL_TMPDIR_ROOT}/lm${lm}"
  wait_pair_idle "${pair}"
  echo "[RUN] lm=${lm} gpu_pair=${pair} max_new_tokens=${max_new}"
  RUN_TAG_OVERRIDE="${RUN_STAMP}_medicine_hardraw_de_lm${lm}" \
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  LANG_CODE_OVERRIDE=de \
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
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
  VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
  VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  SRC_LIST_OVERRIDE="${MED_INPUTS}/medicine.source__medicine5_hardraw.txt" \
  TGT_LIST_OVERRIDE="${MED_INPUTS}/medicine.target.de__medicine5_hardraw.txt" \
  SOURCE_TEXT_FILE_OVERRIDE="${MED_INPUTS}/medicine.source_text.en__medicine5_hardraw.txt" \
  REF_FILE_OVERRIDE="${MED_INPUTS}/medicine.ref.de__medicine5_hardraw.txt" \
  AUDIO_YAML_OVERRIDE="${MED_INPUTS}/medicine.audio__medicine5_hardraw.yaml" \
  GLOSSARY_PATH_OVERRIDE="${HARD_RAW_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${MED_INPUTS}/hard_medicine_raw__medicine5.json" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
  GLOSSARY_TAG_OVERRIDE=hard_medicine_raw__medicine5 \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_DEVICE_OVERRIDE=cuda:0 \
  RAG_BATCH_RETRIEVAL_OVERRIDE=1 \
  INDEX_BUILD_DEVICE_OVERRIDE=cuda:0 \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
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
  bash "${BATCH_LAUNCHER}" > "${lm_log}/launcher.out" 2> "${lm_log}/launcher.err"
  write_lm_summary "${lm}" "${max_new}" | tee "${lm_log}/result.txt"
}

require_file "${BATCH_LAUNCHER}"
require_file "${NOTES_FILE}"
require_file "${MODEL_NAME}/config.json"
require_file "${MODEL_NAME}/generation_config.json"
require_file "${MODEL_NAME}/model.safetensors.index.json"
require_file "${HN1024_CKPT}"
require_file "${HARD_RAW_GLOSSARY}"
for f in \
  medicine.source__medicine5_hardraw.txt \
  medicine.target.de__medicine5_hardraw.txt \
  medicine.source_text.en__medicine5_hardraw.txt \
  medicine.ref.de__medicine5_hardraw.txt \
  medicine.audio__medicine5_hardraw.yaml \
  hard_medicine_raw__medicine5.json; do
  require_file "${MED_INPUTS}/${f}"
done
shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
[[ "${shard_count}" == "15" ]] || fail "Expected 15 HF safetensor shards in ${MODEL_NAME}, found ${shard_count}"

mkdir -p "${OUTPUT_BASE}" "${LOG_ROOT}" "${EVAL_TMPDIR_ROOT}"
df -h /mnt/gemini/data1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_prelaunch.csv"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "gpu_pairs=${GPU_PAIRS_CSV}"
  echo "model=${MODEL_NAME}"
  echo "input_lists=${MED_INPUTS}"
  echo "output_base=${OUTPUT_BASE}"
  echo "index_cache_dir=${INDEX_CACHE_DIR}"
  echo "max_new_tokens_policy=${MAX_NEW_TOKENS_PER_LM}*lm"
  echo "max_cache_chunks=${MAX_CACHE_CHUNKS}"
  echo "keep_cache_chunks=${KEEP_CACHE_CHUNKS}"
  echo "vllm_limit_audio=${VLLM_LIMIT_AUDIO}"
  echo "vllm_max_model_len=${VLLM_MAX_MODEL_LEN}"
} | tee "${OUTPUT_BASE}/run_meta.txt"

IFS=';' read -r -a pairs <<< "${GPU_PAIRS_CSV}"
( run_lm 3 "${pairs[0]}" ) &
pid3="$!"
( run_lm 4 "${pairs[1]}" ) &
pid4="$!"
status=0
wait "${pid3}" || status=1
wait "${pid4}" || status=1
[[ "${status}" == "0" ]] || fail "lm3/lm4 batch failed; see ${LOG_ROOT}"
merge_summaries
date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_BASE}/.success"
echo "[ALL DONE] ${OUTPUT_BASE}/__summary__/summary_medicine_hardraw_de_lm3_lm4.tsv"
