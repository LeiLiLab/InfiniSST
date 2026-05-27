#!/usr/bin/env bash
set -euo pipefail

# Serial SimulEval readout: medicine hardraw En-De, cap16-denoise tagged-term SLM,
# lm=1, cache chunks 30/30, empty term maps omitted, max_new_tokens=40.
# Default placement is two GPUs: vLLM TP=2 on cuda:0,1 and MaxSim shared on
# cuda:1. Set RAG_GPU= to let eval_density_unified.sh auto-place RAG on a third
# visible GPU when running with three GPUs.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260525T185309_medicine_de_cap16_denoise_lm1_serial_cache30_max40_taurus}"
MODEL_NAME="${MODEL_NAME:-/mnt/data1/jiaxuanluo/slm_local_cache/de_tagged_acl_20260525/cap16_denoise_ttag/v0-20260525-203735-hf}"
MODEL_LABEL="${MODEL_LABEL:-de_cap16_denoise_ttag_hn1024_tau078_omit_serial_medicine_lm1_cache30_max40}"
TRAIN_EVENT_ID="${TRAIN_EVENT_ID:-20260525T1236__speech_llm_train__de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6}"
DATA_PREP_EVENT_ID="${DATA_PREP_EVENT_ID:-20260525T1225__data_prepare__de_cap16_denoise_budget_ttag}"

EVAL_SCRIPT="${EVAL_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__medicine_de_cap16_denoise_lm1_serial_cache30_max40_taurus.md}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
MED_INPUTS="${MED_INPUTS:-/mnt/gemini/data1/jiaxuanluo/de_cap16_denoise_medicine_acl_batch_taurus_20260525T165125_de_cap16_denoise_med_acl_batch_taurus/medicine_hardraw_de_cap16_denoise_ttag_hn1024_tau078_batch_chunks30/de/__medicine_inputs__/lists}"
HARD_RAW_GLOSSARY="${HARD_RAW_GLOSSARY:-${MED_INPUTS}/hard_medicine_raw__medicine5.json}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_de_cap16_denoise_lm1_serial_cache30_max40_taurus_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_de_cap16_denoise_lm1_serial_cache30_max40_taurus_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/medicine_hardraw_de_cap16_denoise_lm1_serial_cache30_max40}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_md1s40}"

GPU_SET="${GPU_SET:-2,3}"
MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-2048}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-25}"

LANG_CODE="de"
LM="1"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD:-0.78}"
RAG_TOP_K="${RAG_TOP_K:-10}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC:-1.92}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-40}"
MAX_CACHE_CHUNKS="${MAX_CACHE_CHUNKS:-30}"
KEEP_CACHE_CHUNKS="${KEEP_CACHE_CHUNKS:-30}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
RAG_GPU="${RAG_GPU-cuda:1}"
VLLM_LIMIT_AUDIO="${VLLM_LIMIT_AUDIO:-128}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.72}"

DENSITY_TAG="${DENSITY_TAG:-medhard5_cap16denoise_ttag_hn1024_tau0p78_omit_serial_chunks30_max40}"
GLOSSARY_TAG="hard_medicine_raw__medicine5"
SETTING_DIR="${OUTPUT_BASE}/${LANG_CODE}/d${DENSITY_TAG}_lm${LM}_k${RAG_TOP_K}_th${RAG_SCORE_THRESHOLD}_g${GLOSSARY_TAG}"

SRC_LIST="${MED_INPUTS}/medicine.source__medicine5_hardraw.txt"
TGT_LIST="${MED_INPUTS}/medicine.target.de__medicine5_hardraw.txt"
SOURCE_TEXT="${MED_INPUTS}/medicine.source_text.en__medicine5_hardraw.txt"
REF_FILE="${MED_INPUTS}/medicine.ref.de__medicine5_hardraw.txt"
AUDIO_YAML="${MED_INPUTS}/medicine.audio__medicine5_hardraw.yaml"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
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
  (( mem <= MAX_IDLE_GPU_MEM_MB && util <= MAX_IDLE_GPU_UTIL ))
}

check_gpus_idle() {
  local gpu
  IFS=',' read -r -a gpus <<< "${GPU_SET}"
  for gpu in "${gpus[@]}"; do
    gpu_is_idle "${gpu}" || fail "GPU ${gpu} is not idle"
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

write_summary() {
  local summary_dir="${OUTPUT_BASE}/__summary__"
  mkdir -p "${summary_dir}"
  python3 - \
    "${SETTING_DIR}/eval_results.tsv" \
    "${summary_dir}/summary_medicine_hardraw_de_lm1_serial.tsv" \
    "${summary_dir}/summary_medicine_hardraw_de_lm1_serial.md" \
    "${SETTING_DIR}" \
    "${MODEL_LABEL}" \
    "${MAX_NEW_TOKENS}" \
    "${MAX_CACHE_CHUNKS}" \
    "${KEEP_CACHE_CHUNKS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

eval_tsv = Path(sys.argv[1])
out_tsv = Path(sys.argv[2])
out_md = Path(sys.argv[3])
setting_dir = Path(sys.argv[4])
model_label = sys.argv[5]
max_new_tokens = sys.argv[6]
max_cache_chunks = sys.argv[7]
keep_cache_chunks = sys.argv[8]

rows = list(csv.DictReader(eval_tsv.open("r", encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {eval_tsv}, got {len(rows)}")
row = rows[0]
inst = setting_dir / "instances.log"
strip = setting_dir / "instances.strip_term.log"
if not inst.is_file() or not strip.is_file():
    raise SystemExit("missing instances.log or instances.strip_term.log")
inst_rows = sum(1 for _ in inst.open("r", encoding="utf-8"))
strip_rows = [json.loads(x) for x in strip.open("r", encoding="utf-8")]
if inst_rows != 5 or len(strip_rows) != 5:
    raise SystemExit(f"expected 5 rows, got raw={inst_rows} strip={len(strip_rows)}")

fields = [
    "dataset", "method_key", "mode", "lang", "lm", "max_new_tokens",
    "effective_max_cache_chunks", "effective_keep_cache_chunks",
    "BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT",
    "TERM_TOTAL", "REAL_TERM_ADOPT", "TERM_FCR", "instances_log_rows",
    "instances_strip_term_log_rows", "eval_results", "instances_log",
    "instances_strip_term_log",
]
record = {
    "dataset": "medicine_hardraw",
    "method_key": model_label,
    "mode": "serial_simuleval",
    "lang": "de",
    "lm": "1",
    "max_new_tokens": max_new_tokens,
    "effective_max_cache_chunks": max_cache_chunks,
    "effective_keep_cache_chunks": keep_cache_chunks,
    "BLEU": row.get("BLEU", ""),
    "StreamLAAL": row.get("StreamLAAL", ""),
    "StreamLAAL_CA": row.get("StreamLAAL_CA", ""),
    "TERM_ACC": row.get("TERM_ACC", ""),
    "TERM_CORRECT": row.get("TERM_CORRECT", ""),
    "TERM_TOTAL": row.get("TERM_TOTAL", ""),
    "REAL_TERM_ADOPT": row.get("REAL_TERM_ADOPT", ""),
    "TERM_FCR": row.get("TERM_FCR", ""),
    "instances_log_rows": str(inst_rows),
    "instances_strip_term_log_rows": str(len(strip_rows)),
    "eval_results": str(eval_tsv),
    "instances_log": str(inst),
    "instances_strip_term_log": str(strip),
}
with out_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
out_md.write_text(
    "\n".join([
        "# Medicine hardraw de serial lm1: cap16-denoise ttag",
        "",
        f"- method: `{model_label}`",
        f"- max_new_tokens: `{max_new_tokens}`",
        f"- cache chunks: `{max_cache_chunks}/{keep_cache_chunks}`",
        "",
        "| method | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC | correct/total |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| serial | {float(record['BLEU']):.4f} | {float(record['StreamLAAL']):.4f} | {float(record['StreamLAAL_CA']):.4f} | {float(record['TERM_ACC']):.4f} | {record['TERM_CORRECT']}/{record['TERM_TOTAL']} |",
        "",
        f"- eval_results: `{eval_tsv}`",
        f"- instances: `{inst}`",
        f"- strip: `{strip}`",
    ]) + "\n",
    encoding="utf-8",
)
print(f"[SUMMARY] wrote {out_tsv}")
print(f"[SUMMARY] wrote {out_md}")
PY
}

main() {
  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
  validate_model_dir
  for p in \
    "${EVAL_SCRIPT}" \
    "${NOTES_FILE}" \
    "${HN1024_CKPT}" \
    "${HARD_RAW_GLOSSARY}" \
    "${SRC_LIST}" \
    "${TGT_LIST}" \
    "${SOURCE_TEXT}" \
    "${REF_FILE}" \
    "${AUDIO_YAML}"; do
    require_file "${p}"
  done
  check_gpus_idle

  mkdir -p \
    "${OUT_ROOT}" \
    "${OUTPUT_BASE}/__summary__" \
    "${LOG_ROOT}" \
    "${INDEX_CACHE_DIR}" \
    "${EVAL_TMPDIR}" \
    "${EVAL_TMPDIR}/torchinductor" \
    "${EVAL_TMPDIR}/triton" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_runs" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_cache" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_data" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_artifacts"

  while IFS= read -r wav_path; do
    [[ -n "${wav_path}" ]] || continue
    require_file "${wav_path}"
  done < "${SRC_LIST}"

  df -h /mnt/gemini/data1 | tee "${LOG_ROOT}/df_prelaunch.txt"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_prelaunch.csv"
  {
    echo "run_stamp=${RUN_STAMP}"
    echo "host=$(hostname -s)"
    echo "driver=serial_simuleval"
    echo "batch_eval=false"
    echo "model=${MODEL_NAME}"
    echo "model_label=${MODEL_LABEL}"
    echo "train_event_id=${TRAIN_EVENT_ID}"
    echo "data_prep_event_id=${DATA_PREP_EVENT_ID}"
    echo "retriever=${HN1024_CKPT}"
    echo "glossary=${HARD_RAW_GLOSSARY}"
    echo "output_base=${OUTPUT_BASE}"
    echo "setting_dir=${SETTING_DIR}"
    echo "gpu_set=${GPU_SET}"
    echo "rag_gpu=${RAG_GPU:-auto}"
    echo "lm=${LM}"
    echo "tau=${RAG_SCORE_THRESHOLD}"
    echo "lookback_sec=${RAG_TIMELINE_LOOKBACK_SEC}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "empty_term_map_policy=omit"
    echo "system_prompt_style=given_chunks"
    echo "strip_output_tags=term_t"
    echo "effective_cache_chunks=${MAX_CACHE_CHUNKS}/${KEEP_CACHE_CHUNKS}"
  } | tee "${OUT_ROOT}/run_meta.txt"

  export TMPDIR="${EVAL_TMPDIR}"
  export TMP="${EVAL_TMPDIR}"
  export TEMP="${EVAL_TMPDIR}"
  export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
  export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
  export VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}"
  export VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN}"
  export VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO}"
  export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/mnt/taurus/home/jiaxuanluo/.config/wandb}"
  export WANDB_DIR="${WANDB_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_runs}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_cache}"
  export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_data}"
  export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_artifacts}"

  ROOT_DIR="${ROOT_DIR}" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  EVAL_MODE_OVERRIDE="acl6060" \
  LANG_CODE_OVERRIDE="${LANG_CODE}" \
  GLOSSARY_PATH_OVERRIDE="${HARD_RAW_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${HARD_RAW_GLOSSARY}" \
  SRC_LIST_OVERRIDE="${SRC_LIST}" \
  TGT_LIST_OVERRIDE="${TGT_LIST}" \
  REF_FILE_OVERRIDE="${REF_FILE}" \
  SOURCE_TEXT_FILE_OVERRIDE="${SOURCE_TEXT}" \
  AUDIO_YAML_OVERRIDE="${AUDIO_YAML}" \
  LATENCY_MULTIPLIER_OVERRIDE="${LM}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
  RAG_STREAMING_MODE_OVERRIDE="timeline" \
  TERM_MAP_FORMAT_OVERRIDE="plain" \
  TERM_FCR_POLICY="term_map_source_ref_negative_sentence" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_SET}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE}" \
  RAG_GPU_OVERRIDE="${RAG_GPU}" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION}" \
  MAX_CACHE_SECONDS_OVERRIDE=0 \
  KEEP_CACHE_SECONDS_OVERRIDE=0 \
  MAX_CACHE_CHUNKS_OVERRIDE="${MAX_CACHE_CHUNKS}" \
  KEEP_CACHE_CHUNKS_OVERRIDE="${KEEP_CACHE_CHUNKS}" \
  VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN}" \
  MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS}" \
  STRIP_OUTPUT_TAGS_OVERRIDE="term_t" \
  EMPTY_TERM_MAP_POLICY_OVERRIDE="omit" \
  SYSTEM_PROMPT_STYLE_OVERRIDE="given_chunks" \
  DENSITY_TAG="${DENSITY_TAG}" \
  VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME="VLLM_OBJ_${RUN_STAMP}_${RANDOM}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-1}" \
  bash "${EVAL_SCRIPT}" \
    > "${LOG_ROOT}/serial_launcher.out" \
    2> "${LOG_ROOT}/serial_launcher.err"

  require_file "${SETTING_DIR}/eval_results.tsv"
  require_file "${SETTING_DIR}/instances.log"
  require_file "${SETTING_DIR}/instances.strip_term.log"
  require_file "${SETTING_DIR}/term_adoption.json"
  write_summary
  date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.success"
  echo "[ALL DONE] ${OUTPUT_BASE}/__summary__/summary_medicine_hardraw_de_lm1_serial.md"
}

main "$@"
