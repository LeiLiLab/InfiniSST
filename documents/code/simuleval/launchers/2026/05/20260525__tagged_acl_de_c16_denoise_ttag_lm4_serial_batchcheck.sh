#!/usr/bin/env bash
set -euo pipefail

# En-De tagged ACL raw lm=4 serial SimulEval check for the
# de_c16_denoise_ttag_r32a32_ep1 model. This intentionally avoids the
# same-LM batch-vLLM evaluator.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)_de_c16_denoise_ttag_lm4_serial_batchcheck}"
GPU_PAIR="${GPU_PAIR:-0,1}"
RAG_GPU="${RAG_GPU:-}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
MAX_CACHE_CHUNKS="${MAX_CACHE_CHUNKS:-8}"
KEEP_CACHE_CHUNKS="${KEEP_CACHE_CHUNKS:-4}"

MODEL_LABEL="${MODEL_LABEL:-de_c16_denoise_ttag_r32a32_ep1_hn1024_tau078_omit_serial_lm4_max80}"
MODEL_NAME="${MODEL_NAME:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6/keep1.0_r32/v0-20260525-203735-hf}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_de_c16_denoise_ttag_lm4_serial_batchcheck.md}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"

INPUT_DIR="${INPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all}"
OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_de_c16_denoise_ttag_lm4_serial_batchcheck_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_de_c16_denoise_ttag_lm4_serial_batchcheck_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/serial_simuleval}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_d16s4}"
INPUT_WORK_DIR="${OUT_ROOT}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_WORK_DIR}/source.portable.list"

DENSITY_TAG="${DENSITY_TAG:-tagacl_serial_dec16den_ttag_hn1024_tau078_omit}"
GLOSSARY_TAG="acl6060_tagged_gt_raw_min_norm2"
SETTING_DIR="${OUTPUT_BASE}/de/d${DENSITY_TAG}_lm4_k10_th0.78_g${GLOSSARY_TAG}"

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

check_gpu_pair() {
  local gpu
  IFS=',' read -r -a gpus <<< "${GPU_PAIR}"
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

preflight() {
  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
  for p in \
    "${ROOT_DIR}" \
    "${BASE_LAUNCHER}" \
    "${HN1024_CKPT}" \
    "${RAW_GLOSSARY}" \
    "${GS10K_GLOSSARY}" \
    "${NOTES_FILE}" \
    "${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh" \
    "${ROOT_DIR}/documents/code/offline_sst_eval/offline_streamlaal_eval.py" \
    "${PYTHON_BIN}" \
    "${INPUT_DIR}/source.list" \
    "${INPUT_DIR}/target.list" \
    "${INPUT_DIR}/source_text.txt" \
    "${INPUT_DIR}/ref.txt" \
    "${INPUT_DIR}/audio.yaml"; do
    require_path "${p}"
  done
  validate_model_dir
  check_gpu_pair
}

write_summary() {
  local summary_dir="${OUTPUT_BASE}/__summary__"
  mkdir -p "${summary_dir}"
  "${PYTHON_BIN}" - \
    "${SETTING_DIR}/eval_results.tsv" \
    "${summary_dir}/summary_de_lm4_serial.tsv" \
    "${summary_dir}/summary_de_lm4_serial.md" \
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
    "method_key", "lang", "lm", "mode", "max_new_tokens",
    "effective_max_cache_chunks", "effective_keep_cache_chunks",
    "BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT",
    "TERM_TOTAL", "REAL_TERM_ADOPT", "TERM_FCR", "instances_log_rows",
    "instances_strip_term_log_rows", "eval_results", "instances_log",
    "instances_strip_term_log",
]
record = {
    "method_key": model_label,
    "lang": "de",
    "lm": "4",
    "mode": "serial_simuleval",
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
        "# Tagged ACL de raw serial lm4: de c16 denoise ttag",
        "",
        f"- method: `{model_label}`",
        f"- max_new_tokens: `{max_new_tokens}`",
        f"- cache chunks: `{max_cache_chunks}/{keep_cache_chunks}`",
        "",
        "| method | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC | correct/total |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| denoise-ttag serial | {float(record['BLEU']):.4f} | {float(record['StreamLAAL']):.4f} | {float(record['StreamLAAL_CA']):.4f} | {float(record['TERM_ACC']):.4f} | {record['TERM_CORRECT']}/{record['TERM_TOTAL']} |",
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
  preflight
  mkdir -p \
    "${OUT_ROOT}" \
    "${OUTPUT_BASE}/__summary__" \
    "${LOG_ROOT}" \
    "${INDEX_CACHE_DIR}" \
    "${EVAL_TMPDIR}" \
    "${EVAL_TMPDIR}/torchinductor" \
    "${EVAL_TMPDIR}/triton" \
    "${INPUT_WORK_DIR}" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_runs" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_cache" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_data" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_artifacts"

  sed \
    -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
    -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
    "${INPUT_DIR}/source.list" > "${SRC_LIST_PORTABLE}"
  while IFS= read -r wav_path; do
    [[ -n "${wav_path}" ]] || continue
    require_file "${wav_path}"
  done < "${SRC_LIST_PORTABLE}"

  {
    echo "run_stamp=${RUN_STAMP}"
    echo "host=$(hostname -s)"
    echo "driver=serial_simuleval"
    echo "batch_eval=false"
    echo "model=${MODEL_NAME}"
    echo "retriever=${HN1024_CKPT}"
    echo "raw_glossary=${RAW_GLOSSARY}"
    echo "output_base=${OUTPUT_BASE}"
    echo "setting_dir=${SETTING_DIR}"
    echo "gpu_pair=${GPU_PAIR}"
    echo "rag_gpu=${RAG_GPU:-auto}"
    echo "lm=4"
    echo "tau=0.78"
    echo "lookback_sec=1.92"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "empty_term_map_policy=omit"
    echo "strip_output_tags=term_t"
    echo "effective_cache_chunks=${MAX_CACHE_CHUNKS}/${KEEP_CACHE_CHUNKS}"
  } | tee "${OUT_ROOT}/run_meta.txt"

  export TMPDIR="${EVAL_TMPDIR}"
  export TMP="${EVAL_TMPDIR}"
  export TEMP="${EVAL_TMPDIR}"
  export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
  export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
  export VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}"
  export VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-12288}"
  export VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO_OVERRIDE:-128}"
  export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/mnt/taurus/home/jiaxuanluo/.config/wandb}"
  export WANDB_DIR="${WANDB_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_runs}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_cache}"
  export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_data}"
  export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_artifacts}"

  ROOT_DIR="${ROOT_DIR}" \
  MODE="full" \
  RUN_GRANULARITY="full_corpus" \
  HOLD_JOB_ID=0 \
  INSIDE_HOLD_STEP=1 \
  MAX_PARALLEL_OVERRIDE=1 \
  LANGS_OVERRIDE="de" \
  LMS_OVERRIDE="4" \
  GLOSSARY_KINDS_OVERRIDE="raw" \
  PAPERS_OVERRIDE="2022.acl-long.268 2022.acl-long.367 2022.acl-long.590 2022.acl-long.110 2022.acl-long.117" \
  GPU_PAIRS_CSV_OVERRIDE="${GPU_PAIR}" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  TERM_MAP_FORMAT_OVERRIDE="plain" \
  EMPTY_TERM_MAP_POLICY_OVERRIDE="omit" \
  SYSTEM_PROMPT_STYLE_OVERRIDE="given_chunks" \
  OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
  INPUT_ROOT_OVERRIDE="${OUT_ROOT}/__inputs__/${MODEL_LABEL}" \
  LOG_DIR_OVERRIDE="${LOG_ROOT}/${MODEL_LABEL}" \
  SUMMARY_DIR_OVERRIDE="${OUTPUT_BASE}/__summary__" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  RAW_GLOSSARY_OVERRIDE="${RAW_GLOSSARY}" \
  GS10K_GLOSSARY_OVERRIDE="${GS10K_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_GLOBAL_OVERRIDE="${RAW_GLOSSARY}" \
  EVAL_GLOSSARY_FOLLOWS_KIND_OVERRIDE=0 \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-1}" \
  DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
  WANDB_RUN_PREFIX_OVERRIDE="dec16den_ttag_serial_hn1024_t078" \
  WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_serial_hn1024_tau078" \
  WANDB_VARIANT_PREFIX_OVERRIDE="dec16den_ttag_serial_hn1024_t078" \
  WANDB_COMPUTE_TAG_OVERRIDE="compute:$(hostname -s)_gpu${GPU_PAIR//,/}" \
  TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="0.78" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="1.92" \
  RAG_TOP_K_OVERRIDE=10 \
  STRIP_OUTPUT_TAGS_OVERRIDE="term_t" \
  RAG_GPU_OVERRIDE="${RAG_GPU}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE}" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-12288}" \
  MAX_CACHE_CHUNKS_OVERRIDE="${MAX_CACHE_CHUNKS}" \
  KEEP_CACHE_CHUNKS_OVERRIDE="${KEEP_CACHE_CHUNKS}" \
  MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS}" \
  SRC_LIST_OVERRIDE="${SRC_LIST_PORTABLE}" \
  TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
  SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
  REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
  AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
  bash "${BASE_LAUNCHER}" \
    > "${LOG_ROOT}/serial_launcher.out" \
    2> "${LOG_ROOT}/serial_launcher.err"

  require_file "${SETTING_DIR}/eval_results.tsv"
  require_file "${SETTING_DIR}/instances.log"
  require_file "${SETTING_DIR}/instances.strip_term.log"
  require_file "${SETTING_DIR}/term_adoption.json"
  write_summary
  date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.success"
  echo "[ALL DONE] ${OUTPUT_BASE}/__summary__/summary_de_lm4_serial.md"
}

main "$@"
