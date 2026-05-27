#!/usr/bin/env bash
set -euo pipefail

# En-De tagged ACL raw lm=4 serial SimulEval RAG rerun.
# This intentionally avoids the same-lm batch-vLLM evaluator.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-20260524T2310_tagacl_newv9_mfa_np_serial_de_lm4}"
GPU_PAIR="${GPU_PAIR:-0,1}"

MODEL_LABEL="${MODEL_LABEL:-new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64_hn1024_tau078_serial}"
MODEL_NAME="${MODEL_NAME:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_r32a64_tp2_taurus8/keep1.0_r32/v0-20260525-001708-hf}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__tagged_acl_new_v9_mfa_npfilter_hn1024_tau078_raw_de_lm4_serial_aries01.md}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_serial_mfa_npfilter_hn1024_tau078_raw_de_lm4_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_serial_mfa_npfilter_hn1024_tau078_raw_de_lm4_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/serial_simuleval}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_tacl_de4_ser}"

DENSITY_TAG="tagacl_newv9_mfa_np_serial_hn1024_tau078"
GLOSSARY_TAG="acl6060_tagged_gt_raw_min_norm2"
SETTING_DIR="${OUTPUT_BASE}/de/d${DENSITY_TAG}_lm4_k10_th0.78_g${GLOSSARY_TAG}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_path() {
  [[ -e "$1" ]] || fail "Missing required path: $1"
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
  local g0 g1
  IFS=',' read -r g0 g1 <<< "${GPU_PAIR}"
  gpu_is_idle "${g0}" || fail "GPU ${g0} is not idle"
  gpu_is_idle "${g1}" || fail "GPU ${g1} is not idle"
}

validate_model_dir() {
  require_path "${MODEL_NAME}/config.json"
  require_path "${MODEL_NAME}/model.safetensors.index.json"
  require_path "${MODEL_NAME}/tokenizer_config.json"
  local shard_count
  shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || fail "Expected 15 HF safetensor shards in ${MODEL_NAME}, found ${shard_count}"
}

preflight() {
  [[ "$(hostname -s)" == aries* ]] || fail "This launcher must run on aries; current host=$(hostname -s)"
  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
  for p in \
    "${ROOT_DIR}" \
    "${BASE_LAUNCHER}" \
    "${HN1024_CKPT}" \
    "${RAW_GLOSSARY}" \
    "${GS10K_GLOSSARY}" \
    "${NOTES_FILE}" \
    "${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh" \
    "${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py" \
    "${ROOT_DIR}/documents/code/general/wandb_tool.py" \
    "${PYTHON_BIN}"; do
    require_path "${p}"
  done
  validate_model_dir
  check_gpu_pair
}

write_summary() {
  local summary_dir="${OUTPUT_BASE}/__summary__"
  mkdir -p "${summary_dir}"
  "${PYTHON_BIN}" - "${SETTING_DIR}/eval_results.tsv" "${summary_dir}/summary_de_lm4_serial.tsv" "${summary_dir}/summary_de_lm4_serial.md" "${SETTING_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

eval_tsv = Path(sys.argv[1])
out_tsv = Path(sys.argv[2])
out_md = Path(sys.argv[3])
setting_dir = Path(sys.argv[4])

rows = list(csv.DictReader(eval_tsv.open("r", encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {eval_tsv}, got {len(rows)}")
row = rows[0]
fields = [
    "lang", "lm", "glossary", "BLEU", "StreamLAAL", "StreamLAAL_CA",
    "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL", "REAL_TERM_ADOPT",
    "TERM_FCR", "eval_results", "instances_log",
]
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

def pct(x):
    try:
        return f"{float(x) * 100:.2f}"
    except Exception:
        return str(x)

lines = [
    "# Tagged ACL de raw: clean New V9 + HN1024 tau=0.78 serial lm4",
    "",
    "| lang | lm | glossary | BLEU | TERM_ACC | StreamLAAL | StreamLAAL_CA | correct/total |",
    "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    f"| de | 4 | tagged_raw | {float(record['BLEU']):.4f} | {pct(record['TERM_ACC'])} | {float(record['StreamLAAL']):.4f} | {float(record['StreamLAAL_CA']):.4f} | {record['TERM_CORRECT']}/{record['TERM_TOTAL']} |",
    "",
    f"- eval_results: `{eval_tsv}`",
    f"- instances: `{setting_dir / 'instances.log'}`",
]
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_tsv)
print(out_md)
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
    "/mnt/gemini/data1/jiaxuanluo/wandb_runs" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_cache" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_data" \
    "/mnt/gemini/data1/jiaxuanluo/wandb_artifacts"

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
    echo "lm=4"
    echo "tau=0.78"
    echo "lookback_sec=1.92"
    echo "max_new_tokens=40"
    echo "strip_output_tags=term"
  } | tee "${OUT_ROOT}/run_meta.txt"

  export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
  export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
  export VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}"
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
  WANDB_RUN_PREFIX_OVERRIDE="newv9_mfa_np_serial_hn1024_t078" \
  WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_serial_hn1024_tau078" \
  WANDB_VARIANT_PREFIX_OVERRIDE="newv9_mfa_np_serial_hn1024_t078" \
  WANDB_COMPUTE_TAG_OVERRIDE="compute:aries_gpu${GPU_PAIR//,/}" \
  TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="0.78" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="1.92" \
  RAG_TOP_K_OVERRIDE=10 \
  STRIP_OUTPUT_TAGS_OVERRIDE="term" \
  RAG_GPU_OVERRIDE="cuda:1" \
  VLLM_TP_SIZE_OVERRIDE=2 \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
  VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-16384}" \
  MAX_CACHE_SECONDS_OVERRIDE=80 \
  KEEP_CACHE_SECONDS_OVERRIDE=60 \
  MAX_NEW_TOKENS_OVERRIDE=40 \
  bash "${BASE_LAUNCHER}" \
    > "${LOG_ROOT}/serial_launcher.out" \
    2> "${LOG_ROOT}/serial_launcher.err"

  [[ -s "${SETTING_DIR}/eval_results.tsv" ]] || fail "Missing eval_results.tsv: ${SETTING_DIR}/eval_results.tsv"
  [[ -s "${SETTING_DIR}/instances.log" ]] || fail "Missing instances.log: ${SETTING_DIR}/instances.log"
  [[ -s "${SETTING_DIR}/term_adoption.json" ]] || fail "Missing term_adoption.json: ${SETTING_DIR}/term_adoption.json"
  write_summary
  echo "[ALL DONE] ${OUTPUT_BASE}/__summary__/summary_de_lm4_serial.md"
}

main "$@"
