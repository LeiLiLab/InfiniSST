#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260526T000000_tagged_acl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_taurus}"
OUT_ROOT="${OUT_ROOT:-/mnt/data1/jiaxuanluo/tagged_acl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/data1/jiaxuanluo/logs/tagged_acl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_${RUN_STAMP}}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/launcher.pid}"

NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260526__tagged_acl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_taurus.md}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh}"
AGENT_FILE="${AGENT_FILE:-${ROOT_DIR}/agents/infinisst_omni_vllm_maxsim_rag.py}"

MODEL_NAME="${MODEL_NAME:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_ja_r32a64_tp2_taurus4/keep1.0_r32/v0-20260524-123624-hf}"
RAG_MODEL_PATH="${RAG_MODEL_PATH:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
GLOSSARY_PATH="${GLOSSARY_PATH:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
INPUT_ROOT="${INPUT_ROOT:-/mnt/data1/jiaxuanluo/main_result_rasst_serial_de_ja_cache30_max40lm_20260525T215831_main_result_rasst_serial_de_ja_cache30_max40lm_taurus_retry1/__inputs__/acl_ja}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/data1/jiaxuanluo/maxsim_index_cache/tagged_acl_ja_serial_promptfix_cache30_vllmaudio128_max40lm}"

GPU_PAIR="${GPU_PAIR:-0,1}"
LM="${LM:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$((40 * LM))}"
MAX_CACHE_CHUNKS="${MAX_CACHE_CHUNKS:-30}"
KEEP_CACHE_CHUNKS="${KEEP_CACHE_CHUNKS:-30}"
RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD:-0.78}"
RAG_TOP_K="${RAG_TOP_K:-10}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC:-1.92}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.72}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
RAG_GPU="${RAG_GPU:-cuda:1}"
VLLM_LIMIT_AUDIO="${VLLM_LIMIT_AUDIO:-128}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jxja${LM}p128}"
DENSITY_TAG="${DENSITY_TAG:-tagacl_ja_serial_promptfix_cache30_vllmaudio128_max40lm}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_file() {
  local path="$1"
  [[ -s "${path}" ]] || fail "Missing/empty required file: ${path}"
}

require_model_dir() {
  local model="$1"
  require_file "${model}/config.json"
  require_file "${model}/generation_config.json"
  require_file "${model}/model.safetensors.index.json"
  require_file "${model}/tokenizer_config.json"
  local shard_count
  shard_count="$(find "${model}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || fail "Expected 15 HF shards in ${model}, found ${shard_count}"
}

validate_runtime_prompts() {
  python3 - "${OUT_ROOT}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

out_root = Path(sys.argv[1])
runtime_logs = sorted(out_root.glob("**/runtime_omni_vllm_maxsim_rag*.jsonl"))
if len(runtime_logs) != 1:
    raise SystemExit(f"expected one runtime log, found {len(runtime_logs)}: {runtime_logs}")
counts = Counter()
audio_counts = Counter()
total = 0
for line in runtime_logs[0].open("r", encoding="utf-8"):
    row = json.loads(line)
    if row.get("type") != "llm_input":
        continue
    total += 1
    prompt = row.get("prompt", "")
    counts[prompt.count("<|im_start|>system")] += 1
    audio_counts[prompt.count("<|audio_start|>")] += 1
if total == 0:
    raise SystemExit(f"no llm_input rows in {runtime_logs[0]}")
if counts != {1: total}:
    raise SystemExit(f"system prompt count check failed for {runtime_logs[0]}: {dict(counts)}")
print(
    f"[CHECK] runtime prompt ok: {total} llm_input rows; "
    f"system_count={dict(counts)} audio_count_minmax="
    f"{min(audio_counts)}/{max(audio_counts)} in {runtime_logs[0]}"
)
PY
}

write_summary() {
  python3 - "${OUT_ROOT}" "${LM}" "${MAX_NEW_TOKENS}" "${MAX_CACHE_CHUNKS}" "${KEEP_CACHE_CHUNKS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
lm, max_new, max_cache, keep_cache = sys.argv[2:6]
paths = sorted(out_root.glob(f"ja/*_lm{lm}_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one eval_results.tsv, found {len(paths)}: {paths}")
eval_path = paths[0]
rows = list(csv.DictReader(eval_path.open("r", encoding="utf-8", newline=""), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {eval_path}, got {len(rows)}")
row = rows[0]
inst = eval_path.parent / "instances.log"
strip = eval_path.parent / "instances.strip_term.log"
if sum(1 for _ in inst.open("r", encoding="utf-8")) != 5:
    raise SystemExit(f"expected 5 rows in {inst}")
if sum(1 for _ in strip.open("r", encoding="utf-8")) != 5:
    raise SystemExit(f"expected 5 rows in {strip}")
summary_dir = out_root / "__summary__"
summary_dir.mkdir(parents=True, exist_ok=True)
fields = [
    "dataset", "method", "mode", "lang", "lm", "max_new_tokens",
    "max_cache_chunks", "keep_cache_chunks", "BLEU", "StreamLAAL",
    "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL",
    "REAL_TERM_ADOPT", "TERM_FCR", "source_type", "source_path",
    "status", "note",
]
record = {
    "dataset": "acl_tagged_raw",
    "method": "RASST",
    "mode": "serial_simuleval_promptfix",
    "lang": "ja",
    "lm": lm,
    "max_new_tokens": max_new,
    "max_cache_chunks": max_cache,
    "keep_cache_chunks": keep_cache,
    "BLEU": row.get("BLEU", ""),
    "StreamLAAL": row.get("StreamLAAL", ""),
    "StreamLAAL_CA": row.get("StreamLAAL_CA", ""),
    "TERM_ACC": row.get("TERM_ACC", ""),
    "TERM_CORRECT": row.get("TERM_CORRECT", ""),
    "TERM_TOTAL": row.get("TERM_TOTAL", ""),
    "REAL_TERM_ADOPT": row.get("REAL_TERM_ADOPT", ""),
    "TERM_FCR": row.get("TERM_FCR", ""),
    "source_type": "serial_eval_results",
    "source_path": str(eval_path),
    "status": "verified_promptfix",
    "note": "ja NewV9 tagged-term SLM; tagged ACL raw glossary; HN1024 tau=0.78; VLLM_LIMIT_AUDIO=128; empty maps omitted; system prompt duplication fixed",
}
out = summary_dir / f"summary_acl_tagged_raw_ja_lm{lm}_promptfix_vllmaudio128.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
print(
    f"RESULT ja lm{lm} BLEU={float(record['BLEU']):.4f} "
    f"StreamLAAL={float(record['StreamLAAL']):.4f} "
    f"StreamLAAL_CA={float(record['StreamLAAL_CA']):.4f} "
    f"TERM_ACC={float(record['TERM_ACC']):.4f} "
    f"TERM={record['TERM_CORRECT']}/{record['TERM_TOTAL']} "
    f"eval={eval_path}"
)
PY
}

main() {
  [[ "$(hostname -s)" == taurus* ]] || fail "This launcher is Taurus-only; current host=$(hostname -s)"
  mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${EVAL_TMPDIR}" "${EVAL_TMPDIR}/torchinductor" "${EVAL_TMPDIR}/triton"
  echo "$$" > "${PID_FILE}"

  for f in \
    "${NOTES_FILE}" \
    "${EVAL_SCRIPT}" \
    "${AGENT_FILE}" \
    "${RAG_MODEL_PATH}" \
    "${GLOSSARY_PATH}" \
    "${INPUT_ROOT}/source.portable.list" \
    "${INPUT_ROOT}/target.list" \
    "${INPUT_ROOT}/source_text.txt" \
    "${INPUT_ROOT}/ref.txt" \
    "${INPUT_ROOT}/audio.yaml"; do
    require_file "${f}"
  done
  require_model_dir "${MODEL_NAME}"

  {
    echo "run_stamp=${RUN_STAMP}"
    echo "host=$(hostname -s)"
    echo "out_root=${OUT_ROOT}"
    echo "log_root=${LOG_ROOT}"
    echo "gpu_pair=${GPU_PAIR}"
    echo "lm=${LM}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "vllm_limit_audio=${VLLM_LIMIT_AUDIO}"
    echo "cache_chunks=${MAX_CACHE_CHUNKS}/${KEEP_CACHE_CHUNKS}"
    echo "prompt=given_chunks"
    echo "empty_term_map_policy=omit"
    echo "model=${MODEL_NAME}"
    echo "retriever=${RAG_MODEL_PATH}"
    echo "glossary=${GLOSSARY_PATH}"
    echo "input_root=${INPUT_ROOT}"
  } | tee "${OUT_ROOT}/run_meta.txt"

  export TMPDIR="${EVAL_TMPDIR}"
  export TMP="${EVAL_TMPDIR}"
  export TEMP="${EVAL_TMPDIR}"
  export VLLM_LIMIT_AUDIO="${VLLM_LIMIT_AUDIO}"
  export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}"
  export VLLM_LIMIT_AUDIO_OVERRIDE="${VLLM_LIMIT_AUDIO}"
  export VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN}"
  export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
  export VLLM_MOE_USE_DEEP_GEMM=0
  export VLLM_USE_FUSED_MOE_GROUPED_TOPK=0
  export WANDB_MODE=disabled
  export WANDB_DIR="/mnt/data1/jiaxuanluo/wandb_runs"
  export WANDB_CACHE_DIR="/mnt/data1/jiaxuanluo/wandb_cache"
  export WANDB_DATA_DIR="/mnt/data1/jiaxuanluo/wandb_data"
  export WANDB_ARTIFACT_DIR="/mnt/data1/jiaxuanluo/wandb_artifacts"

  ROOT_DIR="${ROOT_DIR}" \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  RAG_MODEL_PATH_OVERRIDE="${RAG_MODEL_PATH}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
  LANG_CODE_OVERRIDE="ja" \
  LATENCY_MULTIPLIER_OVERRIDE="${LM}" \
  OUTPUT_BASE_OVERRIDE="${OUT_ROOT}" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
  VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE}" \
  RAG_GPU_OVERRIDE="${RAG_GPU}" \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION}" \
  GLOSSARY_PATH_OVERRIDE="${GLOSSARY_PATH}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${GLOSSARY_PATH}" \
  SRC_LIST_OVERRIDE="${INPUT_ROOT}/source.portable.list" \
  TGT_LIST_OVERRIDE="${INPUT_ROOT}/target.list" \
  SOURCE_TEXT_FILE_OVERRIDE="${INPUT_ROOT}/source_text.txt" \
  REF_FILE_OVERRIDE="${INPUT_ROOT}/ref.txt" \
  AUDIO_YAML_OVERRIDE="${INPUT_ROOT}/audio.yaml" \
  EVAL_MODE_OVERRIDE="acl6060" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
  DENSITY_TAG="${DENSITY_TAG}" \
  TERM_MAP_FORMAT_OVERRIDE="plain" \
  EMPTY_TERM_MAP_POLICY_OVERRIDE="omit" \
  SYSTEM_PROMPT_STYLE_OVERRIDE="given_chunks" \
  STRIP_OUTPUT_TAGS_OVERRIDE="term_t" \
  TERM_FCR_POLICY="term_map_source_ref_negative_sentence" \
  TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
  MAX_CACHE_SECONDS_OVERRIDE=0 \
  KEEP_CACHE_SECONDS_OVERRIDE=0 \
  MAX_CACHE_CHUNKS_OVERRIDE="${MAX_CACHE_CHUNKS}" \
  KEEP_CACHE_CHUNKS_OVERRIDE="${KEEP_CACHE_CHUNKS}" \
  MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS}" \
  CLEAN_OUTPUT_DIR_OVERRIDE=1 \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
  bash "${EVAL_SCRIPT}" 2>&1 | tee "${LOG_ROOT}/eval_density.stdout"

  validate_runtime_prompts
  write_summary
  date -u +%Y-%m-%dT%H:%M:%SZ > "${LOG_ROOT}/done.txt"
}

main "$@"
