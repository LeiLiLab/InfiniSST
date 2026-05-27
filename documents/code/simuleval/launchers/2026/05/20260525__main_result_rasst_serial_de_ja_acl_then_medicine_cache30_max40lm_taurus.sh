#!/usr/bin/env bash
set -euo pipefail

# Serial main-result RASST readout for En-De/En-Ja.
#
# Stage order:
#   1. tagged ACL raw, de/ja, lm=1,2,3,4
#   2. medicine hardraw, de/ja, lm=1,2,3,4
#
# Each task uses serial SimulEval via eval_density_unified.sh. Taurus is filled
# with four 2-GPU jobs in waves. Aries was not idle at creation time, so this
# launcher intentionally runs on Taurus only instead of stealing busy GPUs.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260525T215343_main_result_rasst_serial_de_ja_cache30_max40lm_taurus}"
OUT_ROOT="${OUT_ROOT:-/mnt/data1/jiaxuanluo/main_result_rasst_serial_de_ja_cache30_max40lm_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/data1/jiaxuanluo/logs/main_result_rasst_serial_de_ja_cache30_max40lm_${RUN_STAMP}}"
STATUS_DIR="${STATUS_DIR:-${LOG_ROOT}/status}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__main_result_rasst_serial_de_ja_acl_then_medicine_cache30_max40lm_taurus.md}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh}"

GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-0,1;2,3;4,5;6,7}"
MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-2048}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-25}"
POLL_SECS="${POLL_SECS:-30}"

HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
ACL_GLOSSARY="${ACL_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"

DE_MODEL="${DE_MODEL:-/mnt/data1/jiaxuanluo/slm_local_cache/de_tagged_acl_20260525/cap16_denoise_ttag/v0-20260525-203735-hf}"
DE_MODEL_LABEL="${DE_MODEL_LABEL:-de_cap16_denoise_ttag_hn1024_tau078_omit_serial_chunks30_max40lm}"
JA_ACL_MODEL="${JA_ACL_MODEL:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_ja_r32a64_tp2_taurus4/keep1.0_r32/v0-20260524-123624-hf}"
JA_ACL_MODEL_LABEL="${JA_ACL_MODEL_LABEL:-ja_new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078_omit_serial_chunks30_max40lm}"
JA_MED_MODEL="${JA_MED_MODEL:-/mnt/data1/jiaxuanluo/slm_local_cache/ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf}"
JA_MED_MODEL_LABEL="${JA_MED_MODEL_LABEL:-ja_cap16_denoise_ttag_hn1024_tau078_omit_serial_chunks30_max40lm}"

ACL_DE_INPUT="${ACL_DE_INPUT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all}"
ACL_JA_INPUT="${ACL_JA_INPUT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_20260524T0933_tagacl_newv9_hn1024_tau078_raw_ja_taurus8/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/ja/all}"
MED_DE_INPUT="${MED_DE_INPUT:-/mnt/gemini/data1/jiaxuanluo/de_cap16_denoise_medicine_acl_batch_taurus_20260525T165125_de_cap16_denoise_med_acl_batch_taurus/medicine_hardraw_de_cap16_denoise_ttag_hn1024_tau078_batch_chunks30/de/__medicine_inputs__/lists}"
MED_JA_INPUT="${MED_JA_INPUT:-/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_20260525T184043_ja_med_cap16den_lm1234_taurus/ja/__medicine_inputs__/lists}"

INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/data1/jiaxuanluo/maxsim_index_cache/main_result_serial_rasst_cache30_max40lm}"

RAG_SCORE_THRESHOLD="${RAG_SCORE_THRESHOLD:-0.78}"
RAG_TOP_K="${RAG_TOP_K:-10}"
RAG_TIMELINE_LOOKBACK_SEC="${RAG_TIMELINE_LOOKBACK_SEC:-1.92}"
MAX_CACHE_CHUNKS="${MAX_CACHE_CHUNKS:-30}"
KEEP_CACHE_CHUNKS="${KEEP_CACHE_CHUNKS:-30}"
MAX_NEW_TOKENS_PER_LM="${MAX_NEW_TOKENS_PER_LM:-40}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.72}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
RAG_GPU="${RAG_GPU:-cuda:1}"
VLLM_LIMIT_AUDIO="${VLLM_LIMIT_AUDIO:-128}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_file() {
  local path="$1"
  [[ -s "${path}" ]] || fail "Missing/empty required file: ${path}"
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || fail "Missing required directory: ${path}"
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

validate_model_dir() {
  local model="$1"
  require_file "${model}/config.json"
  require_file "${model}/generation_config.json"
  require_file "${model}/model.safetensors.index.json"
  require_file "${model}/tokenizer_config.json"
  local shard_count
  shard_count="$(find "${model}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || fail "Expected 15 HF shards in ${model}, found ${shard_count}"
}

prepare_acl_inputs() {
  local lang="$1" src_dir="$2" out_dir="$3"
  mkdir -p "${out_dir}"
  require_file "${src_dir}/source.list"
  require_file "${src_dir}/target.list"
  require_file "${src_dir}/source_text.txt"
  require_file "${src_dir}/ref.txt"
  require_file "${src_dir}/audio.yaml"
  sed \
    -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
    -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
    "${src_dir}/source.list" > "${out_dir}/source.portable.list"
  cp "${src_dir}/target.list" "${out_dir}/target.list"
  cp "${src_dir}/source_text.txt" "${out_dir}/source_text.txt"
  cp "${src_dir}/ref.txt" "${out_dir}/ref.txt"
  cp "${src_dir}/audio.yaml" "${out_dir}/audio.yaml"
  require_file "${out_dir}/source.portable.list"
  echo "[PREP] acl ${lang}: ${out_dir}"
}

resolve_task_config() {
  local dataset="$1" lang="$2"
  TASK_GLOSSARY=""
  TASK_MODEL=""
  TASK_MODEL_LABEL=""
  TASK_OUTPUT_BASE=""
  TASK_DENSITY=""
  TASK_SRC=""
  TASK_TGT=""
  TASK_SOURCE_TEXT=""
  TASK_REF=""
  TASK_AUDIO=""

  if [[ "${dataset}" == "acl_tagged_raw" ]]; then
    TASK_GLOSSARY="${ACL_GLOSSARY}"
    if [[ "${lang}" == "de" ]]; then
      TASK_MODEL="${DE_MODEL}"
      TASK_MODEL_LABEL="${DE_MODEL_LABEL}"
      TASK_OUTPUT_BASE="${OUT_ROOT}/acl_tagged_raw/${DE_MODEL_LABEL}"
      TASK_DENSITY="mainres_acl_serial_de_cap16denoise_ttag_hn1024_tau078_omit_chunks30_max40lm"
      TASK_SRC="${OUT_ROOT}/__inputs__/acl_de/source.portable.list"
      TASK_TGT="${OUT_ROOT}/__inputs__/acl_de/target.list"
      TASK_SOURCE_TEXT="${OUT_ROOT}/__inputs__/acl_de/source_text.txt"
      TASK_REF="${OUT_ROOT}/__inputs__/acl_de/ref.txt"
      TASK_AUDIO="${OUT_ROOT}/__inputs__/acl_de/audio.yaml"
    elif [[ "${lang}" == "ja" ]]; then
      TASK_MODEL="${JA_ACL_MODEL}"
      TASK_MODEL_LABEL="${JA_ACL_MODEL_LABEL}"
      TASK_OUTPUT_BASE="${OUT_ROOT}/acl_tagged_raw/${JA_ACL_MODEL_LABEL}"
      TASK_DENSITY="mainres_acl_serial_ja_newv9_hn1024_tau078_omit_chunks30_max40lm"
      TASK_SRC="${OUT_ROOT}/__inputs__/acl_ja/source.portable.list"
      TASK_TGT="${OUT_ROOT}/__inputs__/acl_ja/target.list"
      TASK_SOURCE_TEXT="${OUT_ROOT}/__inputs__/acl_ja/source_text.txt"
      TASK_REF="${OUT_ROOT}/__inputs__/acl_ja/ref.txt"
      TASK_AUDIO="${OUT_ROOT}/__inputs__/acl_ja/audio.yaml"
    else
      fail "unsupported acl lang=${lang}"
    fi
  elif [[ "${dataset}" == "medicine_hardraw" ]]; then
    if [[ "${lang}" == "de" ]]; then
      TASK_GLOSSARY="${MED_DE_INPUT}/hard_medicine_raw__medicine5.json"
      TASK_MODEL="${DE_MODEL}"
      TASK_MODEL_LABEL="${DE_MODEL_LABEL}"
      TASK_OUTPUT_BASE="${OUT_ROOT}/medicine_hardraw/${DE_MODEL_LABEL}"
      TASK_DENSITY="mainres_medhard_serial_de_cap16denoise_ttag_hn1024_tau078_omit_chunks30_max40lm"
      TASK_SRC="${MED_DE_INPUT}/medicine.source__medicine5_hardraw.txt"
      TASK_TGT="${MED_DE_INPUT}/medicine.target.de__medicine5_hardraw.txt"
      TASK_SOURCE_TEXT="${MED_DE_INPUT}/medicine.source_text.en__medicine5_hardraw.txt"
      TASK_REF="${MED_DE_INPUT}/medicine.ref.de__medicine5_hardraw.txt"
      TASK_AUDIO="${MED_DE_INPUT}/medicine.audio__medicine5_hardraw.yaml"
    elif [[ "${lang}" == "ja" ]]; then
      TASK_GLOSSARY="${MED_JA_INPUT}/hard_medicine_raw__medicine5.json"
      TASK_MODEL="${JA_MED_MODEL}"
      TASK_MODEL_LABEL="${JA_MED_MODEL_LABEL}"
      TASK_OUTPUT_BASE="${OUT_ROOT}/medicine_hardraw/${JA_MED_MODEL_LABEL}"
      TASK_DENSITY="mainres_medhard_serial_ja_cap16denoise_ttag_hn1024_tau078_omit_chunks30_max40lm"
      TASK_SRC="${MED_JA_INPUT}/medicine.source__medicine5_hardraw.txt"
      TASK_TGT="${MED_JA_INPUT}/medicine.target.ja__medicine5_hardraw.txt"
      TASK_SOURCE_TEXT="${MED_JA_INPUT}/medicine.source_text.en__medicine5_hardraw.txt"
      TASK_REF="${MED_JA_INPUT}/medicine.ref.ja__medicine5_hardraw.txt"
      TASK_AUDIO="${MED_JA_INPUT}/medicine.audio__medicine5_hardraw.yaml"
    else
      fail "unsupported medicine lang=${lang}"
    fi
  else
    fail "unsupported dataset=${dataset}"
  fi
}

write_task_summary() {
  local dataset="$1" lang="$2" lm="$3" output_base="$4" model_label="$5" task_id="$6" max_new="$7"
  python3 - \
    "${dataset}" "${lang}" "${lm}" "${output_base}" "${model_label}" "${task_id}" "${max_new}" \
    "${MAX_CACHE_CHUNKS}" "${KEEP_CACHE_CHUNKS}" "${OUT_ROOT}/__summary__" <<'PY'
import csv
import json
import sys
from pathlib import Path

dataset, lang, lm, output_base_s, model_label, task_id, max_new, max_cache, keep_cache, summary_root_s = sys.argv[1:]
output_base = Path(output_base_s)
summary_root = Path(summary_root_s)
paths = sorted(output_base.glob(f"{lang}/*_lm{lm}_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one eval_results.tsv for {task_id}, found {len(paths)}: {paths}")
eval_path = paths[0]
rows = list(csv.DictReader(eval_path.open("r", encoding="utf-8", newline=""), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {eval_path}, got {len(rows)}")
row = rows[0]
inst = eval_path.parent / "instances.log"
strip = eval_path.parent / "instances.strip_term.log"
if not inst.is_file() or not strip.is_file():
    raise SystemExit(f"missing instances logs for {task_id}")
inst_rows = sum(1 for _ in inst.open("r", encoding="utf-8"))
strip_rows = [json.loads(line) for line in strip.open("r", encoding="utf-8")]
if inst_rows != 5 or len(strip_rows) != 5:
    raise SystemExit(f"expected 5 instances for {task_id}, got raw={inst_rows} strip={len(strip_rows)}")
summary_root.mkdir(parents=True, exist_ok=True)
fields = [
    "dataset", "method", "mode", "lang", "lm", "max_new_tokens",
    "max_cache_chunks", "keep_cache_chunks", "BLEU", "StreamLAAL",
    "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL",
    "REAL_TERM_ADOPT", "TERM_FCR", "instances_log_rows",
    "instances_strip_term_log_rows", "source_type", "source_path",
    "status", "note",
]
record = {
    "dataset": dataset,
    "method": "RASST",
    "mode": "serial_simuleval",
    "lang": lang,
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
    "instances_log_rows": str(inst_rows),
    "instances_strip_term_log_rows": str(len(strip_rows)),
    "source_type": "serial_eval_results",
    "source_path": str(eval_path),
    "status": "verified_serial",
    "note": f"{model_label}; cache_chunks={max_cache}/{keep_cache}; empty_term_map_policy=omit; system_prompt_style=given_chunks; max_new_tokens=40*lm",
}
out = summary_root / f"{task_id}.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
print(
    f"RESULT\t{task_id}\tBLEU={float(record['BLEU']):.4f}"
    f"\tStreamLAAL={float(record['StreamLAAL']):.4f}"
    f"\tStreamLAAL_CA={float(record['StreamLAAL_CA']):.4f}"
    f"\tTERM_ACC={float(record['TERM_ACC']):.4f}"
    f"\tTERM={record['TERM_CORRECT']}/{record['TERM_TOTAL']}"
    f"\teval={eval_path}",
    flush=True,
)
PY
}

run_one_task() {
  local dataset="$1" lang="$2" lm="$3" pair="$4"
  local task_id="${dataset}_${lang}_lm${lm}"
  local task_log="${LOG_ROOT}/${task_id}"
  local done_file="${STATUS_DIR}/${task_id}.done"
  local failed_file="${STATUS_DIR}/${task_id}.failed"
  local max_new="$((MAX_NEW_TOKENS_PER_LM * lm))"
  local dataset_short tmpdir
  case "${dataset}" in
    acl_tagged_raw) dataset_short="acl" ;;
    medicine_hardraw) dataset_short="med" ;;
    *) dataset_short="x" ;;
  esac
  tmpdir="/tmp/jxmr_${dataset_short}_${lang}${lm}"
  mkdir -p "${task_log}" "${STATUS_DIR}"
  rm -f "${done_file}" "${failed_file}"

  (
    set -euo pipefail
    exec > "${task_log}/worker.out" 2> "${task_log}/worker.err"
    echo "[TASK] ${task_id} host=$(hostname -s) pair=${pair} max_new_tokens=${max_new}"
    resolve_task_config "${dataset}" "${lang}"
    validate_model_dir "${TASK_MODEL}"
    for f in "${TASK_GLOSSARY}" "${HN1024_CKPT}" "${TASK_SRC}" "${TASK_TGT}" "${TASK_SOURCE_TEXT}" "${TASK_REF}" "${TASK_AUDIO}" "${EVAL_SCRIPT}" "${NOTES_FILE}"; do
      require_file "${f}"
    done
    wait_pair_idle "${pair}"
    clean_shm
    mkdir -p \
      "${TASK_OUTPUT_BASE}" \
      "${OUT_ROOT}/__summary__" \
      "${INDEX_CACHE_DIR}" \
      "${tmpdir}/torchinductor" \
      "${tmpdir}/triton" \
      "/mnt/data1/jiaxuanluo/wandb_runs" \
      "/mnt/data1/jiaxuanluo/wandb_cache" \
      "/mnt/data1/jiaxuanluo/wandb_data" \
      "/mnt/data1/jiaxuanluo/wandb_artifacts"

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
    MODEL_NAME_OVERRIDE="${TASK_MODEL}" \
    RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
    RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
    RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
    RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
    LANG_CODE_OVERRIDE="${lang}" \
    LATENCY_MULTIPLIER_OVERRIDE="${lm}" \
    OUTPUT_BASE_OVERRIDE="${TASK_OUTPUT_BASE}" \
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${pair}" \
    VLLM_TP_SIZE_OVERRIDE="${VLLM_TP_SIZE}" \
    RAG_GPU_OVERRIDE="${RAG_GPU}" \
    GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION}" \
    GLOSSARY_PATH_OVERRIDE="${TASK_GLOSSARY}" \
    EVAL_GLOSSARY_PATH_OVERRIDE="${TASK_GLOSSARY}" \
    SRC_LIST_OVERRIDE="${TASK_SRC}" \
    TGT_LIST_OVERRIDE="${TASK_TGT}" \
    SOURCE_TEXT_FILE_OVERRIDE="${TASK_SOURCE_TEXT}" \
    REF_FILE_OVERRIDE="${TASK_REF}" \
    AUDIO_YAML_OVERRIDE="${TASK_AUDIO}" \
    EVAL_MODE_OVERRIDE="acl6060" \
    INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
    DENSITY_TAG="${TASK_DENSITY}" \
    TERM_MAP_FORMAT_OVERRIDE="plain" \
    EMPTY_TERM_MAP_POLICY_OVERRIDE="omit" \
    SYSTEM_PROMPT_STYLE_OVERRIDE="given_chunks" \
    STRIP_OUTPUT_TAGS_OVERRIDE="term_t" \
    TERM_FCR_POLICY="term_map_source_ref_negative_sentence" \
    MAX_CACHE_SECONDS_OVERRIDE=0 \
    KEEP_CACHE_SECONDS_OVERRIDE=0 \
    MAX_CACHE_CHUNKS_OVERRIDE="${MAX_CACHE_CHUNKS}" \
    KEEP_CACHE_CHUNKS_OVERRIDE="${KEEP_CACHE_CHUNKS}" \
    MAX_NEW_TOKENS_OVERRIDE="${max_new}" \
    CLEAN_OUTPUT_DIR_OVERRIDE=1 \
    EVAL_TMPDIR_OVERRIDE="${tmpdir}" \
    bash "${EVAL_SCRIPT}"

    write_task_summary "${dataset}" "${lang}" "${lm}" "${TASK_OUTPUT_BASE}" "${TASK_MODEL_LABEL}" "${task_id}" "${max_new}"
    echo "ok $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${done_file}"
  ) || {
    local rc=$?
    echo "failed:${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${failed_file}"
    return "${rc}"
  }
}

merge_stage_summary() {
  local stage="$1"
  python3 - "${OUT_ROOT}/__summary__" "${stage}" <<'PY'
import csv
import sys
from pathlib import Path

summary_root = Path(sys.argv[1])
stage = sys.argv[2]
parts = []
if stage == "acl":
    pattern = "acl_tagged_raw_*_lm*.tsv"
elif stage == "medicine":
    pattern = "medicine_hardraw_*_lm*.tsv"
else:
    pattern = "*_lm*.tsv"
for path in sorted(summary_root.glob(pattern)):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8"), delimiter="\t"))
    if len(rows) == 1:
        parts.append(rows[0])
if not parts:
    raise SystemExit(f"no summary rows found for stage={stage}")
parts.sort(key=lambda r: (r["dataset"], r["lang"], int(r["lm"])))
out = summary_root / f"summary_{stage}_serial.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(parts[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(parts)
print(f"[SUMMARY] {out}")
PY
}

run_stage() {
  local stage="$1"
  shift
  local -a tasks=("$@")
  local -a pairs
  IFS=';' read -r -a pairs <<< "${GPU_PAIRS_CSV}"
  local idx=0
  local task dataset lang lm pair
  local -a pids=()

  echo "[STAGE] start ${stage}; tasks=${#tasks[@]}"
  for task in "${tasks[@]}"; do
    IFS=: read -r dataset lang lm <<< "${task}"
    pair="${pairs[$((idx % ${#pairs[@]}))]}"
    echo "[SCHEDULE] ${stage} task=${task} pair=${pair}"
    run_one_task "${dataset}" "${lang}" "${lm}" "${pair}" &
    pids+=("$!")
    idx=$((idx + 1))
    if (( ${#pids[@]} == ${#pairs[@]} )); then
      local rc=0 pid
      for pid in "${pids[@]}"; do
        wait "${pid}" || rc=1
      done
      pids=()
      (( rc == 0 )) || fail "stage=${stage} wave failed"
    fi
  done
  if (( ${#pids[@]} > 0 )); then
    local rc=0 pid
    for pid in "${pids[@]}"; do
      wait "${pid}" || rc=1
    done
    (( rc == 0 )) || fail "stage=${stage} final wave failed"
  fi
  merge_stage_summary "${stage}"
  echo "[STAGE] done ${stage}"
}

main() {
  [[ "$(hostname -s)" == taurus* ]] || fail "This launcher is Taurus-only; current host=$(hostname -s)"
  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
  require_file "${EVAL_SCRIPT}"
  require_file "${NOTES_FILE}"
  require_file "${HN1024_CKPT}"
  require_file "${ACL_GLOSSARY}"
  validate_model_dir "${DE_MODEL}"
  validate_model_dir "${JA_ACL_MODEL}"
  validate_model_dir "${JA_MED_MODEL}"
  require_dir "${ACL_DE_INPUT}"
  require_dir "${ACL_JA_INPUT}"
  require_dir "${MED_DE_INPUT}"
  require_dir "${MED_JA_INPUT}"

  mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${STATUS_DIR}" "${OUT_ROOT}/__inputs__" "${OUT_ROOT}/__summary__"
  prepare_acl_inputs de "${ACL_DE_INPUT}" "${OUT_ROOT}/__inputs__/acl_de"
  prepare_acl_inputs ja "${ACL_JA_INPUT}" "${OUT_ROOT}/__inputs__/acl_ja"

  {
    echo "run_stamp=${RUN_STAMP}"
    echo "host=$(hostname -s)"
    echo "out_root=${OUT_ROOT}"
    echo "log_root=${LOG_ROOT}"
    echo "gpu_pairs=${GPU_PAIRS_CSV}"
    echo "driver=${EVAL_SCRIPT}"
    echo "stage_order=acl_tagged_raw_then_medicine_hardraw"
    echo "langs=de ja"
    echo "lms=1 2 3 4"
    echo "cache_chunks=${MAX_CACHE_CHUNKS}/${KEEP_CACHE_CHUNKS}"
    echo "prompt=given_chunks"
    echo "empty_term_map_policy=omit"
    echo "max_new_tokens=lm*${MAX_NEW_TOKENS_PER_LM}"
    echo "de_model=${DE_MODEL}"
    echo "ja_acl_model=${JA_ACL_MODEL}"
    echo "ja_medicine_model=${JA_MED_MODEL}"
    echo "retriever=${HN1024_CKPT}"
  } | tee "${OUT_ROOT}/run_meta.txt"

  run_stage "acl" \
    "acl_tagged_raw:de:1" \
    "acl_tagged_raw:de:2" \
    "acl_tagged_raw:de:3" \
    "acl_tagged_raw:de:4" \
    "acl_tagged_raw:ja:1" \
    "acl_tagged_raw:ja:2" \
    "acl_tagged_raw:ja:3" \
    "acl_tagged_raw:ja:4"

  run_stage "medicine" \
    "medicine_hardraw:de:1" \
    "medicine_hardraw:de:2" \
    "medicine_hardraw:de:3" \
    "medicine_hardraw:de:4" \
    "medicine_hardraw:ja:1" \
    "medicine_hardraw:ja:2" \
    "medicine_hardraw:ja:3" \
    "medicine_hardraw:ja:4"

  merge_stage_summary "all"
  echo "success $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_DIR}/all.done"
  echo "[DONE] main-result serial RASST readout complete: ${OUT_ROOT}/__summary__"
}

main "$@"
