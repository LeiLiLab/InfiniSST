#!/usr/bin/env bash
set -euo pipefail

# Taurus-only cap16-denoise batch-vLLM readout.
# Runs:
#   1) medicine hardraw En-De, lm=1,2,3,4
#   2) tagged ACL raw En-De, lm=2,3
# using two Taurus GPU pairs in waves.

ROOT_DIR="${ROOT_DIR:-/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260525T165125_de_cap16_denoise_med_acl_batch_taurus}"
RUN_STAMP_SHORT="${RUN_STAMP%%_*}"
MODEL_NAME="${MODEL_NAME:-/mnt/data1/jiaxuanluo/slm_local_cache/de_tagged_acl_20260525/cap16_denoise_ttag/v0-20260525-203735-hf}"
MODEL_LABEL="${MODEL_LABEL:-de_cap16_denoise_ttag_hn1024_tau078_omit_batch_chunks30}"
TRAIN_EVENT_ID="${TRAIN_EVENT_ID:-20260525T1236__speech_llm_train__de_cap16_denoise_budget_ttag_r32a32_ep1_taurus6}"

BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
MEDICINE_PREP_LAUNCHER="${MEDICINE_PREP_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__medicine_hardraw_lm1to4_5samples_hn1024_tau078_new_v9_batch.sh}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__de_cap16_denoise_medicine_acl_batch_taurus.md}"

HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
HARD_RAW_GLOSSARY="${HARD_RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/medicine_eval_hard_terms_llm_judge_manual_20260524/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json}"
ACL_INPUT_DIR="${ACL_INPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all}"
ACL_RAW_GLOSSARY="${ACL_RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/de_cap16_denoise_medicine_acl_batch_taurus_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/de_cap16_denoise_medicine_acl_batch_taurus_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/de_cap16_denoise_medicine_acl_batch_taurus_${RUN_STAMP_SHORT}}"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT:-/tmp/jx_cdmb_${RUN_STAMP_SHORT}}"

MED_OUTPUT_BASE="${MED_OUTPUT_BASE:-${OUT_ROOT}/medicine_hardraw_de_cap16_denoise_ttag_hn1024_tau078_batch_chunks30}"
ACL_OUTPUT_BASE="${ACL_OUTPUT_BASE:-${OUT_ROOT}/tagged_acl_de_cap16_denoise_ttag_hn1024_tau078_batch_chunks30_lm23}"
MED_INPUTS="${MED_OUTPUT_BASE}/de/__medicine_inputs__/lists"
ACL_INPUT_WORK_DIR="${OUT_ROOT}/__acl_inputs__"
ACL_SOURCE_PORTABLE="${ACL_INPUT_WORK_DIR}/source.portable.list"

GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-4,5;6,7}"
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
VLLM_LIMIT_AUDIO="${VLLM_LIMIT_AUDIO:-128}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
MAX_NEW_TOKENS_PER_LM="${MAX_NEW_TOKENS_PER_LM:-40}"

MEDICINE_LMS="${MEDICINE_LMS:-1 2 3 4}"
ACL_LMS="${ACL_LMS:-2 3}"
TARGET_SAMPLES="${TARGET_SAMPLES:-404 545006 596001 605000 606}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_path() {
  local path="$1"
  [[ -e "${path}" ]] || fail "Missing required path: ${path}"
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

write_lm_summary() {
  local dataset="$1" output_base="$2" lang="$3" lm="$4" max_new="$5"
  python - "${dataset}" "${output_base}" "${MODEL_LABEL}" "${lang}" "${lm}" "${max_new}" <<'PY'
import csv
import sys
from pathlib import Path

dataset, output_base_s, model_label, lang, lm, max_new = sys.argv[1:]
output_base = Path(output_base_s)
paths = sorted(output_base.glob(f"{lang}/*_lm{lm}_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one eval_results.tsv for {dataset} lm={lm}, found {len(paths)}: {paths}")
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
    raise SystemExit(f"expected 5 rows for {dataset} lm={lm}, got raw={inst_rows} strip={strip_rows}")
row = rows[0]
summary_dir = output_base / "__summary__"
summary_dir.mkdir(parents=True, exist_ok=True)
record = {
    "dataset": dataset,
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
out = summary_dir / f"summary_{dataset}_de_lm{lm}.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(record), delimiter="\t")
    writer.writeheader()
    writer.writerow(record)
print(
    "RESULT"
    f"\tdataset={dataset}\tlm={lm}"
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
  local dataset="$1" output_base="$2"
  shift 2
  python - "${dataset}" "${output_base}" "$@" <<'PY'
import csv
import sys
from pathlib import Path

dataset, output_base_s, *lms = sys.argv[1:]
summary_dir = Path(output_base_s) / "__summary__"
rows = []
for lm in lms:
    path = summary_dir / f"summary_{dataset}_de_lm{lm}.tsv"
    if not path.is_file():
        raise SystemExit(f"missing lm summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(data)}")
    rows.extend(data)
suffix = "_".join(f"lm{lm}" for lm in lms)
out = summary_dir / f"summary_{dataset}_de_{suffix}.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: int(r["lm"])))
print(f"[SUMMARY] {out}", flush=True)
PY
}

run_batch_lm() {
  local dataset="$1" lm="$2" pair="$3" output_base="$4" density_tag="$5" glossary_path="$6" eval_glossary_path="$7" glossary_tag="$8" src_list="$9"
  shift 9
  local tgt_list="$1" source_text="$2" ref_file="$3" audio_yaml="$4"
  local max_new="$((MAX_NEW_TOKENS_PER_LM * lm))"
  local lm_log="${LOG_ROOT}/${dataset}_lm${lm}"
  mkdir -p "${lm_log}" "${EVAL_TMPDIR_ROOT}/${dataset}_lm${lm}"
  wait_pair_idle "${pair}"
  clean_shm
  echo "[RUN] dataset=${dataset} lm=${lm} gpu_pair=${pair} max_new_tokens=${max_new}"
  RUN_TAG_OVERRIDE="${RUN_STAMP}_${dataset}_de_lm${lm}" \
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  LANG_CODE_OVERRIDE="de" \
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
  GLOSSARY_PATH_OVERRIDE="${glossary_path}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${eval_glossary_path}" \
  OUTPUT_BASE_OVERRIDE="${output_base}" \
  DENSITY_TAG_OVERRIDE="${density_tag}" \
  GLOSSARY_TAG_OVERRIDE="${glossary_tag}" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_DEVICE_OVERRIDE="cuda:0" \
  RAG_BATCH_RETRIEVAL_OVERRIDE=1 \
  INDEX_BUILD_DEVICE_OVERRIDE="cuda:0" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}/${dataset}" \
  TERM_MAP_FORMAT_OVERRIDE=plain \
  EMPTY_TERM_MAP_POLICY_OVERRIDE="${EMPTY_TERM_MAP_POLICY}" \
  RAG_PROMPT_POLICY_OVERRIDE="${RAG_PROMPT_POLICY}" \
  TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
  STRIP_OUTPUT_TAGS_OVERRIDE="${STRIP_OUTPUT_TAGS}" \
  EVAL_MODE_OVERRIDE=acl6060 \
  WANDB_LOG_OVERRIDE=0 \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  LOG_ROOT_OVERRIDE="${lm_log}/batch" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}/${dataset}_lm${lm}" \
  bash "${BATCH_LAUNCHER}" \
    > "${lm_log}/launcher.out" \
    2> "${lm_log}/launcher.err"
  write_lm_summary "${dataset}" "${output_base}" "de" "${lm}" "${max_new}" | tee "${lm_log}/result.txt"
}

prebuild_index() {
  local dataset="$1" output_base="$2" density_tag="$3" glossary_path="$4" eval_glossary_path="$5" glossary_tag="$6" src_list="$7"
  shift 7
  local tgt_list="$1" source_text="$2" ref_file="$3" audio_yaml="$4"
  local pair="${PREBUILD_GPU_PAIR:-4,5}"
  local pre_log="${LOG_ROOT}/${dataset}_index_prebuild"
  mkdir -p "${pre_log}" "${EVAL_TMPDIR_ROOT}/${dataset}_index"
  wait_pair_idle "${pair}"
  clean_shm
  echo "[INDEX PREBUILD] dataset=${dataset} gpu_pair=${pair}"
  RUN_TAG_OVERRIDE="${RUN_STAMP}_${dataset}_index_prebuild" \
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  LANG_CODE_OVERRIDE="de" \
  LMS_OVERRIDE="1" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${pair}" \
  VLLM_TP_SIZE_OVERRIDE=2 \
  MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
  SRC_LIST_OVERRIDE="${src_list}" \
  TGT_LIST_OVERRIDE="${tgt_list}" \
  SOURCE_TEXT_FILE_OVERRIDE="${source_text}" \
  REF_FILE_OVERRIDE="${ref_file}" \
  AUDIO_YAML_OVERRIDE="${audio_yaml}" \
  GLOSSARY_PATH_OVERRIDE="${glossary_path}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${eval_glossary_path}" \
  OUTPUT_BASE_OVERRIDE="${output_base}" \
  DENSITY_TAG_OVERRIDE="${density_tag}" \
  GLOSSARY_TAG_OVERRIDE="${glossary_tag}" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="${RAG_SCORE_THRESHOLD}" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="${RAG_TIMELINE_LOOKBACK_SEC}" \
  RAG_TOP_K_OVERRIDE="${RAG_TOP_K}" \
  RAG_DEVICE_OVERRIDE="cuda:0" \
  INDEX_BUILD_DEVICE_OVERRIDE="cuda:0" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}/${dataset}" \
  TERM_MAP_FORMAT_OVERRIDE=plain \
  EMPTY_TERM_MAP_POLICY_OVERRIDE="${EMPTY_TERM_MAP_POLICY}" \
  RAG_PROMPT_POLICY_OVERRIDE="${RAG_PROMPT_POLICY}" \
  STRIP_OUTPUT_TAGS_OVERRIDE="${STRIP_OUTPUT_TAGS}" \
  EVAL_MODE_OVERRIDE=acl6060 \
  WANDB_LOG_OVERRIDE=0 \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  DRY_RUN_OVERRIDE=1 \
  LOG_ROOT_OVERRIDE="${pre_log}/batch" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}/${dataset}_index" \
  bash "${BATCH_LAUNCHER}" \
    > "${pre_log}/launcher.out" \
    2> "${pre_log}/launcher.err"
}

run_wave() {
  local dataset="$1" output_base="$2" density_tag="$3" glossary_path="$4" eval_glossary_path="$5" glossary_tag="$6" src_list="$7"
  shift 7
  local tgt_list="$1" source_text="$2" ref_file="$3" audio_yaml="$4"
  shift 4
  local lms=("$@")
  IFS=';' read -r -a gpu_pairs <<< "${GPU_PAIRS_CSV}"
  (( ${#gpu_pairs[@]} >= ${#lms[@]} )) || fail "Need ${#lms[@]} GPU pairs, got ${#gpu_pairs[@]}"
  local pids=()
  for idx in "${!lms[@]}"; do
    local lm="${lms[$idx]}" pair="${gpu_pairs[$idx]}"
    ( run_batch_lm "${dataset}" "${lm}" "${pair}" "${output_base}" "${density_tag}" "${glossary_path}" "${eval_glossary_path}" "${glossary_tag}" "${src_list}" "${tgt_list}" "${source_text}" "${ref_file}" "${audio_yaml}" ) &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  [[ "${status}" == "0" ]] || fail "Wave failed for dataset=${dataset}, lms=${lms[*]}; see ${LOG_ROOT}"
}

require_file "${BATCH_LAUNCHER}"
require_file "${MEDICINE_PREP_LAUNCHER}"
require_file "${NOTES_FILE}"
require_file "${MODEL_NAME}/config.json"
require_file "${MODEL_NAME}/generation_config.json"
require_file "${MODEL_NAME}/model.safetensors.index.json"
require_file "${HN1024_CKPT}"
require_file "${HARD_RAW_GLOSSARY}"
require_file "${ACL_RAW_GLOSSARY}"
for f in source.list target.list source_text.txt ref.txt audio.yaml; do
  require_file "${ACL_INPUT_DIR}/${f}"
done
shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
[[ "${shard_count}" == "15" ]] || fail "Expected 15 HF safetensor shards in ${MODEL_NAME}, found ${shard_count}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${INDEX_CACHE_DIR}" "${EVAL_TMPDIR_ROOT}" "${ACL_INPUT_WORK_DIR}"

df -h /mnt/gemini/data1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_prelaunch.csv"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "gpu_pairs=${GPU_PAIRS_CSV}"
  echo "model=${MODEL_NAME}"
  echo "model_label=${MODEL_LABEL}"
  echo "train_event_id=${TRAIN_EVENT_ID}"
  echo "retriever=${HN1024_CKPT}"
  echo "medicine_glossary=${HARD_RAW_GLOSSARY}"
  echo "acl_glossary=${ACL_RAW_GLOSSARY}"
  echo "medicine_output_base=${MED_OUTPUT_BASE}"
  echo "acl_output_base=${ACL_OUTPUT_BASE}"
  echo "max_new_tokens_policy=${MAX_NEW_TOKENS_PER_LM}*lm"
  echo "empty_term_map_policy=${EMPTY_TERM_MAP_POLICY}"
  echo "rag_prompt_policy=${RAG_PROMPT_POLICY}"
  echo "strip_output_tags=${STRIP_OUTPUT_TAGS}"
  echo "max_cache_chunks=${MAX_CACHE_CHUNKS}"
  echo "keep_cache_chunks=${KEEP_CACHE_CHUNKS}"
  echo "vllm_limit_audio=${VLLM_LIMIT_AUDIO}"
  echo "vllm_max_model_len=${VLLM_MAX_MODEL_LEN}"
} | tee "${OUT_ROOT}/run_meta.txt"

echo "[PREP] medicine hardraw five-sample inputs"
ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
RUN_STAMP="${RUN_STAMP}_medicine_prep" \
LANG_CODE_OVERRIDE=de \
TARGET_SAMPLES_OVERRIDE="${TARGET_SAMPLES}" \
MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
MODEL_LABEL="${MODEL_LABEL}" \
HARD_RAW_GLOSSARY_OVERRIDE="${HARD_RAW_GLOSSARY}" \
RUNTIME_GLOSSARY_OVERRIDE="${HARD_RAW_GLOSSARY}" \
FIXED_RAW_GLOSSARY_OVERRIDE="${HARD_RAW_GLOSSARY}" \
HN1024_CKPT_OVERRIDE="${HN1024_CKPT}" \
OUTPUT_BASE_OVERRIDE="${MED_OUTPUT_BASE}" \
LOG_ROOT_OVERRIDE="${LOG_ROOT}/medicine_prepare" \
INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}/medicine_prepare" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}/medicine_prepare" \
PREP_ONLY_OVERRIDE=1 \
FORCE_PREPARE_OVERRIDE="${FORCE_PREPARE_OVERRIDE:-0}" \
bash "${MEDICINE_PREP_LAUNCHER}" \
  > "${LOG_ROOT}/medicine_prepare.out" \
  2> "${LOG_ROOT}/medicine_prepare.err"

MED_EVAL_GLOSSARY="${MED_INPUTS}/hard_medicine_raw__medicine5.json"
MED_SRC_LIST="${MED_INPUTS}/medicine.source__medicine5_hardraw.txt"
MED_TGT_LIST="${MED_INPUTS}/medicine.target.de__medicine5_hardraw.txt"
MED_SOURCE_TEXT="${MED_INPUTS}/medicine.source_text.en__medicine5_hardraw.txt"
MED_REF_FILE="${MED_INPUTS}/medicine.ref.de__medicine5_hardraw.txt"
MED_AUDIO_YAML="${MED_INPUTS}/medicine.audio__medicine5_hardraw.yaml"
for p in "${MED_EVAL_GLOSSARY}" "${MED_SRC_LIST}" "${MED_TGT_LIST}" "${MED_SOURCE_TEXT}" "${MED_REF_FILE}" "${MED_AUDIO_YAML}"; do
  require_file "${p}"
done

sed \
  -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
  -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
  "${ACL_INPUT_DIR}/source.list" > "${ACL_SOURCE_PORTABLE}"
if grep -q '^/mnt/data/' "${ACL_SOURCE_PORTABLE}"; then
  fail "portable source rewrite left node-local /mnt/data paths in ${ACL_SOURCE_PORTABLE}"
fi
while IFS= read -r wav_path; do
  [[ -n "${wav_path}" ]] || continue
  require_file "${wav_path}"
done < "${ACL_SOURCE_PORTABLE}"

MED_DENSITY_TAG="medhard5_cap16denoise_ttag_hn1024_tau0p78_omit_chunks30"
ACL_DENSITY_TAG="tagacl_bv1_decap16den_ttag_hn1024_tau078_omit_chunks30_lm23"

prebuild_index "medicine_hardraw" "${MED_OUTPUT_BASE}" "${MED_DENSITY_TAG}" "${HARD_RAW_GLOSSARY}" "${MED_EVAL_GLOSSARY}" "hard_medicine_raw__medicine5" "${MED_SRC_LIST}" "${MED_TGT_LIST}" "${MED_SOURCE_TEXT}" "${MED_REF_FILE}" "${MED_AUDIO_YAML}"

echo "[WAVE] medicine lm=1,2"
run_wave "medicine_hardraw" "${MED_OUTPUT_BASE}" "${MED_DENSITY_TAG}" "${HARD_RAW_GLOSSARY}" "${MED_EVAL_GLOSSARY}" "hard_medicine_raw__medicine5" "${MED_SRC_LIST}" "${MED_TGT_LIST}" "${MED_SOURCE_TEXT}" "${MED_REF_FILE}" "${MED_AUDIO_YAML}" 1 2

echo "[WAVE] medicine lm=3,4"
run_wave "medicine_hardraw" "${MED_OUTPUT_BASE}" "${MED_DENSITY_TAG}" "${HARD_RAW_GLOSSARY}" "${MED_EVAL_GLOSSARY}" "hard_medicine_raw__medicine5" "${MED_SRC_LIST}" "${MED_TGT_LIST}" "${MED_SOURCE_TEXT}" "${MED_REF_FILE}" "${MED_AUDIO_YAML}" 3 4

merge_summaries "medicine_hardraw" "${MED_OUTPUT_BASE}" 1 2 3 4

echo "[WAVE] tagged ACL lm=2,3"
prebuild_index "acl_tagged_raw" "${ACL_OUTPUT_BASE}" "${ACL_DENSITY_TAG}" "${ACL_RAW_GLOSSARY}" "${ACL_RAW_GLOSSARY}" "acl6060_tagged_gt_raw_min_norm2" "${ACL_SOURCE_PORTABLE}" "${ACL_INPUT_DIR}/target.list" "${ACL_INPUT_DIR}/source_text.txt" "${ACL_INPUT_DIR}/ref.txt" "${ACL_INPUT_DIR}/audio.yaml"
run_wave "acl_tagged_raw" "${ACL_OUTPUT_BASE}" "${ACL_DENSITY_TAG}" "${ACL_RAW_GLOSSARY}" "${ACL_RAW_GLOSSARY}" "acl6060_tagged_gt_raw_min_norm2" "${ACL_SOURCE_PORTABLE}" "${ACL_INPUT_DIR}/target.list" "${ACL_INPUT_DIR}/source_text.txt" "${ACL_INPUT_DIR}/ref.txt" "${ACL_INPUT_DIR}/audio.yaml" 2 3

merge_summaries "acl_tagged_raw" "${ACL_OUTPUT_BASE}" 2 3

date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.success"
echo "[ALL DONE] out_root=${OUT_ROOT}"
echo "[ALL DONE] medicine_summary=${MED_OUTPUT_BASE}/__summary__/summary_medicine_hardraw_de_lm1_lm2_lm3_lm4.tsv"
echo "[ALL DONE] acl_summary=${ACL_OUTPUT_BASE}/__summary__/summary_acl_tagged_raw_de_lm2_lm3.tsv"
