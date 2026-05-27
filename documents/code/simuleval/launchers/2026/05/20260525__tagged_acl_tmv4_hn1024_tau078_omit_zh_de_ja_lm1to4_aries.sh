#!/usr/bin/env bash
set -euo pipefail

# Aries same-lm batch readout for tagged ACL raw with the historical TM-SFT SLM
# plus the HN1024 retriever. Languages run sequentially to avoid GPU-pair races;
# within a language, lm=1..4 run concurrently on four 2-GPU pairs.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)_tagacl_tmv4_hn1024_tau078_omit_zh_de_ja_lm1to4_aries}"
LANGS="${LANGS:-zh de ja}"
LMS="${LMS:-1 2 3 4}"
GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-0,1;2,3;4,5;6,7}"

MODEL_ROOT="${MODEL_ROOT:-/mnt/gemini/data/jiaxuanluo/owaski}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260525__tagged_acl_tmv4_hn1024_tau078_omit_zh_de_ja_lm1to4_aries.md}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/batched_vllm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

EMPTY_TERM_MAP_POLICY="${EMPTY_TERM_MAP_POLICY:-omit}"
OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_tmv4_hn1024_tau078_omit_zh_de_ja_lm1to4_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_tmv4_hn1024_tau078_omit_zh_de_ja_lm1to4_${RUN_STAMP}}"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT:-/tmp/jx_tmv4}"

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

model_for_lang() {
  local lang="$1"
  echo "${MODEL_ROOT}/gigaspeech-${lang}-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4"
}

input_for_lang() {
  local lang="$1"
  case "${lang}" in
    zh)
      echo "/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_zh_lm1_20260524T0507_tagacl_newv9_hn1024_tau078_raw_zh_lm1_aries01/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/zh/all"
      ;;
    de)
      echo "/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all"
      ;;
    ja)
      echo "/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_20260524T0933_tagacl_newv9_hn1024_tau078_raw_ja_taurus8/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/ja/all"
      ;;
    *)
      fail "Unsupported language: ${lang}"
      ;;
  esac
}

validate_wandb_tags() {
  local tag lang lm pair
  for lang in ${LANGS}; do
    local tags=(
      "family:tagged_acl_tmv4_hn1024_tau078_omit"
      "task:eval"
      "data:tagacl_raw_${lang}"
    )
    IFS=';' read -r -a pairs <<< "${GPU_PAIRS_CSV}"
    for pair in "${pairs[@]}"; do
      tags+=("compute:aries_gpu${pair//,/}")
    done
    for lm in ${LMS}; do
      tags+=("variant:tmv4_t078_${lang}_raw_lm${lm}_omit")
    done
    for tag in "${tags[@]}"; do
      (( ${#tag} >= 1 && ${#tag} <= 64 )) || fail "W&B tag length invalid (${#tag}): ${tag}"
    done
  done
}

prepare_lang_inputs() {
  local lang="$1" input_dir="$2" out_root="$3" src_list_portable
  src_list_portable="${out_root}/__inputs__/${lang}/source.portable.list"
  mkdir -p "$(dirname "${src_list_portable}")"
  for f in source.list target.list source_text.txt ref.txt audio.yaml; do
    require_file "${input_dir}/${f}"
  done
  sed \
    -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
    -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
    "${input_dir}/source.list" > "${src_list_portable}"
  if grep -q '^/mnt/data/' "${src_list_portable}"; then
    fail "portable source rewrite left node-local /mnt/data paths in ${src_list_portable}"
  fi
  while IFS= read -r wav_path; do
    [[ -n "${wav_path}" ]] || continue
    require_file "${wav_path}"
  done < "${src_list_portable}"
  echo "${src_list_portable}"
}

write_lm_summary() {
  local lang="$1" output_base="$2" model_label="$3" lm="$4"
  "${PYTHON_BIN}" - "${lang}" "${output_base}" "${model_label}" "${lm}" <<'PY'
import csv
import sys
from pathlib import Path

lang = sys.argv[1]
output_base = Path(sys.argv[2])
model_label = sys.argv[3]
lm = sys.argv[4]
paths = sorted(output_base.glob(f"{lang}/*_lm{lm}_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one eval_results.tsv for {lang}/lm={lm}, found {len(paths)}: {paths}")
eval_path = paths[0]
with eval_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {eval_path}, got {len(rows)}")
out_dir = eval_path.parent
inst = out_dir / "instances.log"
strip = out_dir / "instances.strip_term.log"
if not inst.is_file() or inst.stat().st_size <= 0:
    raise SystemExit(f"missing/empty instances.log for {lang}/lm={lm}: {inst}")
if not strip.is_file() or strip.stat().st_size <= 0:
    raise SystemExit(f"missing/empty instances.strip_term.log for {lang}/lm={lm}: {strip}")
inst_rows = sum(1 for _ in inst.open("r", encoding="utf-8"))
strip_rows = sum(1 for _ in strip.open("r", encoding="utf-8"))
if inst_rows != 5 or strip_rows != 5:
    raise SystemExit(f"expected 5 instance rows for {lang}/lm={lm}, got raw={inst_rows} strip={strip_rows}")
row = rows[0]
summary_dir = output_base / "__summary__"
summary_dir.mkdir(parents=True, exist_ok=True)
out = summary_dir / f"summary_{lang}_lm{lm}.tsv"
record = {
    "method_key": model_label,
    "lang": row.get("lang_code", lang),
    "lm": lm,
    "BLEU": row.get("BLEU", ""),
    "StreamLAAL": row.get("StreamLAAL", ""),
    "StreamLAAL_CA": row.get("StreamLAAL_CA", ""),
    "TERM_ACC": row.get("TERM_ACC", ""),
    "TERM_CORRECT": row.get("TERM_CORRECT", ""),
    "TERM_TOTAL": row.get("TERM_TOTAL", ""),
    "REAL_TERM_ADOPT": row.get("REAL_TERM_ADOPT", ""),
    "TERM_FCR": row.get("TERM_FCR", ""),
    "empty_term_map_policy": "omit",
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
    f"\tlang={lang}"
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

merge_lang_summaries() {
  local lang="$1" output_base="$2"
  "${PYTHON_BIN}" - "${lang}" "${output_base}" ${LMS} <<'PY'
import csv
import sys
from pathlib import Path

lang = sys.argv[1]
output_base = Path(sys.argv[2])
lms = sys.argv[3:]
summary_dir = output_base / "__summary__"
rows = []
for lm in lms:
    path = summary_dir / f"summary_{lang}_lm{lm}.tsv"
    if not path.is_file():
        raise SystemExit(f"missing lm summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(data)}")
    rows.extend(data)
fields = list(rows[0])
out = summary_dir / f"summary_{lang}_lm1to4.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: int(r["lm"])))
print(f"[SUMMARY] {out}", flush=True)
PY
}

merge_all_summaries() {
  "${PYTHON_BIN}" - "${OUT_ROOT}" ${LANGS} <<'PY'
import csv
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
langs = sys.argv[2:]
rows = []
for lang in langs:
    paths = sorted(out_root.glob(f"tmv4_{lang}_bsz4_hn1024_tau078_omit_batch_max80/__summary__/summary_{lang}_lm1to4.tsv"))
    if len(paths) != 1:
        raise SystemExit(f"expected one merged summary for {lang}, found {len(paths)}: {paths}")
    with paths[0].open("r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f, delimiter="\t"))
if not rows:
    raise SystemExit("no rows to merge")
fields = list(rows[0])
out = out_root / "summary_tmv4_hn1024_tau078_omit_zh_de_ja_lm1to4.tsv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda r: (r["lang"], int(r["lm"]))))
print(f"[SUMMARY_ALL] {out}", flush=True)
PY
}

run_one_lm() {
  local lang="$1" lm="$2" pair="$3" model_name="$4" input_dir="$5" output_base="$6" model_label="$7" density_tag="$8" src_list_portable="$9"
  local compact_pair lm_log
  compact_pair="${pair//,/}"
  lm_log="${LOG_ROOT}/${lang}/lm${lm}"
  mkdir -p "${lm_log}" "${EVAL_TMPDIR_ROOT}/${lang}${lm}"
  wait_pair_idle "${pair}"
  clean_shm
  echo "[RUN] lang=${lang} lm=${lm} gpu_pair=${pair}"
  RUN_TAG_OVERRIDE="${RUN_STAMP}_${lang}_lm${lm}" \
  ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
  LANG_CODE_OVERRIDE="${lang}" \
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
  MODEL_NAME_OVERRIDE="${model_name}" \
  SRC_LIST_OVERRIDE="${src_list_portable}" \
  TGT_LIST_OVERRIDE="${input_dir}/target.list" \
  SOURCE_TEXT_FILE_OVERRIDE="${input_dir}/source_text.txt" \
  REF_FILE_OVERRIDE="${input_dir}/ref.txt" \
  AUDIO_YAML_OVERRIDE="${input_dir}/audio.yaml" \
  GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
  OUTPUT_BASE_OVERRIDE="${output_base}" \
  DENSITY_TAG_OVERRIDE="${density_tag}" \
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
  EMPTY_TERM_MAP_POLICY_OVERRIDE="${EMPTY_TERM_MAP_POLICY}" \
  TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
  STRIP_OUTPUT_TAGS_OVERRIDE=term \
  WANDB_LOG_OVERRIDE="${WANDB_LOG_OVERRIDE:-0}" \
  WANDB_RUN_PREFIX_OVERRIDE="tmv4_hn1024_t078_omit" \
  WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_tmv4_hn1024_tau078_omit" \
  WANDB_VARIANT_PREFIX_OVERRIDE="tmv4_t078_omit" \
  WANDB_COMPUTE_TAG_OVERRIDE="compute:aries_gpu${compact_pair}" \
  WANDB_RUNTIME_GLOSSARY_LABEL_OVERRIDE="raw" \
  WANDB_DATA_TAG_OVERRIDE="tagacl_raw_${lang}" \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  LOG_ROOT_OVERRIDE="${lm_log}/batch" \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}/${lang}${lm}" \
  bash "${BATCH_LAUNCHER}" \
    > "${lm_log}/launcher.out" \
    2> "${lm_log}/launcher.err"
  write_lm_summary "${lang}" "${output_base}" "${model_label}" "${lm}" | tee "${lm_log}/result.txt"
}

run_one_lang() {
  local lang="$1" model_name input_dir output_base model_label density_tag src_list_portable shard_count
  model_name="$(model_for_lang "${lang}")"
  input_dir="$(input_for_lang "${lang}")"
  model_label="tmv4_${lang}_bsz4_hn1024_tau078_omit_batch_max80"
  density_tag="tagacl_bv1_tmv4_hn1024_tau078_omit"
  output_base="${OUT_ROOT}/${model_label}"

  require_file "${model_name}/config.json"
  require_file "${model_name}/generation_config.json"
  require_file "${model_name}/model.safetensors.index.json"
  shard_count="$(find "${model_name}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || fail "Expected 15 HF safetensor shards in ${model_name}, found ${shard_count}"
  src_list_portable="$(prepare_lang_inputs "${lang}" "${input_dir}" "${OUT_ROOT}")"

  mkdir -p "${output_base}/__summary__" "${LOG_ROOT}/${lang}"
  {
    echo "run_stamp=${RUN_STAMP}"
    echo "host=$(hostname -s)"
    echo "lang=${lang}"
    echo "gpu_pairs=${GPU_PAIRS_CSV}"
    echo "model=${model_name}"
    echo "model_label=${model_label}"
    echo "density_tag=${density_tag}"
    echo "empty_term_map_policy=${EMPTY_TERM_MAP_POLICY}"
    echo "input_dir=${input_dir}"
    echo "source_list_portable=${src_list_portable}"
    echo "raw_glossary=${RAW_GLOSSARY}"
    echo "retriever=${HN1024_CKPT}"
    echo "output_base=${output_base}"
    echo "lms=${LMS}"
    echo "eval_mode=same_lm_batch_v1_parallel"
    echo "talks_per_lm=5"
    echo "tau=0.78"
    echo "lookback_sec=1.92"
    echo "max_new_tokens=80"
    echo "vllm_limit_audio=128"
    echo "vllm_max_model_len=12288"
    echo "strip_output_tags=term"
  } | tee "${output_base}/run_meta.txt"

  if [[ "${DRY_RUN_OVERRIDE:-0}" == "1" ]]; then
    echo "[DRY_RUN] lang=${lang} validated inputs and wrote run_meta; not launching vLLM"
    return 0
  fi

  IFS=';' read -r -a gpu_pairs <<< "${GPU_PAIRS_CSV}"
  IFS=' ' read -r -a lm_values <<< "${LMS}"
  (( ${#gpu_pairs[@]} >= ${#lm_values[@]} )) || fail "Need ${#lm_values[@]} GPU pairs, got ${#gpu_pairs[@]}"

  local pids=() idx lm pair status
  for idx in "${!lm_values[@]}"; do
    lm="${lm_values[$idx]}"
    pair="${gpu_pairs[$idx]}"
    ( run_one_lm "${lang}" "${lm}" "${pair}" "${model_name}" "${input_dir}" "${output_base}" "${model_label}" "${density_tag}" "${src_list_portable}" ) &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  (( status == 0 )) || fail "At least one ${lang} lm batch eval failed; inspect ${LOG_ROOT}/${lang}"
  merge_lang_summaries "${lang}" "${output_base}" | tee "${output_base}/__summary__/summary_path.txt"
}

require_file "${BATCH_LAUNCHER}"
require_file "${RAW_GLOSSARY}"
require_file "${HN1024_CKPT}"
require_file "${NOTES_FILE}"
validate_wandb_tags

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${EVAL_TMPDIR_ROOT}"
df -h /mnt/gemini/data1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_prelaunch.csv"
{
  echo "run_stamp=${RUN_STAMP}"
  echo "langs=${LANGS}"
  echo "lms=${LMS}"
  echo "gpu_pairs=${GPU_PAIRS_CSV}"
  echo "empty_term_map_policy=${EMPTY_TERM_MAP_POLICY}"
  echo "output_root=${OUT_ROOT}"
  echo "log_root=${LOG_ROOT}"
} | tee "${OUT_ROOT}/run_meta.txt"

for lang in ${LANGS}; do
  echo "[LANG_START] ${lang}"
  run_one_lang "${lang}"
  echo "[LANG_DONE] ${lang}"
done

if [[ "${DRY_RUN_OVERRIDE:-0}" == "1" ]]; then
  echo "[DRY_RUN] validated all requested languages; not merging runtime summaries"
  exit 0
fi

merge_all_summaries | tee "${OUT_ROOT}/summary_path.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_postrun.csv"
echo "[ALL DONE] output_root=${OUT_ROOT}"
