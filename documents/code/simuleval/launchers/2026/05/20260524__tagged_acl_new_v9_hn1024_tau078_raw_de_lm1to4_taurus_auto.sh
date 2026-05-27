#!/usr/bin/env bash
set -euo pipefail

# Tagged ACL main-result evaluation for German.
# Speech LLM: new_v9 assistant term-tag delay, no-gt-zero, old-newv3 base.
# Retriever: HN1024 lh1b88kw, tau=0.78, timeline lookback=1.92s.
# Glossary: fixed raw tagged ACL glossary. Runs lm=1,2,3,4 one at a time,
# waiting for an idle 2-GPU pair on taurus before each setting.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260520__tagged_acl_origin_bsz4_tau073_sweep_taurus45269.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)_tagacl_newv9_hn1024_tau078_raw_de}"

MODEL_LABEL="${MODEL_LABEL:-new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_de_r32a64_tp2_taurus4/keep1.0_r32}"
EXPECTED_MODEL_DIR="${EXPECTED_MODEL_DIR:-${MODEL_ROOT}/v0-20260524-121145-hf}"

HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
GS10K_GLOSSARY="${GS10K_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_union_gs10000_min_norm2_backfill.json}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_taurus_auto.md}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache}"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT:-/tmp/jxl_tacl_de}"

LMS="${LMS:-1 2 3 4}"
PAPERS="${PAPERS:-2022.acl-long.268 2022.acl-long.367 2022.acl-long.590 2022.acl-long.110 2022.acl-long.117}"
GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-0,1;2,3;4,5;6,7}"
MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-1024}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-20}"
POLL_SECS="${POLL_SECS:-120}"
MIN_DATA1_FREE_MB="${MIN_DATA1_FREE_MB:-120000}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_path() {
  local path="$1"
  [[ -e "${path}" ]] || fail "Missing required path: ${path}"
}

validate_model_dir() {
  require_path "${EXPECTED_MODEL_DIR}/config.json"
  local shard_count
  shard_count="$(find "${EXPECTED_MODEL_DIR}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || fail "Expected 15 HF safetensor shards in ${EXPECTED_MODEL_DIR}, found ${shard_count}"
}

check_host() {
  local host
  host="$(hostname -s)"
  [[ "${host}" == taurus* ]] || fail "This launcher should run on taurus; current host=${host}"
}

check_data1_space() {
  local free_mb
  free_mb="$(df -Pm /mnt/gemini/data1 | awk 'NR == 2 {print $4}')"
  [[ -n "${free_mb}" ]] || fail "Could not read free space for /mnt/gemini/data1"
  (( free_mb >= MIN_DATA1_FREE_MB )) || fail "/mnt/gemini/data1 free space ${free_mb} MB < ${MIN_DATA1_FREE_MB} MB"
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

find_idle_pair() {
  local pair g0 g1
  IFS=';' read -r -a pairs <<< "${GPU_PAIRS_CSV}"
  for pair in "${pairs[@]}"; do
    IFS=',' read -r g0 g1 <<< "${pair}"
    if gpu_is_idle "${g0}" && gpu_is_idle "${g1}"; then
      printf '%s\n' "${pair}"
      return 0
    fi
  done
  return 1
}

wait_idle_pair() {
  local lm="$1" pair
  while true; do
    if pair="$(find_idle_pair)"; then
      printf '%s\n' "${pair}"
      return 0
    fi
    echo "[WAIT] lm=${lm}: no idle 2-GPU pair under mem<=${MAX_IDLE_GPU_MEM_MB}MiB util<=${MAX_IDLE_GPU_UTIL}%; retry in ${POLL_SECS}s" >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,name --format=csv,noheader,nounits >&2 || true
    sleep "${POLL_SECS}"
  done
}

validate_tags_for_lm() {
  local lm="$1" pair="$2" compact_pair
  compact_pair="${pair//,/}"
  local tags=(
    "family:tagged_acl_new_v9_hn1024_tau078"
    "task:eval"
    "data:tagged_acl_strict_raw_de"
    "variant:new_v9_hn1024_tau078_de_raw_lm${lm}"
    "compute:taurus_auto_gpu${compact_pair}"
  )
  local tag
  for tag in "${tags[@]}"; do
    (( ${#tag} >= 1 && ${#tag} <= 64 )) || fail "W&B tag length invalid (${#tag}): ${tag}"
  done
}

clean_shm() {
  local me
  me="$(id -un)"
  find /dev/shm -maxdepth 1 -user "${me}" -mmin +1 \
    \( -name 'psm_*' -o -name 'loky-*' -o -name 'torch_*' -o -name 'vllm_*' \) \
    -delete 2>/dev/null || true
}

setting_dir_for_lm() {
  local lm="$1"
  printf '%s/%s/de/dtagacl_new_v9_hn1024_tau078_lm%s_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2\n' \
    "${OUT_ROOT}" "${MODEL_LABEL}" "${lm}"
}

run_one_lm() {
  local lm="$1" pair="$2" compact_pair eval_dir
  compact_pair="${pair//,/}"
  validate_tags_for_lm "${lm}" "${pair}"
  clean_shm
  mkdir -p "${LOG_ROOT}/${MODEL_LABEL}/lm${lm}" "${EVAL_TMPDIR_ROOT}/lm${lm}"
  echo "[RUN] lm=${lm} gpu_pair=${pair} model=${EXPECTED_MODEL_DIR}"
  RUN_STAMP="${RUN_STAMP}_${MODEL_LABEL}_de_lm${lm}" \
  ROOT_DIR="${ROOT_DIR}" \
  MODE="full" \
  RUN_GRANULARITY="full_corpus" \
  HOLD_JOB_ID=0 \
  INSIDE_HOLD_STEP=1 \
  MAX_PARALLEL_OVERRIDE=1 \
  LANGS_OVERRIDE="de" \
  LMS_OVERRIDE="${lm}" \
  GLOSSARY_KINDS_OVERRIDE="raw" \
  PAPERS_OVERRIDE="${PAPERS}" \
  GPU_PAIRS_CSV_OVERRIDE="${pair}" \
  MODEL_NAME_OVERRIDE="${EXPECTED_MODEL_DIR}" \
  TERM_MAP_FORMAT_OVERRIDE="plain" \
  OUTPUT_BASE_OVERRIDE="${OUT_ROOT}/${MODEL_LABEL}" \
  INPUT_ROOT_OVERRIDE="${OUT_ROOT}/__inputs__/${MODEL_LABEL}" \
  LOG_DIR_OVERRIDE="${LOG_ROOT}/${MODEL_LABEL}/lm${lm}" \
  SUMMARY_DIR_OVERRIDE="${OUT_ROOT}/${MODEL_LABEL}/__summary__" \
  INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
  NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
  RAW_GLOSSARY_OVERRIDE="${RAW_GLOSSARY}" \
  GS10K_GLOSSARY_OVERRIDE="${GS10K_GLOSSARY}" \
  EVAL_GLOSSARY_PATH_GLOBAL_OVERRIDE="${RAW_GLOSSARY}" \
  EVAL_GLOSSARY_FOLLOWS_KIND_OVERRIDE=0 \
  EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}/lm${lm}" \
  CLEAN_OUTPUT_DIR_OVERRIDE="${CLEAN_OUTPUT_DIR_OVERRIDE:-1}" \
  DENSITY_TAG_OVERRIDE="tagacl_new_v9_hn1024_tau078" \
  WANDB_RUN_PREFIX_OVERRIDE="new_v9_hn1024_tau078" \
  WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_new_v9_hn1024_tau078" \
  WANDB_VARIANT_PREFIX_OVERRIDE="new_v9_hn1024_tau078" \
  WANDB_COMPUTE_TAG_OVERRIDE="compute:taurus_auto_gpu${compact_pair}" \
  TERM_FCR_POLICY_OVERRIDE="term_map_source_ref_negative_sentence" \
  RAG_MODEL_PATH_OVERRIDE="${HN1024_CKPT}" \
  RAG_SCORE_THRESHOLD_OVERRIDE="0.78" \
  RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE="1.92" \
  RAG_TOP_K_OVERRIDE=10 \
  STRIP_OUTPUT_TAGS_OVERRIDE="term" \
  RAG_GPU_OVERRIDE="cuda:1" \
  VLLM_TP_SIZE_OVERRIDE=2 \
  GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.72}" \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}" \
  VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
  VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/mnt/taurus/home/jiaxuanluo/.config/wandb}" \
  WANDB_DIR="${WANDB_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_runs}" \
  WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_cache}" \
  WANDB_DATA_DIR="${WANDB_DATA_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_data}" \
  WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-/mnt/gemini/data1/jiaxuanluo/wandb_artifacts}" \
  bash "${BASE_LAUNCHER}" \
    > "${LOG_ROOT}/${MODEL_LABEL}/lm${lm}/launcher.out" \
    2> "${LOG_ROOT}/${MODEL_LABEL}/lm${lm}/launcher.err"

  eval_dir="$(setting_dir_for_lm "${lm}")"
  [[ -s "${eval_dir}/eval_results.tsv" ]] || fail "Missing eval_results.tsv after lm=${lm}: ${eval_dir}/eval_results.tsv"
  [[ -s "${eval_dir}/instances.log" ]] || fail "Missing instances.log after lm=${lm}: ${eval_dir}/instances.log"
  [[ -s "${eval_dir}/term_adoption.json" ]] || fail "Missing term_adoption.json after lm=${lm}: ${eval_dir}/term_adoption.json"
  echo "[DONE] lm=${lm}: ${eval_dir}/eval_results.tsv"
}

summarize_results() {
  local summary_tsv="${OUT_ROOT}/${MODEL_LABEL}/__summary__/summary_de_raw_lm1to4.tsv"
  local summary_md="${OUT_ROOT}/${MODEL_LABEL}/__summary__/summary_de_raw_lm1to4.md"
  mkdir -p "$(dirname "${summary_tsv}")"
  python - "${OUT_ROOT}" "${MODEL_LABEL}" "${summary_tsv}" "${summary_md}" ${LMS} <<'PY'
import csv
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
model_label = sys.argv[2]
summary_tsv = Path(sys.argv[3])
summary_md = Path(sys.argv[4])
lms = sys.argv[5:]

rows = []
for lm in lms:
    path = out_root / model_label / "de" / f"dtagacl_new_v9_hn1024_tau078_lm{lm}_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2" / "eval_results.tsv"
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(data)}")
    row = data[0]
    rows.append({
        "lang": "de",
        "lm": lm,
        "glossary": "raw",
        "BLEU": row.get("BLEU", ""),
        "TERM_ACC": row.get("TERM_ACC", ""),
        "REAL_TERM_ADOPT": row.get("REAL_TERM_ADOPT", ""),
        "TERM_FCR": row.get("TERM_FCR", ""),
        "StreamLAAL": row.get("StreamLAAL", ""),
        "eval_results": str(path),
    })

fields = ["lang", "lm", "glossary", "BLEU", "TERM_ACC", "REAL_TERM_ADOPT", "TERM_FCR", "StreamLAAL", "eval_results"]
with summary_tsv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerows(rows)

def pct(x):
    try:
        return f"{float(x) * 100:.2f}"
    except Exception:
        return str(x)

lines = [
    "# Tagged ACL de raw: new_v9 + HN1024 tau=0.78",
    "",
    "| lang | lm | glossary | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | StreamLAAL |",
    "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    lines.append(
        f"| {r['lang']} | {r['lm']} | {r['glossary']} | "
        f"{float(r['BLEU']):.2f} | {pct(r['TERM_ACC'])} | {pct(r['REAL_TERM_ADOPT'])} | "
        f"{pct(r['TERM_FCR'])} | {float(r['StreamLAAL']):.0f} |"
    )
lines += ["", f"Summary TSV: `{summary_tsv}`", f"Output root: `{out_root}`"]
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_tsv)
print(summary_md)
PY
}

preflight() {
  check_host
  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
  check_data1_space
  for p in \
    "${ROOT_DIR}" \
    "${BASE_LAUNCHER}" \
    "${NOTES_FILE}" \
    "${HN1024_CKPT}" \
    "${RAW_GLOSSARY}" \
    "${GS10K_GLOSSARY}" \
    "${ROOT_DIR}/documents/code/simuleval/eval_density_unified.sh" \
    "${ROOT_DIR}/documents/code/offline_evaluation/wandb_eval_logger.py" \
    "${ROOT_DIR}/documents/code/general/wandb_tool.py"; do
    require_path "${p}"
  done
  validate_model_dir
}

main() {
  preflight
  mkdir -p "${OUT_ROOT}/${MODEL_LABEL}/__summary__" "${LOG_ROOT}"
  {
    echo "run_stamp=${RUN_STAMP}"
    echo "host=$(hostname -s)"
    echo "model_label=${MODEL_LABEL}"
    echo "model=${EXPECTED_MODEL_DIR}"
    echo "retriever=${HN1024_CKPT}"
    echo "raw_glossary=${RAW_GLOSSARY}"
    echo "out_root=${OUT_ROOT}"
    echo "log_root=${LOG_ROOT}"
    echo "lms=${LMS}"
    echo "gpu_pairs_csv=${GPU_PAIRS_CSV}"
    echo "tau=0.78"
    echo "lookback_sec=1.92"
    echo "strip_output_tags=term"
  } | tee "${OUT_ROOT}/${MODEL_LABEL}/run_meta.txt"

  local lm pair eval_dir
  for lm in ${LMS}; do
    eval_dir="$(setting_dir_for_lm "${lm}")"
    if [[ -s "${eval_dir}/eval_results.tsv" ]]; then
      echo "[SKIP] lm=${lm} already complete: ${eval_dir}/eval_results.tsv"
      continue
    fi
    pair="$(wait_idle_pair "${lm}")"
    run_one_lm "${lm}" "${pair}"
  done
  summarize_results
  echo "[ALL DONE] Tagged ACL New V9 HN1024 tau0.78 raw de lm=${LMS}"
  echo "[SUMMARY] ${OUT_ROOT}/${MODEL_LABEL}/__summary__/summary_de_raw_lm1to4.md"
}

main "$@"
