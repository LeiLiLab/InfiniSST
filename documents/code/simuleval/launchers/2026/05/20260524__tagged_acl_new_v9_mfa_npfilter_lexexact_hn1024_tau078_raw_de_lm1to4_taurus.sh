#!/usr/bin/env bash
set -euo pipefail

# Tagged ACL de raw readout for the clean MFA/source-filtered New V9 Speech LLM.
# Uses the same-lm batched evaluator: each lm runs all five ACL talks together
# in one vLLM process, instead of launching one process per talk.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)_tagacl_newv9_mfa_npfilter_de_batch}"

MODEL_LABEL="${MODEL_LABEL:-new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64_hn1024_tau078_same_lm_batch_v1}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_r32a64_tp2_taurus8/keep1.0_r32}"
MODEL_NAME="${MODEL_NAME:-${MODEL_ROOT}/v0-20260525-001708-hf}"

INPUT_DIR="${INPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"

NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_de_lm1to4_taurus.md}"
OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_same_lm_batch_v1_mfa_npfilter_hn1024_tau078_raw_de_lm1to4_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_same_lm_batch_v1_mfa_npfilter_hn1024_tau078_raw_de_lm1to4_${RUN_STAMP}}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/batched_vllm}"
EVAL_TMPDIR_ROOT="${EVAL_TMPDIR_ROOT:-/tmp/jx_bvde_mfa}"

LMS="${LMS:-1 2 3 4}"
GPU_PAIRS_CSV="${GPU_PAIRS_CSV:-0,1;2,3;4,5;6,7}"
WAIT_SECS="${WAIT_SECS:-60}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-14400}"
MAX_IDLE_GPU_MEM_MB="${MAX_IDLE_GPU_MEM_MB:-2048}"
MAX_IDLE_GPU_UTIL="${MAX_IDLE_GPU_UTIL:-25}"
POLL_SECS="${POLL_SECS:-60}"

fail() {
  echo "[ERROR] $*" >&2
  exit 3
}

require_file() {
  local path="$1"
  [[ -s "${path}" ]] || fail "Missing/empty required file: ${path}"
}

hf_ready() {
  [[ -s "${MODEL_NAME}/config.json" ]] || return 1
  [[ -s "${MODEL_NAME}/generation_config.json" ]] || return 1
  [[ -s "${MODEL_NAME}/model.safetensors.index.json" ]] || return 1
  [[ -s "${MODEL_NAME}/tokenizer_config.json" ]] || return 1
  local shard_count
  shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  [[ "${shard_count}" == "15" ]] || return 1
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
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits >&2 || true
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

require_file "${BATCH_LAUNCHER}"
require_file "${RAW_GLOSSARY}"
require_file "${HN1024_CKPT}"
for f in source.list target.list source_text.txt ref.txt audio.yaml; do
  require_file "${INPUT_DIR}/${f}"
done

mkdir -p "${OUT_ROOT}" "${OUTPUT_BASE}/__summary__" "${LOG_ROOT}" "${EVAL_TMPDIR_ROOT}" "$(dirname "${NOTES_FILE}")"

echo "[INFO] waiting for complete HF model: ${MODEL_NAME}"
elapsed=0
while ! hf_ready; do
  shard_count="$(find "${MODEL_NAME}" -maxdepth 1 -type f -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
  echo "[WAIT] hf export incomplete: shards=${shard_count}/15 elapsed=${elapsed}s"
  (( elapsed < MAX_WAIT_SECS )) || fail "Timed out waiting for HF export: ${MODEL_NAME}"
  sleep "${WAIT_SECS}"
  elapsed=$((elapsed + WAIT_SECS))
done

{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "model=${MODEL_NAME}"
  echo "input_dir=${INPUT_DIR}"
  echo "raw_glossary=${RAW_GLOSSARY}"
  echo "retriever=${HN1024_CKPT}"
  echo "output_base=${OUTPUT_BASE}"
  echo "lms=${LMS}"
  echo "eval_mode=same_lm_batch_v1"
  echo "talks_per_lm=5"
  echo "tau=0.78"
  echo "lookback_sec=1.92"
  echo "strip_output_tags=term"
} | tee "${OUT_ROOT}/run_meta.txt"

IFS=';' read -r -a gpu_pairs <<< "${GPU_PAIRS_CSV}"
IFS=' ' read -r -a lm_values <<< "${LMS}"
if (( ${#gpu_pairs[@]} < ${#lm_values[@]} )); then
  fail "Need at least ${#lm_values[@]} GPU pairs, got ${#gpu_pairs[@]} from GPU_PAIRS_CSV=${GPU_PAIRS_CSV}"
fi

pids=()
for idx in "${!lm_values[@]}"; do
  lm="${lm_values[$idx]}"
  pair="${gpu_pairs[$idx]}"
  (
    wait_pair_idle "${pair}"
    clean_shm
    lm_log="${LOG_ROOT}/lm${lm}"
    mkdir -p "${lm_log}" "${EVAL_TMPDIR_ROOT}/lm${lm}"
    echo "[RUN] lm=${lm} gpu_pair=${pair}"
    RUN_TAG_OVERRIDE="${RUN_STAMP}_de_lm${lm}" \
    LANG_CODE_OVERRIDE="de" \
    LMS_OVERRIDE="${lm}" \
    CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${pair}" \
    VLLM_TP_SIZE_OVERRIDE=2 \
    MAX_NUM_SEQS_OVERRIDE=5 \
    SCHEDULER_BATCH_SIZE_OVERRIDE=5 \
    SCHEDULE_MODE_OVERRIDE=round_robin \
    VLLM_ENFORCE_EAGER_OVERRIDE=1 \
    VLLM_ENABLE_PREFIX_CACHING=1 \
    VLLM_LIMIT_AUDIO_OVERRIDE=64 \
    MAX_CACHE_SECONDS_OVERRIDE=80 \
    KEEP_CACHE_SECONDS_OVERRIDE=60 \
    MIN_CACHE_CHUNKS_OVERRIDE=1 \
    MAX_NEW_TOKENS_OVERRIDE=80 \
    MAX_NEW_TOKENS_POLICY_OVERRIDE=fixed \
    TEMPERATURE_OVERRIDE=0.6 \
    TOP_P_OVERRIDE=0.95 \
    TOP_K_DECODE_OVERRIDE=20 \
    GPU_MEMORY_UTILIZATION_OVERRIDE="${GPU_MEMORY_UTILIZATION_OVERRIDE:-0.80}" \
    VLLM_MAX_MODEL_LEN_OVERRIDE="${VLLM_MAX_MODEL_LEN_OVERRIDE:-16384}" \
    VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
    VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}" \
    VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
    MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
    SRC_LIST_OVERRIDE="${INPUT_DIR}/source.list" \
    TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
    SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
    REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
    AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
    GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
    EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
    OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
    DENSITY_TAG_OVERRIDE="tagacl_bv1_mfa_np_hn1024_tau078" \
    GLOSSARY_TAG_OVERRIDE="acl6060_tagged_gt_raw_min_norm2" \
    RAG_SCORE_THRESHOLD_OVERRIDE=0.78 \
    RAG_TIMELINE_LOOKBACK_SEC_OVERRIDE=1.92 \
    RAG_TOP_K_OVERRIDE=10 \
    RAG_DEVICE_OVERRIDE="${RAG_DEVICE_OVERRIDE:-cuda:0}" \
    RAG_BATCH_RETRIEVAL_OVERRIDE=1 \
    INDEX_BUILD_DEVICE_OVERRIDE="${INDEX_BUILD_DEVICE_OVERRIDE:-cuda:0}" \
    INDEX_CACHE_DIR_OVERRIDE="${INDEX_CACHE_DIR}" \
    TERM_MAP_FORMAT_OVERRIDE=plain \
    TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
    STRIP_OUTPUT_TAGS_OVERRIDE=term \
    WANDB_LOG_OVERRIDE="${WANDB_LOG_OVERRIDE:-1}" \
    WANDB_RUN_PREFIX_OVERRIDE="mfa_np_bv1_hn1024_t078" \
    WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_same_lm_batch_v1_hn1024_tau078" \
    WANDB_VARIANT_PREFIX_OVERRIDE="mfa_np_bv1_hn1024_t078" \
    WANDB_COMPUTE_TAG_OVERRIDE="compute:taurus_bv1_gpu${pair//,/}" \
    WANDB_RUNTIME_GLOSSARY_LABEL_OVERRIDE="raw" \
    WANDB_DATA_TAG_OVERRIDE="tagged_acl_strict_raw_de" \
    NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
    LOG_ROOT_OVERRIDE="${lm_log}" \
    EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR_ROOT}/lm${lm}" \
    bash "${BATCH_LAUNCHER}" \
      > "${lm_log}/launcher.out" \
      2> "${lm_log}/launcher.err"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
(( status == 0 )) || fail "At least one lm batch eval failed; inspect ${LOG_ROOT}"

python - "${OUTPUT_BASE}" "${OUT_ROOT}" "${MODEL_LABEL}" ${LMS} <<'PY'
import csv
import re
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
out_root = Path(sys.argv[2])
model_label = sys.argv[3]
lms = sys.argv[4:]
rows = []
for lm in lms:
    paths = sorted(output_base.glob(f"de/**_lm{lm}_*/eval_results.tsv"))
    if len(paths) != 1:
        raise SystemExit(f"expected one eval_results.tsv for lm={lm}, found {len(paths)}: {paths}")
    with paths[0].open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f, delimiter="\t"))
    if len(data) != 1:
        raise SystemExit(f"expected one row in {paths[0]}, got {len(data)}")
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
        "eval_results": str(paths[0]),
    })

summary_dir = output_base / "__summary__"
summary_dir.mkdir(parents=True, exist_ok=True)
fields = ["lang", "lm", "glossary", "BLEU", "TERM_ACC", "REAL_TERM_ADOPT", "TERM_FCR", "StreamLAAL", "eval_results"]
tsv = summary_dir / "summary_de_raw_lm1to4_same_lm_batch_v1.tsv"
with tsv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerows(rows)

def pct(x):
    try:
        return f"{float(x) * 100:.2f}"
    except Exception:
        return str(x)

lines = [
    "# Tagged ACL de raw: clean New V9 + HN1024 tau=0.78, same-lm batch v1",
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
lines += ["", f"Summary TSV: `{tsv}`", f"Output base: `{output_base}`", f"Output root: `{out_root}`"]
md = summary_dir / "summary_de_raw_lm1to4_same_lm_batch_v1.md"
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md)
print(tsv)
PY

echo "[ALL DONE] ${OUTPUT_BASE}/__summary__/summary_de_raw_lm1to4_same_lm_batch_v1.md"
