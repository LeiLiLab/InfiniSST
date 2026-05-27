#!/usr/bin/env bash
set -euo pipefail

# Two-way En-De tagged ACL lm=2 batch probe.
# METHOD_KEY selects the speech LLM while the HN1024 retriever/protocol stays fixed.

ROOT_DIR="${ROOT_DIR:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

METHOD_KEY="${METHOD_KEY:?Set METHOD_KEY=origin_hn1024 or tmv4_hn1024}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%S)_tagacl_de_lm2_${METHOD_KEY}_batch}"
GPU_PAIR="${GPU_PAIR:?Set GPU_PAIR, e.g. 0,1 or 2,7}"

BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
INPUT_DIR="${INPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
HN1024_CKPT="${HN1024_CKPT:-/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/mnt/gemini/data1/jiaxuanluo/maxsim_index_cache/batched_vllm}"

case "${METHOD_KEY}" in
  origin_hn1024)
    MODEL_NAME="${MODEL_NAME:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4}"
    MODEL_LABEL="${MODEL_LABEL:-origin_de_bsz4_hn1024_tau078_batch_lm2}"
    WANDB_PREFIX="${WANDB_PREFIX:-orig_hn1024_t078}"
    NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__tagged_acl_de_lm2_origin_hn1024_batch.md}"
    ;;
  tmv4_hn1024)
    MODEL_NAME="${MODEL_NAME:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4}"
    MODEL_LABEL="${MODEL_LABEL:-tmv4_de_bsz4_hn1024_tau078_batch_lm2}"
    WANDB_PREFIX="${WANDB_PREFIX:-tmv4_hn1024_t078}"
    NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__tagged_acl_de_lm2_tmv4_hn1024_batch.md}"
    ;;
  *)
    echo "[ERROR] Unknown METHOD_KEY=${METHOD_KEY}" >&2
    exit 2
    ;;
esac

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_de_lm2_${METHOD_KEY}_batch_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/${MODEL_LABEL}}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_de_lm2_${METHOD_KEY}_batch_${RUN_STAMP}}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_de2_${METHOD_KEY}}"
COMPUTE_TAG="${COMPUTE_TAG:-compute:aries${GPU_PAIR//,/}}"
INPUT_WORK_DIR="${OUT_ROOT}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_WORK_DIR}/source.portable.list"

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
  (( mem <= 2048 && util <= 25 ))
}

require_file "${BATCH_LAUNCHER}"
require_file "${MODEL_NAME}/config.json"
require_file "${RAW_GLOSSARY}"
require_file "${HN1024_CKPT}"
for f in source.list target.list source_text.txt ref.txt audio.yaml; do
  require_file "${INPUT_DIR}/${f}"
done

IFS=',' read -r gpu0 gpu1 <<< "${GPU_PAIR}"
gpu_is_idle "${gpu0}" || fail "GPU ${gpu0} is not idle"
gpu_is_idle "${gpu1}" || fail "GPU ${gpu1} is not idle"

mkdir -p "${OUT_ROOT}" "${OUTPUT_BASE}/__summary__" "${LOG_ROOT}" "${EVAL_TMPDIR}" "${INPUT_WORK_DIR}"

sed \
  -e 's#^/mnt/data/siqiouyang/#/mnt/taurus/data/siqiouyang/#' \
  -e 's#^/mnt/data1/siqiouyang/#/mnt/taurus/data1/siqiouyang/#' \
  "${INPUT_DIR}/source.list" > "${SRC_LIST_PORTABLE}"
if grep -q '^/mnt/data/' "${SRC_LIST_PORTABLE}"; then
  fail "portable source rewrite left node-local /mnt/data paths in ${SRC_LIST_PORTABLE}"
fi
while IFS= read -r wav_path; do
  [[ -n "${wav_path}" ]] || continue
  require_file "${wav_path}"
done < "${SRC_LIST_PORTABLE}"

df -h /mnt/gemini/data1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_prelaunch.csv"

{
  echo "run_stamp=${RUN_STAMP}"
  echo "method_key=${METHOD_KEY}"
  echo "host=$(hostname -s)"
  echo "gpu_pair=${GPU_PAIR}"
  echo "model=${MODEL_NAME}"
  echo "input_dir=${INPUT_DIR}"
  echo "source_list_portable=${SRC_LIST_PORTABLE}"
  echo "raw_glossary=${RAW_GLOSSARY}"
  echo "retriever=${HN1024_CKPT}"
  echo "output_base=${OUTPUT_BASE}"
  echo "lm=2"
  echo "eval_mode=same_lm_batch_v1"
  echo "talks_per_lm=5"
  echo "tau=0.78"
  echo "lookback_sec=1.92"
  echo "max_new_tokens=80"
  echo "strip_output_tags=term"
} | tee "${OUT_ROOT}/run_meta.txt"

RUN_TAG_OVERRIDE="${RUN_STAMP}_de_lm2" \
ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
LANG_CODE_OVERRIDE="de" \
LMS_OVERRIDE="2" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
VLLM_TP_SIZE_OVERRIDE=2 \
MAX_NUM_SEQS_OVERRIDE=5 \
SCHEDULER_BATCH_SIZE_OVERRIDE=5 \
SCHEDULE_MODE_OVERRIDE=round_robin \
VLLM_ENFORCE_EAGER_OVERRIDE=1 \
VLLM_ENABLE_PREFIX_CACHING=1 \
VLLM_LIMIT_AUDIO_OVERRIDE=128 \
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
SRC_LIST_OVERRIDE="${SRC_LIST_PORTABLE}" \
TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
DENSITY_TAG_OVERRIDE="tagacl_bv1_${METHOD_KEY}_hn1024_tau078" \
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
TERM_FCR_POLICY_OVERRIDE=term_map_source_ref_negative_sentence \
STRIP_OUTPUT_TAGS_OVERRIDE=term \
WANDB_LOG_OVERRIDE="${WANDB_LOG_OVERRIDE:-1}" \
WANDB_RUN_PREFIX_OVERRIDE="${WANDB_PREFIX}" \
WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_de_lm2_hn1024_batch_cmp" \
WANDB_VARIANT_PREFIX_OVERRIDE="${WANDB_PREFIX}" \
WANDB_COMPUTE_TAG_OVERRIDE="${COMPUTE_TAG}" \
WANDB_RUNTIME_GLOSSARY_LABEL_OVERRIDE="raw" \
WANDB_DATA_TAG_OVERRIDE="tagacl_raw_de" \
NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
LOG_ROOT_OVERRIDE="${LOG_ROOT}/batch" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
bash "${BATCH_LAUNCHER}" \
  > "${LOG_ROOT}/launcher.out" \
  2> "${LOG_ROOT}/launcher.err"

python - "${OUTPUT_BASE}" "${METHOD_KEY}" <<'PY'
import csv
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
method_key = sys.argv[2]
paths = sorted(output_base.glob("de/**_lm2_*/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one lm2 eval_results.tsv, found {len(paths)}: {paths}")
with paths[0].open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {paths[0]}, got {len(rows)}")
row = rows[0]
summary = output_base / "__summary__" / "summary_de_lm2.tsv"
fields = [
    "method_key", "lang", "lm", "BLEU", "StreamLAAL", "StreamLAAL_CA",
    "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL", "REAL_TERM_ADOPT", "TERM_FCR",
    "eval_results", "instances_log",
]
with summary.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerow({
        "method_key": method_key,
        "lang": row.get("lang_code", "de"),
        "lm": "2",
        "BLEU": row.get("BLEU", ""),
        "StreamLAAL": row.get("StreamLAAL", ""),
        "StreamLAAL_CA": row.get("StreamLAAL_CA", ""),
        "TERM_ACC": row.get("TERM_ACC", ""),
        "TERM_CORRECT": row.get("TERM_CORRECT", ""),
        "TERM_TOTAL": row.get("TERM_TOTAL", ""),
        "REAL_TERM_ADOPT": row.get("REAL_TERM_ADOPT", ""),
        "TERM_FCR": row.get("TERM_FCR", ""),
        "eval_results": str(paths[0]),
        "instances_log": row.get("instances_log", ""),
    })
print(summary)
PY

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_postrun.csv"
echo "[ALL DONE] ${OUTPUT_BASE}/__summary__/summary_de_lm2.tsv"
