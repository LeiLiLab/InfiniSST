#!/usr/bin/env bash
set -euo pipefail

# En-De tagged ACL raw lm=2 InfiniSST/no-RAG batch verification.

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
cd "${ROOT_DIR}"

RUN_STAMP="${RUN_STAMP:-20260524T2200_tagacl_origin_norag_de_lm2_batch_max80_aries01}"
GPU_PAIR="${GPU_PAIR:-0,1}"

BATCH_LAUNCHER="${BATCH_LAUNCHER:-${ROOT_DIR}/documents/code/simuleval/launchers/2026/05/20260524__batched_vllm_rag_eval.sh}"
MODEL_NAME="${MODEL_NAME:-/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4}"
INPUT_DIR="${INPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_20260524T075533_tagacl_newv9_hn1024_tau078_raw_de_retry1/__inputs__/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/full_all/de/all}"
RAW_GLOSSARY="${RAW_GLOSSARY:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json}"
NOTES_FILE="${NOTES_FILE:-${ROOT_DIR}/documents/code/simuleval/notes/2026/05/20260524__tagged_acl_origin_norag_de_lm2_batch_max80_aries01.md}"

OUT_ROOT="${OUT_ROOT:-/mnt/gemini/data1/jiaxuanluo/tagged_acl_origin_norag_de_lm2_batch_max80_aries01_${RUN_STAMP}}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUT_ROOT}/origin_norag_de_lm2_batch_max80}"
LOG_ROOT="${LOG_ROOT:-/mnt/gemini/data1/jiaxuanluo/logs/tagged_acl_origin_norag_de_lm2_batch_max80_aries01_${RUN_STAMP}}"
CACHE_BASE="${CACHE_BASE:-/mnt/gemini/data1/jiaxuanluo/cache/tagged_acl_origin_norag_de_lm2_batch_max80_aries01_${RUN_STAMP}}"
EVAL_TMPDIR="${EVAL_TMPDIR:-/tmp/jx_tacl_de2_norag}"
INPUT_WORK_DIR="${OUT_ROOT}/__inputs__"
SRC_LIST_PORTABLE="${INPUT_WORK_DIR}/source.portable.list"
DENSITY_TAG="tagacl_origin_norag_batch_max80"
GLOSSARY_TAG="acl6060_tagged_gt_raw_min_norm2"
BATCH_DIR="${OUTPUT_BASE}/de/d${DENSITY_TAG}_lm2_k0_th0.0_g${GLOSSARY_TAG}"

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

if [[ -e "${OUT_ROOT}/.started" && "${FORCE_RERUN_OVERRIDE:-0}" != "1" ]]; then
  fail "output root already started: ${OUT_ROOT}"
fi

require_file "${BATCH_LAUNCHER}"
require_file "${MODEL_NAME}/config.json"
require_file "${RAW_GLOSSARY}"
for f in source.list target.list source_text.txt ref.txt audio.yaml; do
  require_file "${INPUT_DIR}/${f}"
done

IFS=',' read -r gpu0 gpu1 <<< "${GPU_PAIR}"
gpu_is_idle "${gpu0}" || fail "GPU ${gpu0} is not idle"
gpu_is_idle "${gpu1}" || fail "GPU ${gpu1} is not idle"

mkdir -p "${OUT_ROOT}" "${OUTPUT_BASE}" "${LOG_ROOT}" "${CACHE_BASE}" "${EVAL_TMPDIR}" "${INPUT_WORK_DIR}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.started"

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

export XDG_CACHE_HOME="${CACHE_BASE}/xdg"
export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/torchinductor"
export CUDA_CACHE_PATH="${CACHE_BASE}/cuda"
export HF_HOME="${CACHE_BASE}/hf"
export HF_HUB_CACHE="${CACHE_BASE}/hf/hub"
export TRANSFORMERS_CACHE="${CACHE_BASE}/hf/transformers"
export VLLM_CACHE_ROOT="${CACHE_BASE}/vllm"
export NUMBA_CACHE_DIR="${CACHE_BASE}/numba"
export WANDB_CACHE_DIR="${CACHE_BASE}/wandb_cache"
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" \
  "${NUMBA_CACHE_DIR}" "${WANDB_CACHE_DIR}"

df -h /mnt/gemini/data1 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_prelaunch.csv"

{
  echo "run_stamp=${RUN_STAMP}"
  echo "host=$(hostname -s)"
  echo "gpu_pair=${GPU_PAIR}"
  echo "model=${MODEL_NAME}"
  echo "input_dir=${INPUT_DIR}"
  echo "source_list_portable=${SRC_LIST_PORTABLE}"
  echo "raw_glossary=${RAW_GLOSSARY}"
  echo "output_base=${OUTPUT_BASE}"
  echo "batch_dir=${BATCH_DIR}"
  echo "lang=de"
  echo "lm=2"
  echo "eval_mode=same_lm_batch_v1"
  echo "talks_per_lm=5"
  echo "disable_rag=1"
  echo "max_new_tokens=80"
  echo "strip_output_tags=term"
} | tee "${OUT_ROOT}/run_meta.txt"

ROOT_DIR_OVERRIDE="${ROOT_DIR}" \
RUN_TAG_OVERRIDE="${RUN_STAMP}_de_lm2" \
LANG_CODE_OVERRIDE="de" \
LMS_OVERRIDE="2" \
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE="${GPU_PAIR}" \
VLLM_TP_SIZE_OVERRIDE=2 \
MAX_NUM_SEQS_OVERRIDE=5 \
SCHEDULER_BATCH_SIZE_OVERRIDE=5 \
SCHEDULE_MODE_OVERRIDE=round_robin \
DISABLE_RAG_OVERRIDE=1 \
RAG_TOP_K_OVERRIDE=0 \
RAG_SCORE_THRESHOLD_OVERRIDE=0 \
RAG_BATCH_RETRIEVAL_OVERRIDE=0 \
MAX_NEW_TOKENS_OVERRIDE=80 \
MAX_NEW_TOKENS_POLICY_OVERRIDE=fixed \
TEMPERATURE_OVERRIDE=0.6 \
TOP_P_OVERRIDE=0.95 \
TOP_K_DECODE_OVERRIDE=20 \
MAX_CACHE_SECONDS_OVERRIDE=80 \
KEEP_CACHE_SECONDS_OVERRIDE=60 \
MIN_CACHE_CHUNKS_OVERRIDE=1 \
VLLM_LIMIT_AUDIO_OVERRIDE=128 \
VLLM_ENFORCE_EAGER_OVERRIDE=1 \
VLLM_ENABLE_PREFIX_CACHING=1 \
VLLM_MAX_MODEL_LEN_OVERRIDE=16384 \
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
VLLM_MOE_USE_DEEP_GEMM=0 \
VLLM_USE_FUSED_MOE_GROUPED_TOPK=0 \
GPU_MEMORY_UTILIZATION_OVERRIDE=0.80 \
MODEL_NAME_OVERRIDE="${MODEL_NAME}" \
SRC_LIST_OVERRIDE="${SRC_LIST_PORTABLE}" \
TGT_LIST_OVERRIDE="${INPUT_DIR}/target.list" \
SOURCE_TEXT_FILE_OVERRIDE="${INPUT_DIR}/source_text.txt" \
REF_FILE_OVERRIDE="${INPUT_DIR}/ref.txt" \
AUDIO_YAML_OVERRIDE="${INPUT_DIR}/audio.yaml" \
GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
EVAL_GLOSSARY_PATH_OVERRIDE="${RAW_GLOSSARY}" \
OUTPUT_BASE_OVERRIDE="${OUTPUT_BASE}" \
DENSITY_TAG_OVERRIDE="${DENSITY_TAG}" \
GLOSSARY_TAG_OVERRIDE="${GLOSSARY_TAG}" \
TERM_FCR_POLICY_OVERRIDE=source_ref_negative_sentence \
STRIP_OUTPUT_TAGS_OVERRIDE=term \
WANDB_LOG_OVERRIDE="${WANDB_LOG_OVERRIDE:-1}" \
WANDB_RUN_PREFIX_OVERRIDE="origin_norag" \
WANDB_EXPERIMENT_FAMILY_OVERRIDE="tagged_acl_origin_norag_batch_cmp" \
WANDB_VARIANT_PREFIX_OVERRIDE="origin_norag_de_lm2_batch" \
WANDB_COMPUTE_TAG_OVERRIDE="compute:aries01" \
WANDB_RUNTIME_GLOSSARY_LABEL_OVERRIDE="raw" \
WANDB_DATA_TAG_OVERRIDE="tagacl_raw_de" \
NOTES_FILE_OVERRIDE="${NOTES_FILE}" \
LOG_ROOT_OVERRIDE="${LOG_ROOT}/batch" \
EVAL_TMPDIR_OVERRIDE="${EVAL_TMPDIR}" \
bash "${BATCH_LAUNCHER}" \
  > "${LOG_ROOT}/launcher.out" \
  2> "${LOG_ROOT}/launcher.err"

python - "${OUTPUT_BASE}" <<'PY'
import csv
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
paths = sorted(output_base.glob("de/**/eval_results.tsv"))
if len(paths) != 1:
    raise SystemExit(f"expected one eval_results.tsv, found {len(paths)}: {paths}")
rows = list(csv.DictReader(paths[0].open("r", encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {paths[0]}, got {len(rows)}")
row = rows[0]
summary = output_base / "__summary__" / "summary_de_lm2.tsv"
summary.parent.mkdir(parents=True, exist_ok=True)
fields = [
    "method_key", "lang", "lm", "BLEU", "StreamLAAL", "StreamLAAL_CA",
    "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL", "REAL_TERM_ADOPT", "TERM_FCR",
    "eval_results", "instances_log",
]
with summary.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    w.writeheader()
    w.writerow({
        "method_key": "origin_norag",
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

date -u +%Y-%m-%dT%H:%M:%SZ > "${OUT_ROOT}/.success"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee "${LOG_ROOT}/gpu_postrun.csv"
echo "[ALL DONE] ${OUTPUT_BASE}/__summary__/summary_de_lm2.tsv"
